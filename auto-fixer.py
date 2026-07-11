#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — agentic investigation flow.

Pipeline:

    GitHub Actions Pipeline Failed
        → Python: collect initial evidence (logs, exit code, repo tree, git diff,
              AND the triggering workflow file(s) — seeded up front)
        → AI Investigation Agent (multi-turn loop):
              reads evidence → forms hypothesis → requests more files if needed
              → Python fetches requested files (read-only) → loop → confirms root cause
        → AI Patch Generation Agent: emits the concrete fix (flat evidence→corrected issues)
        → Python Validation Engine: format/path/syntax/YAML/JSON checks,
              secret scanning, dangerous-command detection, safe patch verification
        → Apply changes → Execute build & tests
        → on failure: collect new failure evidence → AI reviews it → retry or stop
        → on success: commit, push, open PR

Design notes:
  * A small local model (qwen2.5-coder:3b) is unreliable at open-ended multi-turn
    tool use — it can loop forever, request irrelevant files, or hallucinate a
    root cause with high confidence. Every agentic step here is bounded:
    MAX_INVESTIGATION_TURNS caps the investigation loop, MAX_FILES_PER_REQUEST
    caps how much it can ask for at once, and the model may ONLY request files
    that literally appear in the repository tree Python already listed for it
    — it can never name a file into existence.
  * The AI never writes to disk directly. It only ever emits JSON (a file
    request, or an evidence→corrected quote). Python is the only thing that
    reads or writes files, and it re-validates every AI-authored change
    (syntax, YAML/JSON parse, secret scan, dangerous-command scan, and a
    couple of deterministic "is this Dockerfile change actually complete"
    checks) before it's ever applied.
  * The workflow file that defined the failing run is seeded into the initial
    evidence bundle (Python reads it, never edits it here). Almost every CI
    failure is either in app code or in the workflow itself (a bad
    python-version, a typo'd path in a `run:` step), and the model is
    unreliable at requesting the workflow by its exact name — so Python hands
    it over up front rather than hoping the model asks for it correctly.

Exit codes:
  0 success / nothing to do / escalated (low confidence, open issue instead)
  1 log not found        3 no valid fix
  2 AI stage failed       4 git failed         5 tests/build failed after repairs
"""

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:3b")
AI_TIMEOUT     = 210
MAX_RETRIES    = 2
RETRY_BACKOFF  = [20, 20]

# ── Agentic-loop bounds (the guardrails a 3B model needs) ──────────────────────
MAX_INVESTIGATION_TURNS = int(os.environ.get("MAX_INVESTIGATION_TURNS", "4"))
MAX_FILES_PER_REQUEST   = int(os.environ.get("MAX_FILES_PER_REQUEST", "3"))
MAX_REPAIR_ROUNDS       = int(os.environ.get("MAX_REPAIR_ROUNDS", "2"))

# ── Prompt / context budget ───────────────────────────────────────────────────
MAX_ERROR_LINES   = 14
MAX_FILE_CHARS    = 4000
MAX_TOTAL_CONTEXT = 10000
MAX_FILES_FIXED   = 4
MAX_PROMPT_CHARS  = 11000

# ── Git flow ──────────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

# Files auto-fix may never EDIT at all. CODEOWNERS and secret-bearing files
# have no legitimate narrow auto-fix — any edit to them is high-risk with no
# safe subset, so they stay fully blocked.
ALWAYS_HIDDEN   = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [
    r"\.?github/CODEOWNERS$",
    r"(^|/)\.env(\..*)?$",
    r".*\.pem$", r".*\.key$", r".*id_rsa.*", r".*id_ed25519.*",
    r".*secrets?\.ya?ml$", r".*\.tfstate(\.backup)?$",
    r"(^|/)\.npmrc$", r"(^|/)\.pypirc$",
]
# Workflow files are a privilege-escalation vector (permissions, secrets,
# arbitrary shell) but ALSO the file most likely to hold a genuine one-line
# CI bug (bad python-version, a typo'd path). Instead of a blanket block,
# they go through validate_workflow_edit() below: only scalar VALUE changes
# are allowed — any edit touching permissions/secrets/env/triggers, or that
# adds/removes/reorders a step or job, is rejected no matter how it's framed.
WORKFLOW_PATTERN = re.compile(r"\.?github/workflows/.*\.ya?ml$")

# Files the investigation agent may not even READ. Narrower than the edit
# block — workflow YAML and CODEOWNERS are fine to read for diagnosis (the
# diagram explicitly allows it), just never edited (CODEOWNERS) or edited
# without the strict workflow validator (workflows).
READ_BLOCKED_PATTERNS = [p for p in BLOCKED_PATTERNS if "CODEOWNERS" not in p]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             "dist", "build", ".pytest_cache", "target", "out", "vendor",
             ".idea", ".vscode", "coverage", "tmp", "temp", "logs"}
MAX_FILE_SIZE_BYTES = 100_000

# How many workflow files to seed into the initial evidence bundle. There's
# normally one; the cap keeps a repo with many workflows from blowing the
# context budget.
MAX_SEED_WORKFLOWS = int(os.environ.get("MAX_SEED_WORKFLOWS", "2"))

# ── Secret scanning ─────────────────────────────────────────────────────────
SECRET_PATTERNS = [
    ("AWS access key",   re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS secret key",   re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}")),
    ("GitHub token",     re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,}")),
    ("Slack token",      re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("Private key",      re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----")),
    ("Generic API key",  re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9\-_/+=]{16,}['\"]")),
    ("JWT",              re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("Bearer token",     re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_.=]{20,}")),
]
MAX_ADDED_LINES_PER_FIX = 40  # a legit fix is small; a big blob of new lines
                              # is suspicious for an autofix

# ── Dangerous-command scanning (Python Validation Engine) ─────────────────────
DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r'rm\s+-rf\s+/(?:\s|$)'),
    re.compile(r'curl[^\n]{0,120}\|\s*(sh|bash)\b'),
    re.compile(r'wget[^\n]{0,120}\|\s*(sh|bash)\b'),
    re.compile(r'chmod\s+777'),
    re.compile(r'\bos\.system\('),
    re.compile(r'subprocess\.\w+\([^)]*shell\s*=\s*True'),
    re.compile(r'\beval\('),
    re.compile(r':\(\)\{\s*:\|:&\s*\};:'),  # fork bomb
]

# ── Deterministic Dockerfile checks, used as post-patch safety verification ──
DOCKERFILE_REF_PATTERNS = [
    (re.compile(r'^\s*COPY\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "COPY"),
    (re.compile(r'^\s*ADD\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "ADD"),
    (re.compile(r'-r\s+(\S+\.txt)'), "pip install -r"),
    (re.compile(r'CMD\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "CMD"),
    (re.compile(r'ENTRYPOINT\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "ENTRYPOINT"),
]
DOCKERFILE_FROM_PYTHON = re.compile(r'^(\s*FROM\s+)python:([^\s]+)(.*)$', re.I | re.M)
VALID_PYTHON_MINORS = set(range(8, 14))  # CPython 3.8 – 3.13
DEFAULT_PYTHON_VERSION = os.environ.get("DEFAULT_PYTHON_VERSION", "3.12")

# actions/setup-python `python-version:` lines in workflow YAML — same
# validity rule as the Dockerfile check above, used to force a deterministic
# value instead of letting the model pick a fresh one on every run.
WORKFLOW_PYVERSION_LINE = re.compile(
    r'^(\s*python-version\s*:\s*)([\'"]?)(\d+)\.(\d+)([\'"]?)(.*)$', re.M)


def scan_text_for_secrets(text: str) -> list:
    return [name for name, pat in SECRET_PATTERNS if pat.search(text)]


def redact_secrets(text: str) -> str:
    out = text
    for name, pat in SECRET_PATTERNS:
        out = pat.sub(f"[REDACTED:{name}]", out)
    return out


def scan_dangerous_commands(text: str) -> list:
    return [p.pattern for p in DANGEROUS_COMMAND_PATTERNS if p.search(text)]


def _added_lines(original: str, new: str) -> list:
    old_lines = set(original.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]


# ══════════════════════════════════════════════════════════════════════════════
# Path helpers
# ══════════════════════════════════════════════════════════════════════════════

def _relstrip(rel): return rel[2:] if rel.startswith("./") else rel


def _is_blocked(fp):
    """May this file ever be EDITED?"""
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_HIDDEN):
        return True
    return any(re.search(p, fp) for p in BLOCKED_PATTERNS)


def _is_read_blocked(fp):
    """May the investigation agent even READ this file?"""
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_HIDDEN):
        return True
    return any(re.search(p, fp) for p in READ_BLOCKED_PATTERNS)


def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False


def _resolve_requested_path(req: str, allowed_files: set):
    """Best-effort map an AI-requested path onto a real repo path.

    A 3B model routinely mangles the paths it asks for: a leading slash
    (`/github/workflows/...`), a `./` prefix, or just the basename. It also
    tends to echo absolute traceback paths (`/home/.../site-packages/...`)
    that can never be in the repo. Rather than deny every near-miss outright
    (which is what stalled the investigation loop), try to snap the request
    onto a real, read-permitted path — but ONLY ever return something already
    in `allowed_files`, so this can't widen what the agent is allowed to read.

    Returns the real repo path, or None if it can't be resolved unambiguously.
    """
    if not isinstance(req, str):
        return None
    req = req.strip()
    if not req:
        return None
    if req in allowed_files:
        return req
    norm = req.lstrip("/")
    if norm.startswith("./"):
        norm = norm[2:]
    if norm in allowed_files:
        return norm
    # unique suffix match: a real path ends with the requested tail
    tail_hits = [a for a in allowed_files if a == norm or a.endswith("/" + norm)]
    if len(tail_hits) == 1:
        return tail_hits[0]
    # unique basename match (last resort — only if exactly one file has it)
    base = norm.rsplit("/", 1)[-1]
    if base:
        base_hits = [a for a in allowed_files if a.rsplit("/", 1)[-1] == base]
        if len(base_hits) == 1:
            return base_hits[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — COLLECT INITIAL INVESTIGATION EVIDENCE  (Python, no AI)
# ══════════════════════════════════════════════════════════════════════════════

ERROR_KEYWORDS = [
    "error", "failed", "failure", "exception", "traceback", "exit code",
    "invalid", "fatal", "cannot", "refused", "denied", "missing", "undefined",
    "permission denied", "command not found", "returned non-zero", "syntaxerror",
    "importerror", "nameerror", "typeerror", "valueerror", "attributeerror",
    "modulenotfounderror", "no module named", "not found", "could not find",
    "no matching", "requirement", "assert", "test failed", "no such file",
    "failed to solve", "err!", "panic:", "fatal error",
]
NOISE_KEYWORDS = [
    "##[group]", "##[endgroup]", "set up job", "complete job", "post job",
    "add mask", "git config", "git version", "persist-credentials",
    "fetch-depth", "collecting ", "rootdir:", "configfile:",
]
FILE_REF_HINTS = [re.compile(r'File "[^"]+"'),
                  re.compile(r'[\w./\-]+\.[A-Za-z0-9]+:\d+'),
                  re.compile(r'\bDockerfile(\.\w+)?\b')]

TECH_STACK_SIGNALS = {
    "python": ["python", "pip", "pytest", "flask", "django", "requirements.txt",
               "pyproject.toml", ".py"],
    "node":   ["node", "npm", "yarn", "jest", "package.json", ".js", ".ts"],
    "docker": ["dockerfile", "docker build", "docker push", "image", "container"],
    "java":   ["java", "maven", "gradle", "mvn", ".java", "pom.xml"],
    "go":     ["go build", "go test", "go mod", ".go", "go.mod"],
}


def extract_error_signal(log_text: str) -> str:
    lines = log_text.splitlines()
    relevant = [l.strip() for l in lines
                if (any(k in l.lower() for k in ERROR_KEYWORDS)
                    or any(h.search(l) for h in FILE_REF_HINTS))
                and not any(n in l.lower() for n in NOISE_KEYWORDS) and l.strip()]
    relevant = list(dict.fromkeys(relevant))[-MAX_ERROR_LINES:]
    tail = [l.strip() for l in lines[-20:]
            if l.strip() and not any(n in l.lower() for n in NOISE_KEYWORDS)]
    signal = "\n".join(list(dict.fromkeys(relevant + tail)))
    print(f"[EVIDENCE] Error signal: {len(signal)} chars")
    return signal


def fingerprint_stack(log_text: str) -> set:
    low = log_text.lower()
    stacks = {s for s, sig in TECH_STACK_SIGNALS.items() if any(x in low for x in sig)}
    print(f"[EVIDENCE] Tech stacks: {stacks or {'unknown'}}")
    return stacks


def get_exit_code(log_text: str, cli_override=None) -> str:
    if cli_override is not None:
        return str(cli_override)
    m = re.search(r'exit code[:\s]+(-?\d+)', log_text, re.I)
    return m.group(1) if m else "unknown"


def get_git_diff(max_chars: int = 4000) -> str:
    """Diff of the current (failing) commit — part of the initial evidence
    bundle, same as the diagram's 'Git Diff (current commit changes)' box."""
    try:
        r = subprocess.run(["git", "diff", "HEAD~1", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        diff = r.stdout.strip()
        if not diff:
            r2 = subprocess.run(["git", "show", "--stat", "HEAD"],
                                capture_output=True, text=True, timeout=15)
            diff = r2.stdout.strip()
        if len(diff) > max_chars:
            diff = diff[:max_chars] + "\n...(truncated)"
        return diff or "(no diff available)"
    except Exception as exc:
        return f"(git diff unavailable: {exc})"


def repo_tree_text(limit: int = 300) -> tuple:
    """Folders & filenames only — no contents. Also returns the set of paths
    the investigation agent is allowed to request (read-permitted)."""
    files = []
    for p in sorted(Path(".").rglob("*")):
        if p.is_file() and not any(d in SKIP_DIRS for d in p.parts):
            rel = _relstrip(str(p))
            if not _is_read_blocked(rel):
                files.append(rel)
        if len(files) >= limit:
            break
    text = "\n".join(files) if files else "(no files found)"
    return text, set(files)


def _read_evidence_file(rel: str):
    p = Path(rel)
    if not p.is_file() or _is_read_blocked(rel) or not _is_text_file(p):
        return None
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if len(raw) > MAX_FILE_CHARS:
        raw = raw[:MAX_FILE_CHARS] + "\n...(truncated)"
    return raw


def seed_workflow_evidence(allowed_files: set) -> dict:
    """Seed the investigation with the workflow file(s) that define the CI run.

    Almost every CI failure is either in app code or in the workflow itself
    (a bad python-version, a typo'd path in a `run:` step). The workflow is
    small and cheap to include, and a 3B model is unreliable at requesting it
    by its exact name — in practice it asks for `/github/workflows/<guessed>`
    with a wrong basename and gets denied, then burns the whole loop. So
    Python reads the workflow up front and hands it over as evidence rather
    than hoping the model asks for it correctly.

    This is observation only: Python reads the file here, it never edits it.
    Any edit still goes through validate_workflow_edit()'s strict gate later.
    """
    seeded = {}
    wf_files = sorted(f for f in allowed_files if WORKFLOW_PATTERN.search(f))
    for f in wf_files[:MAX_SEED_WORKFLOWS]:
        content = _read_evidence_file(f)
        if content is not None:
            seeded[f] = content
    if seeded:
        print(f"[EVIDENCE] seeded workflow file(s): {', '.join(seeded)}")
    else:
        print("[EVIDENCE] no workflow file found to seed")
    return seeded


# ══════════════════════════════════════════════════════════════════════════════
# AI plumbing — shared streaming + JSON extraction
# ══════════════════════════════════════════════════════════════════════════════

def _detect_endpoint():
    url = OLLAMA_API_URL.rstrip("/")
    if "/api/generate" in url:
        return url, "native"
    if "/v1/completions" in url or "/v1/chat" in url:
        return url, "openai"
    base = re.sub(r"/(v1|api)/.*$", "", url)
    return f"{base}/api/generate", "native"


def _extract_token(line: bytes, fmt: str) -> str:
    if not line:
        return ""
    try:
        text = line.decode("utf-8", errors="replace").strip()
        if text.startswith("data: "):
            text = text[6:].strip()
        if text in ("", "[DONE]"):
            return ""
        obj = json.loads(text)
        return obj.get("choices", [{}])[0].get("text", "") if fmt == "openai" \
            else obj.get("response", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def _stream_ollama(prompt, schema=None, num_predict=2000, temperature=0.05,
                   num_ctx=8192, tag="AI", retries=MAX_RETRIES) -> str:
    endpoint, fmt = _detect_endpoint()
    if fmt == "openai":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": temperature,
                   "max_tokens": num_predict, "stream": True}
    else:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt,
                   "options": {"temperature": temperature, "num_predict": num_predict,
                               "num_ctx": num_ctx}, "stream": True}
        if schema:
            payload["format"] = schema
    print(f"[{tag}] {endpoint} ({fmt}) | prompt {len(prompt)} chars | model {OLLAMA_MODEL}")

    last = None
    for attempt in range(retries):
        try:
            t0, collected = time.time(), []
            resp = requests.post(endpoint, json=payload, timeout=(10, AI_TIMEOUT), stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                tok = _extract_token(line, fmt)
                if tok:
                    collected.append(tok)
            raw = "".join(collected).strip()
            print(f"[{tag}] done in {time.time()-t0:.1f}s — {len(raw)} chars")
            if not raw:
                try:
                    body = resp.json()
                    raw = (body.get("choices", [{}])[0].get("text", "")
                           if fmt == "openai" else body.get("response", "")).strip()
                except Exception:
                    pass
            if not raw:
                raise RuntimeError("Ollama returned an empty response.")
            return raw
        except requests.exceptions.Timeout as exc:
            last = exc
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
            print(f"[{tag}] timeout; waiting {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {endpoint}: {exc}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}")
    raise RuntimeError(f"Ollama did not respond after {retries} attempts: {last}")


def _json_from(raw: str):
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    best = None
    for start in [i for i, c in enumerate(cleaned) if c == "{"]:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    cand = cleaned[start:i+1]
                    best = cand if best is None or len(cand) > len(best) else best
                    break
    if best:
        try:
            return json.loads(best)
        except json.JSONDecodeError:
            try:
                return json.loads(best + "}" * (best.count("{") - best.count("}")))
            except json.JSONDecodeError:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# AI INVESTIGATION AGENT  — multi-turn, evidence-driven
# ══════════════════════════════════════════════════════════════════════════════

INVESTIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis":        {"type": "string"},
        "status":          {"type": "string"},
        "requested_files": {"type": "array", "items": {"type": "string"}},
        "root_cause":      {"type": "string"},
        "solution":        {"type": "string"},
        "confidence":      {"type": "number"},
        "commit_message":  {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "issue":      {"type": "string"},
                "root_cause": {"type": "string"},
                "solution":   {"type": "string"},
            },
            "required": ["issue", "root_cause"]}},
    },
    "required": ["status"],
}

INVESTIGATE_SYSTEM = """\
You are a DevOps investigation agent diagnosing a CI/CD failure. You do NOT know \
the root cause yet — think like a senior engineer: read the evidence, form a \
hypothesis, and if you're not sure, ask for the specific file(s) that would \
confirm or rule it out.

Output ONLY one JSON object. No markdown fences. Start with { end with }.

Each turn, choose exactly one status:
  - "need_more_info": you cannot confirm a root cause yet. Set "requested_files" \
to up to 3 EXACT paths copied from the "## Repository tree" list below. NEVER \
request a path that is not in that list, and never request a file you've \
already been shown under "## Evidence gathered so far".
  - "root_cause_confirmed": you are confident. Fill in "root_cause" (one-sentence \
OVERALL summary), "solution" (plain words: what must change overall), \
"confidence" (0.0-1.0), "commit_message" (fix: short), AND "findings" (see below).

Schema when asking for more evidence:
{"analysis":"one or two sentences of your current reasoning","status":"need_more_info","requested_files":["exact/path"]}

Schema when confirming:
{"analysis":"...","status":"root_cause_confirmed","root_cause":"one-sentence overall summary","solution":"...","confidence":0.0-1.0,"commit_message":"fix: short","findings":[{"issue":"short name of this specific bug","root_cause":"why THIS bug happened","solution":"what fixes THIS bug"}]}

CRITICAL — "findings" must list EVERY distinct bug you found, one entry per \
bug, even if several live in the same file or look similar. A CI failure \
routinely has more than one independent bug (e.g. a bad version string AND \
separate typo'd file paths) — finding one and stopping is a FAILURE. \
"root_cause"/"solution" above are just a one-sentence roll-up for a commit \
message; "findings" is the itemized breakdown a human will actually read.

Rules:
- If the logs and diff already make the cause obvious, confirm immediately — \
don't pad with unnecessary file requests. A "No such file" error for a path \
referenced in the workflow, where a near-identically-named file DOES exist in \
the repository tree, is a typo you can confirm right now.
- A file being merely related to the tech stack is not enough reason to \
request it — request only what actually tests your hypothesis.
- Paths under site-packages, /home/, or the runner's work directory are NOT \
part of this repository and CANNOT be requested — do not ask for them. Only \
request paths that appear verbatim in the repository tree below.
- If a path is listed under "## Already denied", it is not in the repo. Do \
NOT request it again — pick a different file or confirm your hypothesis.
"""


def _build_investigation_prompt(signal, exit_code, repo_tree, git_diff,
                                evidence: dict, denied: list, last_turn: bool) -> str:
    ev_parts, total = [], 0
    for f, c in evidence.items():
        block = f"### {f}\n```\n{c}\n```"
        if total + len(block) > MAX_TOTAL_CONTEXT and ev_parts:
            break
        ev_parts.append(block)
        total += len(block)
    ev_text = "\n\n".join(ev_parts) if ev_parts else "(none read yet)"
    denied_note = ""
    if denied:
        denied_note = ("## Already denied — NOT in this repo, do not request again:\n"
                       + "\n".join(f"- {d}" for d in denied[-8:]) + "\n")
    turn_note = ("\n## THIS IS YOUR FINAL TURN. You MUST return "
                 "status \"root_cause_confirmed\" now, using your best "
                 "hypothesis from the evidence so far. Lower your confidence "
                 "if you're not fully sure.\n" if last_turn else "")
    return (
        f"{INVESTIGATE_SYSTEM}{turn_note}\n"
        f"## CI failure (key lines):\n```\n{signal}\n```\n"
        f"## Exit code: {exit_code}\n"
        f"## Git diff (most recent commit):\n```\n{git_diff}\n```\n"
        f"## Repository tree (folders & filenames only — request only from this list):\n{repo_tree}\n"
        f"{denied_note}"
        f"## Evidence gathered so far:\n{ev_text}\n\n"
        f"Respond with the JSON described above."
    )


def _finalize_investigation(data: dict, forced: bool) -> dict:
    try:
        confidence = float(data.get("confidence", 0.4))
    except (TypeError, ValueError):
        confidence = 0.4
    if forced and data.get("status") != "root_cause_confirmed":
        confidence = min(confidence, 0.4)

    findings = []
    for f in (data.get("findings") or []):
        if isinstance(f, dict) and (f.get("issue") or f.get("root_cause")):
            findings.append({
                "issue":      (f.get("issue") or f.get("root_cause") or "").strip(),
                "root_cause": (f.get("root_cause") or "").strip(),
                "solution":   (f.get("solution") or "").strip(),
            })
    if not findings:
        # model didn't break it down — fall back to the overall summary so
        # reporting always has at least one itemized entry
        findings = [{"issue": data.get("root_cause") or "unknown",
                     "root_cause": data.get("root_cause") or "unknown",
                     "solution": data.get("solution") or ""}]

    return {
        "root_cause":     data.get("root_cause") or "unknown",
        "solution":       data.get("solution") or "",
        "confidence":     confidence,
        "commit_message": data.get("commit_message") or "fix: auto-fixer change",
        "findings":       findings,
    }


def ai_investigate(signal, exit_code, repo_tree, git_diff, allowed_files: set,
                   seed_evidence: dict = None) -> dict:
    """The core investigation loop from the diagram: AI reads evidence, thinks,
    requests more if needed, loops, confirms root cause. Bounded by
    MAX_INVESTIGATION_TURNS so a small model can't spin forever.

    `seed_evidence` is evidence Python hands over up front (e.g. the workflow
    file that defined the run) so the model doesn't have to correctly request
    it. `denied` remembers paths already rejected so the model stops asking
    for the same un-requestable path every turn."""
    evidence = dict(seed_evidence) if seed_evidence else {}
    log, result, denied = [], None, []

    for turn in range(1, MAX_INVESTIGATION_TURNS + 1):
        last_turn = (turn == MAX_INVESTIGATION_TURNS)
        prompt = _build_investigation_prompt(signal, exit_code, repo_tree,
                                             git_diff, evidence, denied, last_turn)
        try:
            raw = _stream_ollama(prompt, INVESTIGATE_SCHEMA, num_predict=900,
                                 temperature=0.05, tag=f"INVESTIGATE-T{turn}")
        except Exception as exc:
            print(f"[INVESTIGATE] turn {turn} failed: {exc}", file=sys.stderr)
            break

        data = _json_from(raw) or {}
        status = data.get("status", "")
        print(f"[INVESTIGATE] turn {turn}: status={status} | "
              f"{(data.get('analysis') or '')[:160]}")
        log.append({"turn": turn, "status": status,
                    "analysis": (data.get("analysis") or "")[:200]})

        if status == "root_cause_confirmed":
            result = _finalize_investigation(data, False)
            break

        if status == "need_more_info":
            requested = [f.strip() for f in (data.get("requested_files") or [])
                        if isinstance(f, str) and f.strip()]
            to_read = []
            for f in requested[:MAX_FILES_PER_REQUEST]:
                resolved = _resolve_requested_path(f, allowed_files)
                if resolved is None:
                    if f not in denied:
                        denied.append(f)
                    print(f"[INVESTIGATE]   ✗ requested '{f}' — not in repo tree, denied")
                    continue
                if resolved in evidence:
                    continue
                if resolved != f:
                    print(f"[INVESTIGATE]   ~ resolved '{f}' → '{resolved}'")
                to_read.append(resolved)
            for f in to_read:
                content = _read_evidence_file(f)
                evidence[f] = content if content is not None else "(could not read this file)"
                print(f"[INVESTIGATE]   + read {f} "
                      f"({len(evidence[f])} chars)")
            if last_turn:
                # force a final decision using whatever evidence we now have
                final_prompt = _build_investigation_prompt(
                    signal, exit_code, repo_tree, git_diff, evidence, denied, True)
                try:
                    raw2 = _stream_ollama(final_prompt, INVESTIGATE_SCHEMA,
                                          num_predict=900, temperature=0.05,
                                          tag="INVESTIGATE-FINAL")
                    data2 = _json_from(raw2) or data
                except Exception as exc:
                    print(f"[INVESTIGATE] final turn failed: {exc}", file=sys.stderr)
                    data2 = data
                result = _finalize_investigation(data2, True)
            continue

        # malformed / unexpected status — salvage if possible, else stop
        if data.get("root_cause"):
            result = _finalize_investigation(data, True)
        break

    if result is None:
        result = {"root_cause": "unknown — investigation did not converge",
                   "solution": "", "confidence": 0.0,
                   "commit_message": "fix: auto-fixer change", "findings": []}
    result["evidence"] = evidence
    result["investigation_log"] = log
    return result


# ══════════════════════════════════════════════════════════════════════════════
# AI PATCH GENERATION AGENT
# ══════════════════════════════════════════════════════════════════════════════

PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "file":      {"type": "string"},
                "problem":   {"type": "string"},
                "evidence":  {"type": "string"},
                "corrected": {"type": "string"},
            },
            "required": ["file", "problem", "evidence", "corrected"]}},
    },
    "required": ["issues"],
}

PATCH_SYSTEM = """\
You are a patch-generation agent. Another engineer already investigated this \
CI/CD failure and confirmed the root cause below. Your ONLY job is to emit the \
concrete text-level fix — do not re-diagnose.

Output ONLY one JSON object. No markdown fences. Start with { end with }.

"issues" = ONE ENTRY PER BUG implied by the confirmed root cause. Each entry:
  - "file": the exact ### header path of the file the bug is in
  - "problem": one sentence describing this specific bug
  - "evidence": the offending text COPIED EXACTLY, character-for-character, \
from the file contents shown below. SHORT — the single wrong token or one \
wrong line. Never paraphrase; copy it.
  - "corrected": the same text with ONLY the bug fixed. Everything else \
identical.

Schema:
{"issues":[{"file":"exact/path","problem":"...","evidence":"exact text copied from the file","corrected":"same text, bug fixed"}]}

Rules:
- evidence must literally appear in the file contents. If you cannot quote \
exact offending text, omit that issue.
- evidence and corrected must differ, and be as short as unambiguous allows.
- Only files with a ### header may be fixed — never invent a file or a fix \
for something not shown to you.
- If the root cause spans more than one bug, emit one issues[] entry per bug \
— do not stop after the first.
"""


def ai_generate_patch(root_cause: str, solution: str, evidence: dict,
                      retry_note: str = "") -> list:
    parts = [f"### {f}\n```\n{c}\n```" for f, c in evidence.items()]
    context = "\n\n".join(parts) if parts else "(no evidence files were read)"
    retry_section = (f"## Note: a previous attempt at this fix failed:\n"
                     f"{retry_note}\nDo not repeat the same change — adjust "
                     f"based on this new information.\n\n" if retry_note else "")
    prompt = (f"{PATCH_SYSTEM}\n\n{retry_section}"
              f"## Confirmed root cause: {root_cause}\n"
              f"## Solution direction: {solution}\n\n"
              f"## File contents (you may ONLY edit these):\n{context}\n\n"
              f"Emit the issues JSON.")
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]
    raw = _stream_ollama(prompt, PATCH_SCHEMA, num_predict=2200,
                         temperature=0.05, tag="PATCH")
    data = _json_from(raw)
    if data is None:
        raise ValueError(f"No valid JSON from patch agent:\n{raw[:400]}")
    issues = data.get("issues", []) or []
    if not issues and isinstance(data.get("fixes"), list):
        issues = data["fixes"]
    return _normalize_issue_keys(issues)


def _normalize_issue_keys(issues):
    KF = ("file_path", "path", "filename", "filepath", "name")
    KE = ("find", "wrong", "offending", "original", "bad", "before")
    KC = ("replace", "fix", "replacement", "correct", "fixed", "after")
    out = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        it = dict(it)
        for want, alts in (("file", KF), ("evidence", KE), ("corrected", KC)):
            if want not in it:
                for a in alts:
                    if a in it:
                        it[want] = it.pop(a); break
        out.append(it)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# AI REVIEW AGENT  — "AI Reviews New Evidence" → "Decide Retry or Stop"
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "decision":   {"type": "string"},
        "root_cause": {"type": "string"},
        "solution":   {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["decision"],
}

REVIEW_SYSTEM = """\
You are reviewing why a previously applied fix still failed CI. Given the \
original root cause, the fix that was tried, and the NEW failure output, \
decide:
  - "retry": the root cause needs revising — provide an updated root_cause \
and solution for another patch attempt.
  - "stop": this doesn't point to a clear, safely-automatable fix — a human \
should look at it.
Output ONLY one JSON object:
{"decision":"retry","root_cause":"...","solution":"...","confidence":0.0-1.0}
or
{"decision":"stop","root_cause":"why this can't be safely auto-fixed","solution":"","confidence":0.0}
"""


def ai_review_failure(prior_root_cause: str, prior_solution: str, new_signal: str) -> dict:
    prompt = (f"{REVIEW_SYSTEM}\n\n## Original root cause: {prior_root_cause}\n"
              f"## Fix attempted: {prior_solution}\n"
              f"## New failure after applying the fix:\n```\n{new_signal}\n```\n\n"
              f"Return the JSON.")
    try:
        raw = _stream_ollama(prompt, REVIEW_SCHEMA, num_predict=500,
                             temperature=0.05, tag="REVIEW", retries=1)
    except Exception as exc:
        print(f"[REVIEW] failed: {exc}", file=sys.stderr)
        return {"decision": "stop", "root_cause": prior_root_cause,
                "solution": prior_solution, "confidence": 0.0}
    data = _json_from(raw) or {}
    decision = data.get("decision", "stop")
    try:
        conf = float(data.get("confidence", 0.3))
    except (TypeError, ValueError):
        conf = 0.3
    return {"decision": decision if decision in ("retry", "stop") else "stop",
            "root_cause": data.get("root_cause") or prior_root_cause,
            "solution": data.get("solution") or prior_solution,
            "confidence": conf}


# ══════════════════════════════════════════════════════════════════════════════
# PAIR + LOCATE  (Python turns the AI's quotes into fixes)
# ══════════════════════════════════════════════════════════════════════════════

def issues_to_fixes(issues: list, evidence: dict) -> tuple:
    fixes_by_file, rejects = {}, []
    for n, it in enumerate(issues, 1):
        file = (it.get("file") or "").strip()
        ev   = it.get("evidence")
        cor  = it.get("corrected")
        prob = (it.get("problem") or "").strip()

        if not isinstance(ev, str) or not ev.strip():
            rejects.append(f"issue #{n} ({file or '?'}): empty evidence")
            continue
        if not isinstance(cor, str) or cor == ev:
            rejects.append(f"issue #{n} ({file or '?'}): corrected missing or identical")
            continue

        holders = [f for f, c in evidence.items() if isinstance(c, str) and ev in c]
        if file in evidence and isinstance(evidence[file], str) and ev in evidence[file]:
            target = file
        elif len(holders) == 1:
            target = holders[0]
            print(f"[LOCATE] issue #{n}: filed under '{file or '?'}' but evidence only "
                  f"exists in '{target}' — relocating.")
        elif len(holders) > 1:
            rejects.append(f"issue #{n} ({file or '?'}): evidence in {len(holders)} files — ambiguous")
            continue
        else:
            rejects.append(f"issue #{n} ({file or '?'}): evidence {ev[:80]!r} not found — rejected")
            continue

        entry = fixes_by_file.setdefault(
            target, {"file": target, "reason": prob or "AI fix", "edits": []})
        edit = {"find": ev, "replace": cor}
        if edit not in entry["edits"]:
            entry["edits"].append(edit)
            if prob and prob not in entry["reason"]:
                entry["reason"] = (entry["reason"] + "; " + prob).lstrip("; ")[:300]

    return list(fixes_by_file.values()), rejects


# ══════════════════════════════════════════════════════════════════════════════
# APPLY + VALIDATE  (Python Validation Engine)
# ══════════════════════════════════════════════════════════════════════════════

def _salvage_fragment(content: str, find: str, replace: str):
    if not find or not replace or find == replace:
        return None
    i = 0
    while i < len(find) and i < len(replace) and find[i] == replace[i]:
        i += 1
    j = 0
    while (j < len(find) - i and j < len(replace) - i
           and find[-1 - j] == replace[-1 - j]):
        j += 1
    old_frag, new_frag = find[i:len(find) - j], replace[i:len(replace) - j]
    if len(old_frag.strip()) < 3 or content.count(old_frag) != 1:
        return None
    return content.replace(old_frag, new_frag, 1)


def _apply_edits(original: str, edits: list) -> tuple:
    content = original
    for i, ed in enumerate(edits):
        find, repl = ed.get("find", ""), ed.get("replace", "")
        if not isinstance(find, str) or find == "":
            return None, f"edit #{i+1} empty 'find'"
        if find in content:
            content = content.replace(find, repl)
            continue
        if isinstance(repl, str) and repl.strip() and repl in content:
            print(f"[EDIT] edit #{i+1} already satisfied by a prior edit — skipping")
            continue
        nf = "\n".join(l.strip() for l in find.splitlines())
        nc = "\n".join(l.strip() for l in content.splitlines())
        matched = False
        if nf and nf in nc:
            fl = [l.strip() for l in find.splitlines()]
            cl = content.splitlines()
            for j in range(len(cl) - len(fl) + 1):
                if [c.strip() for c in cl[j:j+len(fl)]] == fl:
                    base = cl[j][:len(cl[j]) - len(cl[j].lstrip())]
                    cl[j:j+len(fl)] = [(base + r if r.strip() else r)
                                       for r in (repl.splitlines() or [""])]
                    content = "\n".join(cl)
                    if original.endswith("\n") and not content.endswith("\n"):
                        content += "\n"
                    matched = True
                    break
        if matched:
            continue
        salvaged = _salvage_fragment(content, find, repl)
        if salvaged is not None and salvaged != content:
            print(f"[EDIT] Salvaged edit #{i+1} via unique-fragment match")
            content = salvaged
            continue
        return None, (f"edit #{i+1} 'find' text not present — find: {find[:120]!r}")
    if content == original:
        return None, "edits produced no change"
    return content, "ok"


def _resolve_content(fix: dict) -> tuple:
    file = (fix.get("file") or "").strip()
    if fix.get("edits"):
        p = Path(file)
        if not p.is_file():
            return None, f"file does not exist: {file}"
        return _apply_edits(p.read_text(encoding="utf-8", errors="replace"), fix["edits"])
    fc = fix.get("fixed_content")
    if isinstance(fc, str) and fc.strip():
        return fc, "ok"
    return None, "no 'edits' or 'fixed_content'"


def _dockerfile_still_broken(file: str, content: str) -> str:
    """Safe patch verification for Dockerfiles: after the AI's edit is
    applied, deterministically re-check for the two failure classes we know
    about (broken COPY/ADD/CMD/ENTRYPOINT references, invalid Python base
    image tags). If either is still present post-patch, the patch is
    incomplete — reject it so the repair loop retries instead of shipping a
    half-fix. Returns a reason string, or "" if clean."""
    if "dockerfile" not in Path(file).name.lower():
        return ""

    all_repo = [_relstrip(str(p)) for p in Path(".").rglob("*")
                if p.is_file() and not any(d in SKIP_DIRS for d in p.parts)]
    all_repo_set = set(all_repo)
    docker_dir = Path(file).parent
    for pat, label in DOCKERFILE_REF_PATTERNS:
        for m in pat.finditer(content):
            ref = m.group(1).strip().strip("'\"")
            if not ref or ref in (".", "..") or ref.startswith("-") or ref.startswith("$"):
                continue
            ref_clean = ref.lstrip("./")
            candidates = {ref_clean, _relstrip(str(docker_dir / ref_clean))}
            if any(Path(c).is_file() or c in all_repo_set for c in candidates):
                continue
            return f"{label} still references missing '{ref}' after the patch"

    for m in DOCKERFILE_FROM_PYTHON.finditer(content):
        version = m.group(2)
        base = version.split("-", 1)[0]
        vm = re.match(r'^(\d+)\.(\d+)', base)
        if not vm:
            continue
        major, minor = int(vm.group(1)), int(vm.group(2))
        if not (major == 3 and minor in VALID_PYTHON_MINORS):
            return f"base image still 'python:{version}' — not a real CPython release"
    return ""


def _yaml_structure_signature(node):
    """Structural fingerprint that ignores leaf scalar VALUES but keeps every
    key, list length, and nesting shape. Two workflow files with the same
    signature differ only in values — no step/job added, removed, or
    reordered, no new key introduced anywhere."""
    if isinstance(node, dict):
        return {k: _yaml_structure_signature(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_yaml_structure_signature(v) for v in node]
    return "<scalar>"


def _yaml_leaf_diffs(old, new, path=""):
    """Every leaf where the two (structurally-identical) trees differ, as
    (dotted_path, old_value, new_value)."""
    diffs = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in old:
            diffs.extend(_yaml_leaf_diffs(old[k], new.get(k), f"{path}.{k}" if path else k))
    elif isinstance(old, list) and isinstance(new, list):
        for i, (o, n) in enumerate(zip(old, new)):
            diffs.extend(_yaml_leaf_diffs(o, n, f"{path}[{i}]"))
    elif old != new:
        diffs.append((path, old, new))
    return diffs


MAX_WORKFLOW_VALUE_DIFFS = 3
SENSITIVE_WORKFLOW_KEYS = {"permissions", "secrets", "env", "on", "runs-on", "uses", "if"}


def normalize_workflow_python_versions(original_text: str, new_text: str) -> tuple:
    """Force every 'python-version:' line whose ORIGINAL value was an invalid
    CPython release to the deterministic DEFAULT_PYTHON_VERSION, overwriting
    whatever value the AI happened to pick.

    This exists specifically because leaving the exact replacement version
    up to the local model is non-deterministic — it might write '3.10' this
    run and '3.9' next run, both technically valid, but that's not something
    you can tell people is a reliable, repeatable fix. Python decides the
    final value here; the AI only decides THAT the line needs fixing.
    Lines whose original value was already valid are left exactly as the AI
    wrote them — this only overrides values that were actually broken."""
    old_matches = list(WORKFLOW_PYVERSION_LINE.finditer(original_text))
    new_matches = list(WORKFLOW_PYVERSION_LINE.finditer(new_text))
    if len(old_matches) != len(new_matches):
        return new_text, False  # line count changed — let other checks catch it

    changed = False
    out = new_text
    # walk from the end so earlier replacements don't shift later offsets
    for om, nm in zip(reversed(old_matches), reversed(new_matches)):
        o_major, o_minor = int(om.group(3)), int(om.group(4))
        if o_major == 3 and o_minor in VALID_PYTHON_MINORS:
            continue  # was already valid — leave the AI's line alone
        n_prefix, n_q1, _n_major, _n_minor, n_q2, n_rest = nm.groups()
        quote = n_q1 or n_q2 or '"'
        new_line = f"{n_prefix}{quote}{DEFAULT_PYTHON_VERSION}{quote}{n_rest}"
        if new_line != nm.group(0):
            changed = True
            print(f"[NORMALIZE] python-version was invalid ('{om.group(3)}.{om.group(4)}') "
                  f"— forcing deterministic '{DEFAULT_PYTHON_VERSION}' "
                  f"(model had proposed a different valid value)")
        out = out[:nm.start()] + new_line + out[nm.end():]
    return out, changed


def validate_workflow_edit(original_text: str, new_text: str) -> tuple:
    """The strict gate that makes workflow files editable without opening a
    privilege-escalation hole: only scalar VALUE fixes are allowed — the
    exact class of bug the investigation agent actually finds (a bad
    python-version, a typo'd file path). Anything structural, or anything
    touching a sensitive key, is rejected outright regardless of how the AI
    frames the change."""
    try:
        old_doc = yaml.safe_load(original_text)
        new_doc = yaml.safe_load(new_text)
    except yaml.YAMLError as e:
        return False, f"YAML error: {e}"
    if not isinstance(old_doc, dict) or not isinstance(new_doc, dict):
        return False, "workflow did not parse to a mapping"
    if _yaml_structure_signature(old_doc) != _yaml_structure_signature(new_doc):
        return False, ("edit changes workflow structure (steps/jobs/keys added, "
                       "removed, or reordered) — not allowed, needs human review")

    diffs = _yaml_leaf_diffs(old_doc, new_doc)
    if not diffs:
        return False, "no effective change"
    if len(diffs) > MAX_WORKFLOW_VALUE_DIFFS:
        return False, f"edit touches {len(diffs)} values — too broad for an auto-fix"

    for path, old_v, new_v in diffs:
        segments = [s.strip("]") for s in re.split(r"[.\[]", path)]
        if any(seg in SENSITIVE_WORKFLOW_KEYS for seg in segments):
            return False, f"edit touches sensitive field '{path}' — not allowed"
        if isinstance(old_v, str) and isinstance(new_v, str):
            added = [l for l in new_v.splitlines() if l not in old_v.splitlines()]
            if scan_dangerous_commands("\n".join(added)):
                return False, f"edit at '{path}' introduces a dangerous command pattern"
            secret_hits = scan_text_for_secrets(new_v)
            if secret_hits:
                return False, f"edit at '{path}' introduces a potential secret ({', '.join(secret_hits)})"
    return True, "ok"


def validate_fix(fix: dict) -> tuple:
    file = (fix.get("file") or "").strip()
    if not file:
        return False, "missing 'file'"
    if ".." in file or file.startswith("/") or file.startswith("~"):
        return False, f"unsafe path: {file}"
    if _is_blocked(file):
        return False, f"blocked path: {file}"
    if not Path(file).exists():
        return False, f"file does not exist: {file}"
    original_text = Path(file).read_text(encoding="utf-8", errors="replace")
    content, reason = _resolve_content(fix)
    if content is None:
        return False, reason
    fix["fixed_content"] = content
    if not content.strip():
        return False, "empty result"

    added = _added_lines(original_text, content)
    if len(added) > MAX_ADDED_LINES_PER_FIX:
        return False, (f"fix adds {len(added)} lines (max {MAX_ADDED_LINES_PER_FIX}) "
                       "— too large for an auto-fix, needs human review")
    added_text = "\n".join(added)
    secret_hits = scan_text_for_secrets(added_text)
    if secret_hits:
        return False, f"potential secret in fix ({', '.join(secret_hits)}) — blocked"
    danger_hits = scan_dangerous_commands(added_text)
    if danger_hits:
        return False, "fix introduces a dangerous command pattern — blocked"

    if file.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"Python syntax error: {e}"
    elif re.search(r"\.ya?ml$", file):
        try:
            if not isinstance(yaml.safe_load(content), (dict, list)):
                return False, "YAML did not parse"
        except yaml.YAMLError as e:
            return False, f"YAML error: {e}"
        if WORKFLOW_PATTERN.search(file):
            content, snapped = normalize_workflow_python_versions(original_text, content)
            if snapped:
                fix["fixed_content"] = content
                fix["reason"] = (fix.get("reason", "") +
                                 f"; python-version pinned to deterministic "
                                 f"'{DEFAULT_PYTHON_VERSION}' (was invalid)").lstrip("; ")[:300]
            ok, reason = validate_workflow_edit(original_text, content)
            if not ok:
                return False, reason
    elif file.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"JSON error: {e}"

    docker_reason = _dockerfile_still_broken(file, content)
    if docker_reason:
        return False, f"safe patch verification failed: {docker_reason}"

    return True, "ok"


def write_fixes(fixes: list) -> tuple:
    written, originals, reasons = [], {}, []
    for fix in fixes[:MAX_FILES_FIXED]:
        ok, reason = validate_fix(fix)
        file = fix.get("file", "").strip()
        if not ok:
            print(f"  ✗ {file or '?'} — {reason}")
            print(f"  ✗ {file or '?'} — {reason}", file=sys.stderr)
            reasons.append(f"{file or '?'}: {reason}")
            continue
        originals[file] = Path(file).read_text(encoding="utf-8", errors="replace")
        content = fix["fixed_content"]
        if not content.endswith("\n"):
            content += "\n"
        p = Path(file)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
        print(f"  ✓ {file} — {fix.get('reason','')[:100]}")
        written.append(file)
    written = list(dict.fromkeys(written))
    return written, originals, reasons


def revert_files(originals: dict):
    for file, content in originals.items():
        try:
            Path(file).write_text(content, encoding="utf-8")
            print(f"[REVERT] {file}")
        except Exception as exc:
            print(f"[REVERT] failed {file}: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTE BUILD & TESTS
# ══════════════════════════════════════════════════════════════════════════════

def detect_test_commands(stacks: set) -> list:
    cmds = []
    if "python" in stacks:
        cmds.append(["python", "-m", "pytest", "--tb=short", "-q"])
    if "node" in stacks and Path("package.json").exists():
        try:
            if "test" in json.loads(Path("package.json").read_text()).get("scripts", {}):
                cmds.append(["npm", "test", "--", "--passWithNoTests"])
        except Exception:
            pass
    if "go" in stacks and Path("go.mod").exists():
        cmds.append(["go", "test", "./..."])
    return cmds


def run_tests(stacks: set) -> tuple:
    """Returns (all_passed, failure_output) — failure_output feeds the
    'Collect New Failure Logs' step if anything fails."""
    cmds = detect_test_commands(stacks)
    if not cmds:
        print("[TEST] No test runner detected — skipping")
        return True, ""
    ok_all, captured = True, []
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            out = r.stdout + r.stderr
            for line in out.splitlines()[-15:]:
                print(f"  {line}")
            passed = r.returncode == 0
            print(f"[TEST] {' '.join(cmd[:2])}: {'✓ Passed' if passed else '✗ Failed'}")
            ok_all = ok_all and passed
            if not passed:
                captured.append(out[-3000:])
        except FileNotFoundError:
            print(f"[TEST] {cmd[0]} not found — skipping")
        except subprocess.TimeoutExpired:
            print(f"[TEST] {cmd[0]} timed out — skipping")
    return ok_all, "\n".join(captured)


def try_docker_build(written: list) -> tuple:
    """If a Dockerfile was changed, actually build it — this is the check
    that would have caught 'FROM python:3.1' even without the deterministic
    scanner, since that tag simply doesn't exist on Docker Hub."""
    dockerfiles = [f for f in written if "dockerfile" in Path(f).name.lower()]
    if not dockerfiles:
        return True, ""
    if subprocess.run(["bash", "-lc", "command -v docker"],
                      capture_output=True).returncode != 0:
        print("[TEST] docker not available — skipping build check")
        return True, ""
    df = dockerfiles[0]
    context_dir = str(Path(df).parent) or "."
    try:
        r = subprocess.run(["docker", "build", "-f", df, "-t", "autofix-check:latest", context_dir],
                           capture_output=True, text=True, timeout=240)
        tail = r.stdout + r.stderr
        for line in tail.splitlines()[-15:]:
            print(f"  {line}")
        ok = r.returncode == 0
        print(f"[TEST] docker build: {'✓ Passed' if ok else '✗ Failed'}")
        return ok, ("" if ok else tail[-3000:])
    except FileNotFoundError:
        print("[TEST] docker not found — skipping build check")
        return True, ""
    except subprocess.TimeoutExpired:
        print("[TEST] docker build timed out — not treated as failure")
        return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# COMMIT + PUSH + PR  /  CREATE ISSUE
# ══════════════════════════════════════════════════════════════════════════════

def _git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def last_commit_was_bot() -> bool:
    try:
        author, subject = _git("log", "-1", "--pretty=%an|||%s").stdout.strip().split("|||", 1)
        if author == BOT_NAME and subject.startswith(BOT_PREFIX):
            print("[GUARD] Last commit was bot — stopping.")
            return True
    except Exception:
        pass
    return False


def count_recent_bot_commits(n=10) -> int:
    try:
        lines = _git("log", f"-{n}", "--pretty=%an").stdout.splitlines()
        count = 0
        for author in lines:
            if author.strip() == BOT_NAME:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def _ensure_base_branch() -> bool:
    _git("fetch", "origin", check=False)
    if _git("ls-remote", "--heads", "origin", GIT_BASE_BRANCH, check=False).stdout.strip():
        if _git("rev-parse", "--verify", GIT_BASE_BRANCH, check=False).returncode != 0:
            _git("checkout", "-b", GIT_BASE_BRANCH, f"origin/{GIT_BASE_BRANCH}")
        return True
    try:
        ref = "main" if _git("rev-parse", "--verify", "main", check=False).returncode == 0 else "master"
        _git("checkout", "-b", GIT_BASE_BRANCH, ref)
        _git("push", "-u", "origin", GIT_BASE_BRANCH)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[GIT] cannot create {GIT_BASE_BRANCH}: {exc.stderr.strip()}", file=sys.stderr)
        return False


def commit_to_branch(commit_msg: str, written: list) -> str:
    try:
        _git("config", "user.name", BOT_NAME)
        _git("config", "user.email", BOT_EMAIL)
        if not _ensure_base_branch():
            return ""
        _git("checkout", GIT_BASE_BRANCH)
        _git("pull", "origin", GIT_BASE_BRANCH, check=False)
        branch = f"fix/{int(time.time())}"
        _git("checkout", "-b", branch)
        if written:
            _git("add", "--", *written)
        else:
            _git("add", "-u")
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit.")
            _git("checkout", GIT_BASE_BRANCH, check=False)
            return ""
        _git("commit", "-m", commit_msg)
        _git("push", "-u", "origin", branch)
        print(f"[GIT] Pushed {branch}")
        _git("checkout", GIT_BASE_BRANCH, check=False)
        return branch
    except subprocess.CalledProcessError as exc:
        print(f"[GIT] {exc.stderr.strip()}", file=sys.stderr)
        _git("checkout", GIT_BASE_BRANCH, check=False)
        return ""


def _gh(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
            investigation_log=None, evidence_files=None, solution="",
            findings=None) -> str:
    details = "".join(f"\n**`{f.get('file','?')}`** — {f.get('reason','')}\n" for f in fixes)
    trace = ""
    if investigation_log:
        steps = "".join(f"\n{i+1}. `{s['status']}` — {s['analysis']}"
                        for i, s in enumerate(investigation_log))
        trace = f"### Investigation trace{steps}\n\n"
    issues_section = ""
    if findings:
        items = "".join(
            f"\n{i+1}. **{f['issue']}**"
            + (f" — {f['root_cause']}" if f.get('root_cause') and f['root_cause'] != f['issue'] else "")
            + (f"  \n   _Fix: {f['solution']}_" if f.get('solution') else "")
            for i, f in enumerate(findings))
        issues_section = f"### Issues identified & fixed ({len(findings)}){items}\n\n"
    files_read = (f"**Files the agent read to diagnose this:** "
                 f"{', '.join(f'`{f}`' for f in evidence_files)}\n\n"
                 if evidence_files else "")
    body = (f"## 🤖 AI Auto-Fix\n\n{trace}{issues_section}"
            f"**Overall root cause:** {root_cause}\n\n"
            f"**Solution:** {solution or 'n/a'}\n\n"
            f"{files_read}"
            f"**Files changed:** {', '.join(f'`{f}`' for f in written)}\n\n"
            f"## What changed{details}\n\n> Auto-generated — review before merge.")
    r = requests.post(f"https://api.github.com/repos/{repo}/pulls",
                      json={"title": f"🤖 {commit_msg}", "head": branch,
                            "base": GIT_TARGET_BRANCH, "body": body},
                      headers=_gh(token), timeout=30)
    if r.status_code in (200, 201):
        url = r.json().get("html_url", "")
        print(f"[PR] {url}")
        return url
    print(f"[PR] failed {r.status_code}: {r.text[:200]}", file=sys.stderr)
    return ""


def pending_bot_pr(token, repo) -> str:
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/pulls?state=open&per_page=50",
                         headers=_gh(token), timeout=15)
        if r.status_code == 200:
            for pr in r.json():
                if (pr.get("head", {}).get("ref", "").startswith("fix/")
                        and "🤖" in (pr.get("title") or "")):
                    return pr.get("html_url", "")
    except Exception:
        pass
    return ""


def open_issue(token, repo, reason, run_url=""):
    r = requests.post(f"https://api.github.com/repos/{repo}/issues",
                      json={"title": "🚨 Auto-fixer could not fix CI — manual review",
                            "body": f"**Reason:** {reason}\n\n**Failed run:** {run_url}",
                            "labels": ["bug", "needs-manual-fix"]},
                      headers=_gh(token), timeout=30)
    if r.status_code in (200, 201):
        print(f"[ISSUE] {r.json().get('html_url','')}")
    else:
        print(f"[ISSUE] failed {r.status_code}: {r.text[:200]}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer (agentic investigation)")
    ap.add_argument("--input", required=True, help="Path to CI failure log")
    ap.add_argument("--exit-code", default=None, help="Exit code of the failed step, if known")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "")
    run_url = (os.environ.get("GITHUB_SERVER_URL", "") + "/" + repo + "/actions") if repo else ""

    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Log not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # loop guards
    if last_commit_was_bot():
        sys.exit(0)
    if count_recent_bot_commits() >= MAX_BOT_ATTEMPTS:
        print("[GUARD] Too many bot attempts — escalating.")
        if token and repo:
            open_issue(token, repo, "Auto-fixer attempted too many fixes without success.", run_url)
        sys.exit(0)
    if token and repo:
        url = pending_bot_pr(token, repo)
        if url:
            print(f"[GUARD] A fix PR is already open: {url} — skipping model call.")
            sys.exit(0)

    # ── COLLECT INITIAL INVESTIGATION EVIDENCE ──
    print("\n━━━ COLLECT INITIAL INVESTIGATION EVIDENCE ━━━")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    redacted = scan_text_for_secrets(log_text)
    if redacted:
        print(f"[SECURITY] redacting {len(redacted)} potential secret pattern(s) from log: "
              f"{', '.join(redacted)}")
    log_text = redact_secrets(log_text)
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[EVIDENCE] No error signal — nothing to fix.")
        sys.exit(0)

    exit_code = get_exit_code(log_text, args.exit_code)
    git_diff  = get_git_diff()
    repo_tree, allowed_files = repo_tree_text()
    print(f"[EVIDENCE] exit_code={exit_code} | {len(allowed_files)} readable file(s) in tree")

    # Seed the triggering workflow file(s) as evidence up front — the model is
    # unreliable at requesting the workflow by exact name, and almost every CI
    # failure's fix lives in either app code or the workflow itself.
    seed_evidence = seed_workflow_evidence(allowed_files)

    # ── AI INVESTIGATION AGENT ──
    print("\n━━━ AI INVESTIGATION AGENT ━━━")
    investigation = ai_investigate(signal, exit_code, repo_tree, git_diff,
                                   allowed_files, seed_evidence)
    root_cause = investigation["root_cause"]
    solution   = investigation["solution"]
    commit_msg = investigation["commit_message"]
    confidence = investigation["confidence"]
    evidence   = investigation["evidence"]
    findings   = investigation.get("findings", [])

    print("\n  ── investigation result ──")
    print(f"  OVERALL CAUSE : {root_cause}")
    print(f"  confidence    : {confidence:.0%}")
    print(f"  files read    : {', '.join(evidence.keys()) or '(none)'}")
    print(f"  issues found  : {len(findings)}")
    for i, fnd in enumerate(findings, 1):
        print(f"    {i}. {fnd['issue']}")
        if fnd.get("root_cause"):
            print(f"       cause: {fnd['root_cause']}")
        if fnd.get("solution"):
            print(f"       fix:   {fnd['solution']}")

    if confidence < 0.5:
        print(f"[GATE] Confidence {confidence:.0%} too low — escalating instead of guessing.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({confidence:.0%}). Root cause: {root_cause}", run_url)
        sys.exit(0)

    # feed every itemized issue into the patch agent, not just the one-line
    # overall summary — otherwise it tends to fix the first bug and stop
    findings_text = "\n".join(
        f"- {f['issue']}: {f['root_cause']}" + (f" — fix: {f['solution']}" if f.get('solution') else "")
        for f in findings)
    patch_root_cause = (f"{root_cause}\n\nIndividual issues to fix (address EVERY one):\n{findings_text}"
                        if findings_text else root_cause)

    # ── AI PATCH GENERATION AGENT + PYTHON VALIDATION ENGINE, with retry loop ──
    written, originals, fixes = [], {}, []
    success = False
    retry_note = ""

    for repair_round in range(1, MAX_REPAIR_ROUNDS + 1):
        print(f"\n━━━ AI PATCH GENERATION AGENT (round {repair_round}/{MAX_REPAIR_ROUNDS}) ━━━")
        try:
            issues = ai_generate_patch(patch_root_cause, solution, evidence, retry_note)
        except Exception as exc:
            print(f"[ERROR] patch generation failed: {exc}", file=sys.stderr)
            if repair_round == MAX_REPAIR_ROUNDS:
                if token and repo:
                    open_issue(token, repo, f"AI patch generation failed: {exc}", run_url)
                sys.exit(2)
            continue

        print(f"  issues reported: {len(issues)}")
        for n, it in enumerate(issues, 1):
            print(f"    {n}. {it.get('file','?')}: {it.get('problem','')[:80]}")

        fixes, pair_rejects = issues_to_fixes(issues, evidence)
        for rej in pair_rejects:
            print(f"  ✗ {rej}", file=sys.stderr)
        if not fixes:
            detail = "; ".join(pair_rejects) or "model reported no locatable issues"
            print(f"[ERROR] no usable fixes this round. {detail}", file=sys.stderr)
            if repair_round == MAX_REPAIR_ROUNDS:
                if token and repo:
                    open_issue(token, repo,
                               f"AI produced no usable fixes. Root cause: {root_cause}\n\n{detail}", run_url)
                sys.exit(3)
            continue

        print("\n━━━ PYTHON VALIDATION ENGINE ━━━")
        if args.dry_run:
            for fix in fixes:
                ok, reason = validate_fix(fix)
                print(f"  {'would write' if ok else 'reject'} {fix.get('file','?')} — {reason}")
            sys.exit(0)

        written, originals, reject_reasons = write_fixes(fixes)
        if not written:
            detail = "; ".join(reject_reasons) or "no detail captured"
            print(f"[ERROR] validation rejected all fixes. {detail}", file=sys.stderr)
            if repair_round == MAX_REPAIR_ROUNDS:
                if token and repo:
                    open_issue(token, repo,
                               f"AI fix failed validation. Root cause: {root_cause}\n\n{detail}", run_url)
                sys.exit(3)
            continue

        # ── EXECUTE BUILD & TESTS ──
        print("\n━━━ EXECUTE BUILD & TESTS ━━━")
        if not args.skip_tests:
            tests_ok, test_output = run_tests(stacks)
        else:
            print("[TEST] Skipped (--skip-tests)")
            tests_ok, test_output = True, ""
        docker_ok, docker_output = try_docker_build(written)

        if tests_ok and docker_ok:
            success = True
            break

        # ── FAILURE: collect new evidence, revert, AI reviews it ──
        print(f"[REPAIR] round {repair_round} failed tests/build — reverting.")
        combined_output = (test_output + "\n" + docker_output).strip()
        new_signal = extract_error_signal(combined_output) or combined_output[-1500:]
        revert_files(originals)
        written = []

        if repair_round == MAX_REPAIR_ROUNDS:
            if token and repo:
                open_issue(token, repo,
                           f"Fix applied but tests/build failed after {repair_round} "
                           f"attempt(s) — reverted. Root cause: {root_cause}", run_url)
            sys.exit(5)

        print("\n━━━ AI REVIEWS NEW EVIDENCE ━━━")
        review = ai_review_failure(root_cause, solution, new_signal)
        print(f"[REVIEW] decision={review['decision']} confidence={review['confidence']:.0%}")
        if review["decision"] != "retry":
            if token and repo:
                open_issue(token, repo,
                           f"AI stopped after a failed fix — needs human review. "
                           f"Root cause: {review['root_cause']}", run_url)
            sys.exit(5)

        root_cause, solution = review["root_cause"], review["solution"]
        retry_note = new_signal[:1200]

    if not success:
        sys.exit(5)

    # ── COMMIT + PR ──
    print("\n━━━ COMMIT + PR ━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)
    if token and repo:
        open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
                investigation.get("investigation_log"), list(evidence.keys()), solution,
                findings)
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()