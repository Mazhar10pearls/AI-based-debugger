#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — AI-authored, Python-guarded.

Architecture (the AI reasons and writes; Python observes, guards, executes):

    ┌─ PYTHON: CONTEXT LAYER ────────────────────────────────────────────┐
    │  collect logs → redact secrets → extract error signal              │
    │  fingerprint stack → discover files                                │
    │  static OBSERVATION scan (verifiable facts only — never fixes)     │
    └────────────────────────────────────────────────────────────────────┘
          ↓
      AI Stage 1: extract facts
      AI Stage 2: root cause (+ confidence)     ── gate < 0.5 → escalate
      AI Stage 3: WHOLE-FILE REWRITE — one call per file; the AI fixes
                  EVERY bug in that file in a single pass (multi-bug,
                  multi-category), emitting the full corrected file
                  between sentinel markers
      AI Stage 4: self-review of its own unified diff (keep/drop)
          ↓
    ┌─ PYTHON: SAFETY + EXECUTION LAYER ─────────────────────────────────┐
    │  sentinel extraction → similarity floor → line-change caps         │
    │  secret scan on added lines → syntax validation (py/yaml/json)     │
    │  OBSERVATION VERIFICATION: re-run the static scans on the AI's     │
    │  output. Unresolved observation → RE-PROMPT the AI with exactly    │
    │  what it missed (never patch it in Python). Still unresolved       │
    │  after MAX_AI_ROUNDS → escalate to a human via open_issue().       │
    │  tests → revert-on-fail → branch → commit → push → PR              │
    └────────────────────────────────────────────────────────────────────┘

Design principle — "observe, don't pre-solve":
  Python NEVER authors a change. The static scanners that used to write
  corrected lines (base-image version fixer, correct-the-line pass) are now
  observation emitters. An observation is a checkable ground-truth fact:

      "sample_app/Dockerfile: base image tag 'python:3.1' does not exist.
       Existing CPython minor releases: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13."

      "sample_app/Dockerfile: COPY references 'appp.py' — no such file in
       the build context 'sample_app/'. Files that DO exist in that
       context: app.py, requirements.txt. (Dockerfile paths resolve
       against the build context, NOT the repo root.)"

  Note what's absent: the corrected line. The AI decides the fix. The same
  observation is then re-checked against the AI's rewrite — so Python is a
  grader, not a ghostwriter. This keeps the "python:3.1 gets missed" class
  of bug covered (the checklist won't let it slip through) without Python
  ever doing the AI's job.

Why whole-file rewrite instead of flat find/replace issues:
  A 3B model reliably finds ONE bug per file and stops when asked for a
  list of issues. Asked instead to re-emit the entire (small) file with all
  bugs fixed, it must re-read every line — multi-bug fixes in one file, or
  across several files, come out of a single pass per file. Structured JSON
  escaping of file bodies is what mangled output before; sentinel markers
  avoid JSON entirely for the rewrite payload.

Escape hatches:
  FAST_MODE=1        → collapse Stage 1+2 into one call
  SKIP_SELF_VERIFY=1 → drop Stage 4 (Python validation is still the hard gate)

Exit codes:
  0 success / nothing to do   2 AI failed        4 git failed
  1 log not found             3 no valid fix     5 tests failed (reverted)
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
MAX_AI_ROUNDS  = 3   # rewrite rounds per run (round 2+ = observation-driven re-prompts)

FAST_MODE        = os.environ.get("FAST_MODE", "").lower() in ("1", "true", "yes")
SKIP_SELF_VERIFY = os.environ.get("SKIP_SELF_VERIFY", "").lower() in ("1", "true", "yes")

# ── Prompt / context budget ───────────────────────────────────────────────────
MAX_ERROR_LINES   = 14
MAX_FILE_CHARS    = 4000
MAX_TOTAL_CONTEXT = 9000
MAX_CONTEXT_FILES = 3
MAX_FILES_FIXED   = 4
MAX_REWRITE_CHARS = 6000   # never ask the 3B to re-emit a file bigger than this

# ── Rewrite validation guards (Python = safety, not authorship) ───────────────
REWRITE_SIMILARITY_FLOOR = 0.55  # SequenceMatcher ratio vs. original; a real
                                 # bug-fix rewrite is mostly the same file
MAX_ADDED_LINES_PER_FIX  = 40    # a legit typo/version/port fix is tiny
MAX_LINE_COUNT_DRIFT     = 10    # rewrite must not grow/shrink the file much

# ── Git flow ──────────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_BLOCKED   = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [
    # ALL workflow files — a "fix" to a deploy/CI workflow is a privilege-
    # escalation vector. Workflow breakage escalates to a human via
    # open_issue(), never through auto-fix.
    r"\.?github/workflows/.*\.ya?ml$",
    r"\.?github/CODEOWNERS$",
    r"(^|/)\.env(\..*)?$",
    r".*\.pem$", r".*\.key$", r".*id_rsa.*", r".*id_ed25519.*",
    r".*secrets?\.ya?ml$", r".*\.tfstate(\.backup)?$",
    r"(^|/)\.npmrc$", r"(^|/)\.pypirc$",
]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             "dist", "build", ".pytest_cache", "target", "out", "vendor",
             ".idea", ".vscode", "coverage", "tmp", "temp", "logs"}
MAX_FILE_SIZE_BYTES = 100_000

# ── Secret scanning (deterministic safety net — unchanged) ───────────────────
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


def scan_text_for_secrets(text: str) -> list:
    return [name for name, pat in SECRET_PATTERNS if pat.search(text)]


def redact_secrets(text: str) -> str:
    out = text
    for name, pat in SECRET_PATTERNS:
        out = pat.sub(f"[REDACTED:{name}]", out)
    return out


def _added_lines(original: str, new: str) -> list:
    old_lines = set(original.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON CONTEXT LAYER — read logs, detect stack  (unchanged, proven)
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
    print(f"[DETECT] Error signal: {len(signal)} chars")
    return signal


def fingerprint_stack(log_text: str) -> set:
    low = log_text.lower()
    stacks = {s for s, sig in TECH_STACK_SIGNALS.items() if any(x in low for x in sig)}
    print(f"[DETECT] Tech stacks: {stacks or {'unknown'}}")
    return stacks


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON CONTEXT LAYER — discover files  (unchanged, proven)
# ══════════════════════════════════════════════════════════════════════════════

REFERENCED_PATH_PATTERNS = [
    r'File "([^"]+\.\w+)"',
    r'([\w./\-]+\.[A-Za-z0-9]+):\d+(?::\d+)?',
    r'((?:[\w./\-]+/)?Dockerfile(?:\.\w+)?)\b',
    r'([\w./\-]+/requirements[\w.\-]*\.txt)',
    r'(?:in|from|at|open)\s+([\w./\-]+\.[A-Za-z0-9]+)',
]
STACK_FILE_SIGNALS = {
    "python": [".py", "requirements.txt", "pyproject.toml", "setup.py"],
    "node":   [".js", ".ts", "package.json"],
    "docker": ["Dockerfile", "docker-compose"],
    "java":   [".java", "pom.xml", "build.gradle"],
    "go":     [".go", "go.mod"],
}
HIGH_VALUE = {"dockerfile", "requirements.txt", "package.json", "pom.xml",
              "go.mod", "pyproject.toml", "setup.py", "docker-compose.yml"}


def _relstrip(rel): return rel[2:] if rel.startswith("./") else rel
def _is_blocked(fp):
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_BLOCKED):
        return True
    return any(re.search(p, fp) for p in BLOCKED_PATTERNS)


def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False


def extract_referenced_paths(log_text: str, root: Path = Path(".")) -> list:
    hits = []
    for pat in REFERENCED_PATH_PATTERNS:
        for m in re.finditer(pat, log_text):
            c = m.group(1).strip().strip("'\"")
            if c:
                hits.append(c)
    by_name = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            by_name.setdefault(p.name, []).append(str(p))
    order, counts = [], {}
    def add(rel):
        rel = _relstrip(rel)
        if _is_blocked(rel):
            return
        if rel not in counts:
            order.append(rel)
        counts[rel] = counts.get(rel, 0) + 1
    for h in hits:
        hn = _relstrip(h)
        if Path(hn).is_file():
            add(hn); continue
        for rel in by_name.get(Path(h).name, []):
            add(rel)
    resolved = sorted(order, key=lambda r: -counts[r])
    if resolved:
        print(f"[DISCOVER] Referenced in log: {resolved}")
    return resolved


def _score_file(path: Path, signal: str, stacks: set) -> int:
    name, low, score = path.name.lower(), signal.lower(), 0
    if name in low or str(path).lower() in low:
        score += 50
    for st in stacks:
        for pat in STACK_FILE_SIGNALS.get(st, []):
            if pat.startswith(".") and name.endswith(pat):
                score += 20
            elif pat.lower() == name:
                score += 25
    if name in HIGH_VALUE:
        score += 15
    return score


def find_ci_workflow_files(root: Path = Path(".")) -> list:
    found = []
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return found
    for p in sorted(wf_dir.glob("*.y*ml")):
        if not p.is_file():
            continue
        rel = _relstrip(str(p))
        if _is_blocked(rel):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "auto-fixer.py" in text or "auto_fixer.py" in text:
            continue
        found.append(rel)
    return found


def discover_context(signal: str, stacks: set, forced: list) -> tuple:
    parts, included, contents, total = [], [], {}, 0

    def read_whole(path, cap=MAX_FILE_CHARS):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            return raw if len(raw) <= cap else None
        except Exception:
            return None

    def try_add(rel, content):
        nonlocal total
        block = f"### {rel}\n```\n{content}\n```"
        if total + len(block) > MAX_TOTAL_CONTEXT and included:
            return False
        parts.append(block); included.append(rel); contents[rel] = content; total += len(block)
        print(f"[DISCOVER] + {rel} ({len(content)} chars)")
        return True

    for rel in forced:
        if len(included) >= MAX_CONTEXT_FILES:
            break
        p = Path(rel)
        if p.is_file() and _is_text_file(p):
            c = read_whole(p, cap=8000)
            if c is not None:
                try_add(rel, c)

    if len(included) < MAX_CONTEXT_FILES:
        cand = []
        for p in Path(".").rglob("*"):
            if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
                continue
            if not _is_text_file(p):
                continue
            rel = _relstrip(str(p))
            if rel in included or _is_blocked(rel):
                continue
            sc = _score_file(p, signal, stacks)
            if sc > 0:
                cand.append((sc, p))
        cand.sort(key=lambda x: (-x[0], len(str(x[1]))))
        for _, p in cand:
            if len(included) >= MAX_CONTEXT_FILES:
                break
            rel = _relstrip(str(p))
            c = read_whole(p)
            if c is not None:
                try_add(rel, c)

    context = "\n\n".join(parts)
    print(f"[DISCOVER] {len(included)} files, {len(context)} chars")
    return context, included, contents


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON CONTEXT LAYER — OBSERVATIONS  (facts only, never fixes)
# ══════════════════════════════════════════════════════════════════════════════
# This is the architectural pivot. The old code had two Python-authored fix
# paths (deterministic base-image correction, correct-the-line prefills).
# Both are now OBSERVATION EMITTERS. An observation is:
#   { "key":  stable identity tuple → lets us re-check it after a rewrite,
#     "file": which context file it concerns,
#     "text": a ground-truth fact for the prompt — states the CONSTRAINT
#             ("this tag does not exist", "these files exist in the build
#             context"), never the ANSWER (no corrected line). }
# The AI authors every change; observations are used twice:
#   1. injected into every AI prompt as grounding context, and
#   2. re-scanned against the AI's rewritten content as a verification
#      checklist. Unresolved → re-prompt the AI. Never patch in Python.

DOCKERFILE_REF_PATTERNS = [
    (re.compile(r'^\s*COPY\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "COPY"),
    (re.compile(r'^\s*ADD\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "ADD"),
    (re.compile(r'-r\s+(\S+\.txt)'), "pip install -r"),
    (re.compile(r'CMD\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "CMD"),
    (re.compile(r'ENTRYPOINT\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "ENTRYPOINT"),
]

DOCKERFILE_FROM_PYTHON = re.compile(r'^\s*FROM\s+python:([^\s]+)', re.I | re.M)
VALID_PYTHON_MINORS = set(range(8, 14))   # CPython 3.8 – 3.13 (update as releases land)


def _build_context_files(docker_dir: Path) -> list:
    """Files that exist inside a Dockerfile's build context, as in-context
    relative paths. This is the ground truth the AI needs to pick a correct
    COPY/ADD path — Dockerfile paths resolve against the build context
    (conventionally the Dockerfile's own directory in this repo layout),
    NOT the git repo root."""
    out = []
    for p in sorted(docker_dir.rglob("*")):
        if p.is_file() and not any(d in SKIP_DIRS for d in p.parts):
            try:
                out.append(str(p.relative_to(docker_dir)))
            except ValueError:
                continue
        if len(out) >= 40:
            break
    return out


def scan_observations(included_contents: dict) -> list:
    """Re-runnable static scan. Returns the CURRENT list of observations for
    the given contents — calling it again on rewritten content tells you
    which observations the AI resolved. Purely descriptive: no observation
    ever contains a corrected line."""
    obs = []
    for rel, content in included_contents.items():
        name = Path(rel).name.lower()
        is_dockerish = ("dockerfile" in name or "docker-compose" in name
                        or "compose.y" in name)
        if not is_dockerish:
            continue
        docker_dir = Path(rel).parent
        ctx_files = _build_context_files(docker_dir)
        ctx_set = set(ctx_files)

        # 1) references to files that don't exist in the build context
        for pat, label in DOCKERFILE_REF_PATTERNS:
            for m in pat.finditer(content):
                ref = m.group(1).strip().strip("'\"")
                if not ref or ref in (".", "..") or ref.startswith(("-", "$")):
                    continue
                ref_clean = ref.lstrip("./")
                if ref_clean in ctx_set or (docker_dir / ref_clean).is_file():
                    continue
                listing = ", ".join(ctx_files) or "(none)"
                obs.append({
                    "key": (rel, "ref", label, ref),
                    "file": rel,
                    "text": (f"{rel}: {label} references '{ref}' — no such file in "
                             f"the build context '{docker_dir}/'. Files that DO exist "
                             f"in that context: {listing}. (Dockerfile paths resolve "
                             f"against the build context, NOT the repo root — a "
                             f"repo-relative path like '{docker_dir}/...' will not "
                             f"resolve inside the container.)"),
                })

        # 2) base image tags that name a nonexistent CPython release
        for m in DOCKERFILE_FROM_PYTHON.finditer(content):
            version = m.group(1)
            base = version.split("-", 1)[0]
            vm = re.match(r'^(\d+)\.(\d+)', base)
            if not vm:
                continue
            major, minor = int(vm.group(1)), int(vm.group(2))
            if major == 3 and minor in VALID_PYTHON_MINORS:
                continue
            valid = ", ".join(f"3.{n}" for n in sorted(VALID_PYTHON_MINORS))
            obs.append({
                "key": (rel, "baseimg", version),
                "file": rel,
                "text": (f"{rel}: base image tag 'python:{version}' does not exist — "
                         f"there is no such CPython release. Existing CPython minor "
                         f"releases: {valid}. Any -slim/-alpine style suffix on the "
                         f"tag is fine to keep."),
            })
    if obs:
        print("[OBSERVE] " + f"{len(obs)} observation(s):")
        for o in obs:
            print(f"  • {o['text'][:140]}")
    return obs


def observations_text(obs: list) -> str:
    return "\n".join(f"- {o['text']}" for o in obs)


# ══════════════════════════════════════════════════════════════════════════════
# AI plumbing — shared streaming + JSON extraction  (unchanged)
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
# Repo-file listing + shared prompt fragments
# ══════════════════════════════════════════════════════════════════════════════

def repo_file_list(limit: int = 200) -> str:
    files = []
    for p in sorted(Path(".").rglob("*")):
        if p.is_file() and not any(d in SKIP_DIRS for d in p.parts):
            rel = _relstrip(str(p))
            if not _is_blocked(rel):
                files.append(rel)
        if len(files) >= limit:
            break
    return ", ".join(files) if files else "(none found)"


def _context_block(signal, context, stacks, obs_text):
    repo_files = repo_file_list()
    obs_section = (f"## Verified observations (ground truth — every one of these "
                   f"IS a real problem you must fix):\n{obs_text}\n"
                   if obs_text else "")
    return (
        f"## Tech stack: {', '.join(sorted(stacks)) or 'unknown'}\n"
        f"## CI failure (key lines):\n```\n{signal}\n```\n"
        f"## Repo files (these exist — anything referenced but NOT here is a typo):\n{repo_files}\n"
        f"{obs_section}"
        f"## File contents (you may ONLY edit these):\n{context}"
    )


def _cap_prompt(system: str, body: str, context: str, rebuild, cap=9500) -> str:
    full = f"{system}\n\n{body}"
    if len(full) <= cap:
        return full
    allowed = cap - len(system) - len(rebuild("")) - 100
    trimmed = context[:max(allowed, 1000)] + "\n...(trimmed)"
    return f"{system}\n\n{rebuild(trimmed)}"


# ══════════════════════════════════════════════════════════════════════════════
# AI STAGE 1 — EXTRACT FACTS
# ══════════════════════════════════════════════════════════════════════════════

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "error_type":    {"type": "string"},
        "failing_files": {"type": "array", "items": {"type": "string"}},
        "key_symbols":   {"type": "array", "items": {"type": "string"}},
        "summary":       {"type": "string"},
    },
    "required": ["error_type", "failing_files", "summary"],
}

FACTS_SYSTEM = """\
You are a CI/CD triage engineer. Extract FACTS ONLY — do not diagnose or fix yet.
Output ONLY one JSON object. No markdown fences. Start with { end with }.
{"error_type":"short category e.g. missing_file / bad_version / port_mismatch / import_error","failing_files":["exact/path from a ### header, if any"],"key_symbols":["the literal tokens the error names — filenames, versions, ports"],"summary":"one sentence of what the log shows"}
Copy tokens EXACTLY from the log and file contents. Do not invent files or symbols."""


def ai_extract_facts(signal, context, stacks, obs_text) -> dict:
    def rebuild(ctx):
        return _context_block(signal, ctx, stacks, obs_text) + \
               "\n\nExtract the facts as JSON."
    prompt = _cap_prompt(FACTS_SYSTEM, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, FACTS_SCHEMA, num_predict=800,
                         temperature=0.0, tag="S1-FACTS")
    data = _json_from(raw) or {}
    data.setdefault("error_type", "unknown")
    data.setdefault("failing_files", [])
    data.setdefault("key_symbols", [])
    data.setdefault("summary", "")
    print(f"[S1-FACTS] {data.get('error_type')} | files={data.get('failing_files')} "
          f"| symbols={data.get('key_symbols')}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# AI STAGE 2 — ROOT CAUSE
# ══════════════════════════════════════════════════════════════════════════════

CAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause":     {"type": "string"},
        "solution":       {"type": "string"},
        "confidence":     {"type": "number"},
        "commit_message": {"type": "string"},
    },
    "required": ["root_cause", "solution", "confidence", "commit_message"],
}

CAUSE_SYSTEM = """\
You are a senior CI/CD debugging engineer. Given the extracted facts, the verified
observations, and the file contents, state the ROOT CAUSE. There may be SEVERAL
independent causes — name all of them. Do NOT write the fix yet.
Output ONLY one JSON object.
{"root_cause":"WHY it failed — list every independent cause","solution":"plain words: WHAT must change, per cause","confidence":0.0-1.0,"commit_message":"fix: short"}
Set confidence below 0.5 if the facts do not clearly pin the cause(s)."""


def ai_root_cause(facts, signal, context, stacks, obs_text) -> dict:
    facts_line = (f"## Extracted facts:\nerror_type={facts.get('error_type')}; "
                  f"failing_files={facts.get('failing_files')}; "
                  f"key_symbols={facts.get('key_symbols')}; "
                  f"summary={facts.get('summary')}\n")
    def rebuild(ctx):
        return facts_line + _context_block(signal, ctx, stacks, obs_text) + \
               "\n\nGive root cause as JSON."
    prompt = _cap_prompt(CAUSE_SYSTEM, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, CAUSE_SCHEMA, num_predict=700,
                         temperature=0.05, tag="S2-CAUSE")
    data = _json_from(raw) or {}
    data.setdefault("root_cause", "unknown")
    data.setdefault("solution", "")
    data.setdefault("commit_message", "fix: auto-fixer change")
    try:
        data["confidence"] = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        data["confidence"] = 0.5
    print(f"[S2-CAUSE] {data['root_cause']} (confidence {data['confidence']:.0%})")
    return data


def ai_facts_and_cause(signal, context, stacks, obs_text) -> tuple:
    """FAST_MODE: one call producing facts + cause together."""
    schema = {"type": "object", "properties": {
        **FACTS_SCHEMA["properties"], **CAUSE_SCHEMA["properties"]},
        "required": ["error_type", "failing_files", "summary",
                     "root_cause", "solution", "confidence", "commit_message"]}
    system = (FACTS_SYSTEM.split("Output ONLY")[0] +
              "Extract facts AND state every independent root cause in one JSON "
              "object. Output ONLY one JSON object.\n"
              '{"error_type":"...","failing_files":[...],"key_symbols":[...],'
              '"summary":"...","root_cause":"...","solution":"...",'
              '"confidence":0.0-1.0,"commit_message":"fix: short"}')
    def rebuild(ctx):
        return _context_block(signal, ctx, stacks, obs_text) + \
               "\n\nRespond with the combined JSON."
    prompt = _cap_prompt(system, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, schema, num_predict=1000, temperature=0.05, tag="S1S2")
    data = _json_from(raw) or {}
    facts = {"error_type": data.get("error_type", "unknown"),
             "failing_files": data.get("failing_files", []) or [],
             "key_symbols": data.get("key_symbols", []) or [],
             "summary": data.get("summary", "")}
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    cause = {"root_cause": data.get("root_cause", "unknown"),
             "solution": data.get("solution", ""),
             "confidence": conf,
             "commit_message": data.get("commit_message", "fix: auto-fixer change")}
    print(f"[S1S2] {facts['error_type']} → {cause['root_cause']} "
          f"(confidence {conf:.0%})")
    return facts, cause


# ══════════════════════════════════════════════════════════════════════════════
# AI STAGE 3 — WHOLE-FILE REWRITE  (one call per file; AI authors everything)
# ══════════════════════════════════════════════════════════════════════════════
# Sentinel markers instead of JSON: asking a 3B model to escape a full file
# body inside a JSON string is exactly what mangled output in earlier
# iterations. Between the markers the model just writes the file.

FILE_START = "===FILE_START==="
FILE_END   = "===FILE_END==="

REWRITE_SYSTEM = f"""\
You are a senior CI/CD debugging engineer. You will rewrite ONE file so that EVERY
bug in it is fixed. The root cause has already been diagnosed — apply it, and also
fix any other bug you find while re-reading the file.

OUTPUT FORMAT — exactly this, nothing else before, between, or after:
{FILE_START}
<the complete corrected file, every line, top to bottom>
{FILE_END}
No JSON. No markdown fences. No commentary. Everything between the markers is
written to disk verbatim.

FIND EVERY BUG — a file routinely contains MORE THAN ONE unrelated bug, and fixing
one then stopping is a FAILURE. After each fix, keep scanning the remaining lines.
Check every line that names a file, a version, or a port independently:
- File references: the observations and repo file list are ground truth. A
  referenced name that does not exist is a typo — point it at the file that
  really exists. NEVER invent a new file; fix the reference.
- Dockerfile COPY/ADD/CMD/ENTRYPOINT paths resolve against the BUILD CONTEXT
  (the Dockerfile's own directory), NEVER the repo root. If the Dockerfile is
  at "sample_app/Dockerfile" and the real file is "sample_app/app.py", the
  correct in-file reference is "app.py" — "sample_app/app.py" would break at
  container runtime even though it looks right against the repo tree.
- Version tags must name releases that actually exist (see observations).
- Ports: if the app binds one port and the Dockerfile EXPOSEs another, make the
  app bind the EXPOSEd port.
- If you touch `if __name__ == "__main__":`, it must still start a long-running
  server.

RULES:
- Change ONLY buggy text. Every other line stays byte-identical — same comments,
  same order, same spacing, same blank lines.
- Do not add features, refactor, reformat, reorder, or "improve" anything.
- Do not add new lines unless the fix strictly requires them.
- If the file genuinely contains no bug, return it unchanged between the markers."""


def _extract_rewrite(raw: str):
    m = re.search(re.escape(FILE_START) + r"\s*\n(.*?)\n?\s*" + re.escape(FILE_END),
                  raw, re.S)
    if m:
        return m.group(1)
    # tolerant fallback: model wrapped the file in a fence instead of markers
    fenced = re.search(r"```[a-zA-Z0-9]*\n(.*?)```", raw, re.S)
    if fenced and FILE_START not in raw:
        print("[S3-REWRITE] no sentinel markers — salvaged fenced block")
        return fenced.group(1)
    return None


def ai_rewrite_file(rel, content, facts, cause, signal, stacks,
                    file_obs_text, extra_note="") -> str:
    """One Stage-3 call: the AI re-emits `rel` with all bugs fixed.
    Returns the rewritten text, or None if nothing usable came back.
    Python does not touch the content — extraction only."""
    obs_section = (f"## Verified observations about THIS file (ground truth — "
                   f"each one is a real problem; your rewrite must resolve ALL "
                   f"of them):\n{file_obs_text}\n" if file_obs_text else "")
    note = f"## Reviewer note from the previous round:\n{extra_note}\n" if extra_note else ""
    body = (f"## Root cause (already determined): {cause.get('root_cause')}\n"
            f"## Solution direction: {cause.get('solution')}\n"
            f"## Facts: error_type={facts.get('error_type')}, "
            f"symbols={facts.get('key_symbols')}\n"
            f"## CI failure (key lines):\n```\n{signal}\n```\n"
            f"## Repo files:\n{repo_file_list()}\n"
            f"{obs_section}{note}"
            f"## File to rewrite: {rel}\n"
            f"{FILE_START}\n{content}\n{FILE_END}\n\n"
            f"Now output the corrected {rel} between the markers.")
    prompt = f"{REWRITE_SYSTEM}\n\n{body}"
    # generous budget: the model must re-emit the whole file
    n_predict = min(6000, max(1500, int(len(content) / 2.5) + 600))
    raw = _stream_ollama(prompt, schema=None, num_predict=n_predict,
                         temperature=0.05, num_ctx=8192, tag="S3-REWRITE")
    new = _extract_rewrite(raw)
    if new is None:
        print(f"[S3-REWRITE] {rel}: no extractable file in response", file=sys.stderr)
    return new


# ══════════════════════════════════════════════════════════════════════════════
# PYTHON SAFETY LAYER — validate the AI's rewrite  (guards, not authorship)
# ══════════════════════════════════════════════════════════════════════════════

def validate_rewrite(rel: str, original: str, new: str) -> tuple:
    """Every check here rejects; none rewrites. Failure modes guarded:
    truncation, hallucinated rewrite-from-scratch, runaway additions,
    secret injection, syntax breakage."""
    if not isinstance(new, str) or not new.strip():
        return False, "empty rewrite"
    if new.strip() == original.strip():
        return False, "no change"

    ratio = difflib.SequenceMatcher(None, original, new).ratio()
    if ratio < REWRITE_SIMILARITY_FLOOR:
        return False, (f"similarity {ratio:.0%} below floor "
                       f"{REWRITE_SIMILARITY_FLOOR:.0%} — looks like a "
                       "rewrite-from-scratch or truncation, not a fix")

    drift = abs(len(new.splitlines()) - len(original.splitlines()))
    if drift > MAX_LINE_COUNT_DRIFT:
        return False, (f"line count drifted by {drift} (max {MAX_LINE_COUNT_DRIFT}) "
                       "— too much added/removed for an auto-fix")

    added = _added_lines(original, new)
    if len(added) > MAX_ADDED_LINES_PER_FIX:
        return False, (f"fix adds {len(added)} lines (max {MAX_ADDED_LINES_PER_FIX}) "
                       "— too large, needs human review")
    secret_hits = scan_text_for_secrets("\n".join(added))
    if secret_hits:
        return False, f"potential secret in fix ({', '.join(secret_hits)}) — blocked"

    if rel.endswith(".py"):
        try:
            ast.parse(new)
        except SyntaxError as e:
            return False, f"Python syntax error: {e}"
    elif re.search(r"\.ya?ml$", rel):
        try:
            if not isinstance(yaml.safe_load(new), (dict, list)):
                return False, "YAML did not parse"
        except yaml.YAMLError as e:
            return False, f"YAML error: {e}"
    elif rel.endswith(".json"):
        try:
            json.loads(new)
        except json.JSONDecodeError as e:
            return False, f"JSON error: {e}"
    return True, "ok"


def _diff_stat(original: str, new: str) -> str:
    d = list(difflib.unified_diff(original.splitlines(), new.splitlines(), lineterm=""))
    plus = sum(1 for l in d if l.startswith("+") and not l.startswith("+++"))
    minus = sum(1 for l in d if l.startswith("-") and not l.startswith("---"))
    return f"+{plus}/-{minus}"


# ══════════════════════════════════════════════════════════════════════════════
# AI STAGE 4 — SELF-REVIEW of the diff  (AI judges the AI; Python only drops)
# ══════════════════════════════════════════════════════════════════════════════

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "keep":       {"type": "boolean"},
        "reason":     {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["keep"],
}

VERIFY_SYSTEM = """\
You are reviewing a proposed patch before it is applied. You will see the unified
diff of ONE file. keep=true only if every changed line is a genuine fix of a real
bug and no change introduces a new bug, removes needed code, or alters unrelated
lines. Deleting or rewriting things that were not broken → keep=false.
Output ONLY one JSON object: {"keep":true,"reason":"short","confidence":0.0-1.0}"""


def ai_verify_diff(rel: str, original: str, new: str) -> tuple:
    """Return (keep, confidence). On any failure keep the patch — Python
    validation and tests remain the hard gates."""
    if SKIP_SELF_VERIFY:
        return True, None
    diff = "\n".join(difflib.unified_diff(
        original.splitlines(), new.splitlines(),
        fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm=""))[:5000]
    prompt = f"{VERIFY_SYSTEM}\n\n## Diff for {rel}:\n```\n{diff}\n```\n\nReturn the JSON."
    try:
        raw = _stream_ollama(prompt, VERIFY_SCHEMA, num_predict=400,
                             temperature=0.0, num_ctx=4096, tag="S4-VERIFY", retries=1)
    except Exception as exc:
        print(f"[S4-VERIFY] failed ({exc}) — keeping patch", file=sys.stderr)
        return True, None
    data = _json_from(raw) or {}
    keep = data.get("keep")
    try:
        conf = float(data.get("confidence")) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        conf = None
    if keep is False:
        print(f"[S4-VERIFY] {rel}: model rejected its own patch — "
              f"{data.get('reason','')[:100]}")
        return False, conf
    return True, conf


# ══════════════════════════════════════════════════════════════════════════════
# WRITE + REVERT
# ══════════════════════════════════════════════════════════════════════════════

def write_rewrites(rewrites: dict, reasons: dict) -> tuple:
    """rewrites: {rel: new_content}. Writes atomically; returns
    (written_files, originals_for_revert)."""
    written, originals = [], {}
    for rel, content in list(rewrites.items())[:MAX_FILES_FIXED]:
        p = Path(rel)
        originals[rel] = p.read_text(encoding="utf-8", errors="replace")
        if not content.endswith("\n"):
            content += "\n"
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
        print(f"  ✓ {rel} ({_diff_stat(originals[rel], content)}) — "
              f"{reasons.get(rel, '')[:100]}")
        written.append(rel)
    return written, originals


def revert_files(originals: dict):
    for rel, content in originals.items():
        try:
            Path(rel).write_text(content, encoding="utf-8")
            print(f"[REVERT] {rel}")
        except Exception as exc:
            print(f"[REVERT] failed {rel}: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# RUN TESTS  (unchanged)
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


def run_tests(stacks: set) -> bool:
    cmds = detect_test_commands(stacks)
    if not cmds:
        print("[TEST] No test runner detected — skipping")
        return True
    ok_all = True
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            for line in (r.stdout + r.stderr).splitlines()[-15:]:
                print(f"  {line}")
            passed = r.returncode == 0
            print(f"[TEST] {' '.join(cmd[:2])}: {'✓ Passed' if passed else '✗ Failed'}")
            ok_all = ok_all and passed
        except FileNotFoundError:
            print(f"[TEST] {cmd[0]} not found — skipping")
        except subprocess.TimeoutExpired:
            print(f"[TEST] {cmd[0]} timed out — skipping")
    return ok_all


# ══════════════════════════════════════════════════════════════════════════════
# COMMIT + PUSH + PR  /  CREATE ISSUE  (unchanged)
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


def open_pr(token, repo, branch, commit_msg, root_cause, written, reasons,
            diff_stats, error_analysis="", solution="", unresolved_note="") -> str:
    details = "".join(f"\n**`{f}`** ({diff_stats.get(f,'')}) — {reasons.get(f,'')}\n"
                      for f in written)
    diag = ""
    if error_analysis or solution:
        diag = (f"### AI diagnosis\n**Error:** {error_analysis or 'n/a'}\n\n"
                f"**Solution:** {solution or 'n/a'}\n\n")
    warn = (f"\n\n### ⚠️ Unresolved (needs human eyes)\n{unresolved_note}\n"
            if unresolved_note else "")
    body = (f"## 🤖 AI Auto-Fix\n\n{diag}"
            f"**Root cause:** {root_cause}\n\n"
            f"**Files changed:** {', '.join(f'`{f}`' for f in written)}\n\n"
            f"## What changed{details}{warn}\n\n> Auto-generated — review before merge.")
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
# MAIN — orchestration
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer (AI-authored, Python-guarded)")
    ap.add_argument("--input", required=True, help="Path to CI failure log")
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

    # ── PYTHON: COLLECT LOGS + CONTEXT ──
    print("\n━━━ CONTEXT LAYER: LOGS + FILES + OBSERVATIONS ━━━")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    redacted = scan_text_for_secrets(log_text)
    if redacted:
        print(f"[SECURITY] redacting {len(redacted)} potential secret pattern(s) from log: "
              f"{', '.join(redacted)}")
    log_text = redact_secrets(log_text)
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[DETECT] No error signal — nothing to fix.")
        sys.exit(0)

    forced = extract_referenced_paths(log_text)
    if not forced:
        wf = find_ci_workflow_files()
        if wf:
            print(f"[DISCOVER] No source file referenced — falling back to CI workflow: {wf}")
            forced = wf
    context, included, included_contents = discover_context(signal, stacks, forced)
    if not included:
        print("[DISCOVER] WARNING: no files resolved.")

    observations = scan_observations(included_contents)
    obs_text = observations_text(observations)

    # ── AI STAGE 1 + 2: DIAGNOSIS ──
    print("\n━━━ AI STAGE 1: FACTS / STAGE 2: ROOT CAUSE ━━━")
    try:
        if FAST_MODE:
            facts, cause = ai_facts_and_cause(signal, context, stacks, obs_text)
        else:
            facts = ai_extract_facts(signal, context, stacks, obs_text)
            cause = ai_root_cause(facts, signal, context, stacks, obs_text)
    except Exception as exc:
        print(f"[ERROR] AI stage 1/2 failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI diagnosis failed: {exc}", run_url)
        sys.exit(2)

    root_cause = cause["root_cause"]
    solution   = cause["solution"]
    commit_msg = cause["commit_message"]
    confidence = cause["confidence"]

    print("\n  ── diagnosis ──")
    print(f"  CAUSE      : {root_cause}")
    print(f"  SOLUTION   : {solution or '(none)'}")
    print(f"  confidence : {confidence:.0%}")

    # ── CONFIDENCE GATE (before spending rewrite calls) ──
    if confidence < 0.5 and not observations:
        # Observations are verified facts — if they exist, there IS something
        # concrete to fix regardless of how unsure the model's diagnosis was.
        print(f"[GATE] Confidence {confidence:.0%} too low and no verified "
              "observations — escalating instead of guessing.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({confidence:.0%}). Root cause: {root_cause}",
                       run_url)
        sys.exit(0)

    # ── AI STAGE 3: WHOLE-FILE REWRITES ──
    # Candidate files: files the AI's diagnosis implicates + files with
    # verified observations + everything shown in context (the AI signals
    # "no bug here" by returning the file unchanged, which we then skip —
    # the decision of what needs changing stays with the AI).
    candidates = []
    for rel in included:
        if rel not in candidates:
            candidates.append(rel)
    for rel in facts.get("failing_files", []) or []:
        rel = _relstrip(str(rel).strip())
        if rel in included_contents and rel not in candidates:
            candidates.append(rel)
    obs_by_file = {}
    for o in observations:
        obs_by_file.setdefault(o["file"], []).append(o)
    # files with observations go first — most likely to need work
    candidates.sort(key=lambda r: (r not in obs_by_file,))
    candidates = candidates[:MAX_FILES_FIXED]

    rewrites, reasons, verify_confs, reject_reasons = {}, {}, [], []

    def attempt_rewrite(rel, extra_note=""):
        original = included_contents.get(rel)
        if original is None or len(original) > MAX_REWRITE_CHARS:
            reject_reasons.append(f"{rel}: too large to rewrite safely")
            return False
        if _is_blocked(rel) or ".." in rel or rel.startswith(("/", "~")):
            reject_reasons.append(f"{rel}: blocked/unsafe path")
            return False
        file_obs = observations_text(obs_by_file.get(rel, []))
        new = ai_rewrite_file(rel, rewrites.get(rel, original), facts, cause,
                              signal, stacks, file_obs, extra_note)
        if new is None:
            reject_reasons.append(f"{rel}: no extractable rewrite")
            return False
        base = included_contents[rel]  # always validate against the on-disk original
        if new.strip() == base.strip() and rel not in rewrites:
            print(f"[S3-REWRITE] {rel}: AI returned file unchanged — no bug found here")
            return False
        ok, why = validate_rewrite(rel, base, new)
        if not ok:
            print(f"  ✗ {rel} — {why}", file=sys.stderr)
            reject_reasons.append(f"{rel}: {why}")
            return False
        rewrites[rel] = new
        reasons[rel] = (f"AI whole-file rewrite ({_diff_stat(base, new)}); "
                        f"root cause: {root_cause}")[:300]
        return True

    print("\n━━━ AI STAGE 3: WHOLE-FILE REWRITE (one call per file) ━━━")
    for rel in candidates:
        print(f"\n[S3-REWRITE] → {rel}"
              + (f" ({len(obs_by_file.get(rel, []))} observation(s))" if rel in obs_by_file else ""))
        try:
            attempt_rewrite(rel)
        except Exception as exc:
            print(f"[ERROR] rewrite of {rel} failed: {exc}", file=sys.stderr)
            reject_reasons.append(f"{rel}: {exc}")

    # ── OBSERVATION VERIFICATION LOOP ──
    # Python grades the AI's work: re-run the static scans on the rewritten
    # content. Anything unresolved goes BACK to the AI with a pointed note —
    # Python never patches it. Bounded by MAX_AI_ROUNDS.
    unresolved = []
    if observations:
        for round_no in range(2, MAX_AI_ROUNDS + 1):
            merged_contents = {rel: rewrites.get(rel, c)
                               for rel, c in included_contents.items()}
            unresolved = scan_observations(merged_contents)
            if not unresolved:
                print("[VERIFY-OBS] all observations resolved by the AI's rewrites ✓")
                break
            print(f"\n━━━ ROUND {round_no}/{MAX_AI_ROUNDS}: RE-PROMPT — "
                  f"{len(unresolved)} observation(s) unresolved ━━━")
            progressed = False
            by_file = {}
            for o in unresolved:
                by_file.setdefault(o["file"], []).append(o)
            for rel, olist in by_file.items():
                note = ("Your previous rewrite did NOT resolve these verified "
                        "problems — they are still present. Fix ALL of them this "
                        "time, in addition to keeping your earlier fixes:\n"
                        + observations_text(olist))
                print(f"[ROUND {round_no}] re-prompting {rel} "
                      f"({len(olist)} unresolved)")
                try:
                    if attempt_rewrite(rel, extra_note=note):
                        progressed = True
                except Exception as exc:
                    print(f"[ROUND {round_no}] {rel} failed: {exc}", file=sys.stderr)
            if not progressed:
                print(f"[ROUND {round_no}] no progress — stopping re-prompts.")
                break
        else:
            round_no = MAX_AI_ROUNDS
        merged_contents = {rel: rewrites.get(rel, c)
                           for rel, c in included_contents.items()}
        unresolved = scan_observations(merged_contents)

    if not rewrites:
        detail = "; ".join(reject_reasons) or "AI proposed no change to any shown file"
        print(f"[ERROR] AI produced no usable fixes. {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI produced no usable fixes. Root cause: {root_cause}\n\nDetail: {detail}",
                       run_url)
        sys.exit(3)

    # ── AI STAGE 4: SELF-REVIEW (diff-level, per file) ──
    if not SKIP_SELF_VERIFY:
        print("\n━━━ AI STAGE 4: SELF-REVIEW OF DIFFS ━━━")
        for rel in list(rewrites.keys()):
            keep, vconf = ai_verify_diff(rel, included_contents[rel], rewrites[rel])
            if vconf is not None:
                verify_confs.append(vconf)
            if not keep:
                # exception: a rewrite that resolves verified observations is
                # backed by ground truth — don't let the (weaker) reviewer
                # veto objectively-confirmed progress
                had_obs = rel in obs_by_file
                still_bad = any(o["file"] == rel for o in unresolved)
                if had_obs and not still_bad:
                    print(f"[S4-VERIFY] overriding rejection of {rel}: rewrite "
                          "verifiably resolved its observations")
                else:
                    print(f"[S4-VERIFY] dropping rewrite of {rel}")
                    del rewrites[rel]

    if not rewrites:
        print("[ERROR] Self-review rejected every rewrite.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI rejected its own fixes on review. Root cause: {root_cause}", run_url)
        sys.exit(3)

    # ── FINAL CONFIDENCE GATE ──
    vc = (sum(verify_confs) / len(verify_confs)) if verify_confs else None
    combined = confidence if vc is None else (confidence + vc) / 2
    resolved_any_obs = bool(observations) and len(unresolved) < len(observations)
    if combined < 0.5 and not resolved_any_obs:
        print(f"[GATE] Combined confidence {combined:.0%} too low — escalating.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({combined:.0%}). Root cause: {root_cause}", run_url)
        sys.exit(0)

    # ── APPLY ──
    print("\n━━━ APPLY ━━━")
    if args.dry_run:
        for rel, new in rewrites.items():
            print(f"  would write {rel} ({_diff_stat(included_contents[rel], new)})")
        if unresolved:
            print(f"  unresolved observations: {len(unresolved)}")
        sys.exit(0)

    written, originals = write_rewrites(rewrites, reasons)
    if not written:
        sys.exit(3)

    # ── RUN TESTS ──
    print("\n━━━ RUN TESTS ━━━")
    if not args.skip_tests:
        if not run_tests(stacks):
            revert_files(originals)
            if token and repo:
                open_issue(token, repo,
                           f"Fix applied but tests failed — reverted. Root cause: {root_cause}",
                           run_url)
            sys.exit(5)
    else:
        print("[TEST] Skipped (--skip-tests)")

    # ── COMMIT + PR ──
    print("\n━━━ COMMIT + PR ━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)

    unresolved_note = ""
    if unresolved:
        unresolved_note = observations_text(unresolved)
        print(f"[WARN] {len(unresolved)} observation(s) remain unresolved — "
              "flagging in PR and opening an issue.")
        if token and repo:
            open_issue(token, repo,
                       "Auto-fixer resolved part of the failure but these verified "
                       f"problems remain after {MAX_AI_ROUNDS} AI rounds:\n\n"
                       f"{unresolved_note}", run_url)

    diff_stats = {rel: _diff_stat(included_contents[rel], rewrites[rel]) for rel in written}
    if token and repo:
        open_pr(token, repo, branch, commit_msg, root_cause, written, reasons,
                diff_stats, facts.get("summary", ""), solution, unresolved_note)
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")
    if unresolved:
        print(f"  unresolved : {len(unresolved)} observation(s) — see issue")


if __name__ == "__main__":
    main()