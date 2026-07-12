#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — agentic investigation flow.

Pipeline:

    GitHub Actions Pipeline Failed
        → Python: collect initial evidence (logs, exit code, repo tree, git diff)
        → Python: distill a FOCUSED issue statement (primary error, failing
              step, AND the command that triggered it)
        → AI Investigation Agent (multi-turn loop, ALWAYS runs):
              reads the focused issue + evidence → forms hypothesis →
              requests more files if needed → Python fetches requested files
              (read-only) → loop → confirms root cause
        → AI Patch Generation Agent: emits the concrete fix (flat evidence→corrected issues)
        → Python Validation Engine: format/path/syntax/YAML/JSON checks,
              secret scanning, dangerous-command detection, safe patch verification
        → Apply changes → Execute build & tests
        → on failure: collect new failure evidence → AI reviews it → retry or stop
        → on success: commit, push, open PR

Design notes:
  * Model defaults to gemma3:4b. Override with OLLAMA_MODEL.
  * No deterministic pre‑scan ever. AI drives every investigation step.
  * The investigation prompt now explicitly instructs the model to confirm
    immediately when a workflow YAML shows a misspelled filename.
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
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "gemma3:4b")
AI_TIMEOUT     = int(os.environ.get("AI_TIMEOUT", "210"))
MAX_RETRIES    = int(os.environ.get("AI_MAX_RETRIES", "2"))
RETRY_BACKOFF  = [20, 20]
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")

INVESTIGATION_TIMEOUT = int(os.environ.get("INVESTIGATION_TIMEOUT", str(AI_TIMEOUT)))
INVESTIGATION_RETRIES = int(os.environ.get("INVESTIGATION_RETRIES", "1"))
INVESTIGATION_FIRST_TURN_EXTRA = int(os.environ.get("INVESTIGATION_FIRST_TURN_EXTRA", "60"))

TOTAL_TIME_BUDGET = int(os.environ.get("TOTAL_TIME_BUDGET", "600"))
_run_start_time = None


def _elapsed() -> float:
    return time.time() - _run_start_time if _run_start_time else 0.0


def _budget_exceeded() -> bool:
    return _run_start_time is not None and _elapsed() >= TOTAL_TIME_BUDGET


# ── Agentic-loop bounds ──────────────────────────────────────────────────────
MAX_INVESTIGATION_TURNS = int(os.environ.get("MAX_INVESTIGATION_TURNS", "4"))
MAX_FILES_PER_REQUEST   = int(os.environ.get("MAX_FILES_PER_REQUEST", "3"))
MAX_REPAIR_ROUNDS       = int(os.environ.get("MAX_REPAIR_ROUNDS", "2"))

# ── Prompt / context budget ───────────────────────────────────────────────────
MAX_ERROR_LINES   = 14
MAX_SUPPORTING_LINES = 10
MAX_FILE_CHARS    = int(os.environ.get("MAX_FILE_CHARS", "2500"))
MAX_TOTAL_CONTEXT = int(os.environ.get("MAX_TOTAL_CONTEXT", "5000"))
MAX_FILES_FIXED   = 4
MAX_PROMPT_CHARS  = int(os.environ.get("MAX_PROMPT_CHARS", "8000"))
INVESTIGATION_DIFF_CHARS = int(os.environ.get("INVESTIGATION_DIFF_CHARS", "1200"))
MAX_TRACEBACK_LINES = 12

# ── Git flow ──────────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_HIDDEN   = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [
    r"\.?github/CODEOWNERS$",
    r"(^|/)\.env(\..*)?$",
    r".*\.pem$", r".*\.key$", r".*id_rsa.*", r".*id_ed25519.*",
    r".*secrets?\.ya?ml$", r".*\.tfstate(\.backup)?$",
    r"(^|/)\.npmrc$", r"(^|/)\.pypirc$",
]
WORKFLOW_PATTERN = re.compile(r"\.?github/workflows/.*\.ya?ml$")

READ_BLOCKED_PATTERNS = [p for p in BLOCKED_PATTERNS if "CODEOWNERS" not in p]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             "dist", "build", ".pytest_cache", "target", "out", "vendor",
             ".idea", ".vscode", "coverage", "tmp", "temp", "logs"}
MAX_FILE_SIZE_BYTES = 100_000

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
MAX_ADDED_LINES_PER_FIX = 40

DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r'rm\s+-rf\s+/(?:\s|$)'),
    re.compile(r'curl[^\n]{0,120}\|\s*(sh|bash)\b'),
    re.compile(r'wget[^\n]{0,120}\|\s*(sh|bash)\b'),
    re.compile(r'chmod\s+777'),
    re.compile(r'\bos\.system\('),
    re.compile(r'subprocess\.\w+\([^)]*shell\s*=\s*True'),
    re.compile(r'\beval\('),
    re.compile(r':\(\)\{\s*:\|:&\s*\};:'),
]

DOCKERFILE_REF_PATTERNS = [
    (re.compile(r'^\s*COPY\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "COPY"),
    (re.compile(r'^\s*ADD\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "ADD"),
    (re.compile(r'-r\s+(\S+\.txt)'), "pip install -r"),
    (re.compile(r'CMD\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "CMD"),
    (re.compile(r'ENTRYPOINT\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "ENTRYPOINT"),
]
DOCKERFILE_FROM_PYTHON = re.compile(r'^(\s*FROM\s+)python:([^\s]+)(.*)$', re.I | re.M)
VALID_PYTHON_MINORS = set(range(8, 14))
DEFAULT_PYTHON_VERSION = os.environ.get("DEFAULT_PYTHON_VERSION", "3.12")

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

# Path helpers
def _relstrip(rel): return rel[2:] if rel.startswith("./") else rel

def _is_blocked(fp):
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_HIDDEN):
        return True
    return any(re.search(p, fp) for p in BLOCKED_PATTERNS)

def _is_read_blocked(fp):
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

# ── STAGE 0 – COLLECT EVIDENCE ────────────────────────────────────────────────
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

GH_ERROR_ANNOTATION = re.compile(r'^##\[error\](.*)$', re.M)
GH_GROUP_START       = re.compile(r'^##\[group\](.*)$', re.M)
TRACEBACK_START      = re.compile(r'^Traceback \(most recent call last\):', re.M)
EXCEPTION_LINE       = re.compile(r'^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception|Warning)\b')
GENERIC_GH_ERROR_ANNOTATION = re.compile(r'^Process completed with exit code', re.I)

STRONG_ERROR_LINE_PATTERNS = [
    re.compile(r'\bNo such file or directory\b', re.I),
    re.compile(r'\bcan(?:no|\')t open file\b', re.I),
    re.compile(r'\b(?:ModuleNotFoundError|ImportError)\b'),
    re.compile(r'\bNo module named\b', re.I),
    re.compile(r'\bERROR: Could not (?:open|find|install)\b', re.I),
    re.compile(r'\bcould not find\b', re.I),
    re.compile(r'\bnot found\b', re.I),
    re.compile(r'^npm ERR!', re.I),
    re.compile(r'\bfailed to solve\b', re.I),
    re.compile(r'\bcommand not found\b', re.I),
    re.compile(r'\bpermission denied\b', re.I),
    re.compile(r'^\s*E\s+\w+Error\b'),
    re.compile(r'\b\w+Error\b:'),
    re.compile(r'^\s*error\b[: ]', re.I),
    re.compile(r'\bfatal\b', re.I),
]


def _best_error_line(log_text: str) -> str:
    lines = [l.strip() for l in log_text.splitlines() if l.strip()]
    lines = [l for l in lines if not any(n in l.lower() for n in NOISE_KEYWORDS)]
    for pat in STRONG_ERROR_LINE_PATTERNS:
        hits = [l for l in lines if pat.search(l)]
        if hits:
            return hits[-1][:300]
    return ""


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


def extract_focused_failure(log_text: str) -> dict:
    gh_errors = [m.group(1).strip() for m in GH_ERROR_ANNOTATION.finditer(log_text)
                 if m.group(1).strip()]

    failing_step = ""
    groups = list(GH_GROUP_START.finditer(log_text))
    if groups:
        anchor_match = GH_ERROR_ANNOTATION.search(log_text)
        anchor = anchor_match.start() if anchor_match else len(log_text)
        for g in groups:
            if g.start() < anchor:
                failing_step = g.group(1).strip()
            else:
                break

    traceback_text = ""
    tb_match = TRACEBACK_START.search(log_text)
    if tb_match:
        tb_lines = log_text[tb_match.start():].splitlines()
        end = len(tb_lines)
        for i in range(1, len(tb_lines)):
            l = tb_lines[i]
            if l.strip() == "":
                end = i
                break
            if EXCEPTION_LINE.match(l.strip()):
                end = i + 1
                break
        tb_lines = tb_lines[:end]
        if len(tb_lines) > MAX_TRACEBACK_LINES:
            tb_lines = tb_lines[:1] + ["    ...(truncated)..."] + tb_lines[-(MAX_TRACEBACK_LINES - 1):]
        traceback_text = "\n".join(tb_lines)

    specific_gh_errors = [e for e in gh_errors if not GENERIC_GH_ERROR_ANNOTATION.match(e)]
    exc_lines = []
    if traceback_text:
        exc_lines = [l.strip() for l in traceback_text.splitlines()
                     if EXCEPTION_LINE.match(l.strip())]

    if specific_gh_errors:
        primary_message = specific_gh_errors[-1]
    elif exc_lines:
        primary_message = exc_lines[-1]
    else:
        best_plain = _best_error_line(log_text)
        if best_plain:
            primary_message = best_plain
        elif gh_errors:
            primary_message = gh_errors[-1]
        else:
            primary_message = ""

    file_refs = []
    for h in FILE_REF_HINTS:
        for m in h.finditer(log_text):
            ref = m.group(0)
            if ref not in file_refs:
                file_refs.append(ref)
    file_refs = file_refs[:8]

    # Command context: the line immediately before the error (best guess)
    command_context_lines = []
    if primary_message:
        log_lines = log_text.splitlines()
        msg_line_idx = None
        for i, line in enumerate(log_lines):
            if primary_message in line:
                msg_line_idx = i
                break
        if msg_line_idx is not None:
            # grab the line right before the error
            if msg_line_idx > 0:
                prev = log_lines[msg_line_idx - 1].strip()
                if prev and not any(n in prev.lower() for n in NOISE_KEYWORDS):
                    command_context_lines.append(prev)
            # also try to find a line that contains the same filename
            error_filename = None
            for ref in file_refs:
                if ref in primary_message:
                    error_filename = ref
                    break
            if not error_filename:
                m = re.search(r"'([\w./\-]+)'", primary_message)
                if m:
                    error_filename = m.group(1)
            if error_filename and not command_context_lines:
                for j in range(max(0, msg_line_idx - 3), msg_line_idx):
                    candidate = log_lines[j].strip()
                    if candidate and error_filename in candidate:
                        command_context_lines.append(candidate)
                        break

    return {
        "primary_message": primary_message,
        "failing_step": failing_step,
        "traceback": traceback_text,
        "file_refs": file_refs,
        "gh_errors": gh_errors[:5],
        "command_context": command_context_lines,
    }


def format_focused_issue(focused: dict, exit_code: str) -> str:
    parts = [f"Exit code: {exit_code}"]
    if focused.get("failing_step"):
        parts.append(f"Failing step: {focused['failing_step']}")
    if focused.get("command_context"):
        parts.append("Command(s) that triggered the error:")
        for line in focused["command_context"]:
            parts.append(f"  > {line}")
    if focused.get("primary_message"):
        parts.append(f"Primary error: {focused['primary_message']}")
    if focused.get("traceback"):
        parts.append(f"Traceback:\n{focused['traceback']}")
    if focused.get("file_refs"):
        parts.append(f"File references seen in the log: {', '.join(focused['file_refs'])}")
    others = [e for e in focused.get("gh_errors", []) if e != focused.get("primary_message")]
    if others:
        parts.append("Other error annotations:\n" + "\n".join(f"- {o}" for o in others[:4]))
    return "\n".join(parts) if parts else "(no focused signal extracted)"


def trim_supporting_signal(signal: str, focused: dict, max_lines: int = MAX_SUPPORTING_LINES) -> str:
    tb_lines = set(focused.get("traceback", "").splitlines())
    lines = [l for l in signal.splitlines() if l not in tb_lines]
    lines = list(dict.fromkeys(lines))[-max_lines:]
    return "\n".join(lines) if lines else "(fully covered by the issue summary above)"


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


# ── AI plumbing ──────────────────────────────────────────────────────────────
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
                   num_ctx=OLLAMA_NUM_CTX, tag="AI", retries=MAX_RETRIES,
                   timeout=AI_TIMEOUT) -> str:
    endpoint, fmt = _detect_endpoint()
    if fmt == "openai":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": temperature,
                   "max_tokens": num_predict, "stream": True}
    else:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt,
                   "options": {"temperature": temperature, "num_predict": num_predict,
                               "num_ctx": num_ctx},
                   "keep_alive": OLLAMA_KEEP_ALIVE, "stream": True}
        if schema:
            payload["format"] = schema
    print(f"[{tag}] {endpoint} ({fmt}) | prompt {len(prompt)} chars | model {OLLAMA_MODEL}")

    last = None
    for attempt in range(retries):
        try:
            t0, collected = time.time(), []
            resp = requests.post(endpoint, json=payload, timeout=(10, timeout), stream=True)
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
            if attempt < retries - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
                print(f"[{tag}] timeout after {timeout}s; retrying in {wait}s "
                      f"(attempt {attempt+1}/{retries})...")
                time.sleep(wait)
            else:
                print(f"[{tag}] timeout after {timeout}s (no retries left).")
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {endpoint}: {exc}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}")
    raise RuntimeError(f"Ollama did not respond after {retries} attempts: {last}")


def warm_up_model() -> bool:
    endpoint, fmt = _detect_endpoint()
    warm_timeout = int(os.environ.get("OLLAMA_WARMUP_TIMEOUT", "240"))
    print(f"[WARMUP] pinging {OLLAMA_MODEL} to load it into memory "
          f"(timeout {warm_timeout}s) ...")
    t0 = time.time()
    try:
        if fmt == "openai":
            payload = {"model": OLLAMA_MODEL, "prompt": "ok", "max_tokens": 1, "stream": False}
        else:
            payload = {"model": OLLAMA_MODEL, "prompt": "ok",
                       "options": {"num_predict": 1, "num_ctx": OLLAMA_NUM_CTX},
                       "keep_alive": OLLAMA_KEEP_ALIVE, "stream": False}
        resp = requests.post(endpoint, json=payload, timeout=(10, warm_timeout))
        resp.raise_for_status()
        print(f"[WARMUP] model resident in {time.time()-t0:.1f}s — held for {OLLAMA_KEEP_ALIVE}.")
        return True
    except requests.exceptions.Timeout:
        print(f"[WARMUP] warm-up timed out after {warm_timeout}s — continuing anyway.", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[WARMUP] warm-up call failed ({exc}) — continuing anyway.", file=sys.stderr)
        return False


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


# ── AI INVESTIGATION AGENT (improved prompt) ─────────────────────────────────
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

Focus ONLY on the "## ISSUE TO SOLVE" section below — that is Python's own \
distillation of what actually broke this run. The supporting log lines, diff, \
and repo tree underneath are context.

Output ONLY one JSON object. No markdown fences.

Each turn, choose exactly one status:
  - "need_more_info": you cannot confirm a root cause yet. Set "requested_files" \
to up to 3 EXACT paths from the "## Repository tree" list.
  - "root_cause_confirmed": you are confident. Fill in "root_cause", "solution", \
"confidence", "commit_message", AND "findings".

**CRITICAL RULE**: If the primary error mentions a missing file (e.g., \
`requiements.txt`) and you have already read a workflow YAML file that contains \
that misspelled filename on a `pip install -r ...` line, you MUST answer with \
`root_cause_confirmed` immediately. Do NOT request more files. The typo is the \
root cause. Report every misspelled filename you see as a separate finding.

Schema when confirming:
{"analysis":"...","status":"root_cause_confirmed","root_cause":"...","solution":"...","confidence":0.0-1.0,"commit_message":"fix: short","findings":[{"issue":"...","root_cause":"...","solution":"..."}]}
"""


def _closest_allowed_file(requested: str, allowed_files: set) -> str:
    req_base = Path(requested).name.lower()
    by_base = {}
    for f in allowed_files:
        by_base.setdefault(Path(f).name.lower(), []).append(f)
    close = difflib.get_close_matches(req_base, by_base.keys(), n=1, cutoff=0.6)
    if not close:
        return ""
    candidates = by_base[close[0]]
    if len(candidates) == 1:
        return candidates[0]
    best = difflib.get_close_matches(requested, candidates, n=1, cutoff=0.0)
    return best[0] if best else candidates[0]


def _build_investigation_prompt(signal, exit_code, repo_tree, git_diff,
                                evidence: dict, last_turn: bool,
                                notes: list = None, issue_block: str = "") -> str:
    ev_parts, total = [], 0
    for f, c in evidence.items():
        block = f"### {f}\n```\n{c}\n```"
        if total + len(block) > MAX_TOTAL_CONTEXT and ev_parts:
            break
        ev_parts.append(block)
        total += len(block)
    ev_text = "\n\n".join(ev_parts) if ev_parts else "(none read yet)"
    turn_note = ("\n## THIS IS YOUR FINAL TURN. You MUST return "
                 "status \"root_cause_confirmed\" now, using your best "
                 "hypothesis from the evidence so far. Lower your confidence "
                 "if you're not fully sure.\n" if last_turn else "")
    notes_section = ""
    if notes:
        notes_section = ("\n## Notes on your last request:\n"
                         + "\n".join(f"- {n}" for n in notes) + "\n")
    issue_section = f"\n## ISSUE TO SOLVE\n{issue_block}\n" if issue_block else ""
    diff_trimmed = git_diff if len(git_diff) <= INVESTIGATION_DIFF_CHARS \
        else git_diff[:INVESTIGATION_DIFF_CHARS] + "\n...(diff truncated)"

    def _assemble(ev_block):
        return (
            f"{INVESTIGATE_SYSTEM}{turn_note}{issue_section}{notes_section}\n"
            f"## Supporting log lines:\n```\n{signal}\n```\n"
            f"## Exit code: {exit_code}\n"
            f"## Git diff:\n```\n{diff_trimmed}\n```\n"
            f"## Repository tree:\n{repo_tree}\n"
            f"## Evidence gathered so far:\n{ev_block}\n\n"
            f"Respond with the JSON described above."
        )

    prompt = _assemble(ev_text)
    if len(prompt) > MAX_PROMPT_CHARS:
        overflow = len(prompt) - MAX_PROMPT_CHARS
        if overflow < len(ev_text):
            ev_text = ev_text[:len(ev_text) - overflow] + "\n...(evidence truncated)"
        else:
            ev_text = "(evidence omitted — request specific file)"
        prompt = _assemble(ev_text)
    return prompt


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
                   issue_block: str = "") -> dict:
    evidence = {}
    log, result = [], None
    dead_ends = set()
    pending_notes = []
    stall_count = 0
    got_any_model_response = False
    timed_out_cold = False

    for turn in range(1, MAX_INVESTIGATION_TURNS + 1):
        if _budget_exceeded():
            print(f"[INVESTIGATE] time budget exceeded before turn {turn}.")
            break
        last_turn = (turn == MAX_INVESTIGATION_TURNS)
        turn_timeout = (INVESTIGATION_TIMEOUT + INVESTIGATION_FIRST_TURN_EXTRA
                        if turn == 1 else INVESTIGATION_TIMEOUT)
        prompt = _build_investigation_prompt(signal, exit_code, repo_tree,
                                             git_diff, evidence, last_turn,
                                             pending_notes, issue_block=issue_block)
        pending_notes = []
        try:
            raw = _stream_ollama(prompt, INVESTIGATE_SCHEMA, num_predict=800,
                                 temperature=0.05, tag=f"INVESTIGATE-T{turn}",
                                 timeout=turn_timeout, retries=INVESTIGATION_RETRIES)
        except Exception as exc:
            is_timeout = "timed out" in str(exc).lower() or "timeout" in str(exc).lower()
            print(f"[INVESTIGATE] turn {turn} failed: {exc}", file=sys.stderr)
            if is_timeout and turn == 1 and not got_any_model_response:
                timed_out_cold = True
                if _budget_exceeded():
                    break
                print("[INVESTIGATE] first turn timed out — retrying with lean prompt.")
                lean_prompt = _build_investigation_prompt(
                    "(omitted)", exit_code, repo_tree, "(omitted)", evidence,
                    last_turn, None, issue_block=issue_block)
                try:
                    raw = _stream_ollama(lean_prompt, INVESTIGATE_SCHEMA, num_predict=800,
                                         temperature=0.05, tag="INVESTIGATE-T1-LEAN",
                                         timeout=INVESTIGATION_TIMEOUT + INVESTIGATION_FIRST_TURN_EXTRA,
                                         retries=INVESTIGATION_RETRIES)
                except Exception as exc2:
                    print(f"[INVESTIGATE] lean retry also failed: {exc2}", file=sys.stderr)
                    break
            else:
                break

        data = _json_from(raw) or {}
        if data:
            got_any_model_response = True
        status = data.get("status", "")
        analysis = data.get("analysis", "")
        print(f"[INVESTIGATE] turn {turn}: status={status} | "
              f"{analysis[:160] if analysis else '(no analysis)'}")

        # If the model returned an empty or malformed response, force final decision
        if not status and turn > 1:
            print("[INVESTIGATE] empty status from model — treating as stall.")
            stall_count += 2  # force final decision below

        log.append({"turn": turn, "status": status, "analysis": analysis[:200]})

        if status == "root_cause_confirmed":
            result = _finalize_investigation(data, False)
            break

        if status == "need_more_info":
            requested = [f.strip() for f in (data.get("requested_files") or [])
                        if isinstance(f, str) and f.strip()]
            to_read = []
            for f in requested[:MAX_FILES_PER_REQUEST]:
                if f in evidence or f in dead_ends:
                    continue
                if f in allowed_files:
                    to_read.append(f)
                    continue
                match = _closest_allowed_file(f, allowed_files)
                if match and match not in evidence:
                    print(f"[INVESTIGATE]   ~ requested '{f}' — not in tree, "
                          f"reading closest match '{match}' instead")
                    to_read.append(match)
                    pending_notes.append(f"You asked for '{f}', which doesn't exist. "
                                         f"The closest real file, '{match}', was read instead.")
                else:
                    print(f"[INVESTIGATE]   ✗ requested '{f}' — denied")
                    dead_ends.add(f)
                    pending_notes.append(f"'{f}' does not exist — do NOT request it again.")
            for f in to_read:
                content = _read_evidence_file(f)
                evidence[f] = content if content is not None else "(could not read)"
                print(f"[INVESTIGATE]   + read {f} ({len(evidence[f])} chars)")

            stalled = (not to_read and requested and not last_turn)
            stall_count = stall_count + 1 if stalled else 0
            force_now = last_turn or stall_count >= 2

            if force_now:
                if stall_count >= 2 and not last_turn:
                    print("[INVESTIGATE] no new evidence for 2 turns — forcing final decision.")
                if _budget_exceeded():
                    result = _finalize_investigation(data, True)
                    break
                final_prompt = _build_investigation_prompt(
                    signal, exit_code, repo_tree, git_diff, evidence, True,
                    pending_notes, issue_block=issue_block)
                try:
                    raw2 = _stream_ollama(final_prompt, INVESTIGATE_SCHEMA,
                                          num_predict=800, temperature=0.05,
                                          tag="INVESTIGATE-FINAL",
                                          timeout=INVESTIGATION_TIMEOUT,
                                          retries=INVESTIGATION_RETRIES)
                    data2 = _json_from(raw2) or data
                except Exception as exc:
                    print(f"[INVESTIGATE] final turn failed: {exc}", file=sys.stderr)
                    data2 = data
                result = _finalize_investigation(data2, True)
                break
            continue

        if data.get("root_cause"):
            result = _finalize_investigation(data, True)
        break

    if result is None:
        if not got_any_model_response:
            failure_mode = "infra_timeout" if timed_out_cold else "no_model_response"
            root_cause = ("model did not return any usable response (infra/latency problem)")
        else:
            failure_mode = "not_converged"
            root_cause = "unknown — investigation did not converge"
        result = {"root_cause": root_cause, "solution": "", "confidence": 0.0,
                  "commit_message": "fix: auto-fixer change", "findings": [],
                  "failure_mode": failure_mode}
    else:
        result["failure_mode"] = None
    result["evidence"] = evidence
    result["investigation_log"] = log
    return result


# ── PATCH GENERATION (improved prompt) ──────────────────────────────────────
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

Stay scoped to the "## ISSUE TO SOLVE" section if one is present.

Output ONLY one JSON object. No markdown fences.

"issues" = ONE ENTRY PER BUG. Each entry:
  - "file": exact ### header path
  - "problem": one sentence
  - "evidence": EXACT text from the file contents below (short) – copy the WRONG text.
  - "corrected": same text with ONLY the bug fixed. MUST be different from "evidence".

Schema:
{"issues":[{"file":"...","problem":"...","evidence":"...","corrected":"..."}]}

**REMEMBER**: "evidence" and "corrected" must be DIFFERENT strings.
"""


def ai_generate_patch(root_cause: str, solution: str, evidence: dict,
                      retry_note: str = "", issue_block: str = "") -> list:
    parts = [f"### {f}\n```\n{c}\n```" for f, c in evidence.items()]
    context = "\n\n".join(parts) if parts else "(no evidence files were read)"
    issue_section = (f"## ISSUE TO SOLVE (stay scoped to this):\n{issue_block}\n\n"
                     if issue_block else "")
    retry_section = (f"## Note: a previous attempt at this fix failed:\n"
                     f"{retry_note}\nDo not repeat the same change.\n\n" if retry_note else "")
    prompt = (f"{PATCH_SYSTEM}\n\n{issue_section}{retry_section}"
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


# ── REVIEW AGENT ─────────────────────────────────────────────────────────────
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
  - "retry": provide updated root_cause and solution.
  - "stop": a human should look at it.
Focus on the "## NEW ISSUE" section if present.
Output ONLY one JSON object:
{"decision":"retry","root_cause":"...","solution":"...","confidence":0.0-1.0}
or
{"decision":"stop","root_cause":"why this can't be safely auto-fixed","solution":"","confidence":0.0}
"""


def ai_review_failure(prior_root_cause: str, prior_solution: str, new_signal: str,
                      issue_block: str = "") -> dict:
    issue_section = f"## NEW ISSUE\n{issue_block}\n\n" if issue_block else ""
    prompt = (f"{REVIEW_SYSTEM}\n\n{issue_section}## Original root cause: {prior_root_cause}\n"
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


# ── PAIR + LOCATE / VALIDATE / APPLY ────────────────────────────────────────
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
            print(f"[LOCATE] issue #{n}: relocated to '{target}'")
        elif len(holders) > 1:
            rejects.append(f"issue #{n} ({file or '?'}): evidence ambiguous")
            continue
        else:
            rejects.append(f"issue #{n} ({file or '?'}): evidence not found — rejected")
            continue

        entry = fixes_by_file.setdefault(
            target, {"file": target, "reason": prob or "AI fix", "edits": []})
        edit = {"find": ev, "replace": cor}
        if edit not in entry["edits"]:
            entry["edits"].append(edit)
            if prob and prob not in entry["reason"]:
                entry["reason"] = (entry["reason"] + "; " + prob).lstrip("; ")[:300]

    return list(fixes_by_file.values()), rejects


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
            print(f"[EDIT] edit #{i+1} already satisfied")
            continue
        nf = "\n".join(l.strip() for l in find.splitlines())
        nc = "\n".join(l.strip() for l in content.splitlines())
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
                    break
            continue
        salvaged = _salvage_fragment(content, find, repl)
        if salvaged is not None and salvaged != content:
            print(f"[EDIT] Salvaged edit #{i+1}")
            content = salvaged
            continue
        return None, f"edit #{i+1} 'find' text not present — {find[:120]!r}"
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
    if isinstance(node, dict):
        return {k: _yaml_structure_signature(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_yaml_structure_signature(v) for v in node]
    return "<scalar>"


def _yaml_leaf_diffs(old, new, path=""):
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
    old_matches = list(WORKFLOW_PYVERSION_LINE.finditer(original_text))
    new_matches = list(WORKFLOW_PYVERSION_LINE.finditer(new_text))
    if len(old_matches) != len(new_matches):
        return new_text, False
    changed = False
    out = new_text
    for om, nm in zip(reversed(old_matches), reversed(new_matches)):
        o_major, o_minor = int(om.group(3)), int(om.group(4))
        if o_major == 3 and o_minor in VALID_PYTHON_MINORS:
            continue
        n_prefix, n_q1, _n_major, _n_minor, n_q2, n_rest = nm.groups()
        quote = n_q1 or n_q2 or '"'
        new_line = f"{n_prefix}{quote}{DEFAULT_PYTHON_VERSION}{quote}{n_rest}"
        if new_line != nm.group(0):
            changed = True
            print(f"[NORMALIZE] python-version forced to {DEFAULT_PYTHON_VERSION}")
        out = out[:nm.start()] + new_line + out[nm.end():]
    return out, changed


def validate_workflow_edit(original_text: str, new_text: str) -> tuple:
    try:
        old_doc = yaml.safe_load(original_text)
        new_doc = yaml.safe_load(new_text)
    except yaml.YAMLError as e:
        return False, f"YAML error: {e}"
    if not isinstance(old_doc, dict) or not isinstance(new_doc, dict):
        return False, "workflow did not parse to a mapping"
    if _yaml_structure_signature(old_doc) != _yaml_structure_signature(new_doc):
        return False, "edit changes workflow structure"
    diffs = _yaml_leaf_diffs(old_doc, new_doc)
    if not diffs:
        return False, "no effective change"
    if len(diffs) > MAX_WORKFLOW_VALUE_DIFFS:
        return False, f"edit touches {len(diffs)} values"
    for path, old_v, new_v in diffs:
        segments = [s.strip("]") for s in re.split(r"[.\[]", path)]
        if any(seg in SENSITIVE_WORKFLOW_KEYS for seg in segments):
            return False, f"edit touches sensitive field '{path}'"
        if isinstance(old_v, str) and isinstance(new_v, str):
            added = [l for l in new_v.splitlines() if l not in old_v.splitlines()]
            if scan_dangerous_commands("\n".join(added)):
                return False, f"edit introduces dangerous command pattern"
            secret_hits = scan_text_for_secrets(new_v)
            if secret_hits:
                return False, f"edit introduces a potential secret ({', '.join(secret_hits)})"
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
        return False, f"fix adds {len(added)} lines (max {MAX_ADDED_LINES_PER_FIX})"
    added_text = "\n".join(added)
    secret_hits = scan_text_for_secrets(added_text)
    if secret_hits:
        return False, f"potential secret in fix ({', '.join(secret_hits)})"
    danger_hits = scan_dangerous_commands(added_text)
    if danger_hits:
        return False, "fix introduces a dangerous command pattern"

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
                                 f"; python-version pinned to {DEFAULT_PYTHON_VERSION}").lstrip("; ")[:300]
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
    return list(dict.fromkeys(written)), originals, reasons


def revert_files(originals: dict):
    for file, content in originals.items():
        try:
            Path(file).write_text(content, encoding="utf-8")
            print(f"[REVERT] {file}")
        except Exception as exc:
            print(f"[REVERT] failed {file}: {exc}", file=sys.stderr)


# ── BUILD & TESTS ───────────────────────────────────────────────────────────
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
    dockerfiles = [f for f in written if "dockerfile" in Path(f).name.lower()]
    if not dockerfiles:
        return True, ""
    if subprocess.run(["bash", "-lc", "command -v docker"], capture_output=True).returncode != 0:
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
        print("[TEST] docker not found — skipping")
        return True, ""
    except subprocess.TimeoutExpired:
        print("[TEST] docker build timed out — not treated as failure")
        return True, ""


# ── GIT & PR ────────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    global _run_start_time
    _run_start_time = time.time()

    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer (agentic investigation)")
    ap.add_argument("--input", required=True, help="Path to CI failure log")
    ap.add_argument("--exit-code", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    print(f"[BUDGET] total wall-clock budget: {TOTAL_TIME_BUDGET}s")

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
        print(f"[SECURITY] redacting {len(redacted)} potential secret pattern(s): "
              f"{', '.join(redacted)}")
    log_text = redact_secrets(log_text)

    exit_code = get_exit_code(log_text, args.exit_code)
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[EVIDENCE] No error signal — nothing to fix.")
        sys.exit(0)

    focused = extract_focused_failure(log_text)
    issue_block = format_focused_issue(focused, exit_code)
    supporting_signal = trim_supporting_signal(signal, focused)
    print(f"[EVIDENCE] Focused issue: {(focused.get('primary_message') or '(none)')[:160]}")
    if focused.get("failing_step"):
        print(f"[EVIDENCE] Failing step: {focused['failing_step']}")
    if focused.get("command_context"):
        print(f"[EVIDENCE] Command context: {focused['command_context']}")

    git_diff  = get_git_diff()
    repo_tree, allowed_files = repo_tree_text()
    print(f"[EVIDENCE] exit_code={exit_code} | {len(allowed_files)} readable file(s) in tree")

    # ── AI INVESTIGATION AGENT ──
    print("\n━━━ AI INVESTIGATION AGENT ━━━")
    warm_up_model()

    investigation = ai_investigate(supporting_signal, exit_code, repo_tree, git_diff,
                                   allowed_files, issue_block=issue_block)
    root_cause = investigation["root_cause"]
    solution   = investigation["solution"]
    commit_msg = investigation["commit_message"]
    confidence = investigation["confidence"]
    evidence   = investigation["evidence"]
    findings   = investigation.get("findings", [])
    investigation_log = investigation.get("investigation_log")

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

    failure_mode = investigation.get("failure_mode")
    if failure_mode in ("infra_timeout", "no_model_response"):
        print(f"[GATE] Investigation produced no model response ({failure_mode}) — escalating.")
        if token and repo:
            open_issue(token, repo,
                       f"Auto-fixer could not run the model in time ({failure_mode}). "
                       f"The investigation call(s) to Ollama timed out before the model "
                       f"produced any output.\n\n"
                       f"**Issue Python identified (unused — model never ran):**\n"
                       f"```\n{issue_block[:1200]}\n```",
                       run_url)
        sys.exit(0)

    if confidence < 0.5:
        print(f"[GATE] Confidence {confidence:.0%} too low — escalating.")
        if token and repo:
            escalation_detail = (
                f"AI confidence too low ({confidence:.0%}). Root cause: {root_cause}\n\n"
                f"**Issue Python identified:**\n```\n{issue_block[:1500]}\n```\n\n"
                f"**Files read during investigation:** "
                f"{', '.join(evidence.keys()) or '(none)'}\n"
            )
            open_issue(token, repo, escalation_detail, run_url)
        sys.exit(0)

    # ── PATCH GENERATION + VALIDATION LOOP ──
    findings_text = "\n".join(
        f"- {f['issue']}: {f['root_cause']}" + (f" — fix: {f['solution']}" if f.get('solution') else "")
        for f in findings)
    patch_root_cause = (f"{root_cause}\n\nIndividual issues to fix (address EVERY one):\n{findings_text}"
                        if findings_text else root_cause)

    written, originals, fixes = [], {}, []
    success = False
    retry_note = ""
    retry_issue_block = ""

    for repair_round in range(1, MAX_REPAIR_ROUNDS + 1):
        if _budget_exceeded():
            print(f"[BUDGET] time budget exceeded before repair round {repair_round}.")
            if token and repo:
                open_issue(token, repo,
                           f"Auto-fixer hit its time budget ({TOTAL_TIME_BUDGET}s) before "
                           f"finishing. Root cause so far: {root_cause}", run_url)
            sys.exit(5)

        print(f"\n━━━ AI PATCH GENERATION AGENT (round {repair_round}/{MAX_REPAIR_ROUNDS}) ━━━")
        active_issue_block = retry_issue_block or issue_block
        try:
            issues = ai_generate_patch(patch_root_cause, solution, evidence,
                                       retry_note, active_issue_block)
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

        print(f"[REPAIR] round {repair_round} failed tests/build — reverting.")
        combined_output = (test_output + "\n" + docker_output).strip()
        retry_focused = extract_focused_failure(combined_output)
        retry_issue_block = format_focused_issue(retry_focused, "n/a (post-fix test/build run)")
        new_signal = extract_error_signal(combined_output) or combined_output[-1500:]
        new_signal = trim_supporting_signal(new_signal, retry_focused)
        revert_files(originals)
        written = []

        if repair_round == MAX_REPAIR_ROUNDS:
            if token and repo:
                open_issue(token, repo,
                           f"Fix applied but tests/build failed after {repair_round} "
                           f"attempt(s) — reverted. Root cause: {root_cause}\n\n"
                           f"**New issue after the fix:**\n```\n{retry_issue_block[:1500]}\n```",
                           run_url)
            sys.exit(5)

        print("\n━━━ AI REVIEWS NEW EVIDENCE ━━━")
        review = ai_review_failure(root_cause, solution, new_signal, retry_issue_block)
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

    print("\n━━━ COMMIT + PR ━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)
    if token and repo:
        open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
                investigation_log, list(evidence.keys()), solution, findings)
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()