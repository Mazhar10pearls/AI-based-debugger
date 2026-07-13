#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — agentic investigation flow.
(Hand-off hardening: patch agent now receives only the exact offending lines,
 forced to use them verbatim as evidence, and shown a minimal context window.)
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
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")

PATCH_TIMEOUT            = int(os.environ.get("PATCH_TIMEOUT", str(AI_TIMEOUT + 90)))
MAX_PATCH_EVIDENCE_CHARS = int(os.environ.get("MAX_PATCH_EVIDENCE_CHARS", "2000"))
PATCH_NUM_PREDICT        = int(os.environ.get("PATCH_NUM_PREDICT", "1200"))

INVESTIGATION_TIMEOUT = int(os.environ.get("INVESTIGATION_TIMEOUT", str(AI_TIMEOUT)))
INVESTIGATION_RETRIES = int(os.environ.get("INVESTIGATION_RETRIES", "1"))
INVESTIGATION_FIRST_TURN_EXTRA = int(os.environ.get("INVESTIGATION_FIRST_TURN_EXTRA", "60"))

TOTAL_TIME_BUDGET = int(os.environ.get("TOTAL_TIME_BUDGET", "900"))
_run_start_time = None

MAX_INVESTIGATION_TURNS = int(os.environ.get("MAX_INVESTIGATION_TURNS", "3"))
MAX_FILES_PER_REQUEST   = int(os.environ.get("MAX_FILES_PER_REQUEST", "1"))
MAX_REPAIR_ROUNDS       = int(os.environ.get("MAX_REPAIR_ROUNDS", "4"))
VERIFY_SWEEP            = os.environ.get("VERIFY_SWEEP", "1") == "1"

MAX_ERROR_LINES   = 14
MAX_SUPPORTING_LINES = 10
MAX_FILE_CHARS    = int(os.environ.get("MAX_FILE_CHARS", "8000"))
MAX_TOTAL_CONTEXT = int(os.environ.get("MAX_TOTAL_CONTEXT", "12000"))
MAX_FILES_FIXED   = 4
MAX_PROMPT_CHARS  = int(os.environ.get("MAX_PROMPT_CHARS", "16000"))
INVESTIGATION_DIFF_CHARS = int(os.environ.get("INVESTIGATION_DIFF_CHARS", "1200"))
MAX_TRACEBACK_LINES = 12

MAX_FILES_PER_CATEGORY = int(os.environ.get("MAX_FILES_PER_CATEGORY", "2"))
MAX_PRELOADED_FILES    = int(os.environ.get("MAX_PRELOADED_FILES", "4"))

GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_HIDDEN   = {".git", "auto-fixer.py"}

SELF_WORKFLOW_MARKERS = [
    re.compile(r'auto-?fixer\.py'),
    re.compile(r'OLLAMA_(?:API_URL|MODEL)\b'),
    re.compile(r'auto-?fixer-on-failure', re.I),
]
_SELF_WORKFLOW_CACHE = {}

def _is_self_workflow(rel: str) -> bool:
    if rel in _SELF_WORKFLOW_CACHE:
        return _SELF_WORKFLOW_CACHE[rel]
    result = False
    try:
        if WORKFLOW_PATTERN.search(rel):
            name = Path(rel).name.lower()
            if "auto-fix" in name or "autofix" in name:
                result = True
            else:
                p = Path(rel)
                if p.is_file() and p.stat().st_size <= MAX_FILE_SIZE_BYTES:
                    head = p.read_text(encoding="utf-8", errors="replace")[:8000]
                    result = any(m.search(head) for m in SELF_WORKFLOW_MARKERS)
    except Exception:
        result = False
    _SELF_WORKFLOW_CACHE[rel] = result
    return result

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

REPO_REF_EXT_PATTERN = re.compile(
    r'(?<![\w./\-])((?:[\w.\-]+/)*[\w\-]+\.(?:py|txt|ya?ml|json|toml|cfg|ini|'
    r'js|jsx|ts|tsx|go|java|rb|sh))(?![\w./\-])')

CURRENT_FAILURE_CONTEXT = ""

# --- helpers ----------------------------------------------------------------
def _relstrip(rel): return rel[2:] if rel.startswith("./") else rel
def _is_blocked(fp):
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_HIDDEN):
        return True
    return any(re.search(p, fp) for p in BLOCKED_PATTERNS)
def _is_read_blocked(fp):
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_HIDDEN):
        return True
    if _is_self_workflow(fp):
        return True
    return any(re.search(p, fp) for p in READ_BLOCKED_PATTERNS)
def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False

# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC FAILURE-SHAPE DETECTION & CONTEXT RETRIEVAL
# (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

FAILURE_SIGNATURES = {
    "container_runtime": [
        re.compile(r'-----\s*container logs\s*-----', re.I),
        re.compile(r'OCI runtime exec failed', re.I),
        re.compile(r'exec:\s*".*":\s*stat', re.I),
        re.compile(r'failed to start container', re.I),
        re.compile(r'docker:\s*Error response from daemon', re.I),
        re.compile(r'\bContainerConfig\b'),
    ],
    "container_build": [
        re.compile(r'failed to solve', re.I),
        re.compile(r'^\s*Step \d+/\d+\s*:', re.M),
        re.compile(r'the working directory .* is not found', re.I),
        re.compile(r'^\s*#\d+\s+\[.*\]', re.M),
    ],
    "python_runtime": [
        re.compile(r'Traceback \(most recent call last\)'),
        re.compile(r'ModuleNotFoundError|ImportError'),
        re.compile(r'^\s*E\s+\w+Error\b', re.M),
        re.compile(r'\bpip\b.*(install|ERROR)', re.I),
    ],
    "node_runtime": [
        re.compile(r'^npm ERR!', re.M),
        re.compile(r'Cannot find module'),
        re.compile(r'^yarn error', re.I | re.M),
    ],
    "go_runtime": [
        re.compile(r'^# .*\[build failed\]', re.M),
        re.compile(r'cannot find package'),
        re.compile(r'\.go:\d+:\d+:'),
    ],
    "java_build": [
        re.compile(r'\bBUILD FAILURE\b'),
        re.compile(r'\[ERROR\].*Maven', re.I),
        re.compile(r'Could not resolve dependenc(y|ies)', re.I),
    ],
    "workflow_config": [
        re.compile(r'##\[error\]'),
        re.compile(r'Invalid workflow file'),
        re.compile(r'yaml.*(parse|scan)', re.I),
    ],
}

def detect_failure_categories(log_text: str) -> list:
    return [cat for cat, pats in FAILURE_SIGNATURES.items()
            if any(p.search(log_text) for p in pats)]

def _match_files_by_name(allowed_files: set, *predicates) -> list:
    return [f for f in allowed_files if any(pred(Path(f).name.lower()) for pred in predicates)]

CONTEXT_RETRIEVERS = {
    "container_runtime": lambda files: _match_files_by_name(
        files,
        lambda n: n.startswith("dockerfile"),
        lambda n: n in ("docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"),
    ),
    "container_build": lambda files: _match_files_by_name(
        files,
        lambda n: n.startswith("dockerfile"),
        lambda n: n == ".dockerignore",
    ),
    "python_runtime": lambda files: _match_files_by_name(
        files,
        lambda n: n in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "pipfile"),
    ),
    "node_runtime": lambda files: _match_files_by_name(
        files,
        lambda n: n in ("package.json", "package-lock.json", "yarn.lock", ".npmrc"),
    ),
    "go_runtime": lambda files: _match_files_by_name(
        files,
        lambda n: n in ("go.mod", "go.sum"),
    ),
    "java_build": lambda files: _match_files_by_name(
        files,
        lambda n: n in ("pom.xml", "build.gradle", "build.gradle.kts"),
    ),
    "workflow_config": lambda files: [f for f in files if WORKFLOW_PATTERN.search(f)],
}

def gather_deterministic_context(log_text: str, allowed_files: set) -> dict:
    categories = detect_failure_categories(log_text)
    result = {}
    for cat in categories:
        retriever = CONTEXT_RETRIEVERS.get(cat)
        if not retriever:
            continue
        found = retriever(allowed_files)[:MAX_FILES_PER_CATEGORY]
        if found:
            result[cat] = found
    return result

def flatten_suggested_files(context_map: dict, cap: int = MAX_PRELOADED_FILES) -> list:
    seen, out = set(), []
    for files in context_map.values():
        for f in files:
            if f not in seen:
                seen.add(f)
                out.append(f)
            if len(out) >= cap:
                return out
    return out

# ── STAGE 0 – COLLECT EVIDENCE (unchanged except for increased limits) ────────
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
    "deprecationwarning", "--trace-deprecation", "trace-warnings",
    "node --trace", "(use `node", "punycode", "experimentalwarning",
    "npm warn", "warning:", "deprecated", "notice]", "downloading",
    "already satisfied", "reading package", "resolving", "setup-python",
    "actions/checkout", "actions/setup", "run docker", "##[debug]",
]
COMMAND_LINE_HINTS = [
    re.compile(r'^\s*(?:\$|>|\+)\s'),
    re.compile(r'\b(?:python|pip|pytest|npm|yarn|node|go|mvn|gradle|'
               r'docker|docker\s+buildx|curl|make|bash|sh)\b', re.I),
    re.compile(r'\brun:\s'),
]

def _looks_like_command(line: str) -> bool:
    s = (line or "").strip()
    if not s or any(n in s.lower() for n in NOISE_KEYWORDS):
        return False
    return any(p.search(s) for p in COMMAND_LINE_HINTS)

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

_TS_PREFIX = re.compile(r'^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\s+')

def _strip_ts(line: str) -> str:
    return _TS_PREFIX.sub("", line)

def _strip_log_timestamps(log_text: str) -> str:
    return "\n".join(_strip_ts(l) for l in log_text.splitlines())

_CTRL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_REPLACEMENT_RUN = re.compile(r'[\ufffd\ufeff]+')

def sanitize_log_text(log_text: str) -> tuple:
    corrupted = bool(_CTRL_CHARS.search(log_text) or _REPLACEMENT_RUN.search(log_text))
    clean = _REPLACEMENT_RUN.sub("", _CTRL_CHARS.sub("", log_text))
    return clean, corrupted

def looks_garbled(text: str) -> bool:
    if not text:
        return False
    if "\ufffd" in text or any(ord(c) < 32 and c not in "\t\n" for c in text):
        return True
    printable = sum(1 for c in text if c.isprintable() or c in "\t\n")
    return printable / max(1, len(text)) < 0.85

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
    command_context_lines = []
    if primary_message:
        log_lines = log_text.splitlines()
        msg_line_idx = None
        for i, line in enumerate(log_lines):
            if primary_message in line:
                msg_line_idx = i
                break
        if msg_line_idx is not None:
            for j in range(msg_line_idx - 1, max(-1, msg_line_idx - 8), -1):
                cand = log_lines[j].strip()
                if _looks_like_command(cand):
                    command_context_lines.append(cand)
                    break
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
    return stacks or {"unknown"}

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
    last = None
    for attempt in range(retries):
        try:
            t0, collected = time.time(), []
            deadline = t0 + timeout
            resp = requests.post(endpoint, json=payload, timeout=(10, timeout), stream=True)
            resp.raise_for_status()
            hit_deadline = False
            for line in resp.iter_lines():
                if time.time() > deadline:
                    hit_deadline = True
                    try:
                        resp.close()
                    except Exception:
                        pass
                    break
                tok = _extract_token(line, fmt)
                if tok:
                    collected.append(tok)
            raw = "".join(collected).strip()
            if not raw and not hit_deadline:
                try:
                    body = resp.json()
                    raw = (body.get("choices", [{}])[0].get("text", "")
                           if fmt == "openai" else body.get("response", "")).strip()
                except Exception:
                    pass
            if not raw:
                raise requests.exceptions.Timeout(
                    f"deadline hit with no usable output after {timeout}s")
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
        return True
    except Exception as exc:
        print(f"[WARMUP] warm-up call failed ({exc}) — continuing anyway.", file=sys.stderr)
        return False

def _close_truncated_json(text: str):
    def _close(t: str) -> str:
        stack, in_str, esc = [], False, False
        for ch in t:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    stack.append(ch)
                elif ch in "}]":
                    if stack:
                        stack.pop()
        out = t
        if in_str:
            out += '"'
        out = re.sub(r",\s*$", "", out)
        while stack:
            out += "}" if stack.pop() == "{" else "]"
        return out
    for candidate in (text, ):
        try:
            return json.loads(_close(candidate))
        except json.JSONDecodeError:
            pass
    for cut in range(len(text) - 1, max(0, len(text) - 2000), -1):
        if text[cut] == ",":
            try:
                return json.loads(_close(text[:cut]))
            except json.JSONDecodeError:
                continue
    return None

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
    start = cleaned.find("{")
    if start != -1:
        repaired = _close_truncated_json(cleaned[start:])
        if repaired is not None:
            return repaired
    return None

# ── INVESTIGATION (unchanged except for return augmentation) ─────────────────
INVESTIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "status":          {"type": "string"},
        "confidence":      {"type": "number"},
        "root_cause":      {"type": "string"},
        "solution":        {"type": "string"},
        "all_issues_found": {"type": "boolean"},
        "commit_message":  {"type": "string"},
        "requested_files": {"type": "array", "items": {"type": "string"}},
        "findings": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "issue":      {"type": "string"},
                "root_cause": {"type": "string"},
                "solution":   {"type": "string"},
            },
            "required": ["issue", "root_cause"]}},
        "analysis":        {"type": "string"},
    },
    "required": ["status", "confidence"],
}

INVESTIGATE_SYSTEM = """\
You are a meticulous DevOps investigation agent. Your sole task is to find
EVERY distinct cause of the CI failure described below.

**EXTREMELY IMPORTANT – FOLLOW THESE RULES PRECISELY:**

1. The "## ISSUE TO SOLVE" section tells you what broke. Some files that may be
   relevant have already been provided under "## Evidence gathered so far" —
   read them thoroughly BEFORE requesting anything else.

2. The root cause is NOT necessarily the first thing that looks wrong. Many
   failures are caused by MULTIPLE INDEPENDENT problems in the same file or
   across different files. You must treat each distinct problem as a separate
   bug — even if they are all in the same line or the same file.

3. **LINE-BY-LINE AUDIT:**
   - For EVERY file listed in "## Evidence gathered so far", go through it
     LINE BY LINE.
   - For EVERY path, filename, version string, command, flag, argument, or
     reference that appears in the file, verify whether it:
       a) Points to something that exists in the "## Repository tree", OR
       b) Is a known valid version/tool, OR
       c) Matches the requirements, conventions, or dependencies described
          elsewhere in the evidence.
   - If ANY of those do NOT match reality, you MUST record that as a separate
     issue in the "findings" array. Do not stop after the first mismatch.

4. **CROSS-REFERENCE BETWEEN FILES:**
   - If a Dockerfile references `requirements.txt` or any other file, check
     that the referenced file actually exists and that its content is compatible
     (e.g. Python version in base image vs. `python_requires` in `setup.py` or
     `pyproject.toml`).
   - If a CI workflow mentions a script, check that the script exists AND has
     the correct entry point.
   - If a `COPY` or `ADD` directive uses a path, ensure that path is present
     in the repository tree **exactly as written** (case‑sensitive). A typo in
     the filename, a missing `./` prefix, or an incorrect directory is a bug.

5. **VERSION STRINGS AND TAGS:**
   - Check every version tag (Python, Node, Go, Docker base image, etc.)
     against what is actually available. If a tag is obviously invalid (like
     `python:3.1` instead of `python:3.10`), that is one bug.
   - If a `FROM` line uses a deprecated or non‑existent tag, that is a bug.
   - If a `python-version:` in a workflow file uses an unsupported version,
     that is a bug.

6. **MULTIPLE BUGS IN ONE FILE:**
   - A single Dockerfile can have:
       - A wrong base image tag (e.g. `python:3.1`),
       - A misspelled source path in a `COPY` line (e.g. `appp.py` instead of
         `app.py`),
       - A different misspelling of the SAME file in a `RUN` command,
       - A `WORKDIR` that does not exist.
     These are **FOUR separate bugs**. You must report all of them, each with
     its own entry in `findings`.

7. **WHEN TO REQUEST MORE FILES:**
   - If you suspect a bug but cannot confirm it because a crucial file has not
     been provided, request it with `status: need_more_info`.
   - If you have already seen the file but need to double‑check a specific
     section, request that file again — the system will show the full content
     (if size permits).

8. **CONFIRMATION RULES:**
   - Only respond with `status: root_cause_confirmed` once you have physically
     verified EVERY issue you are reporting, AND you are confident that no
     other problems exist in the evidence you examined.
   - Set `all_issues_found` to `true` ONLY after you have completed a
     full line‑by‑line scan of EVERY file in the evidence and are certain.
     If you are unsure (e.g., file was truncated or you ran out of turns),
     you MUST set `all_issues_found` to `false` and lower your confidence.

9. **FINDINGS FORMAT:**
   - Each entry in `findings` must have:
       - `issue`: a very short name (e.g. "wrong Python base image", "typo in COPY source")
       - `root_cause`: exactly why this particular thing breaks the build
       - `solution`: a precise textual change that would fix it
   - Do NOT merge multiple distinct bugs into one entry.
   - Do NOT skip a bug because you think the fix is "obvious" or "minor".
     If it causes a failure, it belongs in the list.

10. **CONFIDENCE (MANDATORY FIELD — NEVER OMIT IT):**
    - You MUST include a numeric `confidence` between 0.0 and 1.0 in EVERY
      response, and a non-empty `analysis`.
    - Confidence 0.9+ means you have read every relevant file in full and
      found no more issues.
    - Confidence 0.7-0.89 means you are quite sure but there might be
      additional problems in parts of the file you couldn't see.
    - Confidence ≤0.4 means you are guessing without solid evidence.
    - If your findings are directly confirmed by the error log AND you located
      the exact offending text in the evidence files, do not under-report your
      confidence — a verified diagnosis deserves 0.8+.

Output ONLY one JSON object. No markdown fences.

**FIELD ORDER IS CRITICAL — EMIT KEYS IN EXACTLY THIS ORDER:**
`status`, `confidence`, then `root_cause`, `findings`, and finally a SHORT
`analysis` LAST. Put your decision (status + confidence + findings) BEFORE any
long reasoning. Keep `analysis` to ONE sentence — it is optional context, not
the place to think out loud. Emitting a long analysis first will get your
response cut off before the decision is recorded.

Schema when confirming (findings = ONE ENTRY PER DISTINCT PROBLEM, even if
several are in the same file, and even across DIFFERENT files — a Dockerfile
bug AND a workflow bug AND a source-file typo are separate entries):
{"status":"root_cause_confirmed","confidence":0.9,"root_cause":"one-sentence summary covering ALL issues found","solution":"high-level plan to fix all issues","all_issues_found":true,"commit_message":"fix: <brief description>","findings":[{"issue":"short name","root_cause":"why this breaks the build","solution":"exact text change"}],"analysis":"one short sentence"}

Schema when you need another file:
{"status":"need_more_info","confidence":0.2,"requested_files":["exact/path/from/tree"],"analysis":"one short sentence on what you need"}
"""

# ── locate_offending_lines now returns both string and dict ─────────────
def locate_offending_lines(focused: dict, signal: str,
                           evidence_files: dict):
    """Return (offending_string, exact_lines_dict) where exact_lines_dict maps
       file -> the single exact line containing the flagged value."""
    if not evidence_files:
        return "", {}
    haystack = "\n".join(filter(None, [
        focused.get("primary_message", ""),
        "\n".join(focused.get("gh_errors", []) or []),
        signal or "",
    ]))
    tokens = set()
    for m in re.finditer(r"['\"]([\w.\-:/]{2,40})['\"]", haystack):
        tokens.add(m.group(1))
    for m in re.finditer(r"version\s+['\"]?(\d+\.\d+(?:\.\d+)?)['\"]?", haystack, re.I):
        tokens.add(m.group(1))
    if not tokens:
        return "", {}
    found_lines = []
    exact_dict = {}
    for fname, content in evidence_files.items():
        if not isinstance(content, str):
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for tok in tokens:
                if re.search(r'(?<![\w.])' + re.escape(tok) + r'(?![\w.])', line):
                    found_lines.append((fname, line, tok))
                    # store only the first occurrence per file (the most likely)
                    if fname not in exact_dict:
                        exact_dict[fname] = line
                    break
            if len(found_lines) >= 6:
                break
        if len(found_lines) >= 6:
            break
    if not found_lines:
        return "", {}
    lines_str = "\n".join(
        f"- In '{f}', this EXACT line contains the flagged value '{tok}':\n"
        f"    {line}"
        for f, line, tok in found_lines)
    offending_str = (
        "Exact offending line(s) located in the repository (the error names "
        "these value(s); the lines below are copied VERBATIM from the real "
        "files — use the relevant one as your 'evidence' string EXACTLY, and "
        "change ONLY the flagged value in 'corrected'):\n" + lines_str
    )
    return offending_str, exact_dict

# ── PATCH GENERATION – hardened hand-off ────────────────────────────────────
PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "file":      {"type": "string"},
                "evidence":  {"type": "string"},
                "corrected": {"type": "string"},
                "problem":   {"type": "string"},
            },
            "required": ["file", "evidence", "corrected"]}},
    },
    "required": ["issues"],
}

PATCH_SYSTEM = """\
You are a patch-generation agent. Another engineer already investigated this \
CI/CD failure and confirmed the root cause(s) below. Your job is to emit the \
concrete text-level fix for EVERY issue listed — do not re-diagnose, but do
not silently skip any of them either.

Stay scoped to the "## ISSUE TO SOLVE" section if one is present.

**CRITICAL — EVIDENCE MUST BE A SINGLE SHORT LINE (≤150 characters).**
Your "evidence" must be copied EXACTLY from the "## Exact lines that MUST be fixed"
section below, or from the file contents shown. NEVER use a multi-line block.
If you cannot express the fix with such a short evidence, you are picking the
wrong text — re-read the file and choose the single line that contains the bug.

If multiple issues are listed, emit one JSON entry per issue.

- "evidence" = the current WRONG line from the file.
- "corrected" = the SAME line with ONLY the bug fixed — it MUST be different.

Output ONLY one JSON object. No markdown fences.

Schema:
{"issues":[{"file":"...","problem":"...","evidence":"...","corrected":"..."}]}
"""

# Max length for evidence string
MAX_EVIDENCE_LINE_LENGTH = 150

def _extract_fix_window(filepath: str, target_line: str, radius: int = 5) -> str:
    """Return a small window of lines around the first occurrence of `target_line`."""
    try:
        p = Path(filepath)
        if not p.is_file():
            return ""
        all_lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, l in enumerate(all_lines):
            if target_line in l.strip():  # might be a substring
                start = max(0, i - radius)
                end = min(len(all_lines), i + radius + 1)
                return "\n".join(all_lines[start:end])
        # if not found, return whole file but truncated (shouldn't happen)
        return p.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
    except Exception:
        return ""

def _select_patch_evidence(evidence: dict, mention_blob: str,
                           exact_evidence_lines: dict = None) -> dict:
    """Build the evidence snippet for the patch prompt. When exact_evidence_lines
       is provided, we show ONLY the windows around those lines."""
    if exact_evidence_lines:
        selected = {}
        total = 0
        for file, exact_line in exact_evidence_lines.items():
            if file not in evidence:
                continue
            window = _extract_fix_window(file, exact_line, radius=3)
            if not window:
                window = evidence[file][:MAX_FILE_CHARS]
            block_len = len(window) + len(file) + 30
            if total + block_len > MAX_PATCH_EVIDENCE_CHARS:
                break
            selected[file] = window
            total += block_len
        return selected
    # fallback to original behavior if no exact lines provided
    blob = mention_blob.lower()
    mentioned, rest = [], []
    for f, c in evidence.items():
        (mentioned if (f.lower() in blob or Path(f).name.lower() in blob)
         else rest).append((f, c))
    selected, total, dropped = {}, 0, []
    for f, c in mentioned + rest:
        block_len = len(c or "") + len(f) + 16
        if selected and total + block_len > MAX_PATCH_EVIDENCE_CHARS:
            dropped.append(f)
            continue
        selected[f] = c
        total += block_len
    if dropped:
        print(f"[PATCH] evidence budget {MAX_PATCH_EVIDENCE_CHARS} chars — "
              f"omitted (not referenced): {dropped}")
    return selected

def ai_generate_patch(root_cause: str, solution: str, evidence: dict,
                      retry_note: str = "", issue_block: str = "",
                      exact_evidence_lines: dict = None) -> list:
    mention_blob = "\n".join([root_cause or "", solution or "", issue_block or ""])
    scoped = _select_patch_evidence(evidence, mention_blob,
                                    exact_evidence_lines=exact_evidence_lines)

    # Build the "Exact lines that MUST be fixed" section
    exact_lines_prompt = ""
    if exact_evidence_lines:
        exact_lines_prompt = "\n## Exact lines that MUST be fixed (verified):\n"
        for f, line in exact_evidence_lines.items():
            exact_lines_prompt += f"- file: `{f}`\n  evidence (verbatim): `{line}`\n"
        exact_lines_prompt += ("\nYour `evidence` field for each issue MUST be exactly "
                               "one of these lines. `corrected` must be different.\n\n")

    parts = [f"### {f}\n```\n{c}\n```" for f, c in scoped.items()]
    context = "\n\n".join(parts) if parts else "(no evidence files were read)"
    issue_section = (f"## ISSUE TO SOLVE (stay scoped to this):\n{issue_block}\n\n"
                     if issue_block else "")
    retry_section = (f"## Note: a previous attempt at this fix failed:\n"
                     f"{retry_note}\nDo not repeat the same change.\n\n" if retry_note else "")

    def _assemble(ctx):
        return (f"{PATCH_SYSTEM}\n\n{exact_lines_prompt}{issue_section}{retry_section}"
                f"## Confirmed root cause: {root_cause}\n"
                f"## Solution direction: {solution}\n\n"
                f"## File contents (you may ONLY edit these):\n{ctx}\n\n"
                f"Emit the issues JSON.")

    prompt = _assemble(context)
    if len(prompt) > MAX_PROMPT_CHARS:
        overflow = len(prompt) - MAX_PROMPT_CHARS
        context = context[:max(0, len(context) - overflow)] + "\n...(evidence truncated)"
        prompt = _assemble(context)

    remaining = max(0, TOTAL_TIME_BUDGET - _elapsed())
    if remaining < 45:
        raise RuntimeError(f"only {remaining:.0f}s of budget left — not enough "
                           f"for a patch call")
    timeout = min(PATCH_TIMEOUT, int(remaining) - 10)
    retries = MAX_RETRIES if remaining > 2 * timeout else 1
    raw = _stream_ollama(prompt, PATCH_SCHEMA, num_predict=PATCH_NUM_PREDICT,
                         temperature=0.05, tag="PATCH",
                         timeout=timeout, retries=retries)
    data = _json_from(raw)
    if data is None:
        raise ValueError(f"No valid JSON from patch agent:\n{raw[:400]}")
    issues = data.get("issues", []) or []
    if not issues and isinstance(data.get("fixes"), list):
        issues = data["fixes"]
    return _normalize_issue_keys(issues)

# ── issue normalization & validation ───────────────────────────────────────
def _unescape_model_string(s):
    if not isinstance(s, str):
        return s
    out = re.sub(r'\\([._/\-: 0-9A-Za-z])', r'\1', s)
    out = out.replace("\\", "/")
    return re.sub(r'/{2,}', '/', out)

def _normalize_file_field(file: str, evidence: dict) -> str:
    if not file:
        return file
    cand = _unescape_model_string(file).strip().strip("`'\"")
    cand = _relstrip(cand)
    if cand in evidence:
        return cand
    base = Path(cand).name.lower()
    for k in evidence:
        if Path(k).name.lower() == base:
            return k
    match = difflib.get_close_matches(cand, list(evidence.keys()), n=1, cutoff=0.6)
    return match[0] if match else cand

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
        for k in ("file", "evidence", "corrected"):
            if isinstance(it.get(k), str):
                it[k] = _unescape_model_string(it[k])
        out.append(it)
    return out

def _closest_evidence_line(ev: str, evidence: dict) -> tuple:
    needle = (ev or "").strip().splitlines()
    needle = needle[0][:200].lower() if needle else ""
    if not needle:
        return "", "", 0.0
    best = ("", "", 0.0)
    for f, c in evidence.items():
        if not isinstance(c, str):
            continue
        for line in c.splitlines():
            s = line.strip()
            if not s:
                continue
            bare = re.sub(r"\s+#.*$", "", s).strip()
            r = max(
                difflib.SequenceMatcher(None, needle, s.lower()).ratio(),
                difflib.SequenceMatcher(None, needle, bare.lower()).ratio()
                if bare else 0.0)
            if r > best[2]:
                best = (f, s[:200], r)
    return best if best[2] >= 0.4 else ("", "", 0.0)

def issues_to_fixes(issues: list, evidence: dict) -> tuple:
    fixes_by_file, rejects = {}, []
    for n, it in enumerate(issues, 1):
        file = _normalize_file_field((it.get("file") or "").strip(), evidence)
        ev   = it.get("evidence")
        cor  = it.get("corrected")
        prob = (it.get("problem") or "").strip()

        if not isinstance(ev, str) or not ev.strip():
            rejects.append(f"issue #{n} ({file or '?'}): empty evidence")
            continue
        if not isinstance(cor, str) or not cor.strip():
            rejects.append(f"issue #{n} ({file or '?'}): 'corrected' is empty — "
                           f"you must supply the fixed text.")
            continue
        # ── NEW: enforce evidence length limit ──
        if len(ev) > MAX_EVIDENCE_LINE_LENGTH:
            rejects.append(
                f"issue #{n} ({file or '?'}): evidence is {len(ev)} characters — "
                f"maximum allowed is {MAX_EVIDENCE_LINE_LENGTH}. "
                f"Use ONLY the single line that contains the bug.")
            continue
        if cor.strip() == ev.strip():
            rejects.append(
                f"issue #{n} ({file or '?'}): 'evidence' and 'corrected' are "
                f"IDENTICAL. 'corrected' must be the SAME line with the bug fixed. "
                f"Hint: your 'evidence' is likely too long; use only the single "
                f"line that contains the wrong value, like `python-version: \"3.2\"`."
            )
            continue

        holders = [f for f, c in evidence.items() if isinstance(c, str) and ev in c]
        if file in evidence and isinstance(evidence[file], str) and ev in evidence[file]:
            target = file
        elif len(holders) == 1:
            target = holders[0]
        elif len(holders) > 1:
            rejects.append(f"issue #{n} ({file or '?'}): evidence ambiguous")
            continue
        else:
            hint_file, hint_line, _ = _closest_evidence_line(ev, evidence)
            if hint_file:
                rejects.append(
                    f"issue #{n} ({file or '?'}): your 'evidence' text does NOT "
                    f"exist in any provided file — you may have copied it from "
                    f"the error log or invented it. The closest text that "
                    f"ACTUALLY exists is in '{hint_file}': `{hint_line}`. "
                    f"Copy the real file text exactly.")
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

# ── apply, validate, git, etc. (all original) ───────────────────────────────
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

def _missing_ref_map(content: str, file: str) -> dict:
    problems = {}
    if not content:
        return problems
    all_repo = [_relstrip(str(p)) for p in Path(".").rglob("*")
                if p.is_file() and not any(d in SKIP_DIRS for d in p.parts)]
    all_repo_set = set(all_repo)
    parent_dir = Path(file).parent
    seen_tokens = set()
    for m in REPO_REF_EXT_PATTERN.finditer(content):
        token = m.group(1)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        if "${{" in token or token.startswith("."):
            continue
        candidates = {token, _relstrip(str(parent_dir / token))}
        if any(c in all_repo_set or Path(c).is_file() for c in candidates):
            continue
        problems[token] = f"references missing '{token}'"
    return problems

def _dockerfile_problem_map(file: str, content: str) -> dict:
    problems = {}
    if "dockerfile" not in Path(file).name.lower() or not content:
        return problems
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
            problems[ref_clean] = f"{label} references missing '{ref}'"
    for m in DOCKERFILE_FROM_PYTHON.finditer(content):
        version = m.group(2)
        base = version.split("-", 1)[0]
        vm = re.match(r'^(\d+)\.(\d+)', base)
        if not vm:
            continue
        major, minor = int(vm.group(1)), int(vm.group(2))
        if not (major == 3 and minor in VALID_PYTHON_MINORS):
            problems[f"python:{version}"] = (
                f"base image 'python:{version}' — not a real CPython release")
    return problems

def _compare_ref_problems(old_map: dict, new_map: dict, label: str) -> str:
    introduced = {k: v for k, v in new_map.items() if k not in old_map}
    if introduced:
        return (f"{label}: patch INTRODUCES new problem(s): "
                + "; ".join(introduced.values()))
    ctx = CURRENT_FAILURE_CONTEXT.lower()
    persisting = {k: v for k, v in new_map.items() if k in old_map}
    blocking = {k: v for k, v in persisting.items() if k.lower() in ctx}
    if blocking:
        return (f"{label}: the failure evidence mentions these and the patch "
                f"leaves them broken: " + "; ".join(blocking.values()))
    if persisting:
        print(f"[VALIDATE] note ({label}): pre-existing issues NOT mentioned "
              f"in the failure evidence — left for a future run to judge: "
              + "; ".join(persisting.values()))
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

def invalid_workflow_python_versions(text: str) -> list:
    bad = []
    for m in WORKFLOW_PYVERSION_LINE.finditer(text):
        major, minor = int(m.group(3)), int(m.group(4))
        if not (major == 3 and minor in VALID_PYTHON_MINORS):
            bad.append(f"{major}.{minor}")
    return bad

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
                return False, "edit introduces dangerous command pattern"
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
            bad = invalid_workflow_python_versions(content)
            if bad:
                return False, (
                    "workflow still sets python-version to "
                    + ", ".join(f"'{v}'" for v in bad)
                    + " — not a real released CPython version. Choose a real "
                      "release (e.g. 3.10, 3.11, 3.12) in 'corrected'.")
            ok, reason = validate_workflow_edit(original_text, content)
            if not ok:
                return False, reason
    elif file.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"JSON error: {e}"
    docker_reason = _compare_ref_problems(
        _dockerfile_problem_map(file, original_text),
        _dockerfile_problem_map(file, content),
        "dockerfile check")
    if docker_reason:
        return False, docker_reason
    ref_reason = _compare_ref_problems(
        _missing_ref_map(original_text, file),
        _missing_ref_map(content, file),
        "reference check")
    if ref_reason:
        return False, ref_reason
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

def _added_lines(original: str, new: str) -> list:
    old_lines = set(original.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]

def scan_text_for_secrets(text: str) -> list:
    return [name for name, pat in SECRET_PATTERNS if pat.search(text)]

def scan_dangerous_commands(text: str) -> list:
    return [p.pattern for p in DANGEROUS_COMMAND_PATTERNS if p.search(text)]

# ── BUILD & TESTS (unchanged) ─────────────────────────────────────────────
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

# ── GIT & PR (unchanged) ──────────────────────────────────────────────────
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

def detect_failed_branch() -> str:
    for var in ("FAILED_BRANCH", "GITHUB_HEAD_REF"):
        v = os.environ.get(var, "").strip()
        if v:
            return v
    ref = os.environ.get("GITHUB_REF", "")
    if ref.startswith("refs/heads/"):
        return ref[len("refs/heads/"):]
    try:
        b = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        return "" if b == "HEAD" else b
    except Exception:
        return ""

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

def commit_to_existing_branch(commit_msg: str, written: list, branch: str) -> str:
    try:
        _git("config", "user.name", BOT_NAME)
        _git("config", "user.email", BOT_EMAIL)
        _git("fetch", "origin", branch, check=False)
        if _git("rev-parse", "--verify", branch, check=False).returncode != 0:
            _git("checkout", "-b", branch, f"origin/{branch}")
        else:
            _git("checkout", branch)
            _git("pull", "origin", branch, check=False)
        if written:
            _git("add", "--", *written)
        else:
            _git("add", "-u")
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit on existing fix branch.")
            return ""
        _git("commit", "-m", commit_msg)
        _git("push", "origin", branch)
        print(f"[GIT] Pushed follow-up fix to existing {branch}")
        return branch
    except subprocess.CalledProcessError as exc:
        print(f"[GIT] {exc.stderr.strip()}", file=sys.stderr)
        return ""

def _gh(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}

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

def comment_on_bot_pr(token, repo, branch, body) -> bool:
    try:
        owner = repo.split("/")[0]
        r = requests.get(f"https://api.github.com/repos/{repo}/pulls"
                         f"?state=open&head={owner}:{branch}",
                         headers=_gh(token), timeout=15)
        if r.status_code != 200 or not r.json():
            return False
        number = r.json()[0]["number"]
        r2 = requests.post(f"https://api.github.com/repos/{repo}/issues/{number}/comments",
                           json={"body": body}, headers=_gh(token), timeout=15)
        if r2.status_code in (200, 201):
            print(f"[PR] follow-up comment posted on PR #{number}")
            return True
    except Exception as exc:
        print(f"[PR] comment failed: {exc}", file=sys.stderr)
    return False

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

# ── ENRICH HELPERS (unchanged) ────────────────────────────────────────────
def enrich_issue_with_path_checks(focused: dict, signal: str,
                                  allowed_files: set) -> str:
    haystack = "\n".join(filter(None, [
        focused.get("primary_message", ""),
        "\n".join(focused.get("command_context", []) or []),
        "\n".join(focused.get("gh_errors", []) or []),
        signal or "",
    ]))
    path_re = re.compile(r'(?<![\w./\-])((?:[\w.\-]+/)+[\w.\-]+)(?![\w./\-])')
    _skip_prefixes = ("http", "${{", "/home/", "/usr/", "/opt/", "/tmp/",
                      "/var/", "/etc/", "./_", "actions-runner", "_work/",
                      "node_modules/", "site-packages/", "dist-packages/")
    observations = []
    seen = set()
    for m in path_re.finditer(haystack):
        token = m.group(1).strip("'\":,()[]")
        low = token.lower()
        if (not token or token in seen
                or low.startswith(_skip_prefixes)
                or "site-packages" in low or "actions-runner" in low
                or "/_" in token
                or token.count("/") > 4):
            continue
        seen.add(token)
        norm = _relstrip(token)
        if norm in allowed_files or Path(norm).exists():
            continue
        near = _closest_allowed_file(norm, allowed_files)
        if near and near != norm:
            observations.append(
                f"- Path '{token}' referenced in the error does NOT exist in the "
                f"repository. The closest real path is '{near}'.")
        if len(observations) >= 6:
            break
    if not observations:
        return ""
    print(f"[EVIDENCE] Path-existence checks flagged {len(observations)} "
          f"missing reference(s) named in the error.")
    return ("Verified path facts (missing references named in the failure vs. "
            "what actually exists — you MUST reconcile every one of these):\n"
            + "\n".join(observations))

def _read_dot_python_version() -> str:
    p = Path(".python-version")
    if not p.is_file():
        return ""
    try:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                m = re.match(r'(\d+)\.(\d+)', line)
                if m:
                    return f"{m.group(1)}.{m.group(2)}"
    except Exception:
        pass
    return ""

def collect_python_version_declarations(allowed_files: set) -> dict:
    decls = {}
    dot = _read_dot_python_version()
    if dot:
        decls[".python-version"] = dot
    for f in sorted(allowed_files):
        name = Path(f).name.lower()
        try:
            text = Path(f).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if WORKFLOW_PATTERN.search(f):
            for m in WORKFLOW_PYVERSION_LINE.finditer(text):
                decls[f"{f} (python-version:)"] = f"{m.group(3)}.{m.group(4)}"
        if name.startswith("dockerfile"):
            for m in DOCKERFILE_FROM_PYTHON.finditer(text):
                base = m.group(2).split("-", 1)[0]
                vm = re.match(r'(\d+)\.(\d+)', base)
                if vm:
                    decls[f"{f} (FROM python:)"] = f"{vm.group(1)}.{vm.group(2)}"
            for m in re.finditer(
                    r'ARG\s+PYTHON_VERSION\s*=\s*["\']?(\d+)\.(\d+)', text, re.I):
                decls[f"{f} (ARG PYTHON_VERSION)"] = f"{m.group(1)}.{m.group(2)}"
        if name in ("pyproject.toml", "setup.py", "setup.cfg"):
            m = re.search(r'requires[-_]python\s*=?\s*["\'][^0-9]*(\d+)\.(\d+)',
                          text, re.I)
            if m:
                decls[f"{f} (requires-python)"] = f"{m.group(1)}.{m.group(2)}"
    return decls

def check_python_version_coherence(allowed_files: set) -> str:
    decls = collect_python_version_declarations(allowed_files)
    if len(decls) < 2:
        return ""
    values = set(decls.values())
    if len(values) == 1:
        return ""
    canonical = decls.get(".python-version")
    listing = "\n".join(f"- {loc}: {ver}" for loc, ver in decls.items())
    if canonical:
        guidance = (f"The '.python-version' file declares {canonical}, which is "
                    f"the project's source of truth. Align every other "
                    f"declaration to {canonical}.")
    else:
        guidance = ("There is no '.python-version' source-of-truth file; these "
                    "declarations must be made to agree on ONE version.")
    print(f"[EVIDENCE] Python-version coherence: {len(values)} distinct value(s) "
          f"across {len(decls)} declaration(s) — reporting the drift.")
    return ("Python version DISAGREEMENT across the repo (the CI runner's Python "
            "and the image/packaging Python must match, or tests run on a "
            "different interpreter than production ships):\n"
            + listing + "\n" + guidance)

PY_VERSION_FAILURE_SIGNATURES = [
    re.compile(r"version\s+'?[\d.]+'?\s+.*was not found", re.I),
    re.compile(r"python-version", re.I),
    re.compile(r"failed to solve:\s*python:", re.I),
    re.compile(r"docker\.io/library/python:", re.I),
    re.compile(r"no such (?:version|python)", re.I),
    re.compile(r"pyenv:.*version.*not installed", re.I),
    re.compile(r"python[ _-]?version[ _-]?file", re.I),
    re.compile(r"specified python version file.*(?:does\s*n['o]?t|not)\s*exist", re.I),
]

def is_python_version_failure(issue_block: str, signal: str) -> bool:
    blob = f"{issue_block}\n{signal}"
    return any(p.search(blob) for p in PY_VERSION_FAILURE_SIGNATURES)

_VERSION_FILE_REF = re.compile(r"python[-_ ]?version[-_ ]?file\s*:?\s*([.\w][\w./\-]*)", re.I)
_MISSING_FILE_ERR = re.compile(r"file\s+at:?\s*([.\w][\w./\-]*)", re.I)

def _extract_version_file_name(blob: str) -> str:
    candidates = []
    for pat in (_VERSION_FILE_REF, _MISSING_FILE_ERR):
        for m in pat.finditer(blob):
            candidates.append(m.group(1).strip().strip("`'\":,"))
    for cand in candidates:
        if cand and ("." in cand or "/" in cand):
            return cand
    return ""

def missing_version_file(issue_block: str, signal: str, allowed_files: set) -> str:
    cand = _extract_version_file_name(f"{issue_block}\n{signal}")
    if not cand:
        return ""
    norm = _relstrip(cand)
    if norm in allowed_files or Path(norm).is_file():
        return ""
    return cand

def python_version_is_undefined(allowed_files: set) -> bool:
    return not collect_python_version_declarations(allowed_files)

UNDEFINED_PYTHON_VERSION_MESSAGE = (
    "The project doesn't define a Python version in `.python-version`, "
    "`pyproject.toml`, the Dockerfile, or the GitHub Actions workflow. "
    "I can't determine the intended version, and I won't guess one — picking "
    "an arbitrary version could ship or test against the wrong interpreter. "
    "Either add a project version file (e.g. `.python-version` with a value "
    "like `3.12`) or infer the intended version from your package "
    "compatibility (`requires-python`), then re-run."
)

def undefined_python_version_message(issue_block: str, signal: str) -> str:
    blob = f"{issue_block}\n{signal}"
    named = _extract_version_file_name(blob)
    if named:
        return (
            f"The workflow's `setup-python` step is configured with "
            f"`python-version-file: {named}`, but `{named}` does not exist in "
            f"the repository, so there is no Python version to use. I won't "
            f"invent one. To fix this, create `{named}` containing the intended "
            f"version (e.g. `3.12`) and commit it, or replace the "
            f"`python-version-file` reference with an explicit `python-version:` "
            f"value — then re-run. (I can't create a brand-new file here; this "
            f"needs a human decision on which version the project targets.)")
    return UNDEFINED_PYTHON_VERSION_MESSAGE

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

# ── REVIEW AGENT ─────────────────────────────────────────────────────────────
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "decision":   {"type": "string"},
        "root_cause": {"type": "string"},
        "solution":   {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["decision", "confidence"],
}

REVIEW_SYSTEM = """\
You are reviewing why a previously applied fix still failed CI. Given the \
original root cause, the fix that was tried, and the NEW failure output, \
decide:
  - "retry": provide updated root_cause and solution.
  - "stop": a human should look at it.
Focus on the "## NEW ISSUE" section if present.
"confidence" is MANDATORY in every response.
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

# ── investigation loop (unchanged) ────────────────────────────────────────
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

def _finalize_investigation(data: dict, forced: bool, evidence: dict) -> dict:
    raw_conf = data.get("confidence")
    confidence_omitted = raw_conf is None
    try:
        confidence = float(raw_conf if raw_conf is not None else 0.4)
    except (TypeError, ValueError):
        confidence = 0.4
        confidence_omitted = True
    confidence = max(0.0, min(confidence, 1.0))
    if forced and data.get("status") != "root_cause_confirmed":
        confidence = min(confidence, 0.4)
    if not evidence and confidence > 0.4:
        confidence = 0.4
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
    root_cause = (data.get("root_cause") or "").strip()
    if not root_cause and findings and findings[0].get("root_cause") not in ("", "unknown"):
        root_cause = "; ".join(f["root_cause"] for f in findings if f.get("root_cause"))[:400]
    solution = (data.get("solution") or "").strip()
    if not solution and findings:
        solution = "; ".join(f["solution"] for f in findings if f.get("solution"))[:400]
    all_issues_found = data.get("all_issues_found")
    if all_issues_found is False:
        confidence = min(confidence, 0.45)
    return {
        "root_cause":     root_cause or "unknown",
        "solution":       solution,
        "confidence":     confidence,
        "confidence_omitted": confidence_omitted,
        "commit_message": data.get("commit_message") or "fix: auto-fixer change",
        "findings":       findings,
        "all_issues_found": bool(all_issues_found) if all_issues_found is not None else None,
    }

def ai_investigate(signal, exit_code, repo_tree, git_diff, allowed_files: set,
                   issue_block: str = "", suggested_files: list = None) -> dict:
    evidence = {}
    preloaded = []
    for f in (suggested_files or []):
        if f in evidence:
            continue
        content = _read_evidence_file(f)
        if content is not None:
            evidence[f] = content
            preloaded.append(f)
    if preloaded:
        print(f"[INVESTIGATE] evidence pre-loaded via deterministic retrieval: {preloaded}")

    log, result = [], None
    dead_ends = set()
    pending_notes = []
    stall_count = 0
    got_any_model_response = False
    parse_failures = 0
    turn_num_predict = 800

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
            raw = _stream_ollama(prompt, INVESTIGATE_SCHEMA, num_predict=turn_num_predict,
                                 temperature=0.05, tag=f"INVESTIGATE-T{turn}",
                                 timeout=turn_timeout, retries=INVESTIGATION_RETRIES)
        except Exception as exc:
            is_timeout = "timed out" in str(exc).lower() or "timeout" in str(exc).lower()
            print(f"[INVESTIGATE] turn {turn} failed: {exc}", file=sys.stderr)
            if is_timeout and turn == 1 and not got_any_model_response:
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
        if raw:
            got_any_model_response = True
        if not data:
            parse_failures += 1
            print(f"[INVESTIGATE] turn {turn}: model produced {len(raw)} chars "
                  f"but NOT valid JSON (likely truncated at the token cap).")
            log.append({"turn": turn, "status": "unparseable",
                        "analysis": raw[:200]})
            if last_turn or parse_failures >= 2:
                break
            pending_notes.append(
                "Your previous response was NOT valid JSON — it appears to have "
                "been cut off before completion. Respond again with ONLY the "
                "JSON object. Keep 'analysis' to at most 2 short sentences and "
                "each finding brief so the output fits.")
            turn_num_predict = 1500
            continue
        status = data.get("status", "")
        analysis = data.get("analysis", "")
        print(f"[INVESTIGATE] turn {turn}: status={status} | "
              f"confidence={data.get('confidence', '(omitted!)')} | "
              f"{analysis[:160] if analysis else '(no analysis)'}")

        if not status and turn > 1:
            stall_count += 2

        log.append({"turn": turn, "status": status, "analysis": analysis[:200]})

        if status == "root_cause_confirmed":
            result = _finalize_investigation(data, False, evidence)
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
                    to_read.append(match)
                    pending_notes.append(f"You asked for '{f}', which doesn't exist. "
                                         f"The closest real file, '{match}', was read instead.")
                else:
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
                    result = _finalize_investigation(data, True, evidence)
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
                result = _finalize_investigation(data2, True, evidence)
                break
            continue

        if data.get("root_cause"):
            result = _finalize_investigation(data, True, evidence)
        break

    if result is None:
        if not got_any_model_response:
            failure_mode = "infra_timeout"
            root_cause = "model did not return any usable response"
        elif parse_failures:
            failure_mode = "unparseable_output"
            root_cause = "model output was not valid JSON"
        else:
            failure_mode = "not_converged"
            root_cause = "unknown"
        result = {"root_cause": root_cause, "solution": "", "confidence": 0.0,
                  "confidence_omitted": True,
                  "commit_message": "fix: auto-fixer change", "findings": [],
                  "failure_mode": failure_mode}
    else:
        result["failure_mode"] = None
    result["evidence"] = evidence
    result["investigation_log"] = log
    return result

# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    global _run_start_time, CURRENT_FAILURE_CONTEXT
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

    failed_branch = detect_failed_branch()
    chain_mode = failed_branch.startswith("fix/")
    if chain_mode:
        print(f"[CHAIN] failure occurred on bot branch '{failed_branch}' — "
              f"the new error will be fixed on the SAME branch so its open PR "
              f"accumulates fixes until CI is green.")
        if count_recent_bot_commits() >= MAX_BOT_ATTEMPTS:
            print("[GUARD] Too many chained bot attempts on this branch — escalating.")
            if token and repo:
                open_issue(token, repo,
                           f"Auto-fixer made {MAX_BOT_ATTEMPTS} chained fix attempts on "
                           f"`{failed_branch}` and CI still fails — manual review needed.",
                           run_url)
            sys.exit(0)
    else:
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
    log_text = _strip_log_timestamps(log_text)
    log_text, log_corrupted = sanitize_log_text(log_text)
    if log_corrupted:
        print("[SECURITY] log contained control/replacement characters "
              "(mojibake) — stripped before analysis.")

    exit_code = get_exit_code(log_text, args.exit_code)
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[EVIDENCE] No error signal — nothing to fix.")
        sys.exit(0)

    focused = extract_focused_failure(log_text)
    issue_block = format_focused_issue(focused, exit_code)

    primary = focused.get("primary_message", "")
    if log_corrupted and looks_garbled(primary):
        print("[GATE] the primary error line is garbled/corrupted — escalating.")
        if token and repo:
            open_issue(token, repo,
                       "Primary error line is garbled — see log analysis.", run_url)
        sys.exit(0)

    CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    supporting_signal = trim_supporting_signal(signal, focused)
    git_diff  = get_git_diff()
    repo_tree, allowed_files = repo_tree_text()

    path_facts = enrich_issue_with_path_checks(focused, signal, allowed_files)
    if path_facts:
        issue_block = issue_block + "\n\n" + path_facts
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    coherence = check_python_version_coherence(allowed_files)
    if coherence:
        issue_block = issue_block + "\n\n" + coherence
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    if is_python_version_failure(issue_block, signal):
        missing_vf = missing_version_file(issue_block, signal, allowed_files)
        if missing_vf or python_version_is_undefined(allowed_files):
            gate_msg = undefined_python_version_message(issue_block, signal)
            print(f"[GATE] Python-version failure — escalating.")
            if token and repo:
                open_issue(token, repo, gate_msg, run_url)
            sys.exit(0)

    context_map = gather_deterministic_context(log_text, allowed_files)
    suggested_files = flatten_suggested_files(context_map)
    preloaded_contents = {}
    for f in suggested_files:
        c = _read_evidence_file(f)
        if c is not None:
            preloaded_contents[f] = c

    offending_str, exact_evidence_lines = locate_offending_lines(focused, signal, preloaded_contents)
    if offending_str:
        issue_block = issue_block + "\n\n" + offending_str
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    # ── AI INVESTIGATION AGENT ──
    print("\n━━━ AI INVESTIGATION AGENT ━━━")
    warm_up_model()

    investigation = ai_investigate(supporting_signal, exit_code, repo_tree, git_diff,
                                   allowed_files, issue_block=issue_block,
                                   suggested_files=suggested_files)
    root_cause = investigation["root_cause"]
    solution   = investigation["solution"]
    commit_msg = investigation["commit_message"]
    confidence = investigation["confidence"]
    evidence   = investigation["evidence"]
    findings   = investigation.get("findings", [])
    investigation_log = investigation.get("investigation_log")
    confidence_omitted = investigation.get("confidence_omitted", False)

    print("\n  ── investigation result ──")
    print(f"  OVERALL CAUSE : {root_cause}")
    print(f"  confidence    : {confidence:.0%}"
          + ("  (⚠ DEFAULTED)" if confidence_omitted else ""))
    print(f"  files read    : {', '.join(evidence.keys()) or '(none)'}")
    print(f"  issues found  : {len(findings)}")
    for i, fnd in enumerate(findings, 1):
        print(f"    {i}. {fnd['issue']}")

    failure_mode = investigation.get("failure_mode")
    if failure_mode in ("infra_timeout", "no_model_response", "unparseable_output"):
        print(f"[GATE] Investigation produced no usable model output ({failure_mode}) — escalating.")
        sys.exit(0)

    if confidence < 0.5:
        print(f"[GATE] Confidence {confidence:.0%} too low — escalating.")
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
    rejection_history = []
    rejected_fingerprints = set()
    last_patch_duration = 0.0

    def _issues_fingerprint(iss):
        return repr(sorted(
            ((it.get("file") or ""), (it.get("evidence") or ""),
             (it.get("corrected") or ""))
            for it in iss if isinstance(it, dict)))

    for repair_round in range(1, MAX_REPAIR_ROUNDS + 1):
        if _budget_exceeded():
            print(f"[BUDGET] time budget exceeded before repair round {repair_round}.")
            sys.exit(5)

        print(f"\n━━━ AI PATCH GENERATION AGENT (round {repair_round}/{MAX_REPAIR_ROUNDS}) ━━━")
        remaining = TOTAL_TIME_BUDGET - _elapsed()
        if repair_round > 1 and remaining < max(60, last_patch_duration * 1.2 + 20):
            print(f"[BUDGET] {remaining:.0f}s left — not enough for another round.")
            sys.exit(5)

        active_issue_block = retry_issue_block or issue_block
        _patch_t0 = time.time()
        try:
            issues = ai_generate_patch(patch_root_cause, solution, evidence,
                                       retry_note, active_issue_block,
                                       exact_evidence_lines=exact_evidence_lines)
        except Exception as exc:
            last_patch_duration = time.time() - _patch_t0
            print(f"[ERROR] patch generation failed: {exc}", file=sys.stderr)
            if repair_round == MAX_REPAIR_ROUNDS:
                sys.exit(2)
            continue
        last_patch_duration = time.time() - _patch_t0

        print(f"  issues reported: {len(issues)}")
        for n, it in enumerate(issues, 1):
            print(f"    {n}. {it.get('file','?')}: {it.get('problem','')[:80]}")

        fp = _issues_fingerprint(issues)
        if fp in rejected_fingerprints:
            print("[REPAIR] model repeated a previously rejected patch verbatim — escalating.")
            sys.exit(3)
        rejected_fingerprints.add(fp)

        fixes, pair_rejects = issues_to_fixes(issues, evidence)
        for rej in pair_rejects:
            print(f"  ✗ {rej}", file=sys.stderr)

        if not fixes:
            detail = "; ".join(pair_rejects) or "model reported no locatable issues"
            print(f"[ERROR] no usable fixes this round. {detail}", file=sys.stderr)
            if repair_round == MAX_REPAIR_ROUNDS:
                sys.exit(3)
            rejection_history.append(f"round {repair_round}: {detail}")
            retry_note = ("Your previous patch attempts were rejected for these reasons:\n"
                          + "\n".join(f"- {r}" for r in rejection_history)
                          + "\nMake sure 'evidence' is copied EXACTLY from the file "
                            "contents shown, and is ≤150 chars.")
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
                sys.exit(3)
            rejection_history.append(f"round {repair_round}: {detail}")
            retry_note = ("Your previous patches were rejected by validation for these reasons:\n"
                          + "\n".join(f"- {r}" for r in rejection_history))
            continue

        print("\n━━━ EXECUTE BUILD & TESTS ━━━")
        if not args.skip_tests:
            tests_ok, test_output = run_tests(stacks)
        else:
            tests_ok, test_output = True, ""
        docker_ok, docker_output = try_docker_build(written)

        if tests_ok and docker_ok:
            success = True
            break

        print(f"[REPAIR] round {repair_round} failed tests/build — reverting.")
        combined_output = (test_output + "\n" + docker_output).strip()
        retry_focused = extract_focused_failure(combined_output)
        retry_issue_block = format_focused_issue(retry_focused, "n/a")
        new_signal = extract_error_signal(combined_output) or combined_output[-1500:]
        new_signal = trim_supporting_signal(new_signal, retry_focused)
        CURRENT_FAILURE_CONTEXT += "\n" + retry_issue_block + "\n" + new_signal
        revert_files(originals)
        written = []

        if repair_round == MAX_REPAIR_ROUNDS:
            sys.exit(5)

        review = ai_review_failure(root_cause, solution, new_signal, retry_issue_block)
        if review["decision"] != "retry":
            sys.exit(5)
        root_cause, solution = review["root_cause"], review["solution"]
        retry_note = new_signal[:1200]

    if not success:
        sys.exit(5)

    print("\n━━━ COMMIT + PR ━━━")
    if chain_mode:
        branch = commit_to_existing_branch(commit_msg, written, failed_branch)
    else:
        branch = commit_to_branch(commit_msg, written)
        if branch and token and repo:
            open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
                    investigation_log, list(evidence.keys()), solution, findings)

    print("\n━━━ ✅ DONE ━━━")

if __name__ == "__main__":
    main()