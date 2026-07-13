#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — agentic investigation flow.

(Full description unchanged – see original.)
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
# qwen2.5-coder:7b — markedly better exact-copy / JSON-schema compliance than
# gemma3:4b for this pipeline's confirm/patch calls. Override via the
# OLLAMA_MODEL env var (e.g. from a GitHub Actions secret/variable).
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:7b")
# 7B on an 8-core CPU prefills ~2x slower than a 4B; give each call a little
# more wall-clock so a compliant answer isn't deadline-truncated mid-JSON.
AI_TIMEOUT     = int(os.environ.get("AI_TIMEOUT", "240"))
MAX_RETRIES    = int(os.environ.get("AI_MAX_RETRIES", "2"))
RETRY_BACKOFF  = [20, 20]
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))   # larger context window
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")

# Patch generation is a cheaper task than investigation (small JSON output) but
# was drowning in prefill: all evidence files were concatenated up to 16k chars.
# It now gets its own (smaller) evidence budget and its own timeout.
PATCH_TIMEOUT            = int(os.environ.get("PATCH_TIMEOUT", str(AI_TIMEOUT + 90)))
MAX_PATCH_EVIDENCE_CHARS = int(os.environ.get("MAX_PATCH_EVIDENCE_CHARS", "6000"))
PATCH_NUM_PREDICT        = int(os.environ.get("PATCH_NUM_PREDICT", "1200"))

INVESTIGATION_TIMEOUT = int(os.environ.get("INVESTIGATION_TIMEOUT", str(AI_TIMEOUT)))
INVESTIGATION_RETRIES = int(os.environ.get("INVESTIGATION_RETRIES", "1"))
INVESTIGATION_FIRST_TURN_EXTRA = int(os.environ.get("INVESTIGATION_FIRST_TURN_EXTRA", "60"))

# Raised from 600 → 900: a single investigation turn on the 8GB runner has been
# observed taking ~300s, so 600s left no room for a patch round + tests.
TOTAL_TIME_BUDGET = int(os.environ.get("TOTAL_TIME_BUDGET", "900"))
_run_start_time = None


def _elapsed() -> float:
    return time.time() - _run_start_time if _run_start_time else 0.0


def _budget_exceeded() -> bool:
    return _run_start_time is not None and _elapsed() >= TOTAL_TIME_BUDGET


# ── Agentic-loop bounds (tightened for focused investigation) ──────────────────
MAX_INVESTIGATION_TURNS = int(os.environ.get("MAX_INVESTIGATION_TURNS", "3"))   # 3 turns to allow full file scan + confirmation
MAX_FILES_PER_REQUEST   = int(os.environ.get("MAX_FILES_PER_REQUEST", "1"))
MAX_REPAIR_ROUNDS       = int(os.environ.get("MAX_REPAIR_ROUNDS", "4"))
# After the model confirms findings, run ONE extra model pass over the same
# evidence asking it to re-audit its own list for missed issues. Pure
# model-driven discovery — Python contributes nothing but the orchestration.
VERIFY_SWEEP            = os.environ.get("VERIFY_SWEEP", "1") == "1"

# ── Prompt / context budget ───────────────────────────────────────────────────
MAX_ERROR_LINES   = 14
MAX_SUPPORTING_LINES = 10
MAX_FILE_CHARS    = int(os.environ.get("MAX_FILE_CHARS", "8000"))  # large enough to see entire Dockerfile
MAX_TOTAL_CONTEXT = int(os.environ.get("MAX_TOTAL_CONTEXT", "12000"))
MAX_FILES_FIXED   = 4
MAX_PROMPT_CHARS  = int(os.environ.get("MAX_PROMPT_CHARS", "16000"))
INVESTIGATION_DIFF_CHARS = int(os.environ.get("INVESTIGATION_DIFF_CHARS", "1200"))
MAX_TRACEBACK_LINES = 12

# ── Focused-excerpt context shipping ─────────────────────────────────────────
# Ship the AI focused excerpts of evidence files (error-related lines + context)
# instead of whole files. On the 8GB runner every 1k prompt chars costs real
# generation time before the wall-clock deadline; whole-file shipping is what
# kept truncating investigations mid-decision. Selection of "relevant" lines is
# purely mechanical (token match against the error) — no diagnosis.
EXCERPT_CONTEXT_LINES = int(os.environ.get("EXCERPT_CONTEXT_LINES", "4"))
EXCERPT_HEAD_LINES    = int(os.environ.get("EXCERPT_HEAD_LINES", "6"))
EXCERPT_MAX_CHARS     = int(os.environ.get("EXCERPT_MAX_CHARS", "2600"))
EXCERPT_MIN_FILE_CHARS = int(os.environ.get("EXCERPT_MIN_FILE_CHARS", "1800"))
SKIP_MARKER_FMT = "<<<SKIPPED {n} UNRELATED LINES>>>"
SKIP_MARKER_RE  = re.compile(r"<<<SKIPPED \d+ UNRELATED LINES>>>")

# ── Deterministic-context-retrieval budget ───────────────────────────────────
MAX_FILES_PER_CATEGORY = int(os.environ.get("MAX_FILES_PER_CATEGORY", "2"))
MAX_PRELOADED_FILES    = int(os.environ.get("MAX_PRELOADED_FILES", "4"))

# ── Git flow ──────────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_HIDDEN   = {".git", "auto-fixer.py"}

# The auto-fixer's OWN orchestration workflow is not part of the user's app —
# it is the tool's plumbing (it talks about downloading failure.log, retries,
# 404s, calling ollama, running auto-fixer.py). Feeding it to a small model as
# "evidence" makes the model confabulate root causes about the tooling itself
# instead of the actual CI failure. We detect such files by content — anything
# that invokes auto-fixer.py or clearly drives this tool — and hide them from
# the model. Pure observation/filtering; no diagnosis.
SELF_WORKFLOW_MARKERS = [
    re.compile(r'auto-?fixer\.py'),
    re.compile(r'OLLAMA_(?:API_URL|MODEL)\b'),
    re.compile(r'auto-?fixer-on-failure', re.I),
]
_SELF_WORKFLOW_CACHE = {}


def _is_self_workflow(rel: str) -> bool:
    """True if `rel` is one of the auto-fixer's own workflow files (by name or
    by referencing the tool). Cached; safe to call in hot loops."""
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

# ── Pattern for post‑patch "missing reference" validation ────────────────────
REPO_REF_EXT_PATTERN = re.compile(
    r'(?<![\w./\-])((?:[\w.\-]+/)*[\w\-]+\.(?:py|txt|ya?ml|json|toml|cfg|ini|'
    r'js|jsx|ts|tsx|go|java|rb|sh))(?![\w./\-])')

# Set in main() to the actual failure evidence (issue block + log signal).
# Validators use it ONLY to decide whether a pre-existing broken reference is
# part of the error being fixed (blocking) or unrelated noise (non-blocking).
# This keeps validation error-driven, not file-driven.
CURRENT_FAILURE_CONTEXT = ""


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


def retrieve_files_by_error_tokens(error_tokens: list, allowed_files: set,
                                   cap: int = MAX_PRELOADED_FILES) -> list:
    """OBSERVATION-ONLY retrieval that works for ANY error type, independent
    of the failure-category signatures: the error log itself names things
    (paths, values, identifiers) — find the repo files whose NAME matches one
    of those tokens or whose CONTENT contains one. Fully log-driven: no
    knowledge of what kind of error this is is used or needed. This is the
    fallback that guarantees the audit/pinning pipeline gets evidence even
    for failure shapes the category signatures have never seen. Files are
    ranked by how many DISTINCT error tokens they hit."""
    if not error_tokens:
        return []
    scores = {}
    # Prose documentation (README, guides) only MENTIONS the error's tokens in
    # sentences — it never participates in a build/runtime failure. Excluding it
    # keeps the model's prompt focused on files that actually run, and avoids
    # shipping doc prose as "evidence".
    candidate_files = {f for f in allowed_files if not _is_prose_doc(f)}
    # 1) filename matches (cheap)
    for f in candidate_files:
        name = Path(f).name.lower()
        for tok in error_tokens:
            tbase = tok.rsplit("/", 1)[-1].lower()
            if tbase and (tbase == name or (len(tbase) >= 5 and tbase in name)):
                scores[f] = scores.get(f, 0) + 2
    # 2) content matches (bounded scan)
    scanned = 0
    for f in sorted(candidate_files):
        if scanned >= 200:
            break
        p = Path(f)
        if not p.is_file() or not _is_text_file(p):
            continue
        scanned += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        hits = sum(1 for tok in set(error_tokens) if tok and tok in text)
        if hits:
            scores[f] = scores.get(f, 0) + hits
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:cap]
    if ranked:
        print(f"[EVIDENCE] Token-driven retrieval (category-agnostic) matched "
              f"{len(ranked)} file(s) the error itself points at: "
              f"{[f for f, _ in ranked]}")
    return [f for f, _ in ranked]


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
    # deprecation / runtime warnings that are never the actual failure but
    # frequently sit right next to it in GitHub Actions logs (this is what
    # got mis-picked as "command context" for a python-version error):
    "deprecationwarning", "--trace-deprecation", "trace-warnings",
    "node --trace", "(use `node", "punycode", "experimentalwarning",
    "npm warn", "warning:", "deprecated", "notice]", "downloading",
    "already satisfied", "reading package", "resolving", "setup-python",
    "actions/checkout", "actions/setup", "run docker", "##[debug]",
]
# A line that is plausibly the COMMAND that triggered a failure: a shell
# invocation or a known tool call. Used to reject "the line above the error"
# when that line is just a warning or log noise.
COMMAND_LINE_HINTS = [
    re.compile(r'^\s*(?:\$|>|\+)\s'),                       # shell prompt / set -x echo
    re.compile(r'\b(?:python|pip|pytest|npm|yarn|node|go|mvn|gradle|'
               r'docker|docker\s+buildx|curl|make|bash|sh)\b', re.I),
    re.compile(r'\brun:\s'),                                # workflow "run:" line
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


_TS_PREFIX = re.compile(
    r'^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\s+')


def _strip_ts(line: str) -> str:
    """Remove the leading GitHub-Actions ISO-8601 timestamp from a log line so
    the AI sees the clean message, not '2026-07-13T01:51:41.5Z Error: ...'."""
    return _TS_PREFIX.sub("", line)


def _strip_log_timestamps(log_text: str) -> str:
    return "\n".join(_strip_ts(l) for l in log_text.splitlines())


# GitHub-Actions logs occasionally carry mojibake — null bytes, BOMs, or U+FFFD
# replacement chars (e.g. a corrupted `--build-arg PYTHON_VERSION=` producing a
# base image tag like '\x00\x003.12'). If that garbage reaches the AI it drives
# a nonsense diagnosis. Strip control chars and collapse replacement-char runs.
_CTRL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_REPLACEMENT_RUN = re.compile(r'[\ufffd\ufeff]+')


def sanitize_log_text(log_text: str) -> tuple:
    """Remove control/replacement characters from log text. Returns
    (clean_text, corrupted_bool) — corrupted is True if any were found, so the
    caller can lower confidence on a diagnosis built from a garbled signal."""
    corrupted = bool(_CTRL_CHARS.search(log_text) or _REPLACEMENT_RUN.search(log_text))
    clean = _REPLACEMENT_RUN.sub("", _CTRL_CHARS.sub("", log_text))
    return clean, corrupted


def looks_garbled(text: str) -> bool:
    """True if a short string still looks corrupted after sanitizing — e.g. a
    primary error like \"The version '3.12\" that was cut mid-token, or one with
    a high ratio of non-ASCII noise. Used to refuse anchoring a confident
    diagnosis on an unreadable error."""
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

    command_context_lines = []
    if primary_message:
        log_lines = log_text.splitlines()
        msg_line_idx = None
        for i, line in enumerate(log_lines):
            if primary_message in line:
                msg_line_idx = i
                break
        if msg_line_idx is not None:
            # Search a small window ABOVE the error for a line that actually
            # looks like a command (not just whatever line happened to precede
            # it — that was grabbing Node deprecation warnings as "context").
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


# ── FOCUSED-EXCERPT CONTEXT SHIPPING ─────────────────────────────────────────
def extract_error_tokens(focused: dict, signal: str) -> list:
    """Mechanically extract the distinctive tokens the failure names: quoted
    values ('3.2', 'sample_aprrp/test_app.py'), version numbers, and path-like
    references. Shared by the offending-line locator and the excerpt builder.
    Pure text extraction from the error — no diagnosis."""
    haystack = "\n".join(filter(None, [
        focused.get("primary_message", ""),
        "\n".join(focused.get("command_context", []) or []),
        "\n".join(focused.get("gh_errors", []) or []),
        signal or "",
    ]))
    tokens = set()
    for m in re.finditer(r"['\"]([\w.\-:/]{2,60})['\"]", haystack):
        tokens.add(m.group(1))
    for m in re.finditer(r"version\s+['\"]?(\d+\.\d+(?:\.\d+)?)['\"]?", haystack, re.I):
        tokens.add(m.group(1))
    for m in re.finditer(r'(?<![\w./\-])((?:[\w.\-]+/)+[\w.\-]+)(?![\w./\-])', haystack):
        tok = m.group(1).strip("'\":,()[]")
        if tok and not tok.lower().startswith(("http", "/home/", "/usr/", "/opt/")):
            tokens.add(tok)
            # also the basename, so a typo'd dir still matches the file line
            base = tok.rsplit("/", 1)[-1]
            if len(base) >= 4:
                tokens.add(base)
    # drop trivially-common tokens that would match everything
    return [t for t in tokens if len(t) >= 3 and t.lower() not in
            ("run", "yml", "yaml", "true", "false", "with", "name", "uses")]


def focused_excerpt(content: str, tokens: list,
                    context: int = None, head_lines: int = None,
                    max_chars: int = None) -> tuple:
    """Build a focused excerpt of a file: the first few lines (file identity)
    plus every line matching an error token, with N context lines around each.
    Skipped regions are replaced by a marker the model is told never to copy.
    Line SELECTION is mechanical token-matching; the lines themselves are
    verbatim file content, so anything the model copies as 'evidence' still
    matches the real file. Returns (text, was_excerpted)."""
    context = EXCERPT_CONTEXT_LINES if context is None else context
    head_lines = EXCERPT_HEAD_LINES if head_lines is None else head_lines
    max_chars = EXCERPT_MAX_CHARS if max_chars is None else max_chars
    if not content or len(content) <= max(EXCERPT_MIN_FILE_CHARS, max_chars // 2) \
            and len(content) <= max_chars:
        return content, False
    lines = content.splitlines()
    keep = set(range(min(head_lines, len(lines))))
    matched_any = False
    for i, line in enumerate(lines):
        for tok in tokens or []:
            try:
                if re.search(r'(?<![\w.])' + re.escape(tok) + r'(?![\w.])', line):
                    matched_any = True
                    keep.update(range(max(0, i - context),
                                      min(len(lines), i + context + 1)))
                    break
            except re.error:
                continue
    if not matched_any:
        # No token hits — excerpting would hide everything relevant; fall back
        # to a plain head-truncation so the model still sees the file shape.
        if len(content) > max_chars:
            return content[:max_chars] + "\n...(truncated)", True
        return content, False
    out, prev = [], None
    for i in sorted(keep):
        if prev is not None and i > prev + 1:
            out.append(SKIP_MARKER_FMT.format(n=i - prev - 1))
        out.append(lines[i])
        prev = i
    if prev is not None and prev < len(lines) - 1:
        out.append(SKIP_MARKER_FMT.format(n=len(lines) - 1 - prev))
    text = "\n".join(out)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)"
    return text, True


def excerpt_evidence(evidence: dict, tokens: list, tag: str) -> dict:
    """Apply focused_excerpt to every evidence file; log the savings."""
    out, saved = {}, 0
    for f, c in evidence.items():
        if not isinstance(c, str):
            out[f] = c
            continue
        ex, was = focused_excerpt(c, tokens)
        out[f] = ex
        if was:
            saved += len(c) - len(ex)
    if saved > 0:
        print(f"[{tag}] focused excerpts cut evidence by {saved} chars "
              f"(error-token windowing; full files stay on disk for validation).")
    return out


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
            deadline = t0 + timeout
            resp = requests.post(endpoint, json=payload, timeout=(10, timeout), stream=True)
            resp.raise_for_status()
            hit_deadline = False
            for line in resp.iter_lines():
                # Per-read timeout resets on every token, so a steady stream can
                # run far past `timeout` (observed: 355s on a 300s cap). Enforce
                # a hard wall-clock deadline; partial output is still salvageable
                # by the truncated-JSON repair in _json_from.
                if time.time() > deadline:
                    hit_deadline = True
                    print(f"[{tag}] hard wall-clock deadline {timeout}s hit "
                          f"mid-stream — stopping with partial output.")
                    try:
                        resp.close()
                    except Exception:
                        pass
                    break
                tok = _extract_token(line, fmt)
                if tok:
                    collected.append(tok)
            raw = "".join(collected).strip()
            print(f"[{tag}] done in {time.time()-t0:.1f}s — {len(raw)} chars"
                  + (" (deadline-truncated)" if hit_deadline else ""))
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


def _close_truncated_json(text: str):
    """Generic repair for JSON cut off mid-generation (num_predict cap):
    close any open string, strip a trailing incomplete token, and close all
    open brackets. Falls back to backtracking to the last complete value."""
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
    # Backtrack: drop the trailing incomplete key/value and try again.
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
    # Last resort: the output was probably truncated mid-generation.
    start = cleaned.find("{")
    if start != -1:
        repaired = _close_truncated_json(cleaned[start:])
        if repaired is not None:
            print("[JSON] recovered a truncated JSON object from model output.")
            return repaired
    return None


# ── 🔥 HARDENED AI INVESTIGATION PROMPT ──────────────────────────────────────
INVESTIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        # ORDER MATTERS: llama.cpp emits keys in this declared order under a
        # grammar-constrained `format`. Decision fields go FIRST so a slow /
        # truncated generation still yields status + confidence + findings.
        # The verbose `analysis` prose goes LAST and is optional — it must
        # never be the thing that eats the token/time budget before the
        # decision is emitted (that is exactly what escalated a correct
        # diagnosis as "0 issues found").
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

2. Evidence files may be shown as FOCUSED EXCERPTS: lines related to the error
   plus a few lines of context, with omitted regions marked
   `<<<SKIPPED N UNRELATED LINES>>>`. The shown lines are VERBATIM file
   content. Never treat a skip marker as file content, and if the bug seems to
   be in a skipped region, request the file again via `need_more_info`.

3. The root cause is NOT necessarily the first thing that looks wrong. Many
   failures are caused by MULTIPLE INDEPENDENT problems in the same file or
   across different files. You must treat each distinct problem as a separate
   bug — even if they are all in the same line or the same file.

4. **AUDIT EVERY SHOWN LINE:** for every path, filename, version string,
   command, flag, argument, or reference in the evidence, verify it
   (a) points to something in the "## Repository tree", OR (b) is a known
   valid version/tool, OR (c) matches the requirements described elsewhere in
   the evidence. Every mismatch is a separate entry in `findings` — do not
   stop after the first one.

5. **CROSS-REFERENCE BETWEEN FILES:** if a Dockerfile or workflow references
   another file, check that it exists in the tree exactly as written
   (case-sensitive), and that versions are compatible across files.

6. **VERSION STRINGS AND TAGS:** check every version tag against what actually
   exists. `python:3.1` instead of `python:3.10` is a bug. An unsupported
   `python-version:` in a workflow is a bug.

7. **WHEN TO REQUEST MORE FILES:** if you suspect a bug but cannot confirm it
   because a crucial file (or a skipped region) has not been provided, request
   it with `status: need_more_info`.

8. **CONFIRMATION RULES:** only respond `status: root_cause_confirmed` once
   you have verified EVERY issue you report. Set `all_issues_found` to `true`
   ONLY after checking every shown line; if unsure (truncated file, skipped
   regions you couldn't check), set it `false` and lower confidence.

9. **FINDINGS FORMAT:** each entry needs `issue` (short name), `root_cause`
   (why it breaks the build), `solution` (precise text change). One entry per
   distinct bug; never merge bugs; never skip a bug as "minor".

10. **CONFIDENCE (MANDATORY — NEVER OMIT):** numeric 0.0-1.0 in EVERY response.
    0.9+ = read everything, found all issues. 0.7-0.89 = sure but couldn't see
    everything. ≤0.4 = guessing. A diagnosis directly confirmed by the error
    log AND located in the evidence deserves 0.8+.

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


# ── FAST-PATH CONFIRMATION (used when deterministic tooling pinned the bug) ──
# When locate_offending_lines() has pinned the exact line(s) carrying the
# flagged value, the model's job is CONFIRMATION, not investigation. Ship a
# prompt roughly HALF the size of the investigation prompt: the issue block
# (which already contains the pinned lines, path facts, and version-coherence
# report) plus focused excerpts of ONLY the pinned files. No repo tree, no git
# diff, no supporting log, no multi-turn rulebook. One turn; on refusal or
# low confidence we fall back to the full investigation loop unchanged.
# Rationale (observed run): 3 investigation turns × ~190s all returned
# need_more_info on identical 10k-char prompts even though the pinned facts
# already contained the answer — open-ended framing invites hedging.
FAST_CONFIRM = os.environ.get("FAST_CONFIRM", "1") == "1"
MAX_CONFIRM_PROMPT_CHARS = int(os.environ.get("MAX_CONFIRM_PROMPT_CHARS", "7000"))
CONFIRM_TIMEOUT = int(os.environ.get(
    "CONFIRM_TIMEOUT",
    str(INVESTIGATION_TIMEOUT + INVESTIGATION_FIRST_TURN_EXTRA)))

CONFIRM_SYSTEM = """\
You are a DevOps failure-confirmation agent. Deterministic tooling has ALREADY
located the exact offending line(s) and cross-file facts shown in the
"## ISSUE TO SOLVE" section below. Your ONLY job is to confirm the diagnosis
and enumerate EVERY distinct bug — or refute it if the pinned facts genuinely
do not explain the error.

Rules:
- ONE `findings` entry PER distinct bug, even if several share a file or line.
- If the issue lists numbered FACT entries (a repository-side audit of broken
  references), emit one `findings` entry PER FACT — each has already been
  verified against the real repository; do not merge or skip any.
- Each finding's `solution` = the precise text change (quote the exact line to
  change and what it becomes).
- If the pinned facts fully explain the error, respond
  `status: "root_cause_confirmed"` with confidence 0.8 or higher — the hard
  verification work has already been done for you.
- Use `status: "need_more_info"` (with `requested_files`) ONLY if the pinned
  facts cannot explain the error.

Output ONLY one JSON object. No markdown fences.
**EMIT KEYS IN EXACTLY THIS ORDER:** `status`, `confidence`, `root_cause`,
`solution`, `all_issues_found`, `commit_message`, `findings`, and a ONE-sentence
`analysis` LAST.

{"status":"root_cause_confirmed","confidence":0.9,"root_cause":"...","solution":"...","all_issues_found":true,"commit_message":"fix: ...","findings":[{"issue":"...","root_cause":"...","solution":"..."}],"analysis":"one short sentence"}
"""


def ai_confirm_pinned(issue_block: str, exit_code: str, evidence: dict,
                      error_tokens: list):
    """Single lean confirm-or-refute call. Returns a finalized investigation
    dict on success, or None to fall back to the full investigation loop."""
    if not evidence:
        return None
    shipped = excerpt_evidence(dict(evidence), error_tokens or [], "CONFIRM")
    ev_parts, total = [], 0
    for f, c in shipped.items():
        block = f"### {f}\n```\n{c}\n```"
        if ev_parts and total + len(block) > MAX_CONFIRM_PROMPT_CHARS // 2:
            break
        ev_parts.append(block)
        total += len(block)
    ev_text = "\n\n".join(ev_parts) if ev_parts else "(none)"
    prompt = (f"{CONFIRM_SYSTEM}\n"
              f"## ISSUE TO SOLVE (includes pinned offending lines and "
              f"verified cross-file facts)\n{issue_block}\n"
              f"## Exit code: {exit_code}\n\n"
              f"## File contents (focused excerpts of the pinned files — the "
              f"shown lines are VERBATIM; `<<<SKIPPED N UNRELATED LINES>>>` "
              f"markers are NOT file content)\n{ev_text}\n\n"
              f"Respond with the JSON.")
    if len(prompt) > MAX_CONFIRM_PROMPT_CHARS:
        prompt = prompt[:MAX_CONFIRM_PROMPT_CHARS] + "\n\nRespond with the JSON."
    try:
        raw = _stream_ollama(prompt, INVESTIGATE_SCHEMA, num_predict=700,
                             temperature=0.05, tag="CONFIRM",
                             timeout=CONFIRM_TIMEOUT, retries=1)
    except Exception as exc:
        print(f"[CONFIRM] fast path failed ({exc}) — falling back to full "
              f"investigation.", file=sys.stderr)
        return None
    data = _json_from(raw) or {}
    status = data.get("status", "")
    print(f"[CONFIRM] status={status or '(none)'} | "
          f"confidence={data.get('confidence', '(omitted)')} | "
          f"{(data.get('analysis') or '')[:140]}")
    if status != "root_cause_confirmed":
        print("[CONFIRM] model did not confirm on pinned facts — running full "
              "investigation instead.")
        return None
    result = _finalize_investigation(data, False, evidence)
    if result["confidence"] < 0.5:
        print("[CONFIRM] confirmed but confidence below gate — running full "
              "investigation instead.")
        return None
    result["failure_mode"] = None
    result["evidence"] = dict(evidence)
    result["investigation_log"] = [{
        "turn": 1, "status": "root_cause_confirmed (fast-path confirm)",
        "analysis": (data.get("analysis") or "")[:200]}]
    return result


# ── Investigation loop ───────────────────────────────────────────────────────
def locate_offending_lines(focused: dict, signal: str,
                           evidence_files: dict) -> str:
    """OBSERVATION ONLY (not diagnosis, not fix-authoring): the failure log
    often names a specific offending VALUE (a quoted version like '3.1', a
    port, a tag). A slow/small model reliably knows WHAT is wrong but often
    fails to copy the EXACT line into its find/replace pair — it grabs the
    lines above the bug, or produces evidence==corrected. This helper finds
    the exact existing line(s) that contain the flagged value and hands them
    back verbatim, so the model has the precise 'evidence' string to copy.
    Python states the line that EXISTS; it never writes the replacement."""
    if not evidence_files:
        return ""
    tokens = extract_error_tokens(focused, signal)
    if not tokens:
        return ""

    found = []
    seen_lines = set()
    for fname, content in evidence_files.items():
        if not isinstance(content, str):
            continue
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line in seen_lines:
                continue
            for tok in tokens:
                # Match the token as a whole value, not a coincidental substring.
                try:
                    if re.search(r'(?<![\w.])' + re.escape(tok) + r'(?![\w.])', line):
                        found.append((fname, line, tok))
                        seen_lines.add(line)
                        break
                except re.error:
                    continue
            if len(found) >= 6:
                break
        if len(found) >= 6:
            break
    if not found:
        return ""
    print(f"[EVIDENCE] Offending-line locator pinned {len(found)} exact "
          f"line(s) containing the value(s) the error flags.")
    lines = "\n".join(
        f"- In '{f}', this EXACT line contains the flagged value '{tok}':\n"
        f"    {line}"
        for f, line, tok in found)
    return ("Exact offending line(s) located in the repository (the error names "
            "these value(s); the lines below are copied VERBATIM from the real "
            "files — use the relevant one as your 'evidence' string EXACTLY, and "
            "change ONLY the flagged value in 'corrected'):\n" + lines)


def preflight_reference_audit(evidence_files: dict, allowed_files: set) -> tuple:
    """OBSERVATION ONLY (not diagnosis, not fix-authoring): audit the evidence
    files THEMSELVES for references that do not resolve to anything that
    exists, and for invalid version tags — using the exact same observation
    maps the post-patch validator uses. This is what catches MULTI-BUG files:
    a CI log usually names only the FIRST failure (the build dies at the
    first broken COPY), while the same file may carry more broken references
    that would fail the NEXT run — the log's error tokens can never cover
    those. For each broken reference Python reports the fact, the closest
    real path that DOES exist, and the exact verbatim line(s) carrying it;
    it never states which is 'the bug' or authors any replacement.
    Returns (facts_text, audit_tokens, facts)."""
    facts, seen = [], set()
    for fname, content in (evidence_files or {}).items():
        if not isinstance(content, str):
            continue
        problems = {}
        problems.update(_missing_ref_map(content, fname))
        problems.update(_dockerfile_problem_map(fname, content))
        if WORKFLOW_PATTERN.search(fname):
            # reuse the validator's workflow-version observation as facts too
            for v in invalid_workflow_python_versions(content):
                problems[v] = (f"python-version '{v}' is not a real released "
                               f"CPython version")
        for token, desc in problems.items():
            key = (fname, token)
            if key in seen:
                continue
            seen.add(key)
            line_hits = []
            for raw in content.splitlines():
                if token in raw and raw.strip():
                    line_hits.append(raw.strip())
                    if len(line_hits) >= 2:
                        break
            closest = ""
            if not token.startswith("python:"):
                closest = _closest_allowed_file(token, allowed_files)
            facts.append({"token": token, "file": fname, "desc": desc,
                          "lines": line_hits, "closest": closest})
    if not facts:
        return "", [], []
    print(f"[EVIDENCE] Pre-flight reference audit found {len(facts)} broken "
          f"reference(s)/tag(s) across the evidence files — the log likely "
          f"names only the first; ALL will be shipped as numbered facts.")
    parts = [
        "Repository-side audit of the evidence files (OBSERVED FACTS — every "
        "numbered item is a reference or tag that does NOT resolve to anything "
        "that exists. The CI log may only show the FIRST one, because the "
        "build stops there. You MUST reconcile EVERY numbered fact — one "
        "findings/issues entry EACH; do not stop after the first):"]
    for i, f in enumerate(facts, 1):
        line = f"- FACT {i}: In '{f['file']}', {f['desc']}."
        if f["closest"]:
            line += f" The closest path that DOES exist is '{f['closest']}'."
        for l in f["lines"]:
            line += f"\n    exact line (copy VERBATIM as 'evidence'): {l}"
        parts.append(line)
    tokens = []
    for f in facts:
        tokens.append(f["token"])
        base = f["token"].rsplit("/", 1)[-1]
        if base and base != f["token"]:
            tokens.append(base)
        if f["closest"]:
            tokens.append(f["closest"])
    return "\n".join(parts), tokens, facts


_DOCKER_WORKDIR = re.compile(r'^\s*WORKDIR\s+(\S+)', re.I | re.M)
_CONTAINER_ABS_PATH = re.compile(r"(/[\w.\-]+(?:/[\w.\-]+)*/[\w.\-]+\.\w{1,6})")


def map_container_paths_to_repo(focused: dict, signal: str,
                                evidence_files: dict) -> str:
    """OBSERVATION ONLY (not diagnosis): a runtime error from INSIDE a
    container names paths in the CONTAINER filesystem (e.g.
    '/app/apsspp.py'), while the file the model must edit shows the
    repo-side spelling (e.g. CMD [\"python\", \"apsspp.py\"]). Small models
    reliably copy the container path from the log as their 'evidence', which
    then matches nothing. Using the WORKDIR value(s) read from the
    Dockerfile(s) already in evidence, report the mapping between each
    container path in the error and the repo-side reference — and the exact
    Dockerfile line that carries it. Python maps paths; it never says what
    the fix is."""
    dockerfiles = {f: c for f, c in (evidence_files or {}).items()
                   if isinstance(c, str) and "dockerfile" in Path(f).name.lower()}
    if not dockerfiles:
        return ""
    blob = "\n".join(filter(None, [
        focused.get("primary_message", ""),
        "\n".join(focused.get("gh_errors", []) or []),
        signal or "",
    ]))
    notes, seen = [], set()
    for m in _CONTAINER_ABS_PATH.finditer(blob):
        cpath = m.group(1)
        low = cpath.lower()
        if cpath in seen or low.startswith(("/home/", "/usr/", "/opt/", "/tmp/",
                                            "/var/", "/etc/", "/proc/", "/sys/")):
            continue
        seen.add(cpath)
        for dfile, dcontent in dockerfiles.items():
            for wm in _DOCKER_WORKDIR.finditer(dcontent):
                workdir = wm.group(1).rstrip("/")
                if not workdir or not cpath.startswith(workdir + "/"):
                    continue
                rel = cpath[len(workdir) + 1:]
                base = rel.rsplit("/", 1)[-1]
                # exact Dockerfile line(s) that reference this file
                carrier = ""
                for raw in dcontent.splitlines():
                    if (rel in raw or base in raw) and raw.strip() \
                            and not raw.strip().upper().startswith("WORKDIR"):
                        carrier = raw.strip()
                        break
                note = (f"- Container path '{cpath}' in the error is "
                        f"'{rel}' relative to WORKDIR '{workdir}' in "
                        f"'{dfile}' — i.e. it refers to the build-context "
                        f"file '{rel}', NOT to a repo path starting with "
                        f"'{workdir}/'.")
                if carrier:
                    note += (f"\n    The Dockerfile line that references it is "
                             f"(copy THIS VERBATIM as 'evidence', never the "
                             f"container path from the log): {carrier}")
                notes.append(note)
                break
        if len(notes) >= 4:
            break
    if not notes:
        return ""
    print(f"[EVIDENCE] Container-path bridge: mapped {len(notes)} container "
          f"path(s) from the error onto repo-side Dockerfile line(s).")
    return ("Container-path → repository mapping (the error was produced "
            "INSIDE the container; these are the repo-side facts):\n"
            + "\n".join(notes))


def unaddressed_audit_facts(issues: list, audit_facts: list) -> list:
    """VALIDATION ONLY: which audited broken references does the model's issue
    list NOT touch? A fact counts as addressed when some issue's 'evidence'
    line contains its token (i.e. the model is editing the line that carries
    it). Python never writes the missing entries itself — it reports the gap
    so the MODEL authors them on the retry."""
    out = []
    for i, f in enumerate(audit_facts or [], 1):
        tok = f.get("token", "")
        if not tok:
            continue
        addressed = any(isinstance(it, dict) and tok in (it.get("evidence") or "")
                        for it in issues or [])
        if not addressed:
            msg = (f"FACT {i}: '{tok}' in '{f.get('file','?')}' — no issue "
                   f"entry edits the line carrying it")
            if f.get("lines"):
                msg += f" (exact line: {f['lines'][0]})"
            out.append(msg)
    return out


def enrich_issue_with_path_checks(focused: dict, signal: str,
                                  allowed_files: set) -> str:
    """OBSERVATION ONLY (not diagnosis): scan the primary error + supporting
    signal for file/dir paths, and for any that do NOT exist in the repo,
    report the closest path that DOES exist. This is the same denial-to-
    evidence enrichment already used elsewhere — it hands the model verified
    facts ('X named in the error is missing; Y exists and is one edit away')
    so a slow/small model can't drift off onto an invented root cause. Python
    never says which is 'the bug' or how to fix it; it only states what exists.
    """
    haystack = "\n".join(filter(None, [
        focused.get("primary_message", ""),
        "\n".join(focused.get("command_context", []) or []),
        "\n".join(focused.get("gh_errors", []) or []),
        signal or "",
    ]))
    # path-like tokens: a/b/c.ext or bare dir/file references
    path_re = re.compile(r'(?<![\w./\-])((?:[\w.\-]+/)+[\w.\-]+)(?![\w./\-])')
    # Prefixes/patterns that are runner or system paths, URLs, or GH expressions
    # — never repo files, and previously the source of false "missing" flags.
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
                # require a real file-ish extension OR a short repo-relative
                # depth; long absolute-looking chains are runner noise.
                or token.count("/") > 4):
            continue
        seen.add(token)
        norm = _relstrip(token)
        if norm in allowed_files or Path(norm).exists():
            continue  # path is real — nothing to note
        near = _closest_allowed_file(norm, allowed_files)
        # Only report when there IS a close real match — an unmatched token in
        # a noisy log is far more likely to be noise than a genuine missing
        # repo file, and reporting it just distracts the model.
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
    """Return the major.minor from a repo-root .python-version file, or ''."""
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
    """OBSERVATION ONLY: gather every CONCRETE python major.minor declared
    across the repo (source of truth file, workflow, Dockerfile, packaging),
    keyed by where it lives. Python reads and reports; it never rewrites."""
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
            # A hardcoded python-version is a concrete declaration. A
            # `python-version-file:` reference is NOT — it defers to the file,
            # which is exactly the coherent setup, so we don't record it.
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
    """OBSERVATION ONLY (not diagnosis, not fix-authoring): if the repo declares
    a Python version in more than one place and they DISAGREE, report the
    mismatch so the model aligns them. When a .python-version file exists it is
    named as the source of truth. Python states the facts and which value is
    canonical; it never edits any file or decides the winner itself."""
    decls = collect_python_version_declarations(allowed_files)
    if len(decls) < 2:
        return ""
    values = set(decls.values())
    if len(values) == 1:
        return ""  # everything already agrees — nothing to say
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


# Signatures of a failure that is specifically about a Python version being
# invalid/unavailable (setup-python, Docker base image, pyenv).
PY_VERSION_FAILURE_SIGNATURES = [
    re.compile(r"version\s+'?[\d.]+'?\s+.*was not found", re.I),   # setup-python
    re.compile(r"python-version", re.I),
    re.compile(r"failed to solve:\s*python:", re.I),               # docker FROM python:X
    re.compile(r"docker\.io/library/python:", re.I),
    re.compile(r"no such (?:version|python)", re.I),
    re.compile(r"pyenv:.*version.*not installed", re.I),
    # setup-python's python-version-file pointing at a missing file:
    re.compile(r"python[ _-]?version[ _-]?file", re.I),
    re.compile(r"specified python version file.*(?:does\s*n['o]?t|not)\s*exist", re.I),
]


def is_python_version_failure(issue_block: str, signal: str) -> bool:
    blob = f"{issue_block}\n{signal}"
    return any(p.search(blob) for p in PY_VERSION_FAILURE_SIGNATURES)


def _extract_version_file_name(blob: str) -> str:
    """Pull the referenced version-file path from either the workflow
    `python-version-file: X` form or the error's `file at: X` form. Tries all
    matches and returns the first that actually looks like a filename (the
    `python-version-file` regex can otherwise capture the stray word 'at' from
    'python version file at:')."""
    candidates = []
    for pat in (_VERSION_FILE_REF, _MISSING_FILE_ERR):
        for m in pat.finditer(blob):
            candidates.append(m.group(1).strip().strip("`'\":,"))
    for cand in candidates:
        if cand and ("." in cand or "/" in cand):
            return cand
    return ""


def missing_version_file(issue_block: str, signal: str, allowed_files: set) -> str:
    """Return the missing version-file path if the failure is 'the workflow
    points at a version file that does not exist', else ''. This is distinct
    from 'no version defined anywhere' — a version may well exist elsewhere
    (e.g. the Dockerfile), but setup-python's python-version-file target is
    absent. The fix is to CREATE that file (or stop referencing it), which the
    edit-only patch engine cannot do — so this must escalate, not go to the
    model. Observation only; Python names the missing file the error/workflow
    already state."""
    cand = _extract_version_file_name(f"{issue_block}\n{signal}")
    if not cand:
        return ""
    norm = _relstrip(cand)
    # Confirm it really is absent (present in neither the readable tree nor disk).
    if norm in allowed_files or Path(norm).is_file():
        return ""
    return cand


def python_version_is_undefined(allowed_files: set) -> bool:
    """True when the repo declares NO concrete Python version anywhere the tool
    can read (.python-version, workflow python-version:, Dockerfile FROM/ARG,
    requires-python). In that case there is nothing to anchor a fix to, so the
    AI must NOT invent a version — Python reports the absence and the run
    escalates for a human to define the intended version."""
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

_VERSION_FILE_REF = re.compile(
    r"python[-_ ]?version[-_ ]?file\s*:?\s*([.\w][\w./\-]*)", re.I)
_MISSING_FILE_ERR = re.compile(
    r"file\s+at:?\s*([.\w][\w./\-]*)", re.I)


def undefined_python_version_message(issue_block: str, signal: str) -> str:
    """Return the escalation text. If the failure names a specific version file
    the workflow expects but which is missing (e.g. `.python-version`), point
    at THAT file explicitly — the intended fix is to create it. Otherwise fall
    back to the generic 'no version defined anywhere' message. Python only
    reports what the error/workflow already state; it does not choose a value."""
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


def _build_investigation_prompt(signal, exit_code, repo_tree, git_diff,
                                evidence: dict, last_turn: bool,
                                notes: list = None, issue_block: str = "",
                                error_tokens: list = None) -> str:
    # Ship focused excerpts, not whole files — the single biggest lever for
    # finishing generation before the wall-clock deadline on a slow runner.
    shipped = excerpt_evidence(evidence, error_tokens or [], "INVESTIGATE")
    ev_parts, total = [], 0
    for f, c in shipped.items():
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
    if confidence_omitted:
        # With the enforced schema this should never happen; if it does, make it
        # loud so a defaulted value is never mistaken for the model's judgment.
        print("[FINALIZE] ⚠ model omitted 'confidence' — defaulting to 0.4 "
              "(this WILL fail the 0.5 gate). Schema enforcement may be off "
              "(non-native endpoint?).", file=sys.stderr)
    try:
        confidence = float(raw_conf if raw_conf is not None else 0.4)
    except (TypeError, ValueError):
        print(f"[FINALIZE] ⚠ model returned non-numeric confidence "
              f"({raw_conf!r}) — defaulting to 0.4.", file=sys.stderr)
        confidence = 0.4
        confidence_omitted = True
    confidence = max(0.0, min(confidence, 1.0))

    if forced and data.get("status") != "root_cause_confirmed":
        confidence = min(confidence, 0.4)

    if not evidence and confidence > 0.4:
        print("[GUARD] root cause confirmed with zero evidence files read — "
              "capping confidence so it can't proceed straight to a patch.")
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

    # If the model filled findings but left root_cause/solution empty,
    # synthesize them from its own findings instead of reporting "unknown".
    root_cause = (data.get("root_cause") or "").strip()
    if not root_cause and findings and findings[0].get("root_cause") not in ("", "unknown"):
        root_cause = "; ".join(f["root_cause"] for f in findings if f.get("root_cause"))[:400]
        print("[FINALIZE] root_cause empty — synthesized from the model's own findings.")
    solution = (data.get("solution") or "").strip()
    if not solution and findings:
        solution = "; ".join(f["solution"] for f in findings if f.get("solution"))[:400]

    all_issues_found = data.get("all_issues_found")
    if all_issues_found is False:
        print("[INVESTIGATE] model flagged it may NOT have found all issues — "
              "capping confidence.")
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
                   issue_block: str = "", suggested_files: list = None,
                   error_tokens: list = None) -> dict:
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
        print(f"[INVESTIGATE] evidence pre-loaded via deterministic retrieval "
              f"(filename/log-shape based, content not inspected): {preloaded}")

    log, result = [], None
    dead_ends = set()
    pending_notes = []
    stall_count = 0
    got_any_model_response = False
    timed_out_cold = False
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
                                             pending_notes, issue_block=issue_block,
                                             error_tokens=error_tokens)
        pending_notes = []
        try:
            raw = _stream_ollama(prompt, INVESTIGATE_SCHEMA, num_predict=turn_num_predict,
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
                    last_turn, None, issue_block=issue_block,
                    error_tokens=error_tokens)
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
            turn_num_predict = 1500  # give the retry more room to finish
            continue
        status = data.get("status", "")
        analysis = data.get("analysis", "")
        print(f"[INVESTIGATE] turn {turn}: status={status} | "
              f"confidence={data.get('confidence', '(omitted!)')} | "
              f"{analysis[:160] if analysis else '(no analysis)'}")

        if not status and turn > 1:
            print("[INVESTIGATE] empty status from model — treating as stall.")
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

            # Carry the model's own hypothesis into the next turn — otherwise
            # every turn restarts the diagnosis from scratch (observed: turn 2
            # re-derived turn 1's conclusion in different words).
            if analysis:
                pending_notes.append(
                    "Your previous-turn hypothesis (BUILD ON IT — do not "
                    "restart from scratch): " + analysis[:220])

            stalled = (not to_read and requested and not last_turn)
            stall_count = stall_count + 1 if stalled else 0
            # At ~190s/turn on this runner, ONE turn that yields no new
            # evidence is already unaffordable — force the decision now
            # instead of allowing a second stalled turn (was: >= 2).
            force_now = last_turn or stall_count >= 1

            if force_now:
                if stall_count >= 1 and not last_turn:
                    print("[INVESTIGATE] a turn yielded no new evidence — "
                          "forcing final decision now (turns are too "
                          "expensive to waste).")
                if _budget_exceeded():
                    result = _finalize_investigation(data, True, evidence)
                    break
                final_prompt = _build_investigation_prompt(
                    signal, exit_code, repo_tree, git_diff, evidence, True,
                    pending_notes, issue_block=issue_block,
                    error_tokens=error_tokens)
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
            failure_mode = "infra_timeout" if timed_out_cold else "no_model_response"
            root_cause = ("model did not return any usable response (infra/latency problem)")
        elif parse_failures:
            failure_mode = "unparseable_output"
            root_cause = ("model responded but its output was not valid JSON "
                          "even after truncation repair (likely cut off "
                          "mid-generation)")
        else:
            failure_mode = "not_converged"
            root_cause = "unknown — investigation did not converge"
        result = {"root_cause": root_cause, "solution": "", "confidence": 0.0,
                  "confidence_omitted": True,
                  "commit_message": "fix: auto-fixer change", "findings": [],
                  "failure_mode": failure_mode}
    else:
        result["failure_mode"] = None
    result["evidence"] = evidence
    result["investigation_log"] = log
    return result


# ── PATCH GENERATION ─────────────────────────────────────────────────────────
PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            # file → evidence → corrected first (the machine-actionable parts);
            # the human-readable "problem" prose comes last so a truncated
            # issue still carries a usable find/replace pair.
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

IMPORTANT: if multiple issues are listed under "## Confirmed root cause",
you MUST emit one "issues" entry per problem — a single file can have
several independent bugs (e.g. a bad version string AND one or more
mismatched filenames used in different commands). Before finishing, re-scan
the file contents you were given for any other reference of the same kind
as the ones already listed (other paths, other version strings, etc.) that
point to something which doesn't match what actually exists elsewhere in
the evidence — and fix those too, using the same file/evidence/corrected
format.

If the "## ISSUE TO SOLVE" section contains numbered FACT entries (a
repository-side audit of broken references), you MUST emit one "issues"
entry PER FACT — the audit already verified each one against the real
repository. Use the "exact line" shown with each FACT as your "evidence"
string, copied VERBATIM. A patch that fixes some FACTs but not all will
be REJECTED.

Output ONLY one JSON object. No markdown fences.

"issues" = ONE ENTRY PER BUG. Each entry:
  - "file": exact ### header path
  - "evidence": the ONE broken line, copied EXACTLY from the file contents.
  - "corrected": that SAME line with only the bug fixed. MUST differ from "evidence".
  - "problem": one sentence (last).

**"evidence" MUST BE EXACTLY ONE LINE — the single line that contains the
bug.** Never include surrounding lines like the step name, `uses:`, or
`with:`. A multi-line block where nothing actually changes will be REJECTED.
Example: if the broken line is `          python-version: "3.2"`, then
evidence is that one line and corrected is the same line with a valid value.

File contents may be shown as focused excerpts; omitted regions appear as
`<<<SKIPPED N UNRELATED LINES>>>`. NEVER copy a skip marker into "evidence"
or "corrected" — it is not file content.

CRITICAL: "evidence" must be copied VERBATIM from the "## File contents"
section below. NEVER use text from the error log, the root cause, or your
own paraphrase as "evidence" — if the exact characters are not in the file
contents shown, the patch will be rejected.

CONTAINER PATHS: runtime errors name paths as they exist INSIDE the
container (e.g. '/app/apsspp.py'). The file contents show the repo-side
line (e.g. `CMD ["python", "apsspp.py"]`). Your "evidence" must be the
repo-side line from "## File contents" — never a container path like
'/app/...' from the log.

DO NOT ESCAPE CHARACTERS. Write paths and text plainly:
  - file: `.github/workflows/ci-local-deploy.yml`  ✔
  - NOT `\\.github\\_workflows\\_ci-local-deploy.yml`  → WRONG
  - evidence: `python-version: "3.1"`  ✔   corrected: `python-version: "3.10"`  ✔
No backslashes before dots, slashes, underscores, or digits. A path uses
forward slashes only. "evidence" and "corrected" must be DIFFERENT — the
whole point is that "corrected" changes the broken part.

Schema:
{"issues":[{"file":"...","evidence":"...","corrected":"...","problem":"..."}]}

**REMEMBER**: "evidence" and "corrected" must be DIFFERENT strings. Emit
a separate entry for each distinct bug — do not merge multiple bugs into
one entry, and do not stop after the first one.
"""


def _select_patch_evidence(evidence: dict, mention_blob: str) -> dict:
    """Keep evidence files the AI's own diagnosis mentions; drop the rest if
    over budget. Selection is driven purely by the model's output text —
    Python adds no opinion about where the bug is."""
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
              f"omitted (not referenced by the diagnosis): {dropped}")
    return selected


def ai_generate_patch(root_cause: str, solution: str, evidence: dict,
                      retry_note: str = "", issue_block: str = "",
                      error_tokens: list = None) -> list:
    mention_blob = "\n".join([root_cause or "", solution or "", issue_block or ""])
    scoped = _select_patch_evidence(evidence, mention_blob)
    # Ship focused excerpts to the patch agent too — smaller prefill means the
    # ~170s patch rounds drop substantially, and the model sees the offending
    # line with a little context instead of a whole file to get lost in.
    scoped = excerpt_evidence(scoped, error_tokens or [], "PATCH")
    parts = [f"### {f}\n```\n{c}\n```" for f, c in scoped.items()]
    context = "\n\n".join(parts) if parts else "(no evidence files were read)"
    issue_section = (f"## ISSUE TO SOLVE (stay scoped to this):\n{issue_block}\n\n"
                     if issue_block else "")
    retry_section = (f"## Note: a previous attempt at this fix failed:\n"
                     f"{retry_note}\nDo not repeat the same change.\n\n" if retry_note else "")

    def _assemble(ctx):
        return (f"{PATCH_SYSTEM}\n\n{issue_section}{retry_section}"
                f"## Confirmed root cause: {root_cause}\n"
                f"## Solution direction: {solution}\n\n"
                f"## File contents (you may ONLY edit these):\n{ctx}\n\n"
                f"Emit the issues JSON.")

    prompt = _assemble(context)
    if len(prompt) > MAX_PROMPT_CHARS:
        # Truncate the EVIDENCE, never the instructions or the trailing
        # "Emit the issues JSON" line (the old blind prompt[:N] slice could
        # cut both, leaving the model without its output directive).
        overflow = len(prompt) - MAX_PROMPT_CHARS
        context = context[:max(0, len(context) - overflow)] + "\n...(evidence truncated)"
        prompt = _assemble(context)

    # Budget-aware timeout: never start a patch call the run can't afford.
    remaining = max(0, TOTAL_TIME_BUDGET - _elapsed())
    if remaining < 45:
        raise RuntimeError(f"only {remaining:.0f}s of budget left — not enough "
                           f"for a patch call")
    # Never let a single patch call consume the whole remaining budget — cap it
    # at ~55% of what's left so a failed/slow round still leaves room for one
    # retry. (The 330s-eats-everything-then-no-round-2 failure mode.)
    budget_cap = max(120, int(remaining * 0.55))
    timeout = min(PATCH_TIMEOUT, budget_cap, int(remaining) - 10)
    retries = MAX_RETRIES if remaining > 2 * timeout else 1
    print(f"[PATCH] timeout {timeout}s, retries {retries} "
          f"(budget remaining {remaining:.0f}s)")
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


def _unescape_model_string(s):
    """Small coder models frequently over-escape string values — treating a
    file path or YAML snippet like a regex/Windows path and emitting things
    like '\\.github\\_workflows\\_ci-local-deploy.yml' or '3\\.10'. This strips
    escapes that JSON/YAML/paths never need, so the value can actually match
    the real file content. Pure text cleanup — no diagnosis, no fix authoring."""
    if not isinstance(s, str):
        return s
    out = s
    # Drop backslashes before characters that are never escaped in a path,
    # YAML scalar, or plain snippet (., _, /, -, :, spaces, digits, letters,
    # quotes, brackets, commas). Quotes/brackets matter for lines like
    # CMD ["python", "app.py"] — a leftover \" must become ", NOT /".
    out = re.sub(r'\\(["\'\[\],._/\-: 0-9A-Za-z])', r'\1', out)
    # A backslash used as a path separator -> forward slash.
    out = out.replace("\\", "/")
    # Collapse accidental doubled separators introduced by the above.
    out = re.sub(r'/{2,}', '/', out)
    return out


def _normalize_file_field(file: str, evidence: dict) -> str:
    """Map a possibly-mangled file field onto a real evidence key."""
    if not file:
        return file
    cand = _unescape_model_string(file).strip().strip("`'\"")
    cand = _relstrip(cand)
    if cand in evidence:
        return cand
    # Match by basename against evidence keys (handles residual path munging).
    base = Path(cand).name.lower()
    for k in evidence:
        if Path(k).name.lower() == base:
            return k
    # Fuzzy last resort.
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
        # Deterministically un-mangle the string values the model over-escaped.
        for k in ("file", "evidence", "corrected"):
            if isinstance(it.get(k), str):
                it[k] = _unescape_model_string(it[k])
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


# ── PAIR + LOCATE / VALIDATE / APPLY ──────────────────────────────────────
def _closest_evidence_line(ev: str, evidence: dict) -> tuple:
    """When the model claims text exists that doesn't, find the closest line
    that ACTUALLY exists in the provided evidence (denial-to-evidence: correct
    a false claim with an observed fact, no diagnosis)."""
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
            # Score both the raw line and a comment-stripped variant —
            # trailing comments otherwise dilute the similarity ratio.
            bare = re.sub(r"\s+#.*$", "", s).strip()
            r = max(
                difflib.SequenceMatcher(None, needle, s.lower()).ratio(),
                difflib.SequenceMatcher(None, needle, bare.lower()).ratio()
                if bare else 0.0)
            if r > best[2]:
                best = (f, s[:200], r)
    return best if best[2] >= 0.4 else ("", "", 0.0)


def _reanchor_pair(ev: str, cor: str, evidence: dict):
    """VALIDATION-SIDE RESCUE (change stays 100% model-authored): the model
    named a change (evidence → corrected) but its 'evidence' text does not
    exist verbatim in any provided file — typically because it copied the
    CONTAINER path from the error log ('/app/apsspp.py') or paraphrased the
    line. Extract the model's OWN changed fragment (diff of its evidence vs
    its corrected) and re-anchor that exact change onto the closest line
    that ACTUALLY exists. Python decides nothing about WHAT changes — only
    WHERE the model's change lands. Same principle as _salvage_fragment,
    applied at pairing time so a fixable patch doesn't cost a full retry
    round. Returns (file, real_line, new_line) or None."""
    if not isinstance(ev, str) or not isinstance(cor, str) or ev == cor:
        return None
    hint_file, hint_line, ratio = _closest_evidence_line(ev, evidence)
    if not hint_file or ratio < 0.55:
        return None
    # The model's intended change: strip the common prefix/suffix.
    i = 0
    while i < len(ev) and i < len(cor) and ev[i] == cor[i]:
        i += 1
    j = 0
    while (j < len(ev) - i and j < len(cor) - i
           and ev[-1 - j] == cor[-1 - j]):
        j += 1
    old_frag, new_frag = ev[i:len(ev) - j], cor[i:len(cor) - j]
    if len(old_frag.strip()) < 2:
        return None
    # Anchor the fragment in the REAL line (must be unambiguous).
    if hint_line.count(old_frag) != 1:
        of, nf = old_frag.strip(), new_frag.strip()
        if of and hint_line.count(of) == 1:
            old_frag, new_frag = of, nf
        else:
            return None
    new_line = hint_line.replace(old_frag, new_frag, 1)
    if new_line == hint_line or not new_line.strip():
        return None
    # The real line must be unambiguous in its file too.
    content = evidence.get(hint_file)
    if not isinstance(content, str) or content.count(hint_line) != 1:
        return None
    return hint_file, hint_line, new_line


def build_issues_from_findings(findings: list, evidence: dict) -> list:
    """VALIDATION-SIDE PAIRING (change stays 100% model-authored): when the
    investigation/confirm step already produced findings whose `solution` is a
    concrete replacement LINE (e.g. 'CMD [\"python\", \"app.py\"]'), pair each
    with the closest existing line in the evidence and emit a ready
    issue-dict — so a successful, high-confidence confirm is turned straight
    into a patch WITHOUT a second, slow, failure-prone patch-generation model
    call. Python only LOCATES the line the model's solution replaces (same
    principle as the re-anchor rescue and _salvage_fragment); it never authors
    the fix. Findings whose solution isn't a locatable single line are skipped
    and left for the model patch call to handle. Returns a list of issue-dicts
    (possibly empty)."""
    issues = []
    for f in findings or []:
        if not isinstance(f, dict):
            continue
        sol = (f.get("solution") or "").strip()
        # Must be a single concrete line, not prose or a multi-line block.
        if not sol or "\n" in sol or len(sol) > 300:
            continue
        # Reject obviously-prose solutions ("change X to Y", "update the ...").
        low = sol.lower()
        if low.split(" ", 1)[0] in ("change", "update", "replace", "set", "add",
                                     "remove", "fix", "modify", "use", "ensure",
                                     "make", "correct", "rename"):
            continue
        hint_file, hint_line, ratio = _closest_evidence_line(sol, evidence)
        if not hint_file or ratio < 0.6:
            continue
        if hint_line.strip() == sol.strip():
            continue  # solution already present — nothing to change
        issues.append({
            "file": hint_file,
            "evidence": hint_line,
            "corrected": sol,
            "problem": (f.get("issue") or f.get("root_cause") or "").strip(),
        })
    if issues:
        print(f"[PATCH] derived {len(issues)} concrete fix(es) directly from the "
              f"confirmed findings (model authored each 'solution'; Python only "
              f"located the verbatim line it replaces).")
    return issues


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
        if SKIP_MARKER_RE.search(ev) or SKIP_MARKER_RE.search(cor):
            rejects.append(
                f"issue #{n} ({file or '?'}): you copied a "
                f"'<<<SKIPPED ...>>>' marker — that marker is NOT file "
                f"content. Copy only real lines from the file.")
            continue
        if cor.strip() == ev.strip():
            rejects.append(
                f"issue #{n} ({file or '?'}): 'evidence' and 'corrected' are "
                f"IDENTICAL (both `{ev.strip()[:80]}`). 'evidence' must be the "
                f"ONE broken line only — not the step name / uses: / with: "
                f"lines around it — and 'corrected' is that SAME line with the "
                f"bug fixed, e.g. evidence `python-version: \"3.1\"` → "
                f"corrected `python-version: \"3.10\"`.")
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
            re_anchored = _reanchor_pair(ev, cor, evidence)
            if re_anchored:
                target, real_line, new_line = re_anchored
                print(f"[LOCATE] issue #{n}: 'evidence' didn't exist verbatim "
                      f"(likely copied from the error log) — re-anchored the "
                      f"model's own change onto the real line in '{target}': "
                      f"`{real_line[:90]}` → `{new_line[:90]}`")
                ev, cor = real_line, new_line
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


def _is_prose_doc(file: str) -> bool:
    """True if `file` is human documentation prose (README, guides, changelog)
    where filenames are MENTIONED in sentences/tables, not REFERENCED as build
    inputs. Such files can never cause a CI build/runtime failure — nothing
    executes them — so scanning them for 'broken references' produces only
    false positives. Observation-only classification by extension/name."""
    p = Path(file)
    if p.suffix.lower() in (".md", ".markdown", ".rst", ".adoc", ".mdx"):
        return True
    name = p.name.lower()
    # prose .txt (README.txt, NOTES.txt) — but NOT dependency manifests like
    # requirements.txt / constraints.txt, whose names ARE real reference targets.
    if p.suffix.lower() == ".txt" and not name.endswith("requirements.txt") \
            and name not in ("constraints.txt",):
        prose_names = ("readme", "notes", "changelog", "changes", "history",
                       "authors", "contributors", "license", "copying", "todo")
        if any(name.startswith(x) for x in prose_names):
            return True
    return False


def _missing_ref_map(content: str, file: str) -> dict:
    """Map of token -> problem message for repo-file references that don't
    exist. Pure observation; blocking decisions happen in _compare_ref_problems."""
    problems = {}
    if not content or _is_prose_doc(file):
        # Documentation prose mentions filenames constantly; those mentions are
        # not references that can break a build. Never audit them.
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


def _compare_ref_problems(old_map: dict, new_map: dict, label: str) -> str:
    """Error-driven verdict on a patched file's reference problems:
      - problems INTRODUCED by the patch  -> always reject (regression)
      - pre-existing problems the FAILURE EVIDENCE mentions -> reject
        (the patch was supposed to fix exactly this)
      - pre-existing problems unrelated to the error -> note only; the next
        CI run is the judge of whether they matter."""
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


def _dockerfile_problem_map(file: str, content: str) -> dict:
    """Map of key -> problem message for Dockerfile references/tags. Pure
    observation; blocking decisions happen in _compare_ref_problems."""
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
    """VALIDATION ONLY: return the list of python-version values in a workflow
    that are NOT real CPython releases. Python observes and reports; it never
    rewrites the value — the model must author a valid version itself."""
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
            # VALIDATION ONLY — Python does not author the version. If the
            # model's patched workflow still carries a python-version that
            # isn't a real CPython release, reject it so the MODEL picks a
            # valid one on retry. (Previously this silently forced the value
            # to DEFAULT_PYTHON_VERSION, which was Python authoring the fix.)
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
    """Branch whose CI run failed. Checked in priority order — all generic."""
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


def commit_to_existing_branch(commit_msg: str, written: list, branch: str) -> str:
    """Chain mode: push a follow-up fix commit to the bot's own open fix branch
    so its existing PR accumulates fixes until CI is green."""
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


def comment_on_bot_pr(token, repo, branch, body) -> bool:
    """Post the follow-up fix summary on the PR whose head is `branch`."""
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

    # loop guards / chain mode
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
    log_text = _strip_log_timestamps(log_text)  # clean AI-facing log lines
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

    # If the PRIMARY error is itself garbled (e.g. a corrupted version tag like
    # '\x00\x003.12' from an empty --build-arg), the log is unreadable at the
    # decisive point. A confident diagnosis on that premise is a hallucination
    # waiting to happen — escalate for a human instead of guessing.
    primary = focused.get("primary_message", "")
    if log_corrupted and looks_garbled(primary):
        print("[GATE] the primary error line is garbled/corrupted — the log is "
              "unreadable at the point that matters. Refusing to diagnose on a "
              "corrupted signal; escalating.")
        msg = ("The CI log is corrupted at the failing line — the primary error "
               "reads as garbled/non-printable text (e.g. a Python version tag "
               "like `\\x00\\x003.12`). This usually means a build argument or "
               "variable resolved to an EMPTY value (for example "
               "`--build-arg PYTHON_VERSION=` with no value, giving `FROM "
               "python:` with no tag). Check that every variable the failing "
               "step references is actually set — a missing workflow step that "
               "was supposed to define it is the most common cause. I can't "
               "safely diagnose from an unreadable error signal.")
        print(f"       {msg}")
        if token and repo:
            open_issue(token, repo,
                       msg + f"\n\n**Garbled signal (first 300 chars):**\n"
                       + f"```\n{signal[:300]}\n```", run_url)
        sys.exit(0)

    # Validation scoping: the error evidence itself (never model output)
    # decides which pre-existing problems a patch MUST fix.
    global CURRENT_FAILURE_CONTEXT
    CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    supporting_signal = trim_supporting_signal(signal, focused)
    print(f"[EVIDENCE] Focused issue: {(focused.get('primary_message') or '(none)')[:160]}")
    if focused.get("failing_step"):
        print(f"[EVIDENCE] Failing step: {focused['failing_step']}")
    if focused.get("command_context"):
        print(f"[EVIDENCE] Command context: {focused['command_context']}")

    git_diff  = get_git_diff()
    repo_tree, allowed_files = repo_tree_text()
    print(f"[EVIDENCE] exit_code={exit_code} | {len(allowed_files)} readable file(s) in tree")

    # Shared mechanical token extraction from the error — drives both the
    # offending-line locator and the focused-excerpt windows.
    error_tokens = extract_error_tokens(focused, signal)
    if error_tokens:
        print(f"[EVIDENCE] Error tokens for context windowing: "
              f"{sorted(error_tokens)[:8]}")

    # ── PATH-EXISTENCE ENRICHMENT (observation, not diagnosis) ──
    path_facts = enrich_issue_with_path_checks(focused, signal, allowed_files)
    if path_facts:
        issue_block = issue_block + "\n\n" + path_facts
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    # ── PYTHON-VERSION COHERENCE (observation, not diagnosis) ──
    # If the runner Python, the image Python, and/or .python-version disagree,
    # surface the drift so the model aligns them to the source of truth.
    coherence = check_python_version_coherence(allowed_files)
    if coherence:
        issue_block = issue_block + "\n\n" + coherence
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    # ── VERSION-FILE / UNDEFINED-VERSION GATE (observation, not diagnosis) ──
    # Two escalate-don't-guess cases the edit-only patch engine can't handle:
    #  (a) the workflow points at a version file that DOESN'T EXIST — the fix is
    #      to CREATE that file (or stop referencing it), not edit anything; or
    #  (b) the failure is a version error and NO version is defined anywhere.
    # In both, sending it to the model just makes it flail on the workflow and
    # produce "no effective change" (exactly what looped here). Escalate first.
    if is_python_version_failure(issue_block, signal):
        missing_vf = missing_version_file(issue_block, signal, allowed_files)
        if missing_vf or python_version_is_undefined(allowed_files):
            gate_msg = undefined_python_version_message(issue_block, signal)
            reason = (f"the workflow references version file '{missing_vf}' which "
                      f"does not exist" if missing_vf
                      else "the repo defines NO usable Python version")
            print(f"[GATE] Python-version failure — {reason}. The fix requires "
                  f"creating/removing a file, which this tool can't author. "
                  f"Refusing to guess; escalating.")
            print(f"       {gate_msg}")
            if token and repo:
                open_issue(token, repo,
                           gate_msg
                           + f"\n\n**Failure Python identified:**\n"
                           + f"```\n{issue_block[:1200]}\n```",
                           run_url)
            sys.exit(0)

    # ── DETERMINISTIC CONTEXT RETRIEVAL (not diagnosis) ──
    context_map = gather_deterministic_context(log_text, allowed_files)
    suggested_files = flatten_suggested_files(context_map)
    if context_map:
        print(f"[EVIDENCE] Deterministic retrieval matched categories: "
              f"{list(context_map.keys())}")
        print(f"[EVIDENCE] Files pre-loaded as evidence (not a diagnosis): {suggested_files}")
    else:
        print("[EVIDENCE] No failure-category signature matched — relying on "
              "token-driven retrieval below.")

    # ── TOKEN-DRIVEN RETRIEVAL FALLBACK (category-agnostic, log-driven) ──
    # Whatever the error names, find the files that name it back. This keeps
    # the audit/pinning/fast-confirm pipeline working for ANY failure shape,
    # not just the categories the signatures know about.
    token_files = retrieve_files_by_error_tokens(error_tokens, allowed_files)
    for f in token_files:
        if f not in suggested_files and len(suggested_files) < MAX_PRELOADED_FILES:
            suggested_files.append(f)
    if not suggested_files:
        print("[EVIDENCE] No retrieval match at all — AI starts from log "
              "evidence only and requests files itself.")

    # ── OFFENDING-LINE LOCATION (observation, not diagnosis) ──
    # Read the deterministically-retrieved files and pin the exact line(s)
    # that contain the value(s) the error names, so the model has the precise
    # 'evidence' string to copy instead of grabbing the wrong lines.
    preloaded_contents = {}
    for f in suggested_files:
        c = _read_evidence_file(f)
        if c is not None:
            preloaded_contents[f] = c
    offending = locate_offending_lines(focused, signal, preloaded_contents)
    if offending:
        issue_block = issue_block + "\n\n" + offending
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    # ── PRE-FLIGHT REFERENCE AUDIT (observation, not diagnosis) ──
    # The log names only the FIRST failure; the same files may carry MORE
    # broken references (multi-bug files). Audit the evidence files with the
    # validator's own observation maps and ship every broken reference as a
    # numbered FACT with its exact verbatim line. The audit tokens also widen
    # the excerpt windows so every buggy line survives excerpting.
    audit_text, audit_tokens, audit_facts = preflight_reference_audit(
        preloaded_contents, allowed_files)
    if audit_text:
        issue_block = issue_block + "\n\n" + audit_text
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal
        for t in audit_tokens:
            if t not in error_tokens:
                error_tokens.append(t)

    # ── CONTAINER-PATH → REPO-PATH BRIDGE (observation, not diagnosis) ──
    # Runtime errors from inside a container name container paths
    # ('/app/apsspp.py'); the editable file shows the repo-side spelling.
    # Map the two so the model copies the real Dockerfile line as evidence.
    bridge = map_container_paths_to_repo(focused, signal, preloaded_contents)
    if bridge:
        issue_block = issue_block + "\n\n" + bridge
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal

    # ── AI INVESTIGATION AGENT ──
    print("\n━━━ AI INVESTIGATION AGENT ━━━")
    warm_up_model()

    investigation = None
    if FAST_CONFIRM and (offending or audit_facts) and preloaded_contents:
        # Deterministic tooling pinned the exact offending line(s) and/or the
        # reference audit produced verified numbered facts: ship a
        # confirm-or-refute brief (~half the prompt, one turn) instead of the
        # open-ended multi-turn investigation. Falls back below on refusal,
        # low confidence, or any failure — the full loop is unchanged.
        print("[CONFIRM] pinned lines / audited facts available — trying "
              "single-turn fast-path confirmation before full investigation.")
        investigation = ai_confirm_pinned(issue_block, exit_code,
                                          preloaded_contents, error_tokens)
    if investigation is None:
        investigation = ai_investigate(supporting_signal, exit_code, repo_tree, git_diff,
                                       allowed_files, issue_block=issue_block,
                                       suggested_files=suggested_files,
                                       error_tokens=error_tokens)
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
          + ("  (⚠ DEFAULTED — model did not report one)" if confidence_omitted else ""))
    print(f"  files read    : {', '.join(evidence.keys()) or '(none)'}")
    print(f"  issues found  : {len(findings)}")
    for i, fnd in enumerate(findings, 1):
        print(f"    {i}. {fnd['issue']}")
        if fnd.get("root_cause"):
            print(f"       cause: {fnd['root_cause']}")
        if fnd.get("solution"):
            print(f"       fix:   {fnd['solution']}")

    failure_mode = investigation.get("failure_mode")
    if failure_mode in ("infra_timeout", "no_model_response", "unparseable_output"):
        print(f"[GATE] Investigation produced no usable model output ({failure_mode}) — escalating.")
        if token and repo:
            mode_detail = {
                "infra_timeout":      "The investigation call(s) to Ollama timed out "
                                      "before the model produced any output.",
                "no_model_response":  "The model returned nothing usable.",
                "unparseable_output": "The model responded, but its output was not "
                                      "valid JSON even after truncation repair — "
                                      "likely cut off mid-generation. Consider "
                                      "raising the token cap or using a stronger model.",
            }.get(failure_mode, "")
            open_issue(token, repo,
                       f"Auto-fixer could not get a usable diagnosis "
                       f"({failure_mode}). {mode_detail}\n\n"
                       f"**Issue Python identified (unused — no usable diagnosis):**\n"
                       f"```\n{issue_block[:1200]}\n```",
                       run_url)
        sys.exit(0)

    # ── RE-AUDIT OVER EVERYTHING THE INVESTIGATION READ ──
    # Files the model requested mid-investigation weren't covered by the
    # pre-flight audit (which only saw the preloaded set). Re-run it over the
    # full evidence so the patch stage's completeness check knows about EVERY
    # broken reference, wherever it lives.
    full_audit_text, full_audit_tokens, full_audit_facts = \
        preflight_reference_audit(evidence, allowed_files)
    if len(full_audit_facts) > len(audit_facts):
        print(f"[EVIDENCE] re-audit over full evidence found "
              f"{len(full_audit_facts) - len(audit_facts)} additional broken "
              f"reference(s) in files read mid-investigation.")
        if audit_text and audit_text in issue_block:
            issue_block = issue_block.replace(audit_text, full_audit_text)
        else:
            issue_block = issue_block + "\n\n" + full_audit_text
        CURRENT_FAILURE_CONTEXT = issue_block + "\n" + signal
        audit_text, audit_facts = full_audit_text, full_audit_facts
        for t in full_audit_tokens:
            if t not in error_tokens:
                error_tokens.append(t)

    # ── CORROBORATION RESCUE (observation, not diagnosis) ──
    # Fires ONLY when confidence was DEFAULTED because the model failed to
    # report one (a reporting failure — e.g. deadline-truncated JSON), AND the
    # diagnosis cites the very value(s) the deterministic locator pinned in
    # the real files. The observed facts corroborate the claim, so floor the
    # confidence at 0.55 instead of escalating a correct answer over a
    # reporting bug. A model that genuinely SAID a low confidence (0.3 etc.)
    # is still gated normally — Python never overrides the model's judgment,
    # only a defaulted placeholder.
    if confidence < 0.5 and confidence_omitted and error_tokens:
        _diag_blob = (root_cause + " " + (solution or "") + " " + " ".join(
            (f.get("root_cause") or "") + " " + (f.get("solution") or "")
            for f in findings)).lower()
        _cited = sorted({t for t in error_tokens if t.lower() in _diag_blob})
        if _cited:
            print(f"[GATE] confidence was defaulted (model reporting failure), "
                  f"but the diagnosis cites pinned value(s) "
                  f"{', '.join(_cited[:4])} that Python located in the real "
                  f"files — corroborated; flooring confidence at 0.55.")
            confidence = 0.55

    if confidence < 0.5:
        gate_note = (" NOTE: the model never reported a confidence value — this is a "
                     "reporting failure, not necessarily a bad diagnosis."
                     if confidence_omitted else "")
        print(f"[GATE] Confidence {confidence:.0%} too low — escalating.{gate_note}")
        if token and repo:
            escalation_detail = (
                f"AI confidence too low ({confidence:.0%}).{gate_note} "
                f"Root cause: {root_cause}\n\n"
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
    rejection_history = []  # accumulated across rounds so the model never
                            # regresses on an already-reported problem
    rejected_fingerprints = set()  # byte-level identity of rejected issue-sets
    last_patch_duration = 0.0

    def _issues_fingerprint(iss):
        return repr(sorted(
            ((it.get("file") or ""), (it.get("evidence") or ""),
             (it.get("corrected") or ""))
            for it in iss if isinstance(it, dict)))

    # ── DIRECT PATCH FROM CONFIRMED FINDINGS (skip the 2nd model call) ──
    # If the confirm/investigation step already produced concrete solution
    # lines that Python can pair with verbatim source lines, AND together they
    # cover every audited FACT, use them directly on round 1 — no separate,
    # slow, timeout-prone patch-generation model call. Falls back to the model
    # loop automatically if these don't validate.
    prebuilt_issues = build_issues_from_findings(findings, evidence)
    if prebuilt_issues and unaddressed_audit_facts(prebuilt_issues, audit_facts):
        # they don't cover all audited bugs — let the model do the full job
        print("[PATCH] findings-derived fixes don't cover every audited FACT — "
              "using the model patch call instead.")
        prebuilt_issues = None
    elif prebuilt_issues:
        print("[PATCH] findings-derived fixes cover every audited FACT — "
              "applying them directly (no patch-generation model call needed).")

    for repair_round in range(1, MAX_REPAIR_ROUNDS + 1):
        if _budget_exceeded():
            print(f"[BUDGET] time budget exceeded before repair round {repair_round}.")
            if token and repo:
                open_issue(token, repo,
                           f"Auto-fixer hit its time budget ({TOTAL_TIME_BUDGET}s) before "
                           f"finishing. Root cause so far: {root_cause}", run_url)
            sys.exit(5)

        print(f"\n━━━ AI PATCH GENERATION AGENT (round {repair_round}/{MAX_REPAIR_ROUNDS}) ━━━")
        remaining = TOTAL_TIME_BUDGET - _elapsed()
        if repair_round > 1 and remaining < max(60, last_patch_duration * 1.2 + 20):
            print(f"[BUDGET] {remaining:.0f}s left but the last patch round took "
                  f"{last_patch_duration:.0f}s — another round can't fit.")
            if token and repo:
                open_issue(token, repo,
                           f"Auto-fixer ran out of budget mid-repair "
                           f"({remaining:.0f}s left, rounds take ~{last_patch_duration:.0f}s). "
                           f"Root cause: {root_cause}\n\nRejections so far:\n"
                           + "\n".join(f"- {r}" for r in rejection_history), run_url)
            sys.exit(5)
        active_issue_block = retry_issue_block or issue_block
        _patch_t0 = time.time()
        if repair_round == 1 and prebuilt_issues:
            issues = prebuilt_issues
            print("  using fixes derived from the confirmed findings "
                  "(no patch-generation model call).")
        else:
            try:
                issues = ai_generate_patch(patch_root_cause, solution, evidence,
                                           retry_note, active_issue_block,
                                           error_tokens=error_tokens)
            except Exception as exc:
                last_patch_duration = time.time() - _patch_t0
                print(f"[ERROR] patch generation failed: {exc}", file=sys.stderr)
                if repair_round == MAX_REPAIR_ROUNDS:
                    if token and repo:
                        open_issue(token, repo, f"AI patch generation failed: {exc}", run_url)
                    sys.exit(2)
                continue
        last_patch_duration = time.time() - _patch_t0

        print(f"  issues reported: {len(issues)}")
        for n, it in enumerate(issues, 1):
            print(f"    {n}. {it.get('file','?')}: {it.get('problem','')[:80]}")

        fp = _issues_fingerprint(issues)
        if fp in rejected_fingerprints:
            print("[REPAIR] model repeated a previously rejected patch "
                  "verbatim — no progress possible, escalating.",
                  file=sys.stderr)
            if token and repo:
                open_issue(token, repo,
                           f"Auto-fixer stalled: the model kept producing the "
                           f"same rejected patch. Root cause: {root_cause}\n\n"
                           f"Rejections:\n"
                           + "\n".join(f"- {r}" for r in rejection_history),
                           run_url)
            sys.exit(3)
        rejected_fingerprints.add(fp)

        # ── COMPLETENESS CHECK vs the audited facts (validation only) ──
        # A partial patch would be rejected by validation anyway (the leftover
        # broken refs are in the failure context) — catching it HERE saves the
        # validation/test time and produces a retry note that names exactly
        # which FACTs were skipped, with their verbatim lines to copy.
        missing_facts = unaddressed_audit_facts(issues, audit_facts)
        if missing_facts:
            detail = (f"patch addresses {len(audit_facts) - len(missing_facts)}"
                      f"/{len(audit_facts)} audited bug(s) — unaddressed: "
                      + "; ".join(missing_facts))
            print(f"[REPAIR] {detail}", file=sys.stderr)
            if repair_round < MAX_REPAIR_ROUNDS:
                rejection_history.append(f"round {repair_round}: {detail}")
                retry_note = (
                    "Your previous patch FIXED SOME BUGS BUT NOT ALL of the "
                    "numbered FACTs. These audited broken references were left "
                    "untouched — emit one issues entry for EACH, using the "
                    "exact line shown as your 'evidence' string:\n"
                    + "\n".join(f"- {m}" for m in missing_facts)
                    + "\nKeep the correct entries you already produced AND add "
                      "the missing ones — the full set must cover every "
                      "numbered FACT in one response.")
                continue
            print("[REPAIR] final round — attempting the partial patch anyway "
                  "and letting validation judge it.", file=sys.stderr)

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
            rejection_history.append(f"round {repair_round}: {detail}")
            retry_note = ("Your previous patch attempts were rejected before they "
                          "could even be applied, for these reasons (fix ALL of "
                          "them):\n"
                          + "\n".join(f"- {r}" for r in rejection_history)
                          + "\nMake sure 'evidence' is copied EXACTLY from the file "
                            "contents shown — ONE line only, the broken line — and "
                            "'corrected' is that same line with the bug fixed.")
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
            rejection_history.append(f"round {repair_round}: {detail}")
            retry_note = ("Your previous patches were rejected by validation for "
                          "these reasons — your next patch must resolve EVERY "
                          "problem listed across ALL rounds, in one set of "
                          "issues:\n"
                          + "\n".join(f"- {r}" for r in rejection_history))
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
        CURRENT_FAILURE_CONTEXT += "\n" + retry_issue_block + "\n" + new_signal
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
    if chain_mode:
        branch = commit_to_existing_branch(commit_msg, written, failed_branch)
        if not branch:
            sys.exit(4)
        if token and repo:
            findings_md = "".join(
                f"\n{i+1}. **{f['issue']}** — {f.get('root_cause','')}"
                for i, f in enumerate(findings))
            comment_on_bot_pr(
                token, repo, branch,
                f"## 🤖 Follow-up auto-fix\n\n"
                f"CI on this branch failed with a new error; fixed it in the "
                f"latest commit.\n\n**Root cause:** {root_cause}\n"
                f"**Files changed:** {', '.join(f'`{f}`' for f in written)}\n"
                f"{('**Issues:**' + findings_md) if findings_md else ''}")
    else:
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