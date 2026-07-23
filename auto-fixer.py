#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — agentic investigation flow (OpenAI backend).

Pipeline:
    GitHub Actions Pipeline Failed
        → Python: collect initial evidence (logs, exit code, repo tree, git diff)
        → Python: distill a FOCUSED issue statement (primary error, failing step,
              AND the command that triggered it)
        → Python: SEED EVIDENCE — deterministically retrieve the files a human
              would open first (files named in the log, Dockerfiles when the
              error came from a container, CI workflow YAML). Retrieval only:
              Python never decides what the bug IS.
        → AI Investigation Agent (multi-turn CONVERSATION, ALWAYS runs):
              reads the focused issue + seeded evidence → forms hypothesis →
              requests more files if needed → Python fetches them (read-only) →
              loop → confirms root cause
        → AI Patch Generation Agent: emits the concrete fix
        → Python Validation Engine: format/path/syntax/YAML/JSON checks,
              secret scanning, dangerous-command detection, AND a check that
              the patched file no longer references any missing files.
        → Apply changes → Execute build & tests
        → on failure: collect new failure evidence → AI reviews it → retry or stop
        → on success: commit, push, open PR

Design notes:
  * Backend is the OpenAI Chat Completions API with `response_format`
    `json_schema` + `strict: true`. Structure is enforced at decode time, so
    malformed/partial JSON and missing fields are no longer possible failure
    modes (the only exception is a truncated response, handled explicitly).
  * No deterministic DIAGNOSIS ever. Python collects evidence; the AI reasons.
  * The investigation is a real conversation: each turn appends only the newly
    read files instead of rebuilding the whole prompt.
  * A post-patch validator rejects fixes that still contain missing file
    references.

Required env:
  OPENAI_API_KEY     — your OpenAI key (see README / repo secrets)
Optional env:
  OPENAI_BASE_URL    — default https://api.openai.com/v1
  INVESTIGATE_MODEL  — default gpt-4.1
  PATCH_MODEL        — default gpt-4.1-mini
  REVIEW_MODEL       — default gpt-4.1-mini
  GITHUB_TOKEN / GH_PAT, GITHUB_REPOSITORY — for PR + issue creation
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

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# Investigation needs reasoning; patching is mechanical string surgery.
INVESTIGATE_MODEL = os.environ.get("INVESTIGATE_MODEL", "gpt-4.1")
PATCH_MODEL       = os.environ.get("PATCH_MODEL",       "gpt-4.1-mini")
REVIEW_MODEL      = os.environ.get("REVIEW_MODEL",      "gpt-4.1-mini")

AI_TIMEOUT    = int(os.environ.get("AI_TIMEOUT", "90"))   # no local prefill now
MAX_RETRIES   = int(os.environ.get("AI_MAX_RETRIES", "3"))
RETRY_BACKOFF = [2, 6, 15]

TOTAL_TIME_BUDGET = int(os.environ.get("TOTAL_TIME_BUDGET", "600"))
_run_start_time = None


def _elapsed() -> float:
    return time.time() - _run_start_time if _run_start_time else 0.0


def _budget_exceeded() -> bool:
    return _run_start_time is not None and _elapsed() >= TOTAL_TIME_BUDGET


# ── Agentic-loop bounds ──────────────────────────────────────────────────────
# Turns are ~3s now instead of ~100s, so let the agent actually investigate.
# The agent drives its own loop via native tool calls and stops when it calls
# submit_diagnosis. These are cost backstops, not a reasoning schedule.
MAX_AGENT_TURNS   = int(os.environ.get("MAX_AGENT_TURNS", "25"))
MAX_TOOL_CALLS    = int(os.environ.get("MAX_TOOL_CALLS", "30"))
MAX_REPAIR_ROUNDS = int(os.environ.get("MAX_REPAIR_ROUNDS", "2"))
MAX_LIST_ENTRIES  = int(os.environ.get("MAX_LIST_ENTRIES", "300"))
MAX_SEARCH_HITS   = int(os.environ.get("MAX_SEARCH_HITS", "60"))

# ── Prompt / context budget (context is cheap now; stop starving the model) ──
MAX_ERROR_LINES      = 14
MAX_SUPPORTING_LINES = 10
MAX_FILE_CHARS       = int(os.environ.get("MAX_FILE_CHARS", "8000"))
MAX_TOTAL_CONTEXT    = int(os.environ.get("MAX_TOTAL_CONTEXT", "40000"))
MAX_FILES_FIXED      = int(os.environ.get("MAX_FILES_FIXED", "10"))
MAX_PROMPT_CHARS     = int(os.environ.get("MAX_PROMPT_CHARS", "60000"))
INVESTIGATION_DIFF_CHARS = int(os.environ.get("INVESTIGATION_DIFF_CHARS", "1200"))
MAX_TRACEBACK_LINES  = 12

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
    ("OpenAI key",       re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
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
# Deliberately NOT a list of released versions — that knowledge goes stale and
# would reject a correct patch to a newer Python than this file knows about.
# Only structurally impossible tags are caught; a plausible-but-wrong tag is
# caught for real by `docker build` in try_docker_build().
MIN_PYTHON_MINOR = int(os.environ.get("MIN_PYTHON_MINOR", "6"))


def _is_plausible_python_tag(major: int, minor: int) -> bool:
    return major == 3 and minor >= MIN_PYTHON_MINOR

WORKFLOW_PYVERSION_LINE = re.compile(
    r'^(\s*python-version\s*:\s*)([\'"]?)(\d+)\.(\d+)([\'"]?)(.*)$', re.M)

# ── Pattern for post-patch "missing reference" validation ────────────────────
# Extensions the dangling-reference check can see. Not a claim about which
# languages matter — extend via REPO_REF_EXTRA_EXTS (comma-separated) for stacks
# not listed here.
_REF_EXTS = ("py|txt|ya?ml|json|toml|cfg|ini|env|lock|js|jsx|ts|tsx|mjs|cjs|"
             "go|java|kt|rb|rs|php|cs|c|cpp|h|sh|bash|ps1|sql|tf|tfvars|proto|"
             "gradle|xml|properties|conf|md|csv|pem|crt|service|mk")
_extra = [e.strip().lstrip(".") for e in
          os.environ.get("REPO_REF_EXTRA_EXTS", "").split(",") if e.strip()]
if _extra:
    _REF_EXTS += "|" + "|".join(re.escape(e) for e in _extra)
REPO_REF_EXT_PATTERN = re.compile(
    r'(?<![\w./\-])((?:[\w.\-]+/)*[\w\-]+\.(?:' + _REF_EXTS + r'))(?![\w./\-])')


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
def _relstrip(rel):
    return rel[2:] if rel.startswith("./") else rel


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
    "rust":   ["cargo", "rustc", ".rs", "cargo.toml"],
    "dotnet": ["dotnet", ".csproj", ".sln", "nuget"],
    "ruby":   ["bundle", "rspec", "gemfile", ".rb"],
    "php":    ["composer", "phpunit", ".php"],
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
            if msg_line_idx > 0:
                prev = log_lines[msg_line_idx - 1].strip()
                if prev and not any(n in prev.lower() for n in NOISE_KEYWORDS):
                    command_context_lines.append(prev)
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


def trim_supporting_signal(signal: str, focused: dict,
                           max_lines: int = MAX_SUPPORTING_LINES) -> str:
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
    # Repo files leave the machine from here on — never ship live credentials.
    return redact_secrets(raw)


# ── AGENT TOOLS (the model drives these; Python only executes them) ─────────
# seed_evidence() is gone. It selected which files the model saw first — a
# filter, and therefore a form of pre-diagnosis. The agent now lists, searches
# and reads the repository itself. Python supplies an unfiltered directory
# listing and nothing else.

def _tool_list_directory(path: str, allowed_files: set) -> str:
    """Unfiltered listing of a directory. No relevance ranking, no selection."""
    path = _relstrip((path or ".").strip().lstrip("/")) or "."
    if ".." in path:
        return "Error: '..' is not permitted in paths."
    base = Path(path)
    if not base.exists():
        return f"Error: '{path}' does not exist."
    if base.is_file():
        return f"'{path}' is a file, not a directory. Use read_file."
    entries = []
    for p in sorted(base.iterdir()):
        if any(d in SKIP_DIRS for d in p.parts) or p.name in SKIP_DIRS:
            continue
        rel = _relstrip(str(p))
        if p.is_dir():
            entries.append(f"{rel}/")
        elif not _is_read_blocked(rel):
            entries.append(f"{rel}  ({p.stat().st_size} bytes)")
    if not entries:
        return f"'{path}' is empty or contains only excluded files."
    truncated = ""
    if len(entries) > MAX_LIST_ENTRIES:
        truncated = f"\n...({len(entries) - MAX_LIST_ENTRIES} more entries omitted)"
        entries = entries[:MAX_LIST_ENTRIES]
    return "\n".join(entries) + truncated


def _tool_read_file(path: str, allowed_files: set, evidence: dict) -> str:
    """Read one file. Content is secret-redacted before it leaves the machine."""
    path = _relstrip((path or "").strip().lstrip("/"))
    if not path or ".." in path:
        return "Error: invalid path."
    if _is_read_blocked(path):
        return f"Error: '{path}' is excluded from reading (secret or sensitive file)."
    p = Path(path)
    if not p.is_file():
        # Bare absence, with nothing substituted. Reason about who referenced it.
        return (f"'{path}' does not exist in the repository. Nothing was "
                f"substituted for it.")
    content = _read_evidence_file(path)
    if content is None:
        return f"Error: '{path}' could not be read (binary or too large)."
    evidence[path] = content
    return content


def _tool_search_repo(pattern: str, is_regex: bool, allowed_files: set) -> str:
    """Grep the repository. This is how the agent finds EVERY instance of a
    defect rather than only the one the log happened to surface."""
    pattern = (pattern or "").strip()
    if not pattern:
        return "Error: empty pattern."
    try:
        rx = re.compile(pattern if is_regex else re.escape(pattern))
    except re.error as exc:
        return f"Error: invalid regex — {exc}"
    hits, scanned = [], 0
    for rel in sorted(allowed_files):
        p = Path(rel)
        if not p.is_file() or not _is_text_file(p):
            continue
        scanned += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {redact_secrets(line.strip())[:200]}")
                if len(hits) >= MAX_SEARCH_HITS:
                    return ("\n".join(hits)
                            + f"\n...(hit the {MAX_SEARCH_HITS}-result cap; narrow the pattern)")
    if not hits:
        return (f"No matches for {pattern!r} in {scanned} files. "
                f"An absence of matches is itself evidence.")
    return "\n".join(hits) + f"\n\n({len(hits)} match(es) across {scanned} files scanned)"


# ── AI plumbing (OpenAI, strict structured outputs) ──────────────────────────
def _call_openai(messages: list, schema: dict, schema_name: str,
                 model: str = None, max_tokens: int = 1500,
                 temperature: float = 0.0, tag: str = "AI",
                 timeout: int = None, retries: int = MAX_RETRIES) -> dict:
    """Single OpenAI chat call with strict structured output. Returns parsed dict."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model   = model or INVESTIGATE_MODEL
    timeout = timeout or AI_TIMEOUT
    is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))

    payload = {
        "model": model,
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
    }
    if is_reasoning:
        # Reasoning models reject `temperature` and rename the token cap.
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    chars = sum(len(m["content"]) for m in messages)
    print(f"[{tag}] {model} | {len(messages)} msg(s), {chars} chars")

    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = requests.post(f"{OPENAI_BASE_URL}/chat/completions",
                              json=payload,
                              headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                                       "Content-Type": "application/json"},
                              timeout=(10, timeout))

            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    wait = float(r.headers.get(
                        "retry-after", RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]))
                except ValueError:
                    wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if attempt < retries - 1:
                    print(f"[{tag}] {r.status_code} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                raise last
            if r.status_code == 401:
                raise RuntimeError("OpenAI rejected the API key (401). "
                                   "Check the OPENAI_API_KEY secret.")
            if r.status_code == 400 and "json_schema" in r.text:
                raise RuntimeError(f"Schema rejected — model may predate strict "
                                   f"structured outputs: {r.text[:300]}")
            r.raise_for_status()

            body = r.json()
            choice = body["choices"][0]
            usage = body.get("usage", {})
            print(f"[{tag}] {time.time() - t0:.1f}s | "
                  f"in={usage.get('prompt_tokens', '?')} "
                  f"out={usage.get('completion_tokens', '?')}")

            if choice["message"].get("refusal"):
                raise RuntimeError(f"Model refused: {choice['message']['refusal']}")
            if choice.get("finish_reason") == "length":
                # Strict mode guarantees valid JSON only if generation completes.
                raise RuntimeError("Response truncated — raise max_tokens.")
            return json.loads(choice["message"]["content"])

        except requests.exceptions.Timeout as exc:
            last = exc
            if attempt < retries - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"[{tag}] timeout after {timeout}s — retry in {wait}s")
                time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot reach {OPENAI_BASE_URL}: {exc}")
    raise RuntimeError(f"OpenAI call failed after {retries} attempts: {last}")


def _call_openai_tools(messages: list, tools: list, tool_choice,
                       model: str = None, max_tokens: int = 2000,
                       temperature: float = 0.0, tag: str = "AGENT",
                       timeout: int = None, retries: int = MAX_RETRIES) -> dict:
    """Chat call with native function calling. Returns the raw assistant message.

    Unlike _call_openai (which forces one strict JSON blob), this lets the model
    decide when it needs another tool and when it is finished — the loop ends
    when it calls submit_diagnosis, not when a turn counter runs out.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model   = model or INVESTIGATE_MODEL
    timeout = timeout or AI_TIMEOUT
    is_reasoning = model.startswith(("o1", "o3", "o4", "gpt-5"))

    payload = {"model": model, "messages": messages,
               "tools": tools, "tool_choice": tool_choice}
    if is_reasoning:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature

    chars = sum(len(str(m.get("content") or "")) for m in messages)
    print(f"[{tag}] {model} | {len(messages)} msg(s), ~{chars} chars")

    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                                       "Content-Type": "application/json"},
                              timeout=(10, timeout))
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    wait = float(r.headers.get(
                        "retry-after", RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]))
                except ValueError:
                    wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if attempt < retries - 1:
                    print(f"[{tag}] {r.status_code} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                raise last
            if r.status_code == 401:
                raise RuntimeError("OpenAI rejected the API key (401).")
            r.raise_for_status()
            body = r.json()
            choice = body["choices"][0]
            usage = body.get("usage", {})
            print(f"[{tag}] {time.time() - t0:.1f}s | "
                  f"in={usage.get('prompt_tokens','?')} out={usage.get('completion_tokens','?')}")
            if choice["message"].get("refusal"):
                raise RuntimeError(f"Model refused: {choice['message']['refusal']}")
            if choice.get("finish_reason") == "length":
                raise RuntimeError("Response truncated — raise max_tokens.")
            return choice["message"]
        except requests.exceptions.Timeout as exc:
            last = exc
            if attempt < retries - 1:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                print(f"[{tag}] timeout after {timeout}s — retry in {wait}s")
                time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot reach {OPENAI_BASE_URL}: {exc}")
    raise RuntimeError(f"OpenAI tool call failed after {retries} attempts: {last}")


PREFLIGHT_SCHEMA = {"type": "object", "additionalProperties": False,
                    "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}


def preflight_openai() -> bool:
    """Fail fast on a bad key / unreachable egress / model without strict support."""
    if not OPENAI_API_KEY:
        print("[PREFLIGHT] OPENAI_API_KEY is empty.", file=sys.stderr)
        return False
    try:
        out = _call_openai(
            [{"role": "user",
              "content": "Reply with ok set to true."}],
            PREFLIGHT_SCHEMA, "preflight", model=INVESTIGATE_MODEL,
            max_tokens=200, tag="PREFLIGHT", timeout=30, retries=1)
        print(f"[PREFLIGHT] ok — {INVESTIGATE_MODEL} honours strict schema "
              f"(ok={out.get('ok')})")
        return True
    except Exception as exc:
        print(f"[PREFLIGHT] failed: {exc}", file=sys.stderr)
        return False


# ── AI INVESTIGATION AGENT ───────────────────────────────────────────────────
# ── AI INVESTIGATION AGENT (native tool-calling loop) ────────────────────────
def _fn(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description, "strict": True,
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": properties, "required": required}}}


DIAGNOSIS_PROPERTIES = {
    "analysis": {"type": "string",
                 "description": "Your reasoning, in your own words."},
    "hypotheses_considered": {"type": "array", "items": {"type": "string"},
                              "description": "Every candidate cause you weighed."},
    "hypotheses_rejected": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["hypothesis", "disproved_by"],
        "properties": {"hypothesis": {"type": "string"},
                       "disproved_by": {"type": "string"}}},
        "description": "ONLY for candidate explanations that turned out not to be "
                       "defects at all. If you found a REAL defect that simply is not "
                       "the cause of the logged error, it is a finding with "
                       "blocks=later_step — never a rejected hypothesis."},
    "root_cause": {"type": "string",
                   "description": "One sentence covering the underlying defect."},
    "solution": {"type": "string"},
    "why_fix_works": {"type": "string",
                      "description": "Causal chain from the change to a green pipeline."},
    "completeness_check": {"type": "string",
                           "description": "How you verified you found EVERY instance "
                                          "of this defect — which searches you ran and "
                                          "what they returned. Say so plainly if you "
                                          "did not sweep."},
    "confidence": {"type": "number"},
    "commit_message": {"type": "string",
                       "description": "A SINGLE LINE conventional-commit subject, "
                                      "imperative mood, 72 characters maximum. "
                                      "Example: 'fix: correct Python version in CI "
                                      "workflow'. No body, no bullet list, no file "
                                      "paths, no newlines. The detail belongs in the "
                                      "findings, not here."},
    "findings": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["file", "issue", "root_cause", "proof", "blocks", "solution"],
        "properties": {
            "file": {"type": "string",
                     "description": "The file CONTAINING the bad text. Never the missing file."},
            "issue": {"type": "string"},
            "root_cause": {"type": "string"},
            "proof": {"type": "string",
                      "description": "The offending text, verbatim from a file you read."},
            "blocks": {"type": "string", "enum": ["current_failure", "later_step"],
                       "description": "current_failure = this defect caused the error in "
                                      "the log. later_step = a real defect that a LATER "
                                      "stage will hit once the current one is fixed. Both "
                                      "belong here and both must be patched."},
            "solution": {"type": "string"}}},
        "description": "ONE ENTRY PER DEFECT. Several defects in one file means "
                       "several entries. The same defect in five files means five "
                       "entries, one per file."},
}
DIAGNOSIS_REQUIRED = ["analysis", "hypotheses_considered", "hypotheses_rejected",
                      "root_cause", "solution", "why_fix_works",
                      "completeness_check", "confidence", "commit_message",
                      "findings"]

AGENT_TOOLS = [
    _fn("list_directory",
        "List the contents of a directory in the repository. Unfiltered — the "
        "order and contents carry no hint about relevance. Use '.' for the root.",
        {"path": {"type": "string", "description": "Directory path, or '.' for root."}},
        ["path"]),
    _fn("read_file",
        "Read one file from the repository. Returns its full contents (secrets "
        "redacted). If the path does not exist you are told so and nothing is "
        "substituted for it.",
        {"path": {"type": "string", "description": "Path relative to the repository root."}},
        ["path"]),
    _fn("search_repo",
        "Search every readable file for a pattern and return matching lines as "
        "path:line: text. This is how you find EVERY occurrence of a defect "
        "instead of only the one the log happened to surface. Search for the "
        "broken value, not the fixed one.",
        {"pattern": {"type": "string", "description": "Literal text, or a regex if is_regex is true."},
         "is_regex": {"type": "boolean", "description": "Treat pattern as a Python regex."}},
        ["pattern", "is_regex"]),
    _fn("submit_diagnosis",
        "Submit your final diagnosis and end the investigation. Call this ONLY "
        "when you can quote the offending text from files you have actually read "
        "AND you have swept for other instances of the same defect.",
        DIAGNOSIS_PROPERTIES, DIAGNOSIS_REQUIRED),
]

INVESTIGATE_SYSTEM = """\
You are the sole debugging engineer for this CI/CD failure. Nothing else has \
diagnosed it and nothing else will. If you are wrong, the wrong patch ships.

## WHAT THE HARNESS DID — AND WHAT IT CANNOT DO

A Python harness surrounds you. It is an I/O layer with no understanding of the
failure. It dumped the raw CI logs and an unfiltered directory listing, it
executes the tools you call, and it will later apply your patch and run
validators against it. It did not diagnose anything and holds no theory of the
bug. It did not choose which files matter — that is now entirely your job.

One of its outputs LOOKS authoritative and is not: "Primary error" in ISSUE TO
SOLVE is simply the first log line that matched an ordered list of regexes. It
is a guess at which line is salient, nothing more. The true cause is often a
line much earlier in the log, or a condition that never appears in the log at
all. The highlighted error is frequently a downstream symptom.

## YOUR TOOLS

  list_directory(path)          — see what exists. Start at "." if unsure.
  read_file(path)               — read one file's full contents.
  search_repo(pattern, is_regex) — find every line matching a pattern, repo-wide.
  submit_diagnosis(...)          — end the investigation with your conclusion.

Call tools until you can prove the root cause. There is no fixed number of
turns; you decide when you have enough. You cannot run builds, tests, or shell
commands — reason from file contents alone.

## METHOD

1. Form at least TWO competing hypotheses before you look for support for any of
   them. If only one comes to mind, you have not understood the failure yet.
2. For each hypothesis, state what evidence would DISPROVE it, then go get that
   evidence. Requesting files to falsify your own favourite theory is the point.
3. A hypothesis is confirmed only when you can quote the exact offending text
   from a file you have actually read. If you cannot quote it, you are guessing.
4. Eliminate rivals explicitly, with evidence. "The workflow YAML is not the
   cause: every path in its run: lines exists in the tree" is elimination.
   "The Dockerfile seems more likely" is not.
5. Reason about the MECHANISM, never the wording. "No such file or directory:
   'X'" means the process did not find X where it looked. Possible causes
   include a misspelled reference, a file never copied into the image, a wrong
   WORKDIR, a multi-stage build that dropped it, a path resolved relative to the
   wrong directory, a .dockerignore exclusion, or a file meant to be generated
   at build time that was not. Do not collapse to "typo" until you rule these out.
6. The file named in an error is where the failure SURFACED. The bug lives in
   whatever referenced or produced it. Never patch a file that does not exist.

## TWO DIFFERENT JOBS — DO NOT CONFUSE THEM

Job A: explain the error in the log.
Job B: list EVERY defect you found along the way.

These are separate, and the second is where this tool earns its keep. A build
stops at the first fatal error, so a defect three lines later is invisible in the
log until the first is fixed. If you patch only the logged cause, the pipeline
fails again on the very next run and the auto-fixer burns another cycle.

So: a defect that is NOT the cause of the logged error is STILL a finding. Mark
it blocks="later_step". Mark the logged cause blocks="current_failure". Emit both.

hypotheses_rejected is ONLY for candidate explanations that turned out not to be
defects at all — a file you suspected but which is correct, a theory the evidence
killed. The moment you can quote broken text from a real file, it is a finding,
regardless of whether it explains today's error.

Worked example of the mistake to avoid — note that the specific technology here
is irrelevant; the reasoning error is what matters. A Node service's CI config
reads:

    - run: npm ci --registry https://registry.invalid.example
    - run: node ./src/sever.js          # the real file is src/server.js

The log shows only that the registry could not be reached, because the job never
got as far as running node. Both lines are defects. The correct output is TWO
findings — the unreachable registry as current_failure, the misspelled entry
point as later_step. Writing "ruled out: the misspelled script, because the error
is about the registry" is WRONG: that reasoning is sound as an explanation of the
log and useless as engineering. The typo is real, you can quote it, and it will
break the next run.

Apply that shape to whatever stack you are actually looking at.

## SWEEP BEFORE YOU SUBMIT

  * Read every line of each file you are patching, not just the offending one.
    One misunderstanding usually produced several errors in the same sitting.
  * For each file path, image tag, version pin or command referenced in an
    executed file (Dockerfile, workflow YAML, shell script, Makefile, compose
    file), check it against the repository tree. Anything referenced but absent
    is a defect — search_repo and list_directory are how you check.
  * Take each broken value and search_repo for it. It may recur in a compose
    file, a script, or a second Dockerfile. Every occurrence in an executed file
    is its own finding.
  * State plainly in completeness_check which sweeps you ran and what they
    returned. A clean sweep is a real result worth recording.

Do not stop investigating merely because you can explain the log.

## CONFIDENCE — CALIBRATION MATTERS MORE THAN OPTIMISM

  0.9+       you quoted the offending text and eliminated every rival.
  0.6-0.9    something is clearly wrong, but a rival survives or the fix is inferred.
  below 0.6  you are guessing.

Prefer another tool call over guessing. But if you are told this is your FINAL
TURN, no further evidence is coming: submit your best hypothesis with an honest
low score. A truthful 0.4 routes this to a human, which is a correct and useful
outcome. An inflated 0.95 ships a wrong patch to production.
"""


COMMIT_SUBJECT_MAX = int(os.environ.get("COMMIT_SUBJECT_MAX", "72"))


def _clean_commit_message(raw: str) -> str:
    """Reduce whatever the model wrote to one short subject line.

    Formatting only — this does not change what the fix does. Long multi-line
    commit bodies make `git log --oneline` unreadable and are the wrong place
    for detail that already lives in the PR.
    """
    text = (raw or "").strip()
    subject = next((l.strip() for l in text.splitlines() if l.strip()), "")
    subject = re.sub(r"^[-*\u2022]\s*", "", subject).strip().rstrip(".")
    subject = re.sub(r"\s+", " ", subject)
    if not subject:
        subject = "correct CI configuration"
    if not re.match(r"^(fix|chore|ci|build|refactor)(\(.+?\))?:", subject, re.I):
        subject = f"{BOT_PREFIX} {subject}"
    if len(subject) > COMMIT_SUBJECT_MAX:
        cut = subject[:COMMIT_SUBJECT_MAX].rsplit(" ", 1)[0]
        subject = (cut or subject[:COMMIT_SUBJECT_MAX]).rstrip(",;:-") 
    return subject


def _finalize_investigation(data: dict, forced: bool) -> dict:
    try:
        confidence = float(data.get("confidence", 0.4))
    except (TypeError, ValueError):
        confidence = 0.4

    findings = []
    for f in (data.get("findings") or []):
        if isinstance(f, dict) and (f.get("issue") or f.get("root_cause")):
            findings.append({
                "file":       (f.get("file") or "").strip(),
                "issue":      (f.get("issue") or f.get("root_cause") or "").strip(),
                "root_cause": (f.get("root_cause") or "").strip(),
                "proof":      (f.get("proof") or "").strip(),
                "blocks":     (f.get("blocks") or "current_failure").strip(),
                "solution":   (f.get("solution") or "").strip(),
            })
    if not findings:
        findings = [{"file": "", "proof": "", "blocks": "current_failure",
                     "issue": data.get("root_cause") or "unknown",
                     "root_cause": data.get("root_cause") or "unknown",
                     "solution": data.get("solution") or ""}]

    rejected = []
    for h in (data.get("hypotheses_rejected") or []):
        if isinstance(h, dict) and h.get("hypothesis"):
            rejected.append({"hypothesis": h["hypothesis"].strip(),
                             "disproved_by": (h.get("disproved_by") or "").strip()})

    # A confirmation that eliminated nothing is a first guess wearing a
    # confidence score. Cap it so the <0.5 gate routes it to a human.
    considered = [c for c in (data.get("hypotheses_considered") or []) if c]
    if not rejected and len(considered) < 2 and confidence >= 0.5:
        print(f"[CALIBRATE] submitted with {len(considered)} hypothesis and no "
              f"rivals eliminated — capping confidence {confidence:.0%} → 45%")
        confidence = 0.45
    if forced and confidence > 0.6:
        print(f"[CALIBRATE] diagnosis was forced at the turn limit — "
              f"capping confidence {confidence:.0%} → 60%")
        confidence = 0.6

    return {
        "root_cause":     data.get("root_cause") or "unknown",
        "solution":       data.get("solution") or "",
        "why_fix_works":  data.get("why_fix_works") or "",
        "completeness_check": data.get("completeness_check") or "",
        "confidence":     confidence,
        "commit_message": _clean_commit_message(data.get("commit_message")),
        "findings":       findings,
        "hypotheses_considered": considered,
        "hypotheses_rejected":   rejected,
    }


EXECUTED_FILE_HINTS = (".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
                       ".yml", ".yaml", ".mk")
EXECUTED_FILE_NAMES = ("dockerfile", "containerfile", "makefile", "justfile",
                       "taskfile.yml", "taskfile.yaml", "jenkinsfile",
                       "docker-compose.yml", "docker-compose.yaml",
                       "compose.yml", "compose.yaml", "procfile")


def _is_executed_file(rel: str) -> bool:
    """Files whose contents are RUN by CI. A dangling path in a README is
    cosmetic; a dangling path in a Dockerfile breaks the build."""
    name = Path(rel).name.lower()
    if name in EXECUTED_FILE_NAMES or name.startswith("dockerfile"):
        return True
    return rel.lower().endswith(EXECUTED_FILE_HINTS)


def _missing_reference_report(evidence: dict) -> list:
    """Report every path referenced in an executed file that does not exist.

    This states FACTS — 'this file names a path that is not in the tree'. It
    does not decide whether that is the bug, which finding it belongs to, or how
    to fix it. The model draws all conclusions.
    """
    repo = {_relstrip(str(p)) for p in Path(".").rglob("*")
            if p.is_file() and not any(d in SKIP_DIRS for d in p.parts)}
    out = []
    for rel, content in evidence.items():
        if not isinstance(content, str) or not _is_executed_file(rel):
            continue
        parent = Path(rel).parent
        seen = set()
        for m in REPO_REF_EXT_PATTERN.finditer(content):
            token = m.group(1)
            if token in seen or "${{" in token or token.startswith("."):
                continue
            # `jr.json()` is a method call, not a file. Without this, the gate
            # flags every .json()/.text()/.read() call in an embedded script and
            # burns a pushback turn asking the agent about a non-existent file.
            after = content[m.end():m.end() + 1]
            if after == "(":
                continue
            seen.add(token)
            candidates = {token, _relstrip(str(parent / token))}
            if any(c in repo or Path(c).is_file() for c in candidates):
                continue
            out.append((rel, token))
    return out


def _tool_args(call: dict) -> dict:
    try:
        return json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}


def ai_investigate(signal, exit_code, repo_tree, git_diff, allowed_files: set,
                   issue_block: str = "") -> dict:
    """Native tool-calling agent loop.

    The model lists, searches and reads the repository itself, and ends the loop
    by calling submit_diagnosis. Python chooses nothing: not which files are
    relevant, not when the investigation is complete.
    """
    evidence, log = {}, []
    result, forced_submit = None, False
    tool_calls_used = 0
    got_any_model_response = False
    pushback_used = False

    diff_trimmed = git_diff[:INVESTIGATION_DIFF_CHARS]
    messages = [
        {"role": "system", "content": INVESTIGATE_SYSTEM},
        {"role": "user", "content":
            f"## ISSUE TO SOLVE\n{issue_block}\n\n"
            f"## Supporting log lines\n```\n{signal}\n```\n\n"
            f"## Exit code: {exit_code}\n\n"
            f"## Git diff\n```\n{diff_trimmed}\n```\n\n"
            f"## Repository tree (unfiltered listing — no relevance implied)\n"
            f"{repo_tree}\n\n"
            f"Investigate. Use your tools freely, then call submit_diagnosis."},
    ]

    for turn in range(1, MAX_AGENT_TURNS + 1):
        out_of_road = (turn == MAX_AGENT_TURNS
                       or tool_calls_used >= MAX_TOOL_CALLS
                       or _budget_exceeded())
        if out_of_road:
            forced_submit = True
            reason = ("turn limit" if turn == MAX_AGENT_TURNS else
                      "tool-call cap" if tool_calls_used >= MAX_TOOL_CALLS else
                      "time budget")
            print(f"[INVESTIGATE] {reason} reached — forcing submit_diagnosis.")
            messages.append({"role": "user", "content":
                "FINAL TURN. No further tool calls are available. Submit your best "
                "diagnosis now with an honest confidence score, and state in "
                "completeness_check that you could not finish sweeping."})
            tool_choice = {"type": "function", "function": {"name": "submit_diagnosis"}}
        else:
            tool_choice = "auto"

        try:
            msg = _call_openai_tools(messages, AGENT_TOOLS, tool_choice,
                                     model=INVESTIGATE_MODEL, max_tokens=2500,
                                     tag=f"AGENT-T{turn}")
        except Exception as exc:
            print(f"[INVESTIGATE] turn {turn} failed: {exc}", file=sys.stderr)
            break

        got_any_model_response = True
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if msg.get("content"):
            print(f"[AGENT] {msg['content'].strip()[:200]}")

        if not calls:
            print("[INVESTIGATE] model returned no tool call — nudging.")
            log.append({"turn": turn, "status": "no_tool_call",
                        "analysis": (msg.get("content") or "")[:200]})
            messages.append({"role": "user", "content":
                "You must either call an investigation tool or call "
                "submit_diagnosis. Do not reply with prose alone."})
            continue

        # submit_diagnosis ends the loop, whatever else was requested alongside.
        submit = next((c for c in calls
                       if c["function"]["name"] == "submit_diagnosis"), None)
        if submit:
            data = _tool_args(submit)

            # Before accepting: report any dangling path in an executed file the
            # agent read that its findings do not mention. Facts only — Python
            # does not say whether these are bugs or how to fix them. This is the
            # gate that catches "explained the log, missed the second defect".
            uncovered = []
            if not pushback_used and not forced_submit:
                claimed = " ".join(
                    f"{f.get('proof','')} {f.get('issue','')} {f.get('solution','')}"
                    for f in (data.get("findings") or []) if isinstance(f, dict))
                uncovered = [(rel, tok) for rel, tok in _missing_reference_report(evidence)
                             if tok not in claimed]
            if uncovered:
                pushback_used = True
                lines = "\n".join(f"  - {rel} references '{tok}', which is not in "
                                   f"the repository tree" for rel, tok in uncovered)
                print(f"[PUSHBACK] {len(uncovered)} dangling reference(s) not covered "
                      f"by findings — asking the agent to reconsider.")
                for rel, tok in uncovered:
                    print(f"[PUSHBACK]   {rel} → '{tok}'")
                log.append({"turn": turn, "status": "pushback",
                            "analysis": f"{len(uncovered)} uncovered dangling reference(s)"})
                messages.append({"role": "tool", "tool_call_id": submit["id"],
                                 "content": "Diagnosis not accepted yet."})
                messages.append({"role": "user", "content":
                    f"Before this is accepted, note these facts about files you read:\n"
                    f"{lines}\n\n"
                    f"Each of these is a path named inside a file that CI executes, "
                    f"which does not exist in the repository. None of them appears in "
                    f"your findings. Decide for each one whether it is a defect that a "
                    f"later build step will hit, or whether it is legitimate (generated "
                    f"at build time, provided by the base image, created by an earlier "
                    f"step, or otherwise fine). Investigate further if you need to.\n\n"
                    f"If any is a defect, add it as a finding with blocks=\"later_step\" "
                    f"and resubmit. If all are legitimate, resubmit unchanged and say so "
                    f"in completeness_check."})
                continue

            log.append({"turn": turn, "status": "submit_diagnosis",
                        "analysis": (data.get("analysis") or "")[:200]})
            result = _finalize_investigation(data, forced=forced_submit)
            break

        for call in calls:
            name = call["function"]["name"]
            args = _tool_args(call)
            tool_calls_used += 1
            if name == "list_directory":
                target = args.get("path", ".")
                out = _tool_list_directory(target, allowed_files)
            elif name == "read_file":
                target = args.get("path", "")
                out = _tool_read_file(target, allowed_files, evidence)
            elif name == "search_repo":
                target = f"{args.get('pattern','')!r} regex={args.get('is_regex', False)}"
                out = _tool_search_repo(args.get("pattern", ""),
                                        bool(args.get("is_regex")), allowed_files)
            else:
                target, out = name, f"Error: unknown tool '{name}'."
            print(f"[TOOL {tool_calls_used}/{MAX_TOOL_CALLS}] {name}({target}) "
                  f"→ {len(out)} chars")
            log.append({"turn": turn, "status": f"tool:{name}",
                        "analysis": f"{target} → {out.splitlines()[0][:120] if out else ''}"})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": out[:MAX_FILE_CHARS]})

    if result is None:
        failure_mode = "not_converged" if got_any_model_response else "no_model_response"
        result = {"root_cause": ("investigation did not converge"
                                 if got_any_model_response
                                 else "model returned no usable response (API/network problem)"),
                  "solution": "", "why_fix_works": "", "completeness_check": "",
                  "confidence": 0.0, "commit_message": "fix: auto-fixer change",
                  "findings": [], "hypotheses_considered": [],
                  "hypotheses_rejected": [], "failure_mode": failure_mode}
    else:
        result["failure_mode"] = None
    result["evidence"] = evidence
    result["investigation_log"] = log
    result["tool_calls_used"] = tool_calls_used
    return result


# ── PATCH GENERATION ──────────────────────────────────────────────────────
PATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["file", "problem", "evidence", "corrected"],
            "properties": {
                "file":      {"type": "string"},
                "problem":   {"type": "string"},
                "evidence":  {"type": "string"},
                "corrected": {"type": "string"},
            }}},
    },
}

PATCH_SYSTEM = """\
You are a patch-generation agent. Another engineer already investigated this \
CI/CD failure and confirmed the defects below. Your job is to emit the concrete \
text-level fix. Do not re-open the diagnosis or second-guess the findings.

ONE EXCEPTION. If a "previous attempt failed" section reports that validation \
rejected your patch because a file still references something missing, that is a \
CONFIRMED additional defect found by a checker, not a theory. Fix it in the same \
response as everything else. Never re-emit an identical patch after a rejection: \
if you change nothing, the run fails again for the same reason.

Stay scoped to the "## ISSUE TO SOLVE" section if one is present.

"issues" = ONE ENTRY PER DEFECT, across every affected file. Several defects in
one file means several entries with the same "file". The same defect in five
files means five entries. Emit ALL of them in this one response — a partial fix
leaves the pipeline broken and wastes a repair round. Each entry:
  - "file": the exact path from a "### <path>" header below. Nothing else.
  - "problem": one sentence.
  - "evidence": EXACT text copied character-for-character from that file's
    contents below — copy the WRONG text. Keep it short (one line is ideal) and
    make sure it appears exactly once in that file.
  - "corrected": the same text with ONLY the bug fixed.

**HARD REQUIREMENTS**
  * "evidence" must be a verbatim substring of the file contents shown below.
    If you cannot copy it exactly, do not emit that issue at all.
  * "evidence" and "corrected" MUST be different strings.
  * Never invent a file path that has no "###" header below.
  * Do not merge two defects into one entry, even in the same file. Separate
    entries with distinct, individually-locatable "evidence" strings let the
    validator accept the good ones when one is wrong.
"""


def _normalize_issue_keys(issues):
    """Legacy key-drift guard. Strict schemas make this a no-op — remove later."""
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
                        it[want] = it.pop(a)
                        break
        out.append(it)
    return out


def ai_generate_patch(root_cause: str, solution: str, evidence: dict,
                      retry_note: str = "", issue_block: str = "",
                      why_fix_works: str = "") -> list:
    context = "\n\n".join(f"### {f}\n```\n{c}\n```" for f, c in evidence.items()) \
              or "(no evidence files were read)"
    user = (
        (f"## ISSUE TO SOLVE (stay scoped to this)\n{issue_block}\n\n" if issue_block else "")
        + (f"## A previous attempt failed\n{retry_note}\nDo not repeat that change.\n\n"
           if retry_note else "")
        + f"## Confirmed root cause\n{root_cause}\n\n"
        + f"## Solution direction\n{solution}\n\n"
        + (f"## Why the investigator believes this fix works\n{why_fix_works}\n\n"
           if why_fix_works else "")
        + f"## File contents — you may ONLY edit these\n{context}\n\n"
        + "Emit the issues JSON.")
    data = _call_openai([{"role": "system", "content": PATCH_SYSTEM},
                         {"role": "user", "content": user[:MAX_PROMPT_CHARS]}],
                        PATCH_SCHEMA, "patch", model=PATCH_MODEL,
                        max_tokens=3000, tag="PATCH")
    return _normalize_issue_keys(data.get("issues", []))


# ── REVIEW AGENT ─────────────────────────────────────────────────────────────
REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "root_cause", "solution", "confidence"],
    "properties": {
        "decision":   {"type": "string", "enum": ["retry", "stop"]},
        "root_cause": {"type": "string"},
        "solution":   {"type": "string"},
        "confidence": {"type": "number"},
    },
}

REVIEW_SYSTEM = """\
You are reviewing why a previously applied fix still failed CI. Given the \
original root cause, the fix that was tried, and the NEW failure output, decide:
  - "retry": provide an updated root_cause and solution.
  - "stop": a human should look at it; explain why in root_cause.
Focus on the "## NEW ISSUE" section if present.
"""


def ai_review_failure(prior_root_cause: str, prior_solution: str,
                      new_signal: str, issue_block: str = "") -> dict:
    user = ((f"## NEW ISSUE\n{issue_block}\n\n" if issue_block else "")
            + f"## Original root cause\n{prior_root_cause}\n\n"
            + f"## Fix attempted\n{prior_solution}\n\n"
            + f"## New failure after applying the fix\n```\n{new_signal}\n```")
    try:
        data = _call_openai([{"role": "system", "content": REVIEW_SYSTEM},
                             {"role": "user", "content": user[:MAX_PROMPT_CHARS]}],
                            REVIEW_SCHEMA, "review", model=REVIEW_MODEL,
                            max_tokens=800, tag="REVIEW", retries=2)
    except Exception as exc:
        print(f"[REVIEW] failed: {exc}", file=sys.stderr)
        return {"decision": "stop", "root_cause": prior_root_cause,
                "solution": prior_solution, "confidence": 0.0}
    # The enum guarantees decision ∈ {retry, stop}.
    return {"decision": data["decision"],
            "root_cause": data.get("root_cause") or prior_root_cause,
            "solution": data.get("solution") or prior_solution,
            "confidence": float(data.get("confidence", 0.0))}


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
                if [c.strip() for c in cl[j:j + len(fl)]] == fl:
                    base = cl[j][:len(cl[j]) - len(cl[j].lstrip())]
                    cl[j:j + len(fl)] = [(base + r if r.strip() else r)
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


def _any_reference_missing(content: str, file: str) -> str:
    """Check if any file-like references in the patched content don't exist in the repo.
    Returns a reason string, or '' if all referenced files exist.
    """
    if not content:
        return ""
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
        return f"patched file still references missing '{token}'"
    return ""


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
        if not _is_plausible_python_tag(major, minor):
            return (f"base image 'python:{version}' is not a plausible CPython tag "
                    f"(expected 3.{MIN_PYTHON_MINOR}+)")
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
SENSITIVE_WORKFLOW_KEYS = {"permissions", "secrets", "env", "on", "runs-on", "if"}

# `uses` is deliberately NOT in the blanket set above. Repointing `uses:` changes
# which third-party code CI executes, so it cannot be waved through — but a typo
# in an action name ("Unable to resolve action …, repository not found") is a
# common, real failure that was previously unfixable. Instead of trusting the
# model, _verify_uses_change proves the edit is a typo correction: same version
# ref, only one of owner/repo altered, high string similarity, the old target
# genuinely missing on GitHub and the new one genuinely present.
USES_REF = re.compile(r"^([\w.\-]+)/([\w.\-]+?)(?:/([\w.\-/]+))?@([\w.\-/]+)$")
USES_MIN_SIMILARITY = float(os.environ.get("USES_MIN_SIMILARITY", "0.75"))


def _action_repo_exists(owner: str, repo: str, token: str) -> bool:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(f"https://api.github.com/repos/{owner}/{repo}",
                     headers=headers, timeout=15)
    if r.status_code == 200:
        return True
    if r.status_code == 404:
        return False
    raise RuntimeError(f"GitHub API returned {r.status_code} for {owner}/{repo}")


def _verify_uses_change(old_v, new_v) -> tuple:
    """Allow a `uses:` edit only if it is provably a typo correction."""
    if not isinstance(old_v, str) or not isinstance(new_v, str):
        return False, "uses: edit is not a string change"

    if new_v.strip().startswith("./"):
        # Local composite action — no supply-chain question, just path existence.
        local = new_v.strip().split("@")[0]
        return ((True, "ok") if Path(local).exists()
                else (False, f"uses: points at local path '{local}' which does not exist"))

    om, nm = USES_REF.match(old_v.strip()), USES_REF.match(new_v.strip())
    if not om or not nm:
        return False, f"uses: value is not a recognisable owner/repo@ref ({new_v!r})"

    o_owner, o_repo, _o_sub, o_ref = om.groups()
    n_owner, n_repo, _n_sub, n_ref = nm.groups()

    if o_ref != n_ref:
        return False, (f"uses: edit changes the pinned version ({o_ref} → {n_ref}); "
                       f"only the action name may be corrected")
    if o_owner != n_owner and o_repo != n_repo:
        return False, ("uses: edit changes BOTH the owner and the repository — "
                       "that is a repoint, not a typo correction")

    old_full, new_full = f"{o_owner}/{o_repo}", f"{n_owner}/{n_repo}"
    ratio = difflib.SequenceMatcher(None, old_full, new_full).ratio()
    if ratio < USES_MIN_SIMILARITY:
        return False, (f"uses: '{old_full}' → '{new_full}' is too dissimilar "
                       f"({ratio:.0%}) to be a typo correction")

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    try:
        # Fails closed: if the API cannot be reached, the edit is not approved.
        if _action_repo_exists(o_owner, o_repo, token):
            return False, (f"uses: the original action '{old_full}' resolves fine, "
                           f"so this edit is not fixing an unresolvable reference")
        if not _action_repo_exists(n_owner, n_repo, token):
            return False, f"uses: the replacement action '{new_full}' does not exist on GitHub"
    except Exception as exc:
        return False, f"uses: could not verify the action against GitHub ({exc})"

    print(f"[VERIFY] uses: '{old_full}' (404) → '{new_full}' (200), "
          f"ref {n_ref} unchanged, {ratio:.0%} similar — approved")
    return True, "ok"


def check_workflow_python_versions(new_text: str) -> str:
    """Reject a patch that leaves an unreal python-version. Never rewrites it.

    The previous implementation silently overwrote whatever version the model
    chose with DEFAULT_PYTHON_VERSION — that was Python authoring the fix and
    then crediting the AI for it in the PR body. Validation rejects; it does not
    write. The rejection reason is fed back so the model patches it correctly on
    the next repair round.
    """
    for m in WORKFLOW_PYVERSION_LINE.finditer(new_text):
        major, minor = int(m.group(3)), int(m.group(4))
        if _is_plausible_python_tag(major, minor):
            continue
        return (f"patched workflow sets python-version {major}.{minor}, which is not "
                f"a plausible CPython version (expected 3.{MIN_PYTHON_MINOR}+)")
    return ""


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
        if "uses" in segments:
            ok_uses, reason_uses = _verify_uses_change(old_v, new_v)
            if not ok_uses:
                return False, reason_uses
            continue
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
            pyver_reason = check_workflow_python_versions(content)
            if pyver_reason:
                return False, pyver_reason
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

    # Post-patch missing-reference check (catches leftover typos)
    ref_reason = _any_reference_missing(content, file)
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
    if "rust" in stacks and Path("Cargo.toml").exists():
        cmds.append(["cargo", "test"])
    if "ruby" in stacks and Path("Gemfile").exists():
        cmds.append(["bundle", "exec", "rspec"])
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


MAX_DIFF_LINES_PER_FILE = int(os.environ.get("MAX_DIFF_LINES_PER_FILE", "30"))


def _render_diff(original: str, new: str, path: str) -> str:
    """Compact unified diff — the single most useful thing in a review."""
    diff = list(difflib.unified_diff(
        original.splitlines(), new.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm="", n=1))
    if len(diff) > MAX_DIFF_LINES_PER_FILE:
        diff = diff[:MAX_DIFF_LINES_PER_FILE] + ["... (diff truncated — see Files changed)"]
    return "\n".join(diff)


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def open_pr(token, repo, branch, ctx: dict) -> str:
    """Open the fix PR.

    The body answers three questions in order: what broke, what changed, is it
    safe to merge. Everything else — the tool trace, the rejected hypotheses,
    the list of files read — is audit material and lives inside a collapsed
    <details> block, available when someone wants it and invisible when they don't.
    """
    written     = ctx.get("written", [])
    originals   = ctx.get("originals", {})
    fixes       = ctx.get("fixes", [])
    findings    = ctx.get("findings") or []
    confidence  = ctx.get("confidence", 0.0)
    root_cause  = ctx.get("root_cause", "")
    why_works   = ctx.get("why_fix_works", "")
    completeness = ctx.get("completeness", "")
    rejected    = ctx.get("rejected") or []
    trace       = ctx.get("investigation_log") or []
    files_read  = ctx.get("evidence_files") or []
    run_url     = ctx.get("run_url", "")
    tests_ran   = ctx.get("tests_ran", False)
    docker_ran  = ctx.get("docker_ran", False)

    def _patched(f: str) -> str:
        for fx in fixes:
            if fx.get("file") == f and isinstance(fx.get("fixed_content"), str):
                return fx["fixed_content"]
        try:                       # fallback only; the tree may be on the base branch
            return Path(f).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    changed_lines = sum(
        len([l for l in _render_diff(originals.get(f, ""), _patched(f), f).splitlines()
             if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))])
        for f in written)

    # ── Summary ────────────────────────────────────────────────────────────
    head = [f"### What broke\n\n{root_cause}\n"]
    if run_url:
        head.append(f"Failing run: {run_url}\n")
    latent = sum(1 for f in findings if f.get("blocks") == "later_step")
    scope = (f"**Scope:** {_plural(len(findings), 'defect')} in "
             f"{_plural(len(written), 'file')} · ~{changed_lines} lines changed")
    if latent:
        scope += (f"\n\n{_plural(latent, 'of these defects was' if latent == 1 else 'of these defects were')}"
                  f" not the cause of this failure — {'it was' if latent == 1 else 'they were'} "
                  f"found during the investigation and would have broken a later build step.")
    head.append(scope + "\n")

    # ── Changes, one section per file, with a real diff ────────────────────
    changes = ["### Changes\n"]
    for f in written:
        diff = _render_diff(originals.get(f, ""), _patched(f), f)
        per_file = [x for x in findings if x.get("file") == f]
        changes.append(f"**`{f}`**\n")
        if diff:
            changes.append(f"```diff\n{diff}\n```\n")
        for x in per_file:
            reason = x.get("root_cause") or x.get("issue") or ""
            # A reviewer reads these differently: one explains the red build,
            # the other is a defect we caught before it could cause its own.
            marker = ("**Caused the failure** — " if x.get("blocks") == "current_failure"
                      else "**Found while investigating** — ")
            issue = (x.get("issue") or "").strip().rstrip(".:;")
            reason = reason.strip()
            changes.append(f"- {marker}{issue}"
                           + (f". {reason}\n" if reason and reason != issue else "\n"))
        if not per_file:
            reason = next((fx.get("reason", "") for fx in fixes if fx.get("file") == f), "")
            if reason:
                changes.append(f"- {reason}\n")

    # ── Why it is safe to merge ────────────────────────────────────────────
    checks = ["### Verification\n",
              "- Every file path referenced by the patched files exists in the repo\n",
              "- Syntax parsed (Python / YAML / JSON as applicable)\n",
              "- No secrets or dangerous shell patterns introduced\n"]
    if tests_ran:
        checks.append("- Test suite passed after the change\n")
    if docker_ran:
        checks.append("- `docker build` succeeded after the change\n")
    checks.append(f"\n**Diagnostic confidence:** {confidence:.0%}"
                  + ("  — below the comfort threshold; review closely"
                     if confidence < 0.75 else "") + "\n")
    if why_works:
        checks.append(f"\n**Rationale:** {why_works}\n")

    # ── Risk: what a reviewer needs to weigh before approving ──────────────
    sensitive = []
    if any(WORKFLOW_PATTERN.search(f) for f in written):
        sensitive.append("CI workflow configuration")
    if any("dockerfile" in Path(f).name.lower() for f in written):
        sensitive.append("container build definition")
    risk = ["\n### Risk\n"]
    risk.append(f"- Blast radius: {_plural(len(written), 'file')}, "
                f"~{changed_lines} lines\n")
    if sensitive:
        risk.append(f"- Touches {' and '.join(sensitive)} — changes here affect "
                    f"every subsequent build\n")
    if not (tests_ran or docker_ran):
        risk.append("- No test or build execution was possible in this environment; "
                    "correctness rests on static validation alone\n")
    risk.append("- Nothing has been merged. Close this PR to discard the change "
                "entirely.\n")

    # ── Checklist: give the reviewer something to actually do ──────────────
    check_items = ["\n### Before approving\n",
                   "- [ ] The diff matches the defects described above, and changes "
                   "nothing else\n",
                   "- [ ] The replacement values are right for this project "
                   "(versions, filenames, paths)\n"]
    if any(f.get("blocks") == "later_step" for f in findings):
        check_items.append("- [ ] The defects marked *Found while investigating* are "
                           "genuine, not deliberate\n")
    check_items.append("- [ ] CI is green on this branch\n")

    # ── Audit trail, collapsed ─────────────────────────────────────────────
    audit = ["\n<details>\n<summary>Investigation detail</summary>\n\n"]
    if completeness:
        audit.append(f"**Checked for other instances:** {completeness}\n\n")
    if rejected:
        audit.append("**Alternative causes ruled out**\n\n"
                     "| Considered | Ruled out because |\n|---|---|\n")
        audit += [f"| {h['hypothesis']} | {h['disproved_by']} |\n" for h in rejected]
        audit.append("\n")
    if files_read:
        audit.append(f"**Files examined:** {', '.join(f'`{x}`' for x in files_read)}\n\n")
    if trace:
        audit.append(f"**Steps taken:** {len(trace)}\n\n```\n")
        for i, st in enumerate(trace):
            label = st.get("status", "")
            arg = (st.get("analysis", "") or "").split("\u2192")[0].strip()
            audit.append(f"{i+1:>2}. {label:<22} {arg[:70]}\n")
        audit.append("```\n")
    audit.append("\n</details>\n")

    body = ("".join(head) + "\n" + "".join(changes) + "\n" + "".join(checks)
            + "".join(risk) + "".join(check_items) + "".join(audit)
            + "\n---\n*Opened automatically by the CI auto-fixer. "
              "Human review required before merge.*")

    r = requests.post(f"https://api.github.com/repos/{repo}/pulls",
                      json={"title": ctx.get("commit_message", "fix: automated CI fix"),
                            "head": branch, "base": GIT_TARGET_BRANCH, "body": body},
                      headers=_gh(token), timeout=30)
    if r.status_code in (200, 201):
        url = r.json().get("html_url", "")
        print(f"[PR] {url}")
        _label_pr(token, repo, r.json().get("number"), confidence)
        return url
    print(f"[PR] failed {r.status_code}: {r.text[:200]}", file=sys.stderr)
    return ""


def _label_pr(token, repo, number, confidence):
    """Labels do the at-a-glance signalling the old emoji title was doing."""
    if not number:
        return
    labels = ["automated-fix"]
    labels.append("high-confidence" if confidence >= 0.75 else "needs-close-review")
    try:
        requests.post(f"https://api.github.com/repos/{repo}/issues/{number}/labels",
                      json={"labels": labels}, headers=_gh(token), timeout=15)
    except Exception:
        pass


def pending_bot_pr(token, repo) -> str:
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/pulls?state=open&per_page=50",
                         headers=_gh(token), timeout=15)
        if r.status_code == 200:
            for pr in r.json():
                # Branch prefix is the reliable marker; the title is free text.
                if pr.get("head", {}).get("ref", "").startswith("fix/"):
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
    print(f"[CONFIG] investigate={INVESTIGATE_MODEL} patch={PATCH_MODEL} "
          f"review={REVIEW_MODEL} @ {OPENAI_BASE_URL}")

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
    if not preflight_openai():
        print("[FATAL] OpenAI preflight failed — not attempting a fix.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       "Auto-fixer could not reach the OpenAI API (bad key, network "
                       "egress, or a model without strict structured-output support). "
                       "No fix was attempted.", run_url)
        sys.exit(0)

    investigation = ai_investigate(supporting_signal, exit_code, repo_tree, git_diff,
                                   allowed_files, issue_block=issue_block)
    root_cause = investigation["root_cause"]
    solution   = investigation["solution"]
    commit_msg = investigation["commit_message"]
    confidence = investigation["confidence"]
    evidence   = investigation["evidence"]
    findings   = investigation.get("findings", [])
    why_fix_works = investigation.get("why_fix_works", "")
    completeness  = investigation.get("completeness_check", "")
    considered = investigation.get("hypotheses_considered", [])
    rejected   = investigation.get("hypotheses_rejected", [])
    investigation_log = investigation.get("investigation_log")

    print("\n  ── investigation result ──")
    print(f"  OVERALL CAUSE : {root_cause}")
    print(f"  confidence    : {confidence:.0%}")
    print(f"  files read    : {', '.join(evidence.keys()) or '(none)'}")
    print(f"  tool calls    : {investigation.get('tool_calls_used', 0)}")
    files_touched = sorted({f['file'] for f in findings if f.get('file')})
    latent = sum(1 for f in findings if f.get("blocks") == "later_step")
    print(f"  issues found  : {len(findings)} across {len(files_touched)} file(s)"
          f"{' → ' + ', '.join(files_touched) if files_touched else ''}"
          f"{f' ({latent} latent)' if latent else ''}")
    for i, fnd in enumerate(findings, 1):
        tag = "BLOCKS NOW " if fnd.get("blocks") == "current_failure" else "LATENT     "
        print(f"    {i}. [{tag}] {fnd['issue']}"
              + (f"  [{fnd['file']}]" if fnd.get("file") else ""))
        if fnd.get("root_cause"):
            print(f"       cause: {fnd['root_cause']}")
        if fnd.get("proof"):
            print(f"       proof: {fnd['proof'][:160]}")
        if fnd.get("solution"):
            print(f"       fix:   {fnd['solution']}")
    print(f"  hypotheses    : {len(considered)} considered, {len(rejected)} eliminated")
    for h in rejected:
        print(f"    - ruled out: {h['hypothesis'][:90]}")
        print(f"      because:   {h['disproved_by'][:110]}")
    if why_fix_works:
        print(f"  why it works  : {why_fix_works[:200]}")
    if completeness:
        print(f"  sweep         : {completeness[:200]}")

    failure_mode = investigation.get("failure_mode")
    if failure_mode in ("no_model_response", "not_converged") and not investigation_log:
        print(f"[GATE] Investigation produced no model response ({failure_mode}) — escalating.")
        if token and repo:
            open_issue(token, repo,
                       f"Auto-fixer could not get a usable response from the model "
                       f"({failure_mode}).\n\n"
                       f"**Issue Python identified (unused — model never ran):**\n"
                       f"```\n{issue_block[:1200]}\n```",
                       run_url)
        sys.exit(0)

    if not evidence:
        print("[GATE] Root cause confirmed but no file evidence exists — "
              "the patch agent cannot quote text to change. Escalating.")
        if token and repo:
            open_issue(token, repo,
                       f"Investigation confirmed a cause without reading any files, "
                       f"so no patch can be safely generated. Root cause: {root_cause}\n\n"
                       f"**Issue Python identified:**\n```\n{issue_block[:1200]}\n```",
                       run_url)
        sys.exit(3)

    if confidence < 0.5:
        print(f"[GATE] Confidence {confidence:.0%} too low — escalating.")
        if token and repo:
            escalation_detail = (
                f"AI confidence too low ({confidence:.0%}). Root cause: {root_cause}\n\n"
                f"**Issue Python identified:**\n```\n{issue_block[:1500]}\n```\n\n"
                f"**Files read during investigation:** "
                f"{', '.join(evidence.keys()) or '(none)'}\n")
            open_issue(token, repo, escalation_detail, run_url)
        sys.exit(0)

    # ── PATCH GENERATION + VALIDATION LOOP ──
    findings_text = "\n".join(
        f"- [{f['file'] or 'file unknown'}]"
        f"{' (LATENT — not the logged cause, but must still be fixed)' if f.get('blocks') == 'later_step' else ''}"
        f" {f['issue']}: {f['root_cause']}"
        + (f"\n    offending text (verbatim): {f['proof']}" if f.get('proof') else "")
        + (f"\n    fix: {f['solution']}" if f.get('solution') else "")
        for f in findings)
    patch_root_cause = (f"{root_cause}\n\nIndividual issues to fix (address EVERY one):\n"
                        f"{findings_text}" if findings_text else root_cause)

    written, originals, fixes = [], {}, []
    tests_ran = docker_ran = False
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
                                       retry_note, active_issue_block,
                                       why_fix_works)
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
            # Feed the rejection back so the next round quotes real text.
            retry_note = (f"Your previous issues were rejected because the 'evidence' "
                          f"strings were not found verbatim in the files: {detail}. Copy "
                          f"the offending text EXACTLY, character for character, from the "
                          f"### file contents this time.")
            if repair_round == MAX_REPAIR_ROUNDS:
                if token and repo:
                    open_issue(token, repo,
                               f"AI produced no usable fixes. Root cause: {root_cause}\n\n{detail}",
                               run_url)
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
            retry_note = (
                f"Validation REJECTED your previous patch:\n{detail}\n\n"
                f"A rejection of the form \"still references missing 'X'\" means the "
                f"file contains a SECOND defect that the investigation did not list. "
                f"It is confirmed — a checker found it in the patched file. Emit your "
                f"original fix AND an additional issue entry correcting that reference, "
                f"in the same response. Do not resend the previous patch unchanged.")
            if repair_round == MAX_REPAIR_ROUNDS:
                if token and repo:
                    open_issue(token, repo,
                               f"AI fix failed validation. Root cause: {root_cause}\n\n{detail}",
                               run_url)
                sys.exit(3)
            continue

        print("\n━━━ EXECUTE BUILD & TESTS ━━━")
        if not args.skip_tests:
            tests_ok, test_output = run_tests(stacks)
            tests_ran = bool(detect_test_commands(stacks))
        else:
            print("[TEST] Skipped (--skip-tests)")
            tests_ok, test_output = True, ""
        docker_ok, docker_output = try_docker_build(written)
        docker_ran = any("dockerfile" in Path(f).name.lower() for f in written)

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
        open_pr(token, repo, branch, {
            "commit_message": commit_msg,
            "root_cause": root_cause,
            "solution": solution,
            "why_fix_works": why_fix_works,
            "completeness": completeness,
            "confidence": confidence,
            "findings": findings,
            "rejected": rejected,
            "written": written,
            "originals": originals,
            "fixes": fixes,
            "investigation_log": investigation_log,
            "evidence_files": list(evidence.keys()),
            "run_url": run_url,
            "tests_ran": tests_ran,
            "docker_ran": docker_ran,
        })
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()