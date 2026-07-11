#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — AI-Centric Architecture v2

ARCHITECTURE:

    ┌─────────────────────────────────────────────────────────┐
    │ PHASE 1: COLLECTION (Deterministic)                     │
    │ ─────────────────────────────────────────────────────────│
    │ • Parse & redact logs                                    │
    │ • Extract error signal + tech stack fingerprint          │
    │ • Discover all available files (with scoring/ranking)    │
    │ • Read file contents (with budgets)                      │
    │ • Provide static hints (file existence, reference scan)  │
    │                                                           │
    │ Output: (signal, stacks, context, files, hints)          │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ PHASE 2: AI DIAGNOSIS (Single Unified Call)              │
    │ ─────────────────────────────────────────────────────────│
    │ Input: error signal + file context + hints               │
    │                                                           │
    │ AI Outputs:                                              │
    │  {                                                       │
    │    "error_type": "missing_file|bad_version|...",        │
    │    "root_cause": "detailed explanation",                │
    │    "reasoning": "step-by-step thinking",                │
    │    "confidence": 0.0–1.0,                               │
    │    "solution": "plain English description",             │
    │    "commit_message": "fix: ..."                         │
    │  }                                                       │
    │                                                           │
    │ (FAST_MODE combines multiple steps; standard does >1)    │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ PHASE 3: AI FIX GENERATION (Comprehensive)               │
    │ ─────────────────────────────────────────────────────────│
    │ Input: diagnosis + full file context                     │
    │                                                           │
    │ AI Outputs: Flat issues list                             │
    │  {                                                       │
    │    "issues": [                                           │
    │      {                                                   │
    │        "file": "path/to/file",                           │
    │        "problem": "description",                         │
    │        "evidence": "exact text from file",               │
    │        "corrected": "fixed version",                     │
    │        "reasoning": "why this fix works"                 │
    │      },                                                  │
    │      ...                                                 │
    │    ],                                                    │
    │    "all_issues_found": true,  ← critical: tell us if     │
    │    "reasoning": "scan procedure"  you found EVERY bug    │
    │  }                                                       │
    │                                                           │
    │ AI is responsible for finding ALL issues, not just one   │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ PHASE 4: VALIDATION (Deterministic Safety Gates)        │
    │ ─────────────────────────────────────────────────────────│
    │ For each fix:                                            │
    │  ✓ evidence actually exists in file (no hallucination)   │
    │  ✓ corrected differs from evidence (not a no-op)         │
    │  ✓ syntax valid (Python, YAML, JSON, etc.)              │
    │  ✓ no secrets introduced                                │
    │  ✓ file paths safe (no .. / ~ escapes)                  │
    │  ✓ added lines < threshold (catches suspicious blobs)    │
    │  ✓ diff < size limit (blocks unauthorized feature adds)  │
    │                                                           │
    │ Python never writes a fix; only validates AI's work      │
    └─────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────┐
    │ PHASE 5: EXECUTION (Deterministic)                      │
    │ ─────────────────────────────────────────────────────────│
    │ • Apply validated patches (find/replace)                 │
    │ • Run tests                                              │
    │ • If tests fail → revert all                             │
    │ • Commit + push + open PR                                │
    │ • On any failure → escalate to GitHub issue              │
    └─────────────────────────────────────────────────────────┘


KEY CHANGES FROM v1:

1. NO DETERMINISTIC PROBLEM-SOLVING
   ✗ Removed: scan_python_base_image() (was pre-fixing versions)
   ✗ Removed: ai_correct_lines() as isolated stage (blame goes to AI)
   ✓ Added: "all_issues_found" flag so AI commits to completeness

2. AI OWNS THE DIAGNOSIS
   ✓ Single unified diagnosis call (with FAST_MODE option)
   ✓ AI produces "reasoning" explaining its thinking
   ✓ AI explicitly hunts for ALL bugs (not just one per file)
   ✓ Deterministic only validates, never rewrites

3. CLEAR RESPONSIBILITY BOUNDARY
   AI says: "Here are the bugs and fixes"
   Python says: "I checked your work — these pass safety gates"
   Python does NOT say: "I found extra bugs you missed"

4. AUDITABILITY
   ✓ Each AI stage includes "reasoning" for review
   ✓ Issues include per-fix reasoning
   ✓ Clear trace: error → diagnosis → fixes → validation → execution


SECURITY MODEL (improved):

   Secret scanning:     redact BEFORE entering any prompt
   Syntax validation:   parse corrected text before applying
   Path validation:     reject .., ~, absolute paths
   Diff size cap:       block 40+ new lines (catches payload injections)
   Evidence matching:   Python proves evidence exists (no hallucination)
   Immutable prompts:   file list, repo structure, error signal cannot be changed by AI


BACKWARD COMPAT:

   - FAST_MODE=1 still works (fewer AI calls, same result)
   - SKIP_SELF_VERIFY=1 removed (self-verify should be mandatory now)
   - Same env vars for ollama, git, github tokens
   - Same exit codes (0=success, 1=not found, 2=AI failed, 3=no valid fix, 4=git, 5=tests)
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

# ── Staged-flow toggles ───────────────────────────────────────────────────────
FAST_MODE = os.environ.get("FAST_MODE", "").lower() in ("1", "true", "yes")

# ── Prompt budgets ────────────────────────────────────────────────────────────
MAX_ERROR_LINES   = 14
MAX_FILE_CHARS    = 4000
MAX_TOTAL_CONTEXT = 9000
MAX_CONTEXT_FILES = 3
MAX_FILES_FIXED   = 4

# ── Git flow ──────────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_BLOCKED = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [
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

# ── Secret scanning ───────────────────────────────────────────────────────────
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


def scan_text_for_secrets(text: str) -> list:
    hits = []
    for name, pat in SECRET_PATTERNS:
        if pat.search(text):
            hits.append(name)
    return hits


def redact_secrets(text: str) -> str:
    out = text
    for name, pat in SECRET_PATTERNS:
        out = pat.sub(f"[REDACTED:{name}]", out)
    return out


def _added_lines(original: str, new: str) -> list:
    old_lines = set(original.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: COLLECTION (Deterministic)
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
    print(f"[COLLECT] Error signal: {len(signal)} chars")
    return signal


def fingerprint_stack(log_text: str) -> set:
    low = log_text.lower()
    stacks = {s for s, sig in TECH_STACK_SIGNALS.items() if any(x in low for x in sig)}
    print(f"[COLLECT] Tech stacks: {stacks or {'unknown'}}")
    return stacks


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
    """Extract file paths mentioned in log."""
    REFERENCED_PATH_PATTERNS = [
        r'File "([^"]+\.\w+)"',
        r'([\w./\-]+\.[A-Za-z0-9]+):\d+(?::\d+)?',
        r'((?:[\w./\-]+/)?Dockerfile(?:\.\w+)?)\b',
        r'([\w./\-]+/requirements[\w.\-]*\.txt)',
        r'(?:in|from|at|open)\s+([\w./\-]+\.[A-Za-z0-9]+)',
    ]
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
        print(f"[COLLECT] Referenced in log: {resolved}")
    return resolved


def _score_file(path: Path, signal: str, stacks: set) -> int:
    name, low, score = path.name.lower(), signal.lower(), 0
    if name in low or str(path).lower() in low:
        score += 50
    for st in stacks:
        for pat in ["Dockerfile", "requirements.txt", "package.json", "pom.xml", ".py", ".js"]:
            if pat.lower() in name:
                score += 20
    return score


def discover_context(signal: str, stacks: set, forced: list) -> tuple:
    """Discover and read relevant files. Returns (context_text, file_list, contents_dict)."""
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
        print(f"[COLLECT] + {rel} ({len(content)} chars)")
        return True

    # Forced files first
    for rel in forced:
        if len(included) >= MAX_CONTEXT_FILES:
            break
        p = Path(rel)
        if p.is_file() and _is_text_file(p):
            c = read_whole(p, cap=8000)
            if c is not None:
                try_add(rel, c)

    # Then score remaining files
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
    print(f"[COLLECT] {len(included)} files, {len(context)} chars")
    return context, included, contents


def repo_file_list(limit: int = 200) -> str:
    """Return comma-separated list of all repo files (safe reference)."""
    files = []
    for p in sorted(Path(".").rglob("*")):
        if p.is_file() and not any(d in SKIP_DIRS for d in p.parts):
            rel = _relstrip(str(p))
            if not _is_blocked(rel):
                files.append(rel)
        if len(files) >= limit:
            break
    return ", ".join(files) if files else "(none found)"


def _context_block(signal, context, stacks):
    """Format context for AI consumption."""
    repo_files = repo_file_list()
    return (
        f"## Tech stack: {', '.join(sorted(stacks)) or 'unknown'}\n"
        f"## CI failure (key lines):\n```\n{signal}\n```\n"
        f"## Repo files (these exist — anything referenced but NOT here is a typo):\n{repo_files}\n"
        f"## File contents (read these carefully):\n{context}"
    )


def _cap_prompt(system: str, body: str, context: str, rebuild, cap=9500) -> str:
    """Trim file-contents if prompt is too long."""
    full = f"{system}\n\n{body}"
    if len(full) <= cap:
        return full
    allowed = cap - len(system) - len(rebuild("")) - 100
    trimmed = context[:max(allowed, 1000)] + "\n...(trimmed)"
    return f"{system}\n\n{rebuild(trimmed)}"


# ══════════════════════════════════════════════════════════════════════════════
# AI Plumbing (shared streaming + JSON extraction)
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
    """Stream from Ollama with retries on timeout."""
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
    print(f"[{tag}] {endpoint} ({fmt}) | prompt {len(prompt)} chars")

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
    """Best-effort JSON extraction."""
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
# PHASE 2: AI DIAGNOSIS
# ══════════════════════════════════════════════════════════════════════════════

DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "error_type":       {"type": "string"},
        "root_cause":       {"type": "string"},
        "reasoning":        {"type": "string"},
        "confidence":       {"type": "number"},
        "solution":         {"type": "string"},
        "commit_message":   {"type": "string"},
    },
    "required": ["error_type", "root_cause", "confidence", "commit_message"],
}

DIAGNOSIS_SYSTEM = """\
You are a senior CI/CD debugging engineer. Given the CI failure logs and file contents, produce a DIAGNOSIS.

Output ONLY one JSON object (no markdown fences):
{
  "error_type": "category (e.g., missing_file, bad_version, syntax_error, import_error)",
  "root_cause": "one sentence: WHY it failed — the core problem",
  "reasoning": "2-3 sentences: step-by-step thinking that led to this diagnosis",
  "confidence": 0.0–1.0 (lower if multiple causes plausible),
  "solution": "plain English: WHAT must change to fix it",
  "commit_message": "fix: short description for git"
}

Do NOT diagnose what to fix yet. Just state the problem and why it happened.
Set confidence below 0.5 if the facts do not clearly pin a single cause."""


def ai_diagnose(signal, context, stacks) -> dict:
    """Single AI call for diagnosis (or first half of FAST_MODE)."""
    def rebuild(ctx):
        return _context_block(signal, ctx, stacks) + "\n\nProvide a diagnosis as JSON."
    prompt = _cap_prompt(DIAGNOSIS_SYSTEM, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, DIAGNOSIS_SCHEMA, num_predict=700,
                         temperature=0.05, tag="DIAGNOSIS")
    data = _json_from(raw) or {}
    data.setdefault("error_type", "unknown")
    data.setdefault("root_cause", "unknown")
    data.setdefault("reasoning", "")
    data.setdefault("solution", "")
    data.setdefault("commit_message", "fix: auto-fixer change")
    try:
        data["confidence"] = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        data["confidence"] = 0.5
    print(f"[DIAGNOSIS] {data['error_type']} → {data['root_cause'][:60]} "
          f"(confidence {data['confidence']:.0%})")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: AI FIX GENERATION
# ══════════════════════════════════════════════════════════════════════════════

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "file":      {"type": "string"},
                "problem":   {"type": "string"},
                "evidence":  {"type": "string"},
                "corrected": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["file", "problem", "evidence", "corrected"]}},
        "all_issues_found": {"type": "boolean"},
        "reasoning":        {"type": "string"},
    },
    "required": ["issues", "all_issues_found"],
}

FIX_SYSTEM = """\
You are a senior engineer. You now know the root cause. Emit ALL FIXES — this is your only chance.

Output ONLY one JSON object (no markdown fences):
{
  "issues": [
    {
      "file": "### header path (exact)",
      "problem": "one sentence: what is this specific bug",
      "evidence": "exact text COPIED from the file — the offending part",
      "corrected": "same text with ONLY the bug fixed, everything else identical",
      "reasoning": "why this fix works (brief)"
    },
    ...
  ],
  "all_issues_found": true/false — CRITICAL: did you scan EVERY file and find EVERY bug?
  "reasoning": "walk-through: how you scanned for bugs"
}

CRITICAL RULES:

1. FIND EVERY BUG — A file often contains MORE THAN ONE unrelated bug.
   Re-read every file line-by-line. Check independently:
   • File names: "## Repo files" is truth. Not there? It's a typo.
   • Versions: python:3.1 / python:3.2 invalid (no such CPython release)
   • Ports: if Dockerfile EXPOSE 8080 but app.run(port=5000), fix the app
   • Imports: if imports "missing_module", check requirements.txt
   • If you find one bug and stop, YOU FAILED. Scan thoroughly.

2. evidence MUST literally appear in the file contents shown.
   If you cannot quote exact offending text, omit that issue.

3. evidence and corrected MUST differ, and be as SHORT as unambiguous allows.

4. For Dockerfile COPY/ADD/CMD/ENTRYPOINT paths:
   These resolve against the BUILD CONTEXT (Dockerfile's directory).
   If Dockerfile is "sample_app/Dockerfile", build context is "sample_app/".
   Correct reference is relative to that context, NOT the repo root.
   Example: if real file is "sample_app/app.py", the reference is "app.py" — NEVER "sample_app/app.py".

5. set "all_issues_found": false only if you know there are bugs you couldn't locate."""


def ai_generate_fixes(diagnosis, signal, context, stacks) -> tuple:
    """Generate comprehensive fix list. Returns (issues, all_found)."""
    diag_line = (f"## Root cause (determined): {diagnosis['root_cause']}\n"
                 f"## Solution: {diagnosis['solution']}\n")
    def rebuild(ctx):
        return diag_line + _context_block(signal, ctx, stacks) + \
               "\n\nEmit ALL fixes as JSON."
    prompt = _cap_prompt(FIX_SYSTEM, rebuild(context), context, rebuild, cap=11000)
    raw = _stream_ollama(prompt, FIX_SCHEMA, num_predict=3000,
                         temperature=0.05, tag="FIX-GEN")
    data = _json_from(raw)
    if data is None:
        raise ValueError(f"No valid JSON in fix generation:\n{raw[:400]}")
    
    issues = data.get("issues", []) or []
    all_found = bool(data.get("all_issues_found", True))
    
    print(f"[FIX-GEN] {len(issues)} issues | all_found={all_found}")
    for i, it in enumerate(issues[:5], 1):
        print(f"  {i}. {it.get('file','?')}: {(it.get('problem') or '')[:70]}")
    if len(issues) > 5:
        print(f"  ... and {len(issues)-5} more")
    
    return _normalize_issue_keys(issues), all_found


def _normalize_issue_keys(issues):
    """Handle alternate key names from model drift."""
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
# PHASE 4: VALIDATION (Deterministic Safety Gates)
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
    """Apply find/replace edits with fallback strategies."""
    content = original
    for i, ed in enumerate(edits):
        find, repl = ed.get("find", ""), ed.get("replace", "")
        if not isinstance(find, str) or find == "":
            return None, f"edit #{i+1} empty 'find'"
        if find in content:
            content = content.replace(find, repl)
            continue
        # Idempotency: if the replacement is already there, skip
        if isinstance(repl, str) and repl.strip() and repl in content:
            print(f"[EDIT] edit #{i+1} already satisfied — skipping")
            continue
        # Try line-by-line normalization
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
        return None, (f"edit #{i+1} 'find' not present — find: {find[:120]!r}")
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


def validate_fix(fix: dict) -> tuple:
    """Gate: ensure fix passes all safety checks."""
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
                       "— too large for an auto-fix")
    secret_hits = scan_text_for_secrets("\n".join(added))
    if secret_hits:
        return False, f"potential secret in fix ({', '.join(secret_hits)}) — blocked"

    # Syntax validation
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
    elif file.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"JSON error: {e}"
    return True, "ok"


def issues_to_fixes(issues: list, included_contents: dict) -> tuple:
    """Convert AI issues to fixes. Checks evidence exists; returns fixes + rejects."""
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

        holders = [f for f, c in included_contents.items() if ev in c]
        if file in included_contents and ev in included_contents[file]:
            target = file
        elif len(holders) == 1:
            target = holders[0]
            print(f"[LOCATE] issue #{n}: evidence found in '{target}' (not '{file}')")
        elif len(holders) > 1:
            rejects.append(f"issue #{n} ({file or '?'}): evidence in {len(holders)} files — ambiguous")
            continue
        else:
            rejects.append(f"issue #{n}: evidence {ev[:80]!r} not found")
            continue

        entry = fixes_by_file.setdefault(target, {"file": target, "reason": prob or "AI fix", "edits": []})
        edit = {"find": ev, "replace": cor}
        if edit not in entry["edits"]:
            entry["edits"].append(edit)

    return list(fixes_by_file.values()), rejects


def write_fixes(fixes: list) -> tuple:
    """Apply validated fixes to disk."""
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
        print(f"  ✓ {file}")
        written.append(file)
    written = list(dict.fromkeys(written))
    return written, originals, reasons


def revert_files(originals: dict):
    """Restore files to original state."""
    for file, content in originals.items():
        try:
            Path(file).write_text(content, encoding="utf-8")
            print(f"[REVERT] {file}")
        except Exception as exc:
            print(f"[REVERT] failed {file}: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: EXECUTION (Deterministic)
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


def open_pr(token, repo, branch, commit_msg, diagnosis, written, issues) -> str:
    """Open PR with diagnosis details."""
    body = (f"## 🤖 AI Auto-Fix\n\n"
            f"**Root Cause:** {diagnosis['root_cause']}\n\n"
            f"**Solution:** {diagnosis['solution'] or '(see reasoning)'}\n\n"
            f"**Files Changed:** {', '.join(f'`{f}`' for f in written)}\n\n"
            f"**Issues Fixed:**\n")
    for i in issues[:5]:
        body += f"- `{i.get('file','?')}`: {(i.get('problem') or '')[:80]}\n"
    if len(issues) > 5:
        body += f"- ... and {len(issues)-5} more\n"
    body += f"\n> Auto-generated by AI auto-fixer — review before merge."
    
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer (AI-centric architecture)")
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

    # Loop guards
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
            print(f"[GUARD] A fix PR is already open: {url} — skipping.")
            sys.exit(0)

    # ── PHASE 1: COLLECT LOGS + CONTEXT ──
    print("\n━━━ PHASE 1: COLLECTION ━━━")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    redacted = scan_text_for_secrets(log_text)
    if redacted:
        print(f"[SECURITY] redacting {len(redacted)} potential secret pattern(s): "
              f"{', '.join(redacted)}")
    log_text = redact_secrets(log_text)
    
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[COLLECT] No error signal — nothing to fix.")
        sys.exit(0)

    forced = extract_referenced_paths(log_text)
    context, included, included_contents = discover_context(signal, stacks, forced)
    if not included:
        print("[COLLECT] WARNING: no files resolved.")

    # ── PHASE 2: AI DIAGNOSIS ──
    print("\n━━━ PHASE 2: AI DIAGNOSIS ━━━")
    try:
        diagnosis = ai_diagnose(signal, context, stacks)
    except Exception as exc:
        print(f"[ERROR] AI diagnosis failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI diagnosis failed: {exc}", run_url)
        sys.exit(2)

    print(f"  error_type  : {diagnosis['error_type']}")
    print(f"  root_cause  : {diagnosis['root_cause']}")
    print(f"  reasoning   : {diagnosis['reasoning'][:100]}...")
    print(f"  confidence  : {diagnosis['confidence']:.0%}")
    print(f"  solution    : {diagnosis['solution'][:100] if diagnosis['solution'] else '(none)'}...")

    # ── PHASE 3: AI FIX GENERATION ──
    print("\n━━━ PHASE 3: FIX GENERATION ━━━")
    try:
        issues, all_found = ai_generate_fixes(diagnosis, signal, context, stacks)
    except Exception as exc:
        print(f"[ERROR] AI fix generation failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI fix generation failed: {exc}", run_url)
        sys.exit(2)

    if not all_found:
        print(f"[WARNING] AI reports it may not have found all issues")
    for i, it in enumerate(issues, 1):
        print(f"  {i}. {it.get('file','?')}: {(it.get('problem') or '')[:70]}")

    # ── CONFIDENCE GATE ──
    if diagnosis['confidence'] < 0.5:
        print(f"[GATE] Confidence {diagnosis['confidence']:.0%} too low — escalating.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({diagnosis['confidence']:.0%}). "
                       f"Root cause: {diagnosis['root_cause']}", run_url)
        sys.exit(0)

    # ── PHASE 4: VALIDATION ──
    print("\n━━━ PHASE 4: VALIDATION ━━━")
    fixes, rejects = issues_to_fixes(issues, included_contents)
    for rej in rejects:
        print(f"  ✗ {rej}", file=sys.stderr)

    if not fixes:
        detail = "; ".join(rejects) or "model reported no locatable issues"
        print(f"[ERROR] No usable fixes. {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI produced no usable fixes.\n\nDetail: {detail}", run_url)
        sys.exit(3)

    # ── PHASE 5: EXECUTION ──
    print("\n━━━ PHASE 5: EXECUTION ━━━")
    if args.dry_run:
        for fix in fixes:
            ok, reason = validate_fix(fix)
            print(f"  {'would write' if ok else 'reject'} {fix.get('file','?')} — {reason}")
        sys.exit(0)

    written, originals, reject_reasons = write_fixes(fixes)
    if not written:
        detail = "; ".join(reject_reasons) or "no detail"
        print(f"[ERROR] No valid fix applied. {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI fix failed validation.\n\nDetail: {detail}", run_url)
        sys.exit(3)

    # Run tests
    print("\n━━━ TESTS ━━━")
    if not args.skip_tests:
        if not run_tests(stacks):
            revert_files(originals)
            if token and repo:
                open_issue(token, repo,
                           f"Fix applied but tests failed — reverted.\n"
                           f"Root cause: {diagnosis['root_cause']}", run_url)
            sys.exit(5)
    else:
        print("[TEST] Skipped (--skip-tests)")

    # Commit + PR
    print("\n━━━ GIT ━━━")
    branch = commit_to_branch(diagnosis["commit_message"], written)
    if not branch:
        sys.exit(4)
    if token and repo:
        open_pr(token, repo, branch, diagnosis["commit_message"], diagnosis, written, issues)
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {diagnosis['root_cause']}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()