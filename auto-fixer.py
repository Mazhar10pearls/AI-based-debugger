#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — straight AI flow.

Flow (matches the pipeline diagram, nothing more):

    Read logs (Python)
        → Detect tech stack (Python)
        → Discover files (Python)
        → Build prompt (Python)
        → Send to Ollama  ══ AI MODEL reasons + generates code changes ══
        → Receive JSON
        → Apply files (Python)
        → Validate syntax (Python)
        → Run tests (Python)
            → Tests pass → Commit + Push + PR
            → Tests fail → Create Issue

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
# A small local model reliably fixes ONE bug per call, even when a file has
# several and is told so explicitly. Rather than rely on a single shot being
# exhaustive, re-ask up to this many times, re-scanning between rounds — each
# round only has to do what the model already does reliably: fix one thing.
MAX_AI_ROUNDS  = 3

# ── Prompt / context budget ───────────────────────────────────────────────────
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
MAX_BOT_ATTEMPTS = 3          # stop the bot fixing its own commits forever

# Never let the AI touch its own tooling.
ALWAYS_BLOCKED   = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [r"\.?github/workflows/auto-fix.*\.ya?ml$",
                    r"\.?github/workflows/self-heal.*\.ya?ml$"]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             "dist", "build", ".pytest_cache", "target", "out", "vendor",
             ".idea", ".vscode", "coverage", "tmp", "temp", "logs"}
MAX_FILE_SIZE_BYTES = 100_000


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — READ LOGS + DETECT TECH STACK
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
# STAGE 2 — DISCOVER FILES
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
    """
    Fallback candidate source for failures that name NO real source file in
    the log — e.g. a fast setup-action failure (bad python-version) that dies
    before pip/build/test ever runs, so nothing in the log points at app code.
    By elimination, a failure with no file trace is a workflow-configuration
    problem, so the repo's own CI workflow YAML(s) become the prime suspect.
    Not tied to any specific bug or filename — this is a general "when nothing
    else is referenced, check the thing that defines the pipeline" rule.

    Excludes the self-heal workflow itself (detected generically by content,
    not by a hardcoded filename) so the fixer never targets its own definition.
    """
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
            continue  # this IS the workflow that runs us — never a fix target
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


# ── static reference scan (hints only — AI still decides what's real) ──────────
# A small local model reliably finds the FIRST bug in a file and then stops.
# This does not diagnose or fix anything itself; it just extracts every file
# path a Dockerfile/compose file references and checks whether that exact path
# exists in the repo, so a second (or third) broken reference in the same file
# can't slip past the model unnoticed. The AI still verifies each candidate,
# decides if it's a real problem, and writes the actual fix.
DOCKERFILE_REF_PATTERNS = [
    (re.compile(r'^\s*COPY\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "COPY"),
    (re.compile(r'^\s*ADD\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "ADD"),
    (re.compile(r'-r\s+(\S+\.txt)'), "pip install -r"),
    (re.compile(r'CMD\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "CMD"),
    (re.compile(r'ENTRYPOINT\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "ENTRYPOINT"),
]


def scan_reference_hints(included_contents: dict) -> str:
    all_repo = [_relstrip(str(p)) for p in Path(".").rglob("*")
                if p.is_file() and not any(d in SKIP_DIRS for d in p.parts)]
    basenames = {}
    for f in all_repo:
        basenames.setdefault(Path(f).name.lower(), []).append(f)
    all_repo_set = set(all_repo)
    lines = []
    for rel, content in included_contents.items():
        name = Path(rel).name.lower()
        if "dockerfile" not in name and "docker-compose" not in name and "compose.y" not in name:
            continue
        docker_dir = Path(rel).parent  # COPY/ADD/-r paths resolve relative to the build
        for pat, label in DOCKERFILE_REF_PATTERNS:              # context (the Dockerfile's own dir), not the repo root
            for m in pat.finditer(content):
                ref = m.group(1).strip().strip("'\"")
                if not ref or ref in (".", "..") or ref.startswith("-") or ref.startswith("$"):
                    continue
                ref_clean = ref.lstrip("./")
                candidates = {ref_clean, _relstrip(str(docker_dir / ref_clean))}
                if any(Path(c).is_file() or c in all_repo_set for c in candidates):
                    continue
                close = difflib.get_close_matches(Path(ref_clean).name.lower(),
                                                   basenames.keys(), n=1, cutoff=0.4)
                suggestion = ", ".join(basenames[close[0]]) if close else "no close match in repo"
                lines.append(f"- {rel}: {label} references '{ref}' — NOT found in repo. "
                             f"Closest existing file: {suggestion}")
    hints = "\n".join(dict.fromkeys(lines))  # de-dupe, preserve order
    if hints:
        print(f"[DISCOVER] Static reference hints:\n{hints}")
    return hints


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — BUILD PROMPT + SEND TO OLLAMA (AI)
# ══════════════════════════════════════════════════════════════════════════════

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "error_analysis": {"type": "string"},
        "root_cause":     {"type": "string"},
        "solution":       {"type": "string"},
        "confidence":     {"type": "number"},
        "commit_message": {"type": "string"},
        "fixes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "file":   {"type": "string"},
                "reason": {"type": "string"},
                "edits":  {"type": "array", "items": {
                    "type": "object",
                    "properties": {"find": {"type": "string"},
                                   "replace": {"type": "string"}},
                    "required": ["find", "replace"]}},
                "fixed_content": {"type": "string"},
            },
            "required": ["file", "reason"]}},
    },
    "required": ["error_analysis", "root_cause", "solution",
                 "confidence", "commit_message", "fixes"],
}

SYSTEM_PROMPT = """\
You are a senior CI/CD debugging engineer. A pipeline failed. YOU diagnose it and YOU fix it. Output ONLY one JSON object. No markdown fences. Start with { end with }.

Think in this exact order, then emit the JSON:
1. error_analysis — state precisely WHAT the error is and WHERE it lives (which file, which line or token). Quote the offending text.
2. root_cause — one sentence: WHY it failed.
3. solution — state in plain words WHAT must change to fix it.
4. fixes — the concrete edits that implement your solution.

Schema:
{"error_analysis":"...","root_cause":"...","solution":"...","confidence":0.0-1.0,"commit_message":"fix: short","fixes":[{"file":"exact/path","reason":"what changed","edits":[{"find":"exact text from the file","replace":"corrected text"}]}]}

HOW TO REASON (do this yourself — nothing is pre-solved for you):
- A "## Repo files (these exist)" list is given. If a Dockerfile/workflow/script references a file whose exact name is NOT in that list, that reference is a TYPO — correct it to the closest real name from the list. NEVER create the missing file.
- Valid Python versions are 3.8 through 3.13. Tags like python:3.1, python:3.2, or python-version "3.1"/"3.2" are INVALID — replace with 3.12.
- If the app's `app.run(port=N)` disagrees with the Dockerfile `EXPOSE M`, change the app to bind M.
- If you change an `if __name__ == "__main__":` block, it MUST still start a long-running server (app.run(host="0.0.0.0", port=<EXPOSE port>)). Never leave it empty.
- A "## Static reference check" section may be given, listing filenames a script references that a naive scan couldn't find in the repo. This is a HINT ONLY, not a diagnosis — verify each one yourself against "## Repo files", and IGNORE any hint that turns out to be a false positive (e.g. a build arg, a URL, or a file created earlier in the same Dockerfile).

MULTIPLE BUGS IN ONE FILE — READ THIS CAREFULLY:
A single file routinely has MORE THAN ONE unrelated bug (e.g. one line references a misspelled requirements file AND a separate line runs the wrong script name). Finding one bug and stopping is a FAILURE. After drafting a fix, go back and re-read the ENTIRE file one more time, line by line, treating it as if you had not looked at it yet — list every remaining line that references a file, a version, or a port, and check each one independently. Every distinct bug in a file goes into that SAME file's `edits` array as its own {{"find":...,"replace":...}} entry — do not open a second fixes[] entry for a file you already opened one for.

RULES:
- Pick ONE mode per file: "edits" (find/replace; `find` copied EXACTLY from the file, kept short — ideally the single wrong token) OR "fixed_content" (the COMPLETE corrected small file).
- file = exact path from a ### header. You may ONLY edit files shown under "## File contents" — never name a file with no ### header, never create a new file.
- Every fixes[] entry MUST carry a non-empty "edits" or "fixed_content". If you decide a file doesn't need a change after all, omit it entirely — don't include an empty entry.
- Preserve everything correct. Change only what is broken.
- Set confidence below 0.5 if you are unsure, rather than guessing."""

USER_PROMPT = """\
## Tech stack: {stacks}
## CI failure (key lines):
```
{signal}
```
## Repo files (these exist — anything referenced but NOT in this list is a typo):
{repo_files}
{hints_section}
## File contents (you may ONLY edit these):
{context}"""


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


def build_prompt(signal, context, stacks, hints=""):
    repo_files = repo_file_list()
    hints_section = f"## Static reference check (verify each — not authoritative):\n{hints}\n" if hints else ""
    def fmt(ctx):
        return USER_PROMPT.format(stacks=", ".join(sorted(stacks)) or "unknown",
                                  signal=signal, repo_files=repo_files,
                                  hints_section=hints_section, context=ctx)
    user = fmt(context)
    cap = 9500
    if len(SYSTEM_PROMPT + "\n\n" + user) > cap:
        allowed = cap - len(SYSTEM_PROMPT) - len(fmt("")) - 100
        user = fmt(context[:max(allowed, 1000)] + "\n...(trimmed)")
    return user


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


def call_ai(signal, context, stacks, hints="") -> str:
    prompt = f"{SYSTEM_PROMPT}\n\n{build_prompt(signal, context, stacks, hints)}"
    endpoint, fmt = _detect_endpoint()
    if fmt == "openai":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": 0.05,
                   "max_tokens": 2800, "stream": True}
    else:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "format": FIX_SCHEMA,
                   "options": {"temperature": 0.05, "num_predict": 2800,
                               "num_ctx": 8192}, "stream": True}
    print(f"[AI] {endpoint} ({fmt}) | prompt {len(prompt)} chars | model {OLLAMA_MODEL}")

    last = None
    for attempt in range(MAX_RETRIES):
        try:
            t0, collected = time.time(), []
            resp = requests.post(endpoint, json=payload,
                                 timeout=(10, AI_TIMEOUT), stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                tok = _extract_token(line, fmt)
                if tok:
                    collected.append(tok)
            raw = "".join(collected).strip()
            print(f"[AI] Done in {time.time()-t0:.1f}s — {len(raw)} chars")
            if not raw:
                try:
                    body = resp.json()
                    raw = (body.get("choices", [{}])[0].get("text", "")
                           if fmt == "openai" else body.get("response", "")).strip()
                except Exception:
                    pass
            if not raw:
                raise RuntimeError("Ollama returned an empty response.")
            print(f"[AI] Preview: {raw[:300]}")
            return raw
        except requests.exceptions.Timeout as exc:
            last = exc
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
            print(f"[AI] Timeout; waiting {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {endpoint}: {exc}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}")
    raise RuntimeError(f"Ollama did not respond after {MAX_RETRIES} attempts: {last}")


# ══════════════════════════════════════════════════════════════════════════════
#   RECEIVE JSON
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_keys(fixes):
    KF = ("file_path", "path", "filename", "filepath", "name")
    KC = ("content", "new_content", "fixed", "updated_content", "code")
    out = []
    for f in fixes:
        f = dict(f)
        if "file" not in f:
            for a in KF:
                if a in f:
                    f["file"] = f.pop(a); break
        if "fixed_content" not in f and "edits" not in f:
            for a in KC:
                if a in f:
                    f["fixed_content"] = f.pop(a); break
        out.append(f)
    return out


def parse_ai_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    if data is None:
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
                data = json.loads(best)
            except json.JSONDecodeError:
                try:
                    data = json.loads(best + "}" * (best.count("{") - best.count("}")))
                except json.JSONDecodeError:
                    pass
    if data is None:
        raise ValueError(f"No valid JSON in AI response:\n{raw[:500]}")
    if isinstance(data.get("fixes"), list):
        data["fixes"] = _normalize_keys(data["fixes"])
    return data


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — APPLY FILES + VALIDATE SYNTAX
# ══════════════════════════════════════════════════════════════════════════════

def _salvage_fragment(content: str, find: str, replace: str) -> str | None:
    """
    Last-resort recovery when `find` doesn't match verbatim. The model often
    paraphrases the surrounding line/quotes but gets the actual changed TOKEN
    right. Strip the longest common prefix/suffix of find vs replace to isolate
    just the changed fragment; apply it ONLY if that fragment occurs exactly
    once in the file (never ambiguous, never a guess).
    """
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
        # whitespace-tolerant fallback
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
        # final salvage: isolate the actual changed fragment, apply only if unique
        salvaged = _salvage_fragment(content, find, repl)
        if salvaged is not None and salvaged != content:
            print(f"[EDIT] Salvaged edit #{i+1} via unique-fragment match "
                  f"(model's 'find' did not match the file verbatim)")
            content = salvaged
            continue
        return None, (f"edit #{i+1} 'find' text not present in file — "
                      f"model's find: {find[:120]!r}")
    if content == original:
        return None, "edits produced no change"
    return content, "ok"


def _resolve_content(fix: dict) -> tuple:
    file = (fix.get("file") or "").strip()
    if fix.get("edits"):
        p = Path(file)
        if not p.is_file():
            return None, f"file does not exist: {file}"
        return _apply_edits(p.read_text(encoding="utf-8", errors="replace"),
                            fix["edits"])
    fc = fix.get("fixed_content")
    if isinstance(fc, str) and fc.strip():
        return fc, "ok"
    return None, "no 'edits' or 'fixed_content'"


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
    content, reason = _resolve_content(fix)
    if content is None:
        return False, reason
    fix["fixed_content"] = content
    if not content.strip():
        return False, "empty result"
    # ── validate syntax ──
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



def write_fixes(fixes: list) -> tuple:
    """Apply the AI's fixes exactly as returned. Python rejects invalid fixes
    (unknown file, bad syntax, unmatched find) but NEVER writes a fix of its
    own — the model's JSON is the only source of any change."""
    written, originals, reasons = [], {}, []
    for fix in fixes[:MAX_FILES_FIXED]:
        ok, reason = validate_fix(fix)
        file = fix.get("file", "").strip()

        if not ok:
            print(f"  ✗ {file or '?'} — {reason}")          # stdout: always visible in Action logs
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

    written = list(dict.fromkeys(written))   # same file can be fixed by >1 entry
    return written, originals, reasons


def revert_files(originals: dict):
    for file, content in originals.items():
        try:
            Path(file).write_text(content, encoding="utf-8")
            print(f"[REVERT] {file}")
        except Exception as exc:
            print(f"[REVERT] failed {file}: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — RUN TESTS
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
# STAGE 6 — COMMIT + PUSH + PR  /  CREATE ISSUE
# ══════════════════════════════════════════════════════════════════════════════

def _git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def last_commit_was_bot() -> bool:
    try:
        author, subject = _git("log", "-1", "--pretty=%an|||%s").stdout.strip().split("|||", 1)
        if author == BOT_NAME and subject.startswith(BOT_PREFIX):
            print(f"[GUARD] Last commit was bot — stopping.")
            return True
    except Exception:
        pass
    return False


def count_recent_bot_commits(n=10) -> int:
    """
    Count a CONSECUTIVE run of bot commits walking back from HEAD — stops at the
    first human commit. This distinguishes a genuine stuck loop (bot fixes →
    fails → bot fixes → fails, back to back with nothing in between) from a
    healthy history where 3+ PAST bot fixes were reviewed and merged by a human
    at various points. Counting "any N bot commits in recent history" (the old
    behavior) permanently blocks the fixer forever once enough fixes have
    accumulated and been merged — that's not a loop, that's the system working.
    """
    try:
        lines = _git("log", f"-{n}", "--pretty=%an").stdout.splitlines()
        count = 0
        for author in lines:
            if author.strip() == BOT_NAME:
                count += 1
            else:
                break  # a human commit breaks the streak — not a loop
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
            error_analysis="", solution="") -> str:
    details = "".join(f"\n**`{f.get('file','?')}`** — {f.get('reason','')}\n" for f in fixes)
    diag = ""
    if error_analysis or solution:
        diag = (f"### AI diagnosis\n"
                f"**Error:** {error_analysis or 'n/a'}\n\n"
                f"**Solution:** {solution or 'n/a'}\n\n")
    body = (f"## 🤖 AI Auto-Fix\n\n{diag}"
            f"**Root cause:** {root_cause}\n\n"
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
    """URL of an already-open bot fix PR, or ''. If a fix is awaiting review,
    calling the LLM again just stacks duplicate PRs — skip the call entirely."""
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
# MAIN — the straight flow
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer")
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

    # loop guard
    if last_commit_was_bot():
        sys.exit(0)
    if count_recent_bot_commits() >= MAX_BOT_ATTEMPTS:
        print("[GUARD] Too many bot attempts — escalating.")
        if token and repo:
            open_issue(token, repo, "Auto-fixer attempted too many fixes without success.", run_url)
        sys.exit(0)
    # minimum-LLM-call guard: a fix is already awaiting review — don't stack another
    if token and repo:
        url = pending_bot_pr(token, repo)
        if url:
            print(f"[GUARD] A fix PR is already open awaiting review: {url}")
            print("[GUARD] Skipping the model call — merge or close that PR first.")
            sys.exit(0)

    # ── STAGE 1: read logs + detect ──
    print("\n━━━ READ LOGS + DETECT ━━━")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[DETECT] No error signal — nothing to fix.")
        sys.exit(0)

    # ── STAGE 2: discover files ──
    print("\n━━━ DISCOVER FILES ━━━")
    forced = extract_referenced_paths(log_text)
    if not forced:
        wf = find_ci_workflow_files()
        if wf:
            print(f"[DISCOVER] No source file referenced in the log — falling back "
                  f"to the CI workflow as the prime suspect: {wf}")
            forced = wf
    context, included, included_contents = discover_context(signal, stacks, forced)
    if not included:
        print("[DISCOVER] WARNING: no files resolved.")
    hints = scan_reference_hints(included_contents)

    # ── STAGE 3: build prompt + send to AI + receive JSON ──
    print("\n━━━ SEND TO AI ━━━")
    try:
        raw = call_ai(signal, context, stacks, hints)
        data = parse_ai_response(raw)
    except Exception as exc:
        print(f"[ERROR] AI failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI analysis failed: {exc}", run_url)
        sys.exit(2)

    error_analysis = data.get("error_analysis", "")
    root_cause = data.get("root_cause", "unknown")
    solution   = data.get("solution", "")
    commit_msg = data.get("commit_message", "fix: auto-fixer change")
    confidence = float(data.get("confidence", 1.0))
    fixes = data.get("fixes", []) or []

    # grounding gate: the model may ONLY touch files it was actually shown —
    # a fix for any other path is by definition a hallucination (it never saw
    # that file's contents), like the invented 'sample_app/Dockerfiless'.
    grounded = []
    for f in fixes:
        fp = (f.get("file") or "").strip()
        if fp in included:
            grounded.append(f)
        else:
            print(f"  ✗ {fp or '?'} — rejected: model was never shown this file "
                  f"(hallucination guard)", file=sys.stderr)
    fixes = grounded

    print("\n  ── AI diagnosis ──")
    print(f"  ERROR    : {error_analysis or '(none given)'}")
    print(f"  CAUSE    : {root_cause}")
    print(f"  SOLUTION : {solution or '(none given)'}")
    print(f"  confidence : {confidence:.0%}")
    print(f"  fixes      : {len(fixes)} file(s)")

    if confidence < 0.5:
        print(f"[GATE] Confidence {confidence:.0%} too low — escalating instead of guessing.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({confidence:.0%}). Root cause: {root_cause}", run_url)
        sys.exit(0)

    if not fixes:
        print("[ERROR] AI returned no grounded fixes.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI produced no usable fixes. Root cause: {root_cause}", run_url)
        sys.exit(3)

    # ── STAGE 4: apply files + validate syntax ──
    print("\n━━━ APPLY + VALIDATE ━━━")
    if args.dry_run:
        for fix in fixes:
            ok, reason = validate_fix(fix)
            print(f"  {'would write' if ok else 'reject'} {fix.get('file','?')} — {reason}")
        sys.exit(0)

    written, originals, reject_reasons = write_fixes(fixes)
    if not written:
        detail = "; ".join(reject_reasons) or "no detail captured"
        print(f"[ERROR] No valid fix applied. {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI fix failed validation. Root cause: {root_cause}\n\n"
                       f"Rejection detail: {detail}", run_url)
        sys.exit(3)

    # ── STAGE 4b: additional rounds — catch bugs round 1 missed ──
    # If the static scanner still finds an unresolved reference mismatch after
    # round 1's fix was applied, there's proof more remains to fix, so ask the
    # model again against the now-partially-fixed files. Stops the moment a
    # round finds nothing left to flag, the model returns no new grounded fix,
    # or a round repeats the exact same unresolved hints as the round before
    # (model is stuck on it — escalating to another call won't help).
    prev_hints = None
    for round_no in range(2, MAX_AI_ROUNDS + 1):
        r_context, r_included, r_contents = discover_context(signal, stacks, forced)
        r_hints = scan_reference_hints(r_contents)
        if not r_hints:
            print(f"[LOOP] Static scan is clean after round {round_no - 1} — no more rounds needed.")
            break
        if r_hints == prev_hints:
            print(f"[LOOP] Round {round_no - 1} left the same issue(s) flagged — "
                  f"model isn't resolving it, stopping.")
            break
        prev_hints = r_hints

        print(f"\n━━━ SEND TO AI (round {round_no}/{MAX_AI_ROUNDS}) ━━━")
        try:
            raw_r = call_ai(signal, r_context, stacks, r_hints)
            data_r = parse_ai_response(raw_r)
        except Exception as exc:
            print(f"[LOOP] round {round_no} AI call failed: {exc} — stopping.")
            break

        fixes_r = data_r.get("fixes", []) or []
        grounded_r = []
        for f in fixes_r:
            fp = (f.get("file") or "").strip()
            if fp in r_included:
                grounded_r.append(f)
            else:
                print(f"  ✗ {fp or '?'} — rejected: model was never shown this file "
                      f"(hallucination guard)", file=sys.stderr)

        conf_r = float(data_r.get("confidence", 1.0))
        print(f"  round {round_no} confidence: {conf_r:.0%}, fixes: {len(grounded_r)} file(s)")
        if conf_r < 0.5 or not grounded_r:
            print(f"[LOOP] Nothing more to apply this round — stopping.")
            break

        print(f"\n━━━ APPLY + VALIDATE (round {round_no}) ━━━")
        written_r, originals_r, _ = write_fixes(grounded_r)
        if not written_r:
            print(f"[LOOP] Round {round_no} produced no valid fix — stopping.")
            break
        for f in written_r:
            originals.setdefault(f, originals_r[f])
        written = list(dict.fromkeys(written + written_r))
        fixes = fixes + grounded_r

    # ── STAGE 5: run tests ──
    print("\n━━━ RUN TESTS ━━━")
    if not args.skip_tests:
        if not run_tests(stacks):
            revert_files(originals)
            if token and repo:
                open_issue(token, repo, f"Fix applied but tests failed — reverted. Root cause: {root_cause}", run_url)
            sys.exit(5)
    else:
        print("[TEST] Skipped (--skip-tests)")

    # ── STAGE 6: commit + push + PR  /  issue ──
    print("\n━━━ COMMIT + PR ━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)
    if token and repo:
        open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
                error_analysis, solution)
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()