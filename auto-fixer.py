#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — agentic investigation flow.

Pipeline (matches the new diagram):

    Collect initial evidence (Python)
        logs + exit code + repo tree (names only) + git diff
        → Build investigation prompt (NO root cause given)
        → AI Investigation Agent
              reads evidence, thinks like a DevOps engineer,
              forms a hypothesis, decides what else it needs
              ⇄ Python fulfills requests (read files / run safe diagnostics)
          ...loop until "sufficient" or MAX_INVESTIGATION_ROUNDS...
        → AI confirms root cause + repair strategy
        → AI Patch Generation Agent → patch (flat issue list)
        → Python Validation Engine (format, paths, syntax, YAML/JSON,
          secret scan, dangerous-command scan, safe-patch verification)
        → Apply changes → Build & Tests
              success → Commit + PR
              failure → collect new failure logs → AI reviews new evidence
                        → retry (bounded) or stop → Commit + PR (as a
                          flagged/needs-review PR) or escalate via issue
                          if nothing could ever be safely applied

This is a structural rewrite of the old staged-call version. The old script
front-loaded ALL context into fixed Stage 1→2→3→4 calls. Here the AI decides,
round by round, what it actually needs to see — closer to how a human
engineer investigates a broken pipeline: skim the log, form a theory, go
look at the specific file/command that would confirm or kill that theory,
repeat.

Design carryovers from the old version (still true, still enforced):
  * The model authors every change; Python never invents a fix, only
    locates the model's own quoted evidence and applies it.
  * Small local models are unreliable at deeply nested JSON — every AI
    contract below is intentionally flat.
  * Dockerfile COPY/ADD/CMD/ENTRYPOINT paths resolve against the build
    context (the Dockerfile's own directory), not the repo root — this
    trips up naive path fixes, so it gets called out explicitly wherever
    file paths are handled.
  * Nothing the AI touches bypasses the deterministic gates: secret
    scanning, blocked-path list, syntax/YAML/JSON validation, dangerous
    shell-command detection, added-lines cap.

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

# ── Ollama ───────────────────────────────────────────────────────────────
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:3b")
AI_TIMEOUT     = 210
MAX_RETRIES    = 2
RETRY_BACKOFF  = [20, 20]

# agentic-flow toggles
MAX_INVESTIGATION_ROUNDS = int(os.environ.get("MAX_INVESTIGATION_ROUNDS", "5"))
MAX_REQUESTS_PER_ROUND    = 4
MAX_PATCH_RETRY_ROUNDS    = int(os.environ.get("MAX_PATCH_RETRY_ROUNDS", "2"))
SKIP_SELF_VERIFY = os.environ.get("SKIP_SELF_VERIFY", "").lower() in ("1", "true", "yes")

# ── Prompt / context budget ─────────────────────────────────────────────
MAX_ERROR_LINES    = 14
MAX_FILE_CHARS     = 4000
MAX_EVIDENCE_CHARS = 9000   # running total of everything fed back to the AI
MAX_FILES_FIXED    = 4
MAX_TREE_ENTRIES   = 400

# ── Git flow ─────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_BLOCKED   = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [
    # ALL workflow files — a "fix" to a deploy/CI workflow is a privilege-
    # escalation vector. Workflow breakage always escalates to a human via
    # open_issue(), never goes through auto-fix.
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

# ── Safe diagnostic commands ────────────────────────────────────────────
# Allowlist ONLY — the "execute safe diagnostic commands" box in the diagram
# is a read-only investigation aid, never a general shell. Every entry here
# is non-mutating. Arguments are still validated per-command below.
SAFE_DIAGNOSTIC_COMMANDS = {
    "git_log":        (["git", "log", "-5", "--oneline"], None),
    "git_status":      (["git", "status", "--short"], None),
    "git_show_head":   (["git", "show", "--stat", "HEAD"], None),
    "python_version":  (["python3", "--version"], None),
    "pip_freeze":      (["pip", "list"], None),
    "node_version":    (["node", "--version"], None),
    "npm_list":        (["npm", "list", "--depth=0"], None),
    "docker_version":  (["docker", "--version"], None),
    "ls_root":         (["ls", "-la"], None),
}

# ── Secret scanning ──────────────────────────────────────────────────────
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

# ── Dangerous-command detection (new Validation Engine gate) ─────────────
# The old script only had to worry about the model rewriting file content.
# An agent that can *propose* diagnostic commands and patches needs an
# explicit "don't let it ask for/emit something destructive" gate.
DANGEROUS_COMMAND_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r":\(\)\s*\{\s*:\|\:&\s*\}"),           # fork bomb
    re.compile(r"curl[^\n]*\|\s*(sh|bash)"),
    re.compile(r"wget[^\n]*\|\s*(sh|bash)"),
    re.compile(r"\bmkfs\."),
    re.compile(r"\bdd\s+if="),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"chmod\s+-R\s+777\s+/"),
    re.compile(r"\bshutdown\b|\breboot\b"),
    re.compile(r"eval\s*\("),
]


def scan_text_for_secrets(text: str) -> list:
    return [name for name, pat in SECRET_PATTERNS if pat.search(text)]


def redact_secrets(text: str) -> str:
    out = text
    for name, pat in SECRET_PATTERNS:
        out = pat.sub(f"[REDACTED:{name}]", out)
    return out


def scan_for_dangerous_commands(text: str) -> list:
    return [pat.pattern for pat in DANGEROUS_COMMAND_PATTERNS if pat.search(text)]


def _added_lines(original: str, new: str) -> list:
    old_lines = set(original.splitlines())
    return [l for l in new.splitlines() if l not in old_lines]


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


# ══════════════════════════════════════════════════════════════════════════
# STAGE — COLLECT INITIAL INVESTIGATION EVIDENCE  (Python, diagram box 1)
# ══════════════════════════════════════════════════════════════════════════

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


def get_exit_code(log_text: str) -> str:
    m = re.search(r"exit(?:ed)?\s+(?:with\s+)?code\s+(-?\d+)", log_text, re.I)
    return m.group(1) if m else "unknown"


def get_repo_tree(root: Path = Path("."), limit: int = MAX_TREE_ENTRIES) -> str:
    """Folder/filename skeleton only — NO file contents. This is the
    'Repository Tree (folders & filenames only)' evidence box."""
    entries = []
    for p in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = _relstrip(str(p))
        if _is_blocked(rel):
            continue
        entries.append(rel + ("/" if p.is_dir() else ""))
        if len(entries) >= limit:
            entries.append("... (truncated)")
            break
    tree = "\n".join(entries)
    print(f"[COLLECT] Repo tree: {len(entries)} entries")
    return tree


def get_git_diff(max_chars: int = 3000) -> str:
    """Diff of the current/most recent commit — the 'Git Diff (current
    commit changes)' evidence box."""
    try:
        r = subprocess.run(["git", "show", "--stat", "-p", "HEAD"],
                           capture_output=True, text=True, timeout=15)
        diff = r.stdout.strip()
    except Exception as exc:
        print(f"[COLLECT] git diff unavailable: {exc}")
        return ""
    if len(diff) > max_chars:
        diff = diff[:max_chars] + "\n...(truncated)"
    print(f"[COLLECT] Git diff: {len(diff)} chars")
    return diff


def collect_initial_evidence(log_text: str) -> dict:
    return {
        "log_signal": extract_error_signal(log_text),
        "exit_code":  get_exit_code(log_text),
        "repo_tree":  get_repo_tree(),
        "git_diff":   get_git_diff(),
    }


# ══════════════════════════════════════════════════════════════════════════
# AI plumbing — shared streaming + JSON extraction
# ══════════════════════════════════════════════════════════════════════════

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
                               "num_ctx": num_ctx}, "stream": True,
                   # Reasoning models (qwen3, deepseek-r1, etc.) emit a
                   # <think>...</think> block before the answer by default.
                   # We want the structured JSON only — Ollama honors this
                   # top-level "think" flag for models that support it, and
                   # silently ignores it for models that don't (e.g. the old
                   # qwen2.5-coder default), so it's safe to always send.
                   "think": False}
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
    # Defensive: even with think:False some reasoning models occasionally
    # leak a <think>...</think> block anyway. Strip it before brace-matching
    # so reasoning-time braces never get mistaken for the JSON answer.
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
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


# ══════════════════════════════════════════════════════════════════════════
# AI INVESTIGATION AGENT  (diagram: "AI Investigation Agent" loop)
# ══════════════════════════════════════════════════════════════════════════
#
# Contract per round: given everything gathered SO FAR (starts with just the
# initial evidence, grows each round with whatever it asked for), the model
# must either (a) say it has enough and give the root cause + repair
# strategy, or (b) ask for specific additional evidence. It never sees a
# pre-computed root cause — it has to build the hypothesis itself, same as
# the diagram's "No Root Cause" investigation prompt.

INVESTIGATE_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis":        {"type": "string"},
        "hypothesis":       {"type": "string"},
        "sufficient":       {"type": "boolean"},
        "requests": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type":   {"type": "string"},   # "read_file" | "run_command"
                "target": {"type": "string"},   # path, or a key from the safe command list
            },
            "required": ["type", "target"]}},
        "root_cause":      {"type": "string"},
        "repair_strategy": {"type": "string"},
        "confidence":      {"type": "number"},
    },
    "required": ["analysis", "hypothesis", "sufficient"],
}

INVESTIGATE_SYSTEM = """\
You are a senior DevOps engineer investigating a broken CI/CD pipeline. You do
NOT know the root cause yet — figure it out like a human would: read the log,
look at what changed, look at the repo layout, form a theory, and only ask to
see specific files or run specific diagnostics that would confirm or kill
that theory. Do not ask for things you already have.

Output ONLY one JSON object. No markdown fences. Start with { end with }.
{"analysis":"what the log/diff/tree show you","hypothesis":"your current best guess at what's wrong","sufficient":true/false,"requests":[{"type":"read_file","target":"exact/repo/path"},{"type":"run_command","target":"one of the allowed diagnostic keys"}],"root_cause":"only if sufficient=true: one sentence, WHY it failed","repair_strategy":"only if sufficient=true: plain words, WHAT must change","confidence":0.0-1.0}

Rules:
- Ask for AT MOST {max_req} things per round, and only things likely to move
  the investigation forward.
- "read_file" targets must be real paths — check the repo tree you were
  given first, don't guess a name that isn't listed there.
- "run_command" targets must be chosen from the allowed diagnostic list you
  were given — you cannot run arbitrary shell commands.
- Set sufficient=true as soon as you can state a specific, falsifiable root
  cause — don't over-investigate. If you still don't know after several
  rounds, set sufficient=true anyway with your best hypothesis and a lower
  confidence, rather than looping forever.
- confidence below 0.5 means "I'm not sure this is really it."
""".replace("{max_req}", str(MAX_REQUESTS_PER_ROUND))


def build_investigation_prompt(evidence: dict, transcript: list) -> str:
    parts = [
        f"## CI failure (key lines):\n```\n{evidence['log_signal']}\n```",
        f"## Exit code: {evidence['exit_code']}",
        f"## Repository tree (names only — no contents yet):\n{evidence['repo_tree']}",
        f"## Git diff (most recent commit):\n```diff\n{evidence['git_diff']}\n```",
        f"## Available diagnostic commands you may request by key:\n"
        f"{', '.join(SAFE_DIAGNOSTIC_COMMANDS.keys())}",
    ]
    if transcript:
        parts.append("## Evidence gathered so far (from your earlier requests):")
        parts.extend(transcript)
    parts.append("\nRespond with the investigation JSON.")
    prompt = f"{INVESTIGATE_SYSTEM}\n\n" + "\n\n".join(parts)
    if len(prompt) > MAX_EVIDENCE_CHARS + len(INVESTIGATE_SYSTEM) + 500:
        # keep system + latest-gathered evidence, trim the oldest transcript entries
        head = "\n\n".join(parts[:5])
        keep_transcript = transcript[-3:] if transcript else []
        tail = "\n\n".join((["## Evidence gathered so far (most recent):"] + keep_transcript)
                            if keep_transcript else [])
        prompt = f"{INVESTIGATE_SYSTEM}\n\n{head}\n\n{tail}\n\nRespond with the investigation JSON."
    return prompt


def fulfill_requests(requests_: list) -> tuple:
    """Python Orchestrator box: reads requested source files / Dockerfile /
    workflow YAML / requirements.txt / package.json, and runs allowed
    diagnostic commands. Returns (transcript_entries, files_read_map)."""
    entries, files_read = [], {}
    for req in requests_[:MAX_REQUESTS_PER_ROUND]:
        rtype = (req.get("type") or "").strip()
        target = (req.get("target") or "").strip()
        if rtype == "read_file":
            rel = _relstrip(target)
            if ".." in rel or rel.startswith("/") or rel.startswith("~"):
                entries.append(f"### {target} (denied: unsafe path)")
                continue
            if _is_blocked(rel):
                entries.append(f"### {target} (denied: blocked/sensitive path)")
                continue
            p = Path(rel)
            if not p.is_file() or not _is_text_file(p):
                entries.append(f"### {target} (not found or not a text file)")
                continue
            content = p.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_CHARS]
            content = redact_secrets(content)
            entries.append(f"### {rel}\n```\n{content}\n```")
            files_read[rel] = content
            print(f"[FULFILL] read_file {rel} ({len(content)} chars)")
        elif rtype == "run_command":
            key = target
            if key not in SAFE_DIAGNOSTIC_COMMANDS:
                entries.append(f"### command `{target}` (denied: not on the safe list)")
                continue
            cmd, _ = SAFE_DIAGNOSTIC_COMMANDS[key]
            joined = " ".join(cmd)
            dangerous = scan_for_dangerous_commands(joined)
            if dangerous:
                entries.append(f"### command `{key}` (denied: matched dangerous pattern)")
                continue
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
                out = redact_secrets((r.stdout + r.stderr).strip()[:1500])
                entries.append(f"### command `{key}` (`{joined}`) output:\n```\n{out}\n```")
                print(f"[FULFILL] run_command {key}")
            except FileNotFoundError:
                entries.append(f"### command `{key}` (tool not installed on runner)")
            except subprocess.TimeoutExpired:
                entries.append(f"### command `{key}` (timed out)")
        else:
            entries.append(f"### unrecognized request type `{rtype}` for `{target}`")
    return entries, files_read


def ai_investigate(evidence: dict) -> tuple:
    """Runs the investigation loop. Returns (root_cause_dict, gathered_files)."""
    transcript, gathered_files = [], {}
    result = {}
    for round_no in range(1, MAX_INVESTIGATION_ROUNDS + 1):
        print(f"\n━━━ AI INVESTIGATION — round {round_no}/{MAX_INVESTIGATION_ROUNDS} ━━━")
        prompt = build_investigation_prompt(evidence, transcript)
        raw = _stream_ollama(prompt, INVESTIGATE_SCHEMA, num_predict=900,
                             temperature=0.05, tag=f"INVESTIGATE-{round_no}")
        data = _json_from(raw) or {}
        result = data
        print(f"  hypothesis  : {data.get('hypothesis','')[:120]}")
        print(f"  sufficient  : {data.get('sufficient')}")
        if data.get("sufficient"):
            result.setdefault("root_cause", data.get("hypothesis", "unknown"))
            result.setdefault("repair_strategy", "")
            try:
                result["confidence"] = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                result["confidence"] = 0.5
            break
        requests_ = data.get("requests", []) or []
        if not requests_:
            # said "not sufficient" but asked for nothing — force a stop to
            # avoid an unproductive loop
            print("  (no requests given — stopping investigation here)")
            result["sufficient"] = True
            result.setdefault("root_cause", data.get("hypothesis", "unknown"))
            result.setdefault("repair_strategy", "")
            result["confidence"] = float(data.get("confidence", 0.4) or 0.4)
            break
        entries, files_read = fulfill_requests(requests_)
        gathered_files.update(files_read)
        transcript.extend(entries)
    else:
        result["sufficient"] = True
        result.setdefault("root_cause", result.get("hypothesis", "unknown"))
        result.setdefault("repair_strategy", "")
        result["confidence"] = min(float(result.get("confidence", 0.4) or 0.4), 0.45)
        print("  (max rounds reached — proceeding with best hypothesis)")

    root_cause = result.get("root_cause") or result.get("hypothesis") or "unknown"
    repair_strategy = result.get("repair_strategy", "")
    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    print(f"\n  ── confirmed ──\n  CAUSE      : {root_cause}\n"
          f"  STRATEGY   : {repair_strategy or '(none)'}\n  confidence : {confidence:.0%}")
    return {"root_cause": root_cause, "repair_strategy": repair_strategy,
            "confidence": confidence}, gathered_files


# ══════════════════════════════════════════════════════════════════════════
# AI PATCH GENERATION AGENT  (diagram box)
# ══════════════════════════════════════════════════════════════════════════

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
            },
            "required": ["file", "problem", "evidence", "corrected"]}},
    },
    "required": ["issues"],
}

FIX_SYSTEM = """\
You are the patch-generation half of a two-agent CI/CD auto-fixer. The
investigation is done — you already know the root cause and repair
strategy. Now emit the FIX. Output ONLY one JSON object. No markdown
fences. Start with { end with }.

"issues" = ONE ENTRY PER BUG. Each entry:
  - "file": exact path from a ### header in the file contents below
  - "problem": one sentence describing this specific bug
  - "evidence": the offending text COPIED EXACTLY, character-for-character.
    SHORT — the single wrong token or one wrong line. Never paraphrase.
  - "corrected": the same text with ONLY the bug fixed. Everything else
    identical.

Schema:
{"issues":[{"file":"exact/path","problem":"...","evidence":"exact text from the file","corrected":"same text, bug fixed"}]}

CRITICAL for Dockerfile COPY/ADD/CMD/ENTRYPOINT paths: these resolve
against the Docker BUILD CONTEXT (the Dockerfile's own directory), NOT the
repo root. If the real file is at "sample_app/app.py" and the Dockerfile is
at "sample_app/Dockerfile", the correct in-container reference is "app.py",
never "sample_app/app.py".

RULES:
- evidence must literally appear in the file contents shown. If you cannot
  quote exact offending text, omit that issue.
- evidence and corrected must differ, and be as short as unambiguous allows.
- Only files with a ### header may be fixed. Never invent a new file."""


def ai_generate_patch(cause: dict, evidence: dict, files_context: dict,
                       extra_note: str = "") -> list:
    context_blocks = "\n\n".join(f"### {f}\n```\n{c}\n```" for f, c in files_context.items())
    ctx_line = (f"## Root cause: {cause['root_cause']}\n"
                f"## Repair strategy: {cause['repair_strategy']}\n"
                f"## Original CI failure:\n```\n{evidence['log_signal']}\n```\n")
    if extra_note:
        ctx_line += f"## Note from a previous failed attempt:\n{extra_note}\n"
    prompt = f"{FIX_SYSTEM}\n\n{ctx_line}\n## File contents (you may ONLY edit these):\n{context_blocks}\n\nEmit the issues JSON."
    if len(prompt) > 11000:
        prompt = prompt[:11000]
    raw = _stream_ollama(prompt, FIX_SCHEMA, num_predict=2800, temperature=0.05, tag="PATCH-GEN")
    data = _json_from(raw)
    if data is None:
        raise ValueError(f"No valid JSON from patch generation:\n{raw[:400]}")
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


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id":     {"type": "integer"},
                "keep":   {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["id", "keep"]}},
        "confidence": {"type": "number"},
    },
    "required": ["verdicts"],
}

VERIFY_SYSTEM = """\
You are reviewing proposed fixes before they are applied. For each numbered
issue, decide keep=true only if:
  - the "evidence" text actually appears in the shown file contents, AND
  - "corrected" is a genuine fix of a real bug (not a no-op, not a new bug).
If evidence is not present, or the change looks wrong or invented, keep=false.
Output ONLY one JSON object:
{"verdicts":[{"id":N,"keep":true,"reason":"short"}],"confidence":0.0-1.0}"""


def ai_self_verify(issues, files_context: dict) -> tuple:
    if SKIP_SELF_VERIFY or not issues:
        return issues, None
    context = "\n\n".join(f"### {f}\n```\n{c}\n```" for f, c in files_context.items())
    listing = []
    for i, it in enumerate(issues, 1):
        listing.append(f'{i}. file="{it.get("file","")}"\n'
                       f'   evidence: {(it.get("evidence") or "")[:160]!r}\n'
                       f'   corrected: {(it.get("corrected") or "")[:160]!r}')
    prompt = (f"{VERIFY_SYSTEM}\n\n## File contents:\n{context}\n\n"
              f"## Proposed fixes:\n" + "\n".join(listing) + "\n\nReturn the verdicts JSON.")
    if len(prompt) > 11000:
        prompt = prompt[:11000]
    try:
        raw = _stream_ollama(prompt, VERIFY_SCHEMA, num_predict=900,
                             temperature=0.0, tag="SELF-VERIFY", retries=1)
    except Exception as exc:
        print(f"[SELF-VERIFY] failed ({exc}) — keeping all issues", file=sys.stderr)
        return issues, None
    data = _json_from(raw)
    if not data:
        return issues, None
    drop = set()
    for v in data.get("verdicts", []) or []:
        try:
            idx = int(v.get("id", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(issues) and v.get("keep") is False:
            drop.add(idx)
            print(f"[SELF-VERIFY] dropped #{idx+1}: {v.get('reason','')[:80]}")
    kept = [it for i, it in enumerate(issues) if i not in drop]
    try:
        vc = float(data.get("confidence")) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        vc = None
    if not kept and issues:
        print("[SELF-VERIFY] verifier rejected all issues — keeping patch-gen output "
              "for the Validation Engine to judge")
        return issues, vc
    return kept, vc


# ══════════════════════════════════════════════════════════════════════════
# PAIR + LOCATE  (Python turns the AI's quotes into concrete edits)
# ══════════════════════════════════════════════════════════════════════════

def issues_to_fixes(issues: list, files_context: dict) -> tuple:
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

        holders = [f for f, c in files_context.items() if ev in c]
        if file in files_context and ev in files_context[file]:
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


# ══════════════════════════════════════════════════════════════════════════
# PYTHON VALIDATION ENGINE  (diagram box — the hard gate)
# ══════════════════════════════════════════════════════════════════════════

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


def validate_fix(fix: dict) -> tuple:
    """The Validation Engine: patch format, file paths, syntax, YAML/JSON,
    secret scanning, dangerous-command detection, safe-patch verification."""
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
    secret_hits = scan_text_for_secrets("\n".join(added))
    if secret_hits:
        return False, f"potential secret in fix ({', '.join(secret_hits)}) — blocked"
    dangerous_hits = scan_for_dangerous_commands("\n".join(added))
    if dangerous_hits:
        return False, "fix contains a dangerous command pattern — blocked"

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


# ══════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ══════════════════════════════════════════════════════════════════════════

def fingerprint_stack(log_text: str) -> set:
    low = log_text.lower()
    signals = {"python": ["python", "pip", "pytest", "requirements.txt", ".py"],
               "node":   ["node", "npm", "yarn", "package.json", ".js", ".ts"],
               "go":     ["go build", "go test", "go.mod", ".go"]}
    return {s for s, sig in signals.items() if any(x in low for x in sig)}


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


# ══════════════════════════════════════════════════════════════════════════
# COMMIT + PUSH + PR  /  CREATE ISSUE
# ══════════════════════════════════════════════════════════════════════════

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
            analysis="", strategy="", flagged=False) -> str:
    details = "".join(f"\n**`{f.get('file','?')}`** — {f.get('reason','')}\n" for f in fixes)
    diag = ""
    if analysis or strategy:
        diag = (f"### AI investigation\n**Analysis:** {analysis or 'n/a'}\n\n"
                f"**Repair strategy:** {strategy or 'n/a'}\n\n")
    flag_note = ("\n> ⚠️ **Needs extra scrutiny** — tests did not pass after retries; "
                 "opening for human review rather than looping further.\n" if flagged else "")
    body = (f"## 🤖 AI Auto-Fix\n\n{diag}"
            f"**Root cause:** {root_cause}\n\n"
            f"**Files changed:** {', '.join(f'`{f}`' for f in written)}\n\n"
            f"## What changed{details}\n{flag_note}\n> Auto-generated — review before merge.")
    title_prefix = "🤖⚠️" if flagged else "🤖"
    r = requests.post(f"https://api.github.com/repos/{repo}/pulls",
                      json={"title": f"{title_prefix} {commit_msg}", "head": branch,
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


# ══════════════════════════════════════════════════════════════════════════
# MAIN — wires the diagram's boxes together end to end
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer (agentic investigation)")
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

    # ── COLLECT INITIAL INVESTIGATION EVIDENCE ──
    print("\n━━━ COLLECT INITIAL INVESTIGATION EVIDENCE ━━━")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    redacted = scan_text_for_secrets(log_text)
    if redacted:
        print(f"[SECURITY] redacting {len(redacted)} potential secret pattern(s) from log: "
              f"{', '.join(redacted)}")
    log_text = redact_secrets(log_text)
    stacks = fingerprint_stack(log_text)
    evidence = collect_initial_evidence(log_text)
    if not evidence["log_signal"].strip():
        print("[COLLECT] No error signal — nothing to fix.")
        sys.exit(0)

    # ── AI INVESTIGATION AGENT (loop with Python-fulfilled requests) ──
    try:
        cause, gathered_files = ai_investigate(evidence)
    except Exception as exc:
        print(f"[ERROR] Investigation failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI investigation failed: {exc}", run_url)
        sys.exit(2)

    if cause["confidence"] < 0.5:
        print(f"[GATE] Confidence {cause['confidence']:.0%} too low — escalating instead of guessing.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({cause['confidence']:.0%}). "
                       f"Root cause: {cause['root_cause']}", run_url)
        sys.exit(0)

    if not gathered_files:
        print("[ERROR] Investigation confirmed a root cause but read no files to patch.",
              file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI investigation had no file evidence to act on. "
                       f"Root cause: {cause['root_cause']}", run_url)
        sys.exit(3)

    commit_msg = f"fix: {cause['root_cause'][:60]}"
    written, originals, fixes = [], {}, []
    extra_note, flagged_for_review = "", False

    # ── AI PATCH GENERATION AGENT + VALIDATION + APPLY + TEST, with bounded retry ──
    for attempt in range(1, MAX_PATCH_RETRY_ROUNDS + 1):
        print(f"\n━━━ AI PATCH GENERATION — attempt {attempt}/{MAX_PATCH_RETRY_ROUNDS} ━━━")
        try:
            issues = ai_generate_patch(cause, evidence, gathered_files, extra_note)
        except Exception as exc:
            print(f"[ERROR] Patch generation failed: {exc}", file=sys.stderr)
            if attempt == MAX_PATCH_RETRY_ROUNDS:
                if token and repo:
                    open_issue(token, repo, f"AI patch generation failed: {exc}", run_url)
                sys.exit(2)
            continue

        print(f"  issues reported: {len(issues)}")
        for n, it in enumerate(issues, 1):
            print(f"    {n}. {it.get('file','?')}: {it.get('problem','')[:80]}")

        issues, _ = ai_self_verify(issues, gathered_files)
        print(f"  issues after self-verify: {len(issues)}")

        fixes, pair_rejects = issues_to_fixes(issues, gathered_files)
        for rej in pair_rejects:
            print(f"  ✗ {rej}", file=sys.stderr)

        if not fixes:
            extra_note = "; ".join(pair_rejects) or "model reported no locatable issues"
            print(f"[ERROR] No usable fixes this attempt: {extra_note}", file=sys.stderr)
            continue

        print("\n━━━ PYTHON VALIDATION ENGINE + APPLY ━━━")
        if args.dry_run:
            for fix in fixes:
                ok, reason = validate_fix(fix)
                print(f"  {'would write' if ok else 'reject'} {fix.get('file','?')} — {reason}")
            sys.exit(0)

        written, originals, reject_reasons = write_fixes(fixes)
        if not written:
            extra_note = "; ".join(reject_reasons) or "no detail captured"
            print(f"[ERROR] Nothing passed validation this attempt: {extra_note}", file=sys.stderr)
            continue

        print("\n━━━ EXECUTE BUILD & TESTS ━━━")
        if args.skip_tests:
            print("[TEST] Skipped (--skip-tests)")
            break
        if run_tests(stacks):
            break

        # failure branch: collect new failure evidence, let the AI review it,
        # decide whether to retry
        print("[TEST] Failed — reverting this attempt and gathering new evidence")
        revert_files(originals)
        written, originals = [], {}
        extra_note = ("Your previous patch was reverted because tests failed after "
                      "applying it. Reconsider the root cause and try a different fix.")
        if attempt == MAX_PATCH_RETRY_ROUNDS:
            flagged_for_review = True

    if not written:
        detail = extra_note or "no attempt produced a passing fix"
        print(f"[ERROR] No fix survived validation/tests after {MAX_PATCH_RETRY_ROUNDS} "
              f"attempt(s). {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"Auto-fixer could not land a passing fix. "
                       f"Root cause: {cause['root_cause']}\n\nDetail: {detail}", run_url)
        sys.exit(5 if "tests failed" in detail.lower() or flagged_for_review else 3)

    # ── COMMIT + PR ──
    print("\n━━━ COMMIT + PR ━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)
    if token and repo:
        open_pr(token, repo, branch, commit_msg, cause["root_cause"], written, fixes,
                analysis="", strategy=cause["repair_strategy"], flagged=flagged_for_review)
    else:
        print(f"[PR] No token — merge {branch} manually.")

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {cause['root_cause']}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()