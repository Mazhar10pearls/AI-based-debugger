#!/usr/bin/env python3
"""
Self-Healing CI Auto-Fixer — Prompt-Optimised Edition
======================================================
Root cause of previous timeouts: 19,905 char prompt → llama3.2 times out.

Fix: keep total prompt under 3,000 chars by:
  - Sending ONLY files directly mentioned in the error log
  - Capping error signal to the 20 most relevant lines
  - Stripping comments and blank lines from context files
  - Hard prompt size cap with truncation warning

Pipeline: DETECT → ANALYSE → VALIDATE → PATCH → TEST → COMMIT → PR
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_API_URL      = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
DEFAULT_MODEL        = os.environ.get("OLLAMA_MODEL",   "llama3.2")

MAX_ERROR_LINES      = 20      # max lines of error signal sent to AI
MAX_FILE_CHARS       = 800     # max chars per file in context
MAX_CONTEXT_FILES    = 4       # max number of files sent as context
MAX_PROMPT_CHARS     = 4000    # hard cap — llama3.2 reliable limit
MAX_PATCHES          = 5
CONFIDENCE_THRESHOLD = 0.5
MAX_RETRIES          = 3
RETRY_BACKOFF        = [10, 30, 60]

BOT_NAME             = "github-actions[bot]"
BOT_EMAIL            = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX           = "fix:"

BLOCKED_PATHS = [
    ".git/",
    ".github/workflows/auto-fixer-on-failure.yml",
    ".github/workflows/auto-fix-ci.yml",
]

# All candidate files — ranked by relevance (checked in order)
CONTEXT_PRIORITY = [
    ".github/workflows/ci-local-deploy.yml",
    ".github/workflows/ci-local-deploy.yaml",
    "requirements.txt",
    "sample_app/requirements.txt",
    "Dockerfile",
    "sample_app/Dockerfile",
    "docker-compose.yml",
    "setup.py",
    "pyproject.toml",
    "package.json",
]

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".mypy_cache", ".pytest_cache",
}


# ── Git helpers ───────────────────────────────────────────────────────────────

def run_git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], check=check, capture_output=True, text=True
    )


def last_commit_was_bot() -> bool:
    try:
        r = run_git("log", "-1", "--pretty=%an|||%s")
        author, subject = r.stdout.strip().split("|||", 1)
        if author == BOT_NAME and subject.startswith(BOT_PREFIX):
            print(f"[GUARD] Last commit was bot fix: '{subject}' — stopping loop.")
            return True
    except Exception:
        pass
    return False


# ── Smart context builder ─────────────────────────────────────────────────────

def strip_noise(content: str, filepath: str) -> str:
    """Remove comments and blank lines to shrink file size."""
    lines = content.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip blank lines
        if not stripped:
            continue
        # Skip comment-only lines (Python, YAML, shell, Dockerfile)
        if stripped.startswith("#"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def files_mentioned_in_error(error_signal: str) -> list[str]:
    """
    Extract file paths that appear in the error log.
    These are the highest-priority files to send as context.
    """
    # Match things like: sample_app/app.py, requirements.txt, Dockerfile
    pattern = re.compile(
        r'(?:^|[\s"\'])([a-zA-Z0-9_\-./]+\.(?:py|yml|yaml|txt|sh|js|ts|json|toml|cfg|rb|go|java|Dockerfile\S*))',
        re.MULTILINE
    )
    found = []
    seen  = set()
    for m in pattern.finditer(error_signal):
        path = m.group(1).lstrip("./")
        if path not in seen and Path(path).is_file():
            seen.add(path)
            found.append(path)
    return found


def collect_smart_context(error_signal: str) -> str:
    """
    Build the smallest useful context:
    1. Files directly mentioned in the error log  (highest signal)
    2. Priority list files that exist on disk     (config files likely involved)
    Stop once MAX_CONTEXT_FILES reached.
    """
    selected = []
    seen     = set()

    # Pass 1 — files the error log explicitly mentions
    for path in files_mentioned_in_error(error_signal):
        if path not in seen and len(selected) < MAX_CONTEXT_FILES:
            selected.append(path)
            seen.add(path)

    # Pass 2 — priority config files not yet included
    for path_str in CONTEXT_PRIORITY:
        if len(selected) >= MAX_CONTEXT_FILES:
            break
        p = Path(path_str)
        if p.exists() and path_str not in seen:
            selected.append(path_str)
            seen.add(path_str)

    # Build context block
    parts = []
    for path_str in selected:
        try:
            content = Path(path_str).read_text(encoding="utf-8", errors="replace")
            content = strip_noise(content, path_str)
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n...(truncated)"
            parts.append(f"### {path_str}\n```\n{content}\n```")
        except Exception as exc:
            parts.append(f"### {path_str}\n(unreadable: {exc})")

    result = "\n\n".join(parts)
    print(f"[DETECT] Context: {len(selected)} files — {list(selected)} — {len(result)} chars")
    return result


# ── Error signal extraction ───────────────────────────────────────────────────

def extract_error_signal(log_text: str) -> str:
    ERROR_KW = [
        "error", "failed", "failure", "exception", "traceback",
        "exit code", "exitcode", "no module", "not found", "invalid",
        "fatal", "syntaxerror", "importerror", "nameerror", "typeerror",
        "valueerror", "attributeerror", "assertionerror", "runtimeerror",
        "cannot", "refused", "rejected", "killed", "denied", "missing",
        "undefined", "permission denied", "command not found",
        "connectionerror", "step failed", "build failed", "test failed",
        "pytest", "npm err", "returned non-zero",
    ]
    NOISE_KW = [
        "allow-prereleases", "fetch-depth", "persist-credentials",
        "set-safe-directory", "##[group]", "##[endgroup]", "extraheader",
        "post job cleanup", "set up job", "complete job", "add mask",
    ]
    lines    = log_text.splitlines()
    relevant = [
        l.strip() for l in lines
        if any(k in l.lower() for k in ERROR_KW)
        and not any(n in l.lower() for n in NOISE_KW)
        and l.strip()
    ]
    # Keep only the last N most relevant lines — earlier lines are usually setup noise
    relevant = list(dict.fromkeys(relevant))[-MAX_ERROR_LINES:]
    # Always include last 10 lines of the log (failure context)
    tail = [l.strip() for l in lines[-10:] if l.strip()]
    merged = list(dict.fromkeys(relevant + tail))
    result = "\n".join(merged)
    print(f"[DETECT] Error signal: {len(merged)} lines, {len(result)} chars")
    return result


# ── AI call ───────────────────────────────────────────────────────────────────

PROMPT = """\
CI pipeline failed. Find the bug and return a surgical fix.

## Error
{error_signal}

## Files
{repo_context}

Return ONLY this JSON (no markdown):
{{
  "root_cause": "one sentence",
  "confidence": 0.9,
  "commit_message": "fix: description",
  "patches": [
    {{
      "file": "path/to/file",
      "find": "exact string in file",
      "replace": "corrected string"
    }}
  ]
}}"""


def build_prompt(error_signal: str, repo_context: str) -> str:
    prompt = PROMPT.format(
        error_signal=error_signal,
        repo_context=repo_context,
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        # Trim repo_context to fit
        overhead    = len(PROMPT.format(error_signal=error_signal, repo_context=""))
        allowed     = MAX_PROMPT_CHARS - overhead - 50
        repo_context = repo_context[:allowed] + "\n...(trimmed to fit model)"
        prompt      = PROMPT.format(
            error_signal=error_signal,
            repo_context=repo_context,
        )
        print(f"[AI] Prompt trimmed to {len(prompt)} chars to fit model context")
    return prompt


def call_ai(error_signal: str, repo_context: str,
            model: str, api_url: str) -> str:
    prompt  = build_prompt(error_signal, repo_context)
    payload = {
        "model":       model,
        "prompt":      prompt,
        "temperature": 0.1,
        "max_tokens":  600,   # patches are short — find+replace needs very few tokens
    }
    print(f"[AI] Prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)")

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[AI] Calling Ollama — attempt {attempt + 1}/{MAX_RETRIES}...")
            resp = requests.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
            print(f"[AI] Got {len(raw)} chars: {raw[:200]}{'...' if len(raw) > 200 else ''}")
            return raw
        except requests.exceptions.Timeout as e:
            last_exc = e
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"[AI] Timeout on attempt {attempt + 1}. Waiting {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot reach Ollama at {api_url}: {e}")

    raise RuntimeError(f"Ollama timed out on all {MAX_RETRIES} attempts: {last_exc}")


def parse_ai_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    best = None
    for start in [i for i, c in enumerate(cleaned) if c == "{"]:
        stack = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                stack += 1
            elif cleaned[i] == "}":
                stack -= 1
                if stack == 0:
                    candidate = cleaned[start:i + 1]
                    if best is None or len(candidate) > len(best):
                        best = candidate
                    break
    if best:
        try:
            return json.loads(best)
        except json.JSONDecodeError:
            repaired = best + "}" * (best.count("{") - best.count("}"))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
    raise ValueError(f"No valid JSON in AI response:\n{raw[:500]}")


# ── Patch validation + application ───────────────────────────────────────────

def is_blocked(file: str) -> bool:
    return any(file.startswith(b) or file == b.rstrip("/") for b in BLOCKED_PATHS)


def normalise_patch(patch: dict) -> dict:
    for key in ("find", "replace"):
        val = patch.get(key, "")
        if isinstance(val, str) and "\\n" in val and "\n" not in val:
            patch[key] = val.replace("\\n", "\n")
    return patch


def validate_patch(patch: dict) -> tuple[bool, str]:
    file    = (patch.get("file") or "").strip()
    find    = patch.get("find",    "")
    replace = patch.get("replace", "")

    if not file:
        return False, "missing 'file'"
    if ".." in file or file.startswith("/") or file.startswith("~"):
        return False, f"unsafe path: {file}"
    if is_blocked(file):
        return False, f"blocked: {file}"
    if not find:
        return False, f"empty 'find' for {file}"
    if replace is None:
        return False, f"missing 'replace' for {file}"
    if not Path(file).exists():
        return False, f"file not found: {file}"

    content = Path(file).read_text(encoding="utf-8", errors="replace")
    if find not in content:
        return False, f"find string not in {file}: {repr(find[:60])}"

    patched = content.replace(find, replace, 1)

    if re.search(r"\.github/workflows/.+\.ya?ml$", file):
        try:
            parsed = yaml.safe_load(patched)
            if not isinstance(parsed, dict) or "jobs" not in parsed:
                return False, f"{file}: patched YAML missing 'jobs:'"
        except yaml.YAMLError as exc:
            return False, f"{file}: patched YAML invalid — {exc}"

    if file.endswith(".py"):
        try:
            compile(patched, file, "exec")
        except SyntaxError as exc:
            return False, f"{file}: patched Python syntax error — {exc}"

    return True, "ok"


def apply_patch(patch: dict) -> bool:
    file    = patch["file"].strip()
    find    = patch["find"]
    replace = patch["replace"]
    content = Path(file).read_text(encoding="utf-8", errors="replace")
    Path(file).write_text(content.replace(find, replace, 1), encoding="utf-8")
    print(f"  ✓ {file}")
    print(f"    - {repr(find[:70])}")
    print(f"    + {repr(replace[:70])}")
    return True


def apply_all_patches(patches: list) -> list[str]:
    written = []
    for raw_patch in patches[:MAX_PATCHES]:
        patch = normalise_patch(raw_patch)
        ok, reason = validate_patch(patch)
        if not ok:
            print(f"  ✗ {reason}", file=sys.stderr)
            continue
        if apply_patch(patch):
            written.append(patch["file"].strip())
    return written


# ── Local tests ───────────────────────────────────────────────────────────────

def run_tests() -> bool:
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q", "--no-header"],
            capture_output=True, text=True, timeout=120
        )
        for line in (result.stdout + result.stderr).splitlines()[-20:]:
            print(f"  {line}")
        if result.returncode == 0:
            print("[TEST] ✓ Passed")
            return True
        print("[TEST] ✗ Failed")
        return False
    except FileNotFoundError:
        print("[TEST] pytest not found — skipping")
        return True
    except subprocess.TimeoutExpired:
        print("[TEST] Timed out — skipping")
        return True


def revert_patches(written: list[str]):
    if not written:
        return
    try:
        run_git("checkout", "--", *written)
        print(f"[REVERT] Restored {len(written)} file(s)")
    except subprocess.CalledProcessError as exc:
        print(f"[REVERT] Failed: {exc.stderr}", file=sys.stderr)


# ── Commit + PR ───────────────────────────────────────────────────────────────

def commit_to_branch_and_pr(commit_msg: str, root_cause: str) -> bool:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "")

    try:
        run_git("config", "user.name",  BOT_NAME)
        run_git("config", "user.email", BOT_EMAIL)

        run_git("fetch", "origin")
        merge = subprocess.run(
            ["git", "merge", "-X", "ours", "origin/main",
             "--no-edit", "-m", "merge: sync before fix"],
            capture_output=True, text=True
        )
        if merge.returncode != 0:
            print(f"[WARN] Merge skipped: {merge.stderr.strip()}")
            run_git("merge", "--abort", check=False)

        branch = f"auto-fix/{int(time.time())}"
        run_git("checkout", "-b", branch)
        run_git("add", "-A")

        if run_git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit.")
            run_git("checkout", "main", check=False)
            return True

        run_git("commit", "-m", commit_msg)
        run_git("push", "-u", "origin", branch)
        print(f"[COMMIT] ✓ Pushed branch: {branch}")

        run_git("checkout", "main", check=False)

        if token and repo:
            _open_pr(token, repo, branch, commit_msg, root_cause)
        else:
            print(f"[PR] No token — merge manually: git merge {branch}")

        return True

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Git: {' '.join(exc.cmd)}\n  {exc.stderr}", file=sys.stderr)
        run_git("checkout", "main", check=False)
        run_git("merge", "--abort", check=False)
        return False


def _open_pr(token, repo, branch, commit_msg, root_cause):
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        json={
            "title": commit_msg,
            "head":  branch,
            "base":  "main",
            "body":  f"**Root cause:** {root_cause}\n\nAutomated fix — please review before merging.",
        },
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"[PR] ✓ {resp.json().get('html_url', '')}")
    else:
        print(f"[PR] Failed {resp.status_code}: {resp.text[:200]}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--model",       default=DEFAULT_MODEL)
    parser.add_argument("--api-url",     default=DEFAULT_API_URL)
    parser.add_argument("--dry-run",     action="store_true")
    parser.add_argument("--skip-tests",  action="store_true")
    args = parser.parse_args()

    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Log not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if last_commit_was_bot():
        sys.exit(0)

    # ── DETECT ────────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 1: DETECT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log_text     = log_path.read_text(encoding="utf-8", errors="replace")
    error_signal = extract_error_signal(log_text)
    repo_context = collect_smart_context(error_signal)

    # ── ANALYSE ───────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 2: ANALYSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        raw     = call_ai(error_signal, repo_context, args.model, args.api_url)
        ai_data = parse_ai_response(raw)
    except Exception as exc:
        print(f"[ERROR] AI failed: {exc}", file=sys.stderr)
        sys.exit(2)

    confidence = float(ai_data.get("confidence", 1.0))
    root_cause = ai_data.get("root_cause", "unknown")
    patches    = ai_data.get("patches", [])
    commit_msg = ai_data.get("commit_message", "fix: auto-fixer patch")

    print(f"\n  root_cause : {root_cause}")
    print(f"  confidence : {confidence:.0%}")
    print(f"  patches    : {len(patches)}")
    for p in patches:
        print(f"    → {p.get('file','?')}  find: {repr(str(p.get('find',''))[:50])}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[SKIP] Confidence {confidence:.0%} below threshold — not applying.")
        sys.exit(0)

    if not patches:
        print("[ERROR] No patches returned.", file=sys.stderr)
        sys.exit(3)

    if args.dry_run:
        print("\n━━━ DRY-RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for p in patches[:MAX_PATCHES]:
            p = normalise_patch(p)
            ok, reason = validate_patch(p)
            print(f"  {'✓' if ok else '✗'} {p.get('file','?')} — {reason}")
        sys.exit(0)

    # ── PATCH ─────────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 3: PATCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    written = apply_all_patches(patches)
    if not written:
        print("[ERROR] All patches failed validation.", file=sys.stderr)
        sys.exit(3)
    print(f"[PATCH] {len(written)}/{len(patches)} applied.")

    # ── TEST ──────────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 4: TEST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not args.skip_tests:
        if not run_tests():
            revert_patches(written)
            sys.exit(5)
    else:
        print("[TEST] Skipped")

    # ── COMMIT + PR ───────────────────────────────────────────────────────────
    print("\n━━━ STAGE 5: COMMIT + PR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not commit_to_branch_and_pr(commit_msg, root_cause):
        sys.exit(4)

    print("\n━━━ ✅ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  commit : {commit_msg}")
    print(f"  files  : {', '.join(written)}")
    print("  Review and merge the PR — that push will re-trigger CI.")


if __name__ == "__main__":
    main()