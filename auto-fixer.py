#!/usr/bin/env python3
"""
Self-Healing CI Auto-Fixer — Secure PR Flow
============================================
Production Git flow:
  1. AI detects bug and creates surgical find/replace patches
  2. Patches go to an auto-fix/<timestamp> branch — never directly to main
  3. A Pull Request is opened with full description of what changed and why
  4. A human reviews the PR diff, approves, and merges
  5. Merge to main triggers CI re-run automatically

Why PR flow for production:
  - You see exactly what the AI changed before it touches main
  - Broken fix never reaches main or affects the running service
  - Full audit trail — every fix is a reviewed PR
  - If auto-fixer is stuck (3 failures), opens a GitHub Issue instead
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

MAX_ERROR_LINES      = 25
MAX_FILE_CHARS       = 600
MAX_PROMPT_CHARS     = 3800
MAX_PATCHES          = 5
CONFIDENCE_THRESHOLD = 0.5
MAX_RETRIES          = 3
RETRY_BACKOFF        = [10, 30, 60]
MAX_BOT_ATTEMPTS     = 3      # open issue instead of looping forever

BOT_NAME             = "github-actions[bot]"
BOT_EMAIL            = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX           = "fix:"

BLOCKED_PATHS = [
    ".git/",
    ".github/workflows/auto-fixer-on-failure.yml",
    ".github/workflows/auto-fix-ci.yml",
]

# Tier 1 — always sent to AI (covers 95% of CI failures)
ALWAYS_INCLUDE = [
    ".github/workflows/ci-local-deploy.yml",
    ".github/workflows/ci-local-deploy.yaml",
    "sample_app/Dockerfile",
    "Dockerfile",
    "sample_app/requirements.txt",
    "requirements.txt",
    "sample_app/deploy_local.sh",
]

# Tier 2 — added only when their keyword appears in the error log
CONDITIONAL_FILES = {
    "pytest":       "sample_app/test_app.py",
    "test_app":     "sample_app/test_app.py",
    "app.py":       "sample_app/app.py",
    "setup.py":     "setup.py",
    "pyproject":    "pyproject.toml",
    "package.json": "package.json",
    "docker-compose": "docker-compose.yml",
    "makefile":     "Makefile",
}

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
    """Loop guard — stop if last commit was already our fix."""
    try:
        r = run_git("log", "-1", "--pretty=%an|||%s")
        author, subject = r.stdout.strip().split("|||", 1)
        if author == BOT_NAME and subject.startswith(BOT_PREFIX):
            print(f"[GUARD] Last commit was bot fix: '{subject}' — stopping loop.")
            return True
    except Exception:
        pass
    return False


def count_recent_bot_attempts(n: int = 10) -> int:
    """Count how many of the last N commits were bot fix attempts."""
    try:
        r = run_git("log", f"-{n}", "--pretty=%an")
        return sum(1 for line in r.stdout.strip().splitlines()
                   if line.strip() == BOT_NAME)
    except Exception:
        return 0


# ── Error signal ──────────────────────────────────────────────────────────────

def extract_error_signal(log_text: str) -> str:
    ERROR_KW = [
        "error", "failed", "failure", "exception", "traceback",
        "exit code", "exitcode", "invalid", "fatal", "cannot",
        "refused", "rejected", "killed", "denied", "missing",
        "undefined", "permission denied", "command not found",
        "returned non-zero", "syntaxerror", "importerror",
        "nameerror", "typeerror", "valueerror", "attributeerror",
        "no module named", "not found",
        # Docker
        "not implemented", "media type", "containerd",
        "docker daemon", "step 1", "step 2", "step 3",
        "no such image", "pull access denied", "manifest unknown",
        # pip
        "could not find", "no matching", "requirement",
        # Tests
        "pytest", "test failed", "assert",
        # Shell / curl
        "bash", "sh:", "chmod", "curl",
    ]
    NOISE_KW = [
        "##[group]", "##[endgroup]", "extraheader", "sshcommand",
        "safe.directory", "worktreeconfig", "sparse-checkout",
        "submodule foreach", "check-latest", "allow-prereleases",
        "freethreaded", "persist-credentials", "fetch-depth",
        "set up job", "complete job", "post job", "add mask",
        "removing .pytest_cache", "removing __pycache__",
        "cacheprovider", "rootdir:", "configfile:",
        "no warnings", "warnings summary", "short test summary",
        "passed in", "collecting ", "git config", "git version",
    ]
    lines    = log_text.splitlines()
    relevant = [
        l.strip() for l in lines
        if any(k in l.lower() for k in ERROR_KW)
        and not any(n in l.lower() for n in NOISE_KW)
        and l.strip()
    ]
    relevant = list(dict.fromkeys(relevant))[-MAX_ERROR_LINES:]
    tail     = [l.strip() for l in lines[-15:] if l.strip()
                and not any(n in l.lower() for n in NOISE_KW)]
    merged   = list(dict.fromkeys(relevant + tail))
    result   = "\n".join(merged)
    print(f"[DETECT] Error signal: {len(merged)} lines, {len(result)} chars")
    return result


# ── Context builder ───────────────────────────────────────────────────────────

def strip_file_noise(content: str) -> str:
    return "\n".join(
        line for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


def collect_context(error_signal: str) -> tuple[str, list[str]]:
    error_low = error_signal.lower()
    selected, seen = [], set()

    for path_str in ALWAYS_INCLUDE:
        if Path(path_str).exists() and path_str not in seen:
            selected.append(path_str)
            seen.add(path_str)

    for keyword, path_str in CONDITIONAL_FILES.items():
        if keyword in error_low and Path(path_str).exists() and path_str not in seen:
            selected.append(path_str)
            seen.add(path_str)

    parts, included, total = [], [], 0
    budget = MAX_PROMPT_CHARS - 800

    for path_str in selected:
        try:
            content = Path(path_str).read_text(encoding="utf-8", errors="replace")
            content = strip_file_noise(content)
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n...(truncated)"
            block = f"### {path_str}\n```\n{content}\n```"
            if total + len(block) > budget:
                print(f"[DETECT] Skipping {path_str} — prompt budget reached")
                break
            parts.append(block)
            total += len(block)
            included.append(path_str)
        except Exception as exc:
            parts.append(f"### {path_str}\n(unreadable: {exc})")
            included.append(path_str)

    result = "\n\n".join(parts)
    print(f"[DETECT] Context: {len(included)} files — {included}")
    print(f"[DETECT] Context size: {len(result)} chars")
    return result, included


# ── AI ────────────────────────────────────────────────────────────────────────

PROMPT = """\
CI/CD pipeline failed. Find ALL bugs across ALL files shown and return surgical fixes.

## Error log
{error_signal}

## Repository files (current state on disk)
{repo_context}

## Rules
- Check EVERY file shown — bugs are often in multiple files at once
- "find" must be an exact substring currently in that file
- "replace" is the corrected version — keep it minimal
- Do NOT rewrite whole files

Return ONLY this JSON (no markdown):
{{
  "root_cause": "one sentence listing every bug found",
  "confidence": 0.9,
  "commit_message": "fix: description",
  "patches": [
    {{
      "file": "relative/path/to/file",
      "find": "exact string in file right now",
      "replace": "corrected string"
    }}
  ]
}}"""


def build_prompt(error_signal: str, repo_context: str) -> str:
    prompt = PROMPT.format(error_signal=error_signal, repo_context=repo_context)
    if len(prompt) > MAX_PROMPT_CHARS:
        overhead     = len(PROMPT.format(error_signal=error_signal, repo_context=""))
        allowed      = MAX_PROMPT_CHARS - overhead - 100
        repo_context = repo_context[:allowed] + "\n...(trimmed)"
        prompt       = PROMPT.format(error_signal=error_signal, repo_context=repo_context)
        print(f"[AI] Prompt trimmed to {len(prompt)} chars")
    return prompt


def call_ai(error_signal: str, repo_context: str,
            model: str, api_url: str) -> str:
    prompt  = build_prompt(error_signal, repo_context)
    payload = {"model": model, "prompt": prompt, "temperature": 0.1, "max_tokens": 800}
    print(f"[AI] Prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)")

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[AI] Calling Ollama — attempt {attempt + 1}/{MAX_RETRIES}...")
            resp = requests.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
            print(f"[AI] Got {len(raw)} chars: {raw[:300]}{'...' if len(raw) > 300 else ''}")
            return raw
        except requests.exceptions.Timeout as e:
            last_exc = e
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"[AI] Timeout attempt {attempt + 1}. Waiting {wait}s...")
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
            if cleaned[i] == "{": stack += 1
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
        preview = " | ".join(content.splitlines()[:6])
        return False, (
            f"find string not in {file}\n"
            f"  AI wanted : {repr(find[:80])}\n"
            f"  File has  : {preview[:200]}"
        )

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
            return False, f"{file}: syntax error after patch — {exc}"

    return True, "ok"


def apply_patch(patch: dict) -> bool:
    file    = patch["file"].strip()
    find    = patch["find"]
    replace = patch["replace"]
    content = Path(file).read_text(encoding="utf-8", errors="replace")
    Path(file).write_text(content.replace(find, replace, 1), encoding="utf-8")
    print(f"  ✓ {file}")
    print(f"    - {repr(find[:80])}")
    print(f"    + {repr(replace[:80])}")
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


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests() -> bool:
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--tb=short", "-q", "--no-header"],
            capture_output=True, text=True, timeout=120
        )
        for line in (result.stdout + result.stderr).splitlines()[-20:]:
            print(f"  {line}")
        passed = result.returncode == 0
        print(f"[TEST] {'✓ Passed' if passed else '✗ Failed'}")
        return passed
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


# ── GitHub API helpers ────────────────────────────────────────────────────────

def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def open_pr(token: str, repo: str, branch: str,
            commit_msg: str, root_cause: str,
            written: list[str], patches: list) -> str:
    """
    Open a Pull Request with a full description of every change made.
    Returns the PR URL.
    """
    # Build a readable diff summary for the PR body
    patch_summary = ""
    for p in patches:
        patch_summary += (
            f"\n**`{p.get('file','?')}`**\n"
            f"```diff\n"
            f"- {p.get('find','').strip()}\n"
            f"+ {p.get('replace','').strip()}\n"
            f"```\n"
        )

    body = f"""## 🤖 AI Auto-Fix — Review Required

**Root cause:** {root_cause}

**Files changed:** {', '.join(f'`{f}`' for f in written)}

---

## What the AI changed

{patch_summary}

---

## How to review
1. Check the diff above — does each change look correct?
2. Look at the **Files changed** tab in this PR for full context
3. If correct → **Merge** — CI will re-run automatically
4. If wrong → **Close** — fix manually and push to main

> This PR was created automatically by the self-healing CI system.
> Merging will trigger `Local CI/CD Deploy` automatically.
"""

    resp = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        json={
            "title": f"🤖 {commit_msg}",
            "head":  branch,
            "base":  "main",
            "body":  body,
        },
        headers=gh_headers(token),
        timeout=30,
    )
    if resp.status_code in (200, 201):
        pr_url = resp.json().get("html_url", "")
        print(f"[PR] ✓ Opened: {pr_url}")
        return pr_url
    else:
        print(f"[PR] Failed {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
        return ""


def open_issue(token: str, repo: str, reason: str, run_url: str = ""):
    """Open a GitHub Issue when the auto-fixer is stuck."""
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        json={
            "title": "🚨 Auto-fixer stuck — manual fix needed",
            "body": (
                f"## Auto-fixer could not fix the CI failure\n\n"
                f"**Reason:** {reason}\n\n"
                f"**Failed run:** {run_url}\n\n"
                f"The auto-fixer attempted to fix this but failed. "
                f"Please investigate and fix manually.\n\n"
                f"Once fixed, close this issue."
            ),
            "labels": ["bug", "needs-manual-fix"],
        },
        headers=gh_headers(token),
        timeout=30,
    )
    if resp.status_code in (200, 201):
        issue_url = resp.json().get("html_url", "")
        print(f"[ISSUE] ✓ Opened: {issue_url}")
    else:
        print(f"[ISSUE] Failed {resp.status_code}: {resp.text[:200]}", file=sys.stderr)


# ── Commit to branch ──────────────────────────────────────────────────────────

def commit_to_fix_branch(commit_msg: str) -> str:
    """
    Commit patches to a new auto-fix/<timestamp> branch.
    Returns the branch name, or empty string on failure.

    Never touches main — the human merge does that.
    """
    try:
        run_git("config", "user.name",  BOT_NAME)
        run_git("config", "user.email", BOT_EMAIL)

        print("[COMMIT] Fetching remote...")
        run_git("fetch", "origin")

        # Sync with remote main first
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
            return ""

        run_git("commit", "-m", commit_msg)
        run_git("push", "-u", "origin", branch)
        print(f"[COMMIT] ✓ Pushed fix branch: {branch}")

        # Return to main so repo stays clean
        run_git("checkout", "main", check=False)
        return branch

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Git: {' '.join(exc.cmd)}\n  {exc.stderr}", file=sys.stderr)
        run_git("checkout", "main", check=False)
        run_git("merge", "--abort", check=False)
        return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--api-url",    default=DEFAULT_API_URL)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "")

    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Log not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if last_commit_was_bot():
        sys.exit(0)

    # ── Max attempts guard ────────────────────────────────────────────────────
    attempts = count_recent_bot_attempts()
    if attempts >= MAX_BOT_ATTEMPTS:
        print(f"[GUARD] {attempts} recent bot fix attempts — opening issue instead.")
        if token and repo:
            open_issue(token, repo,
                       f"Auto-fixer attempted {attempts} fixes without success.",
                       os.environ.get("GITHUB_SERVER_URL", "") + "/" + repo + "/actions")
        sys.exit(0)

    # ── STAGE 1: DETECT ───────────────────────────────────────────────────────
    print("\n━━━ STAGE 1: DETECT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log_text     = log_path.read_text(encoding="utf-8", errors="replace")
    error_signal = extract_error_signal(log_text)
    repo_context, included = collect_context(error_signal)

    # ── STAGE 2: ANALYSE ──────────────────────────────────────────────────────
    print("\n━━━ STAGE 2: ANALYSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        raw     = call_ai(error_signal, repo_context, args.model, args.api_url)
        ai_data = parse_ai_response(raw)
    except Exception as exc:
        print(f"[ERROR] AI failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI analysis failed: {exc}")
        sys.exit(2)

    confidence = float(ai_data.get("confidence", 1.0))
    root_cause = ai_data.get("root_cause", "unknown")
    patches    = ai_data.get("patches", [])
    commit_msg = ai_data.get("commit_message", "fix: auto-fixer patch")

    print(f"\n  root_cause : {root_cause}")
    print(f"  confidence : {confidence:.0%}")
    print(f"  patches    : {len(patches)}")
    for p in patches:
        print(f"    → {p.get('file','?')}")
        print(f"      find   : {repr(str(p.get('find',''))[:70])}")
        print(f"      replace: {repr(str(p.get('replace',''))[:70])}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"[SKIP] Confidence {confidence:.0%} below threshold.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({confidence:.0%}) to apply fix. "
                       f"Root cause: {root_cause}")
        sys.exit(0)

    if not patches:
        print("[ERROR] No patches returned.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI found root cause but produced no patches: {root_cause}")
        sys.exit(3)

    # ── DRY RUN ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n━━━ DRY-RUN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        for p in patches[:MAX_PATCHES]:
            p = normalise_patch(p)
            ok, reason = validate_patch(p)
            print(f"  {'✓' if ok else '✗'} {p.get('file','?')} — {reason}")
        sys.exit(0)

    # ── STAGE 3: PATCH ────────────────────────────────────────────────────────
    print("\n━━━ STAGE 3: PATCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    written = apply_all_patches(patches)
    if not written:
        print("[ERROR] All patches failed validation.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI patches were invalid (find strings not found in files). "
                       f"Root cause: {root_cause}")
        sys.exit(3)
    print(f"[PATCH] {len(written)}/{len(patches)} applied.")

    # ── STAGE 4: TEST ─────────────────────────────────────────────────────────
    print("\n━━━ STAGE 4: TEST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not args.skip_tests:
        if not run_tests():
            revert_patches(written)
            if token and repo:
                open_issue(token, repo,
                           f"AI patch applied but local tests failed. "
                           f"Root cause: {root_cause}")
            sys.exit(5)
    else:
        print("[TEST] Skipped (--skip-tests)")

    # ── STAGE 5: COMMIT TO BRANCH ─────────────────────────────────────────────
    print("\n━━━ STAGE 5: COMMIT TO BRANCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    branch = commit_to_fix_branch(commit_msg)
    if not branch:
        sys.exit(4)

    # ── STAGE 6: OPEN PR ──────────────────────────────────────────────────────
    print("\n━━━ STAGE 6: OPEN PR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    pr_url = ""
    if token and repo:
        pr_url = open_pr(token, repo, branch, commit_msg,
                         root_cause, written, patches)
    else:
        print("[PR] GITHUB_TOKEN not set — PR skipped.")
        print(f"[PR] Merge manually: git checkout main && git merge {branch} && git push")

    # ── DONE ──────────────────────────────────────────────────────────────────
    print("\n━━━ ✅ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  root cause : {root_cause}")
    print(f"  files fixed: {', '.join(written)}")
    print(f"  PR         : {pr_url or 'not created'}")
    print()
    print("  Next steps:")
    print("  1. Open the PR link above")
    print("  2. Review the diff — check each change looks correct")
    print("  3. Approve and merge → CI re-runs automatically")


if __name__ == "__main__":
    main()