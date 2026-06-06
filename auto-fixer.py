#!/usr/bin/env python3
"""
Self-Healing CI Auto-Fixer — Surgical Patch Edition
====================================================
Key design: AI never rewrites files. It only returns:
  { "find": "exact string to find", "replace": "exact replacement" }

Python applies the change with a simple str.replace().
This means:
  - YAML structure is never corrupted by the AI
  - Dockerfile is never mangled
  - Python files are never broken by bad indentation
  - The AI only needs to identify WHAT is wrong — a much easier task

Pipeline: DETECT → ANALYSE → PATCH → COMMIT → (push triggers CI)
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
DEFAULT_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
DEFAULT_MODEL   = os.environ.get("OLLAMA_MODEL",   "llama3.2")

MAX_LOG_CHARS   = 2000
MAX_FILE_CHARS  = 2000
MAX_RETRIES     = 3
RETRY_BACKOFF   = [10, 30, 60]

BOT_NAME        = "github-actions[bot]"
BOT_EMAIL       = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX      = "fix:"

CONTEXT_GLOBS = [
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "Dockerfile*",
    "**/Dockerfile*",
    "docker-compose*.yml",
    "requirements*.txt",
    "**/requirements*.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "package.json",
    "Makefile",
    "*.sh",
    "**/*.sh",
    "**/*.py",
    "**/*.js",
    "**/*.ts",
    "**/*.go",
    "**/*.java",
]

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", ".mypy_cache", ".pytest_cache",
}


# ── Utilities ─────────────────────────────────────────────────────────────────

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


def _should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts)


def collect_repo_context() -> str:
    import glob as globmod
    seen, parts = set(), []
    for pattern in CONTEXT_GLOBS:
        for path_str in globmod.glob(pattern, recursive=True):
            p = Path(path_str)
            if not p.is_file() or str(p.resolve()) in seen or _should_skip(p):
                continue
            seen.add(str(p.resolve()))
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                if len(content) > MAX_FILE_CHARS:
                    content = content[:MAX_FILE_CHARS] + f"\n...(truncated)"
                parts.append(f"### FILE: {p}\n```\n{content}\n```\n")
            except Exception as exc:
                parts.append(f"### FILE: {p}\n(unreadable: {exc})\n")
    result = "\n".join(parts)
    print(f"[DETECT] Repo context: {len(seen)} files, {len(result)} chars")
    return result


def extract_error_signal(log_text: str) -> str:
    ERROR_KW = [
        "error", "failed", "failure", "exception", "traceback",
        "exit code", "exitcode", "no module", "not found", "invalid",
        "fatal", "syntaxerror", "importerror", "nameerror", "typeerror",
        "valueerror", "attributeerror", "assertionerror", "runtimeerror",
        "cannot", "refused", "rejected", "killed", "denied", "missing",
        "undefined", "permission denied", "command not found",
        "connectionerror", "timeout", "step failed", "build failed",
        "test failed", "pytest", "npm err", "returned non-zero",
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
    tail   = [l.strip() for l in lines[-40:] if l.strip()]
    merged = list(dict.fromkeys(relevant + tail))
    result = "\n".join(merged)
    print(f"[DETECT] Error signal: {len(merged)} lines, {len(result)} chars")
    return result[-MAX_LOG_CHARS:]


# ── AI call ───────────────────────────────────────────────────────────────────

PROMPT = """\
You are a senior DevOps engineer. A CI/CD pipeline has failed.
Your job: find every bug and return ONLY the minimal string changes needed to fix them.

## CI Error Log
{error_signal}

## Repository Files
{repo_context}

## CRITICAL INSTRUCTIONS
- Do NOT rewrite entire files
- Do NOT return full file contents
- Return ONLY the exact string to find and its exact replacement
- The "find" value must be an exact substring that exists in the file right now
- The "replace" value must be the corrected version of that exact substring
- Keep find/replace as SHORT as possible — one line is ideal

## Response format
Return ONLY this JSON. No markdown, no explanation outside the JSON.

{{
  "root_cause": "one sentence — what went wrong",
  "explanation": "one sentence — what you changed and why",
  "commit_message": "fix: short description (must start with fix:)",
  "patches": [
    {{
      "file": "relative/path/to/file",
      "find": "exact string currently in the file",
      "replace": "corrected string to replace it with"
    }}
  ]
}}

Examples of good patches:
  Wrong python version in workflow:
    "file": ".github/workflows/ci-local-deploy.yml"
    "find": "python-version: '3.13'"
    "replace": "python-version: '3.12'"

  Wrong Docker base image:
    "file": "sample_app/Dockerfile"
    "find": "FROM python:3.13-slim"
    "replace": "FROM python:3.12-slim"

  Missing pip package:
    "file": "requirements.txt"
    "find": "flask==99.0.0"
    "replace": "flask==2.3.3"

  Python import typo:
    "file": "sample_app/app.py"
    "find": "from flask import render_template_strng"
    "replace": "from flask import render_template_string"

Return ONLY the JSON object."""


def call_ai(error_signal: str, repo_context: str,
            model: str, api_url: str) -> str:
    prompt  = PROMPT.format(
        error_signal=error_signal,
        repo_context=repo_context,
    )
    payload = {
        "model":       model,
        "prompt":      prompt,
        "temperature": 0.1,
        "max_tokens":  1500,   # Much less needed — no full file rewrites
    }
    print(f"[AI] Prompt: {len(prompt)} chars (~{len(prompt)//4} tokens)")

    for attempt in range(MAX_RETRIES):
        try:
            print(f"[AI] Calling Ollama — attempt {attempt + 1}/{MAX_RETRIES}...")
            resp = requests.post(api_url, json=payload, timeout=180)
            resp.raise_for_status()
            raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
            print(f"[AI] Got {len(raw)} chars: {raw[:300]}{'...' if len(raw) > 300 else ''}")
            return raw
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"[AI] Timeout. Retry in {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Cannot reach Ollama at {api_url}: {exc}")

    raise RuntimeError(f"Ollama timed out on all {MAX_RETRIES} attempts.")


def parse_ai_response(raw: str) -> dict:
    """Robustly extract JSON from AI output."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find largest balanced {...} block
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

    raise ValueError(f"No valid JSON in AI response:\n{raw[:600]}")


# ── Surgical patch application ────────────────────────────────────────────────

def validate_patch(patch: dict) -> tuple[bool, str]:
    """Check the patch is safe before touching disk."""
    file    = (patch.get("file") or "").strip()
    find    = patch.get("find",    "")
    replace = patch.get("replace", "")

    if not file:
        return False, "missing 'file' field"
    if ".." in file or file.startswith("/") or file.startswith("~"):
        return False, f"unsafe path: {file}"
    if not find:
        return False, f"empty 'find' for {file}"
    if replace is None:
        return False, f"missing 'replace' for {file}"
    if not Path(file).exists():
        return False, f"file does not exist on disk: {file}"

    # Make sure 'find' actually exists in the file
    content = Path(file).read_text(encoding="utf-8", errors="replace")
    if find not in content:
        return False, f"'find' string not found in {file}: {repr(find[:80])}"

    # After applying, validate YAML structure for workflow files
    if re.search(r"\.github/workflows/.+\.ya?ml$", file):
        patched = content.replace(find, replace, 1)
        try:
            parsed = yaml.safe_load(patched)
            if not isinstance(parsed, dict) or "jobs" not in parsed:
                return False, f"patched {file} would be missing 'jobs:' key"
        except yaml.YAMLError as exc:
            return False, f"patched {file} would be invalid YAML: {exc}"

    # After applying, validate Python syntax for .py files
    if file.endswith(".py"):
        patched = content.replace(find, replace, 1)
        try:
            compile(patched, file, "exec")
        except SyntaxError as exc:
            return False, f"patched {file} would have Python syntax error: {exc}"

    return True, "ok"


def apply_patch(patch: dict) -> bool:
    """
    Apply a single surgical patch.
    Reads the file, does ONE str.replace(), writes back.
    The AI never touches the file structure — only the specific value.
    """
    file    = patch["file"].strip()
    find    = patch["find"]
    replace = patch["replace"]

    content = Path(file).read_text(encoding="utf-8", errors="replace")

    # Count occurrences so we know what we're doing
    count = content.count(find)
    if count > 1:
        print(f"[WARN] '{find[:50]}' found {count} times in {file} — replacing first occurrence only")

    patched = content.replace(find, replace, 1)
    Path(file).write_text(patched, encoding="utf-8")

    print(f"[FIX] ✓ {file}")
    print(f"       - was : {repr(find[:80])}")
    print(f"       + now : {repr(replace[:80])}")
    return True


def apply_all_patches(patches: list) -> list[str]:
    """Validate then apply every patch. Returns list of patched file paths."""
    written = []
    for patch in patches:
        # Normalise — AI sometimes wraps values in extra quotes or escapes
        for key in ("find", "replace"):
            val = patch.get(key, "")
            if isinstance(val, str):
                # Unescape \\n → real newlines if model escaped them
                if "\\n" in val and "\n" not in val:
                    patch[key] = val.replace("\\n", "\n")

        ok, reason = validate_patch(patch)
        if not ok:
            print(f"[FIX] ✗ Skipping patch: {reason}", file=sys.stderr)
            continue

        if apply_patch(patch):
            written.append(patch["file"].strip())

    return written


# ── Git: clean fetch → merge → commit → push ─────────────────────────────────

def commit_and_push(commit_msg: str) -> bool:
    """
    Clean pull strategy:
      1. git fetch origin          (download remote, don't touch working tree)
      2. git merge -X ours ...     (merge remote in; our fix wins on conflict)
      3. git add -A
      4. git commit + push
    No stash, no rebase, no pop — patch files on disk are never disturbed.
    """
    try:
        run_git("config", "user.name",  BOT_NAME)
        run_git("config", "user.email", BOT_EMAIL)

        print("[COMMIT] Fetching remote...")
        run_git("fetch", "origin")

        print("[COMMIT] Merging remote (our fix wins on conflict)...")
        merge = subprocess.run(
            ["git", "merge", "-X", "ours", "origin/main",
             "--no-edit", "-m", f"merge: sync before {commit_msg}"],
            capture_output=True, text=True
        )
        if merge.returncode != 0:
            print(f"[WARN] Merge skipped (non-fatal): {merge.stderr.strip()}")
            run_git("merge", "--abort", check=False)

        run_git("add", "-A")

        if run_git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit — already up to date.")
            return True

        run_git("commit", "-m", commit_msg)
        run_git("push")
        print(f"[COMMIT] ✓ Pushed: {commit_msg}")
        return True

    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        print(f"[ERROR] Git failed: {' '.join(exc.cmd)}\n  {stderr}", file=sys.stderr)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Surgical self-healing CI fixer — AI finds bugs, Python fixes files"
    )
    parser.add_argument("--input",   required=True, help="Path to failure.log")
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show patches without writing or pushing")
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
    repo_context = collect_repo_context()

    # ── ANALYSE ───────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 2: ANALYSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        raw     = call_ai(error_signal, repo_context, args.model, args.api_url)
        ai_data = parse_ai_response(raw)
        patches = ai_data.get("patches", [])

        print(f"\n  root_cause : {ai_data.get('root_cause', 'unknown')}")
        print(f"  explanation: {ai_data.get('explanation', '')}")
        print(f"  patches    : {len(patches)} change(s) identified")
        for p in patches:
            print(f"    → {p.get('file', '?')}")
            print(f"      find   : {repr(str(p.get('find', ''))[:60])}")
            print(f"      replace: {repr(str(p.get('replace', ''))[:60])}")

        commit_msg = ai_data.get("commit_message", "fix: auto-fixer patch")

    except Exception as exc:
        print(f"\n[ERROR] AI analysis failed: {exc}", file=sys.stderr)
        sys.exit(2)

    if not patches:
        print("[ERROR] AI returned no patches.", file=sys.stderr)
        sys.exit(3)

    # ── DRY RUN ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n━━━ DRY-RUN — validating patches ━━━━━━━━━━━━━━━━━━━━━━━")
        for p in patches:
            ok, reason = validate_patch(p)
            print(f"  {'✓' if ok else '✗'} {p.get('file', '?')} — {reason}")
        print("\n[DRY-RUN] Nothing written.")
        sys.exit(0)

    # ── PATCH ─────────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 3: PATCH ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    written = apply_all_patches(patches)
    if not written:
        print("[ERROR] All patches failed validation — nothing written.", file=sys.stderr)
        sys.exit(3)
    print(f"\n[PATCH] {len(written)}/{len(patches)} patches applied successfully.")

    # ── COMMIT ────────────────────────────────────────────────────────────────
    print("\n━━━ STAGE 4: COMMIT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not commit_and_push(commit_msg):
        sys.exit(4)

    # ── DONE ──────────────────────────────────────────────────────────────────
    print("\n━━━ ✅ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  commit : {commit_msg}")
    print(f"  files  : {', '.join(written)}")
    print("  CI will re-run automatically from the push.")


if __name__ == "__main__":
    main()