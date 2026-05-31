#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows
100% AI-driven: no keyword matching, no hardcoded failure types.
Ollama reads the filtered log and decides what to fix.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import requests

DEFAULT_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
DEFAULT_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3")

# Send only the most relevant portion of the log to Ollama.
# CPU-only VMs time out on large contexts — keep this small.
MAX_LOG_CHARS = 4_000


class WorkflowAnalyzer:
    def __init__(self, model: str = DEFAULT_MODEL, api_url: str = DEFAULT_API_URL):
        self.model   = model
        self.api_url = api_url

    def _extract_relevant_lines(self, log_text: str) -> str:
        """
        Filter the log down to lines that actually matter.
        GitHub Actions logs are full of git/setup noise — strip it out
        so Ollama only sees the error-relevant lines.
        """
        error_keywords = [
            "error", "failed", "failure", "exception", "traceback",
            "exit code", "no module", "not found", "invalid", "fatal",
            "warning", "denied", "timeout", "killed", "cannot", "unable",
            "python-version", "setup-python", "syntaxerror", "importerror",
        ]
        lines = log_text.splitlines()

        relevant = [
            line for line in lines
            if any(k in line.lower() for k in error_keywords)
        ]

        # Always include the last 60 lines — the failure summary lives here
        tail = lines[-60:]

        # Merge, deduplicate, preserve order
        seen = set()
        out = []
        for line in relevant + tail:
            if line not in seen:
                seen.add(line)
                out.append(line)

        result = "\n".join(out)
        print(f"[INFO] Filtered log: {len(lines)} lines → {len(out)} relevant lines ({len(result)} chars)")
        return result

    def analyze(self, log_text: str) -> dict:
        """
        Send the filtered CI/CD failure log to Ollama.
        The AI identifies the root cause and returns the exact fix as JSON.
        """
        filtered = self._extract_relevant_lines(log_text)

        # Trim to MAX_LOG_CHARS from the tail (most recent = most relevant)
        if len(filtered) > MAX_LOG_CHARS:
            filtered = filtered[-MAX_LOG_CHARS:]
            print(f"[INFO] Further trimmed to last {MAX_LOG_CHARS} chars for model context.")

        prompt = textwrap.dedent(f"""
            You are an expert DevOps engineer.
            Below is a filtered CI/CD failure log from GitHub Actions.
            Your job:
            1. Identify exactly what caused the failure.
            2. Identify which repository file needs to be changed.
            3. Return the COMPLETE corrected file content.

            Rules:
            - Respond ONLY with valid JSON — no text outside the JSON.
            - Do NOT use markdown code fences.
            - "fixed_content" must be the COMPLETE corrected file, not a diff.
            - Common fixes: wrong python-version in workflow YAML, bad package
              version in requirements.txt, broken Dockerfile base image, etc.
            - If unsure, set fixed_file to "unknown" and fixed_content to "".

            JSON format:
            {{
                "root_cause": "one-line description of what went wrong",
                "severity": "critical|high|medium|low",
                "fix_type": "dependency|dockerfile|github-action|config|code",
                "fixed_file": "relative/path/to/file",
                "fixed_content": "complete corrected file content",
                "commit_message": "fix: short description",
                "explanation": "one or two sentences explaining the fix"
            }}

            Filtered failure log:
            {filtered}
        """).strip()

        payload = {
            "model":       self.model,
            "prompt":      prompt,
            "max_tokens":  2000,
            "temperature": 0.1,
        }

        print(f"[AI] Contacting Ollama at {self.api_url} with model '{self.model}'...")
        print(f"[AI] Prompt size: {len(prompt)} chars")

        try:
            resp = requests.post(self.api_url, json=payload, timeout=180)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                "Ollama timed out (180s). The log may still be too large, "
                "or the model is under heavy load. Try reducing MAX_LOG_CHARS further."
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.api_url}. Is it running?\n{exc}"
            )

        data = resp.json()
        raw  = data.get("choices", [{}])[0].get("text", "")

        print(f"[AI] Response received ({len(raw)} chars)")
        print(f"[AI] Preview: {raw[:300]}{'...' if len(raw) > 300 else ''}")

        # Strip markdown fences if the model added them anyway
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        raw = raw.replace("```", "").strip()

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"Ollama did not return valid JSON.\nFull response:\n{raw}")

        return json.loads(json_match.group())


class AutoFixer:
    """Apply the AI-generated fix and commit it to git."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        os.chdir(repo_path)

    def apply_fix(self, fix_data: dict) -> bool:
        fixed_file    = fix_data.get("fixed_file", "")
        fixed_content = fix_data.get("fixed_content", "")

        if not fixed_file or fixed_file == "unknown":
            print("[WARN] AI could not determine which file to fix.", file=sys.stderr)
            return False

        if not fixed_content:
            print("[WARN] AI returned empty fixed_content.", file=sys.stderr)
            return False

        Path(fixed_file).parent.mkdir(parents=True, exist_ok=True)
        with open(fixed_file, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        print(f"[FIXED] Written: {fixed_file}")
        return True

    def commit_fix(self, fix_data: dict) -> bool:
        commit_msg = fix_data.get("commit_message", "fix: auto-fixer patch")
        try:
            subprocess.run(
                ["git", "config", "user.name", "github-actions[bot]"],
                check=True, capture_output=True
            )
            subprocess.run(
                ["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"],
                check=True, capture_output=True
            )
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)

            # Only commit if there are actual changes
            diff = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True
            )
            if diff.returncode == 0:
                print("[INFO] No file changes to commit.")
                return True

            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                check=True, capture_output=True
            )
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"[COMMITTED] {commit_msg}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Git failed: {exc.stderr.decode()}", file=sys.stderr)
            return False

    def auto_fix(self, fix_data: dict, commit: bool = True) -> bool:
        if not self.apply_fix(fix_data):
            return False
        if commit:
            return self.commit_fix(fix_data)
        return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Auto-Fixer: fully AI-driven CI/CD failure detection and fix."
    )
    parser.add_argument("--input",     "-i", required=True, help="Path to failure log.")
    parser.add_argument("--model",     default=DEFAULT_MODEL,   help="Ollama model name.")
    parser.add_argument("--api-url",   default=DEFAULT_API_URL, help="Ollama API endpoint.")
    parser.add_argument("--no-commit", action="store_true",     help="Apply fix but skip git commit.")
    parser.add_argument("--repo",      default=".",             help="Path to git repository.")
    args = parser.parse_args()

    # ── Step 1: Read the log ──────────────────────────────────────────────────
    print("[STEP 1] Reading failure log...")
    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        return 1
    log_content = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[INFO] Raw log size: {len(log_content)} chars")

    # ── Step 2: Send to Ollama ────────────────────────────────────────────────
    print("[STEP 2] Sending filtered log to Ollama...")
    analyzer = WorkflowAnalyzer(model=args.model, api_url=args.api_url)
    try:
        fix_data = analyzer.analyze(log_content)
    except Exception as exc:
        print(f"[ERROR] AI analysis failed: {exc}", file=sys.stderr)
        return 2

    print(f"\n[ANALYSIS]")
    print(f"  Root cause : {fix_data.get('root_cause', 'unknown')}")
    print(f"  Fix type   : {fix_data.get('fix_type', 'unknown')}")
    print(f"  Severity   : {fix_data.get('severity', 'unknown')}")
    print(f"  File to fix: {fix_data.get('fixed_file', 'unknown')}")
    print(f"  Explanation: {fix_data.get('explanation', '')}")

    # ── Step 3: Apply and commit ──────────────────────────────────────────────
    print("\n[STEP 3] Applying fix...")
    fixer = AutoFixer(repo_path=args.repo)
    if not fixer.auto_fix(fix_data, commit=not args.no_commit):
        return 3

    print("\n=== AUTO-FIX COMPLETE ===")
    print(f"  Root cause    : {fix_data.get('root_cause')}")
    print(f"  Fixed file    : {fix_data.get('fixed_file')}")
    print(f"  Commit message: {fix_data.get('commit_message')}")
    print(f"  Explanation   : {fix_data.get('explanation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())