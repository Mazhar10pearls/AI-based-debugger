#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows
100% AI-driven: no keyword matching, no hardcoded failure types.
Ollama reads the full log and decides what to fix.
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

# How many characters of the log to send to Ollama.
# Logs can be huge; the first 12 000 chars contain the failure in almost every case.
MAX_LOG_CHARS = 12_000


class WorkflowAnalyzer:
    def __init__(self, model: str = DEFAULT_MODEL, api_url: str = DEFAULT_API_URL):
        self.model   = model
        self.api_url = api_url

    def analyze(self, log_text: str) -> dict:
        """
        Send the raw CI/CD failure log to Ollama.
        Ollama identifies the root cause and returns the exact fix as JSON.
        No keyword matching — the AI does all the thinking.
        """
        # Trim log so we stay inside the model context window
        trimmed = log_text[:MAX_LOG_CHARS]
        if len(log_text) > MAX_LOG_CHARS:
            trimmed += "\n... [log truncated] ..."

        prompt = textwrap.dedent(f"""
            You are an expert DevOps engineer.
            Below is a raw CI/CD failure log from GitHub Actions.
            Your job is to:
            1. Identify exactly what caused the failure.
            2. Identify which file in the repository needs to be changed.
            3. Return the COMPLETE corrected file content.

            Important rules:
            - Respond ONLY with valid JSON — no explanation outside the JSON.
            - Do NOT wrap the JSON in markdown code fences.
            - "fixed_content" must be the COMPLETE file, not a diff or snippet.
            - If the fix requires changing a workflow YAML (e.g. wrong python-version),
              return the corrected YAML as fixed_content.
            - If you cannot determine the fix, still return valid JSON with
              fixed_file set to "unknown" and fixed_content set to "".

            Required JSON format:
            {{
                "root_cause": "one-line description of what went wrong",
                "severity": "critical|high|medium|low",
                "fix_type": "dependency|dockerfile|github-action|config|code",
                "fixed_file": "relative path to the file that needs changing",
                "fixed_content": "complete corrected file content",
                "commit_message": "fix: short description of the fix",
                "explanation": "one or two sentences explaining the fix"
            }}

            CI/CD Failure Log:
            {trimmed}
        """).strip()

        payload = {
            "model":       self.model,
            "prompt":      prompt,
            "max_tokens":  3000,
            "temperature": 0.1,
        }

        print(f"[AI] Sending log to Ollama ({self.api_url}) with model '{self.model}'...")
        try:
            resp = requests.post(self.api_url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.api_url}. Is it running?\n{exc}"
            )

        data = resp.json()
        raw  = data.get("choices", [{}])[0].get("text", "")

        print(f"[AI] Raw response ({len(raw)} chars):\n{raw[:500]}{'...' if len(raw) > 500 else ''}")

        # Strip markdown fences if the model wrapped the JSON anyway
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Extract the first JSON object from the response
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

        print(f"[FIXED] Written to {fixed_file}")
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

            # Check if there is actually anything to commit
            diff = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True
            )
            if diff.returncode == 0:
                print("[INFO] No file changes detected — nothing to commit.")
                return True

            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                check=True, capture_output=True
            )
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"[COMMITTED] {commit_msg}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Git operation failed: {exc.stderr.decode()}", file=sys.stderr)
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
    print(f"[INFO] Log size: {len(log_content)} chars")

    # ── Step 2: Send to Ollama ────────────────────────────────────────────────
    print("[STEP 2] Sending log to Ollama for AI analysis...")
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

    # ── Step 3: Apply fix and commit ──────────────────────────────────────────
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