#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows
100% AI-driven. Minimal prompt to work within CPU inference speed limits.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

DEFAULT_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
DEFAULT_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3.2")

# At 20s per 118 tokens on this VM, keep total prompt under 200 tokens (~800 chars)
MAX_LOG_CHARS = 800


class WorkflowAnalyzer:
    def __init__(self, model: str = DEFAULT_MODEL, api_url: str = DEFAULT_API_URL):
        self.model   = model
        self.api_url = api_url

    def _extract_error_lines(self, log_text: str) -> str:
        """Pull only the most critical error lines from the log."""
        error_keywords = [
            "error", "failed", "failure", "exception", "traceback",
            "exit code", "no module", "not found", "invalid", "fatal",
            "python-version", "syntaxerror", "importerror", "cannot",
        ]
        lines = log_text.splitlines()

        # Grab error lines + last 20 lines (failure summary)
        relevant = [
            line.strip() for line in lines
            if any(k in line.lower() for k in error_keywords)
        ]
        tail = [line.strip() for line in lines[-20:]]

        seen = set()
        out = []
        for line in relevant + tail:
            if line and line not in seen:
                seen.add(line)
                out.append(line)

        result = "\n".join(out)
        print(f"[INFO] Extracted {len(out)} error lines ({len(result)} chars)")
        return result[-MAX_LOG_CHARS:]  # take the tail (most recent errors)

    def analyze(self, log_text: str) -> dict:
        error_snippet = self._extract_error_lines(log_text)
        print(f"[INFO] Sending {len(error_snippet)} chars to Ollama...")

        # Minimal prompt — every extra word costs ~2s on this VM
        prompt = (
            "CI/CD failure log:\n"
            f"{error_snippet}\n\n"
            "Return ONLY valid JSON, no other text:\n"
            '{"root_cause":"...","severity":"critical|high|medium|low",'
            '"fix_type":"dependency|dockerfile|github-action|config|code",'
            '"fixed_file":"relative/path/to/file",'
            '"fixed_content":"complete corrected file content",'
            '"commit_message":"fix: ...",'
            '"explanation":"..."}'
        )

        payload = {
            "model":       self.model,
            "prompt":      prompt,
            "max_tokens":  1000,
            "temperature": 0.1,
        }

        print(f"[AI] Prompt size: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        print(f"[AI] Contacting Ollama ({self.model})...")

        try:
            resp = requests.post(self.api_url, json=payload, timeout=180)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama timed out. VM CPU may be overloaded.")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.api_url}.\n{exc}")

        raw = resp.json().get("choices", [{}])[0].get("text", "")
        print(f"[AI] Response ({len(raw)} chars): {raw[:300]}{'...' if len(raw)>300 else ''}")

        # Clean up markdown fences if model added them
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON in Ollama response:\n{raw}")

        return json.loads(json_match.group())


class AutoFixer:
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
            subprocess.run(["git", "config", "user.name",  "github-actions[bot]"], check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True, capture_output=True)
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)

            diff = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
            if diff.returncode == 0:
                print("[INFO] No changes to commit.")
                return True

            subprocess.run(["git", "commit", "-m", commit_msg], check=True, capture_output=True)
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
    parser = argparse.ArgumentParser(description="AI Auto-Fixer for CI/CD failures.")
    parser.add_argument("--input",     "-i", required=True)
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    parser.add_argument("--api-url",   default=DEFAULT_API_URL)
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--repo",      default=".")
    args = parser.parse_args()

    # Step 1: Read log
    print("[STEP 1] Reading failure log...")
    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Not found: {args.input}", file=sys.stderr)
        return 1
    log_content = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[INFO] Raw log: {len(log_content)} chars")

    # Step 2: AI analysis
    print("[STEP 2] Analyzing with Ollama...")
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

    # Step 3: Apply and commit
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