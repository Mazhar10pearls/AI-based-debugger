#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows
100% AI-driven. Minimal prompt optimized for CPU inference speed.
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
MAX_LOG_CHARS   = 800


class WorkflowAnalyzer:
    def __init__(self, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL):
        self.model   = model
        self.api_url = api_url

    def _extract_error_lines(self, log_text):
        """
        Pull only critical error lines from the raw log.
        Excludes noisy-but-normal lines like allow-prereleases, fetch-depth etc.
        """
        # Lines containing these are actual errors
        error_keywords = [
            "error", "failed", "failure", "exception", "traceback",
            "exit code", "no module", "not found", "invalid", "fatal",
            "syntaxerror", "importerror", "cannot", "refused", "rejected",
            "killed", "denied", "timeout", "unavailable", "missing",
        ]

        # Lines containing these are normal setup output — skip them
        noise_keywords = [
            "allow-prereleases",
            "fetch-depth",
            "persist-credentials",
            "set-safe-directory",
            "sparse-checkout",
            "update-environment",
            "freethreaded",
            "check-latest",
            "##[group]",
            "##[endgroup]",
            "git config",
            "git submodule",
        ]

        lines    = log_text.splitlines()
        relevant = []
        for l in lines:
            stripped = l.strip()
            if not stripped:
                continue
            low = stripped.lower()
            # Skip noise lines
            if any(n in low for n in noise_keywords):
                continue
            # Keep error lines
            if any(k in low for k in error_keywords):
                relevant.append(stripped)

        # Always include last 20 lines — job conclusion lives here
        tail = [l.strip() for l in lines[-20:] if l.strip()]

        seen, out = set(), []
        for l in relevant + tail:
            if l not in seen:
                seen.add(l)
                out.append(l)

        result = "\n".join(out)
        print(f"[INFO] Extracted {len(out)} error lines ({len(result)} chars)")
        return result[-MAX_LOG_CHARS:]

    def analyze(self, log_text):
        snippet = self._extract_error_lines(log_text)
        print(f"[INFO] Sending {len(snippet)} chars to Ollama...")

        prompt = (
            "You are a DevOps expert. Analyze this GitHub Actions failure log.\n\n"
            "Failure log:\n"
            f"{snippet}\n\n"
            "Identify the root cause and which file needs to be fixed.\n"
            "Common causes: wrong runner labels in runs-on, wrong python-version,\n"
            "bad package version in requirements.txt, broken Dockerfile.\n\n"
            "Respond with a single JSON object with these fields:\n"
            "- root_cause: what went wrong\n"
            "- severity: critical, high, medium, or low\n"
            "- fix_type: dependency, dockerfile, github-action, config, or code\n"
            "- fixed_file: relative path to the file that needs fixing\n"
            "- fixed_content: the COMPLETE corrected file content\n"
            "- commit_message: short git commit message starting with fix:\n"
            "- explanation: one sentence explaining the fix\n\n"
            "Return ONLY the JSON object, no other text."
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
            raise RuntimeError("Ollama timed out (180s). VM CPU may be overloaded.")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.api_url}.\n{exc}")

        raw = resp.json().get("choices", [{}])[0].get("text", "")
        print(f"[AI] Response ({len(raw)} chars): {raw[:400]}{'...' if len(raw) > 400 else ''}")

        # Strip markdown fences if model added them
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Model sometimes outputs empty {} before the real answer — take the longest match
        matches = re.findall(r"\{.*?\}", raw, re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON found in Ollama response:\n{raw}")

        best = max(matches, key=len)
        return json.loads(best)


class AutoFixer:
    def __init__(self, repo_path="."):
        os.chdir(repo_path)

    def apply_fix(self, fix_data):
        fixed_file    = fix_data.get("fixed_file", "")
        fixed_content = fix_data.get("fixed_content", "")

        if not fixed_file or fixed_file == "unknown":
            print("[WARN] AI could not determine which file to fix.", file=sys.stderr)
            return False
        if not fixed_content:
            print("[WARN] AI returned empty fixed_content.", file=sys.stderr)
            return False

        Path(fixed_file).parent.mkdir(parents=True, exist_ok=True)
        Path(fixed_file).write_text(fixed_content, encoding="utf-8")
        print(f"[FIXED] Written: {fixed_file}")
        return True

    def commit_fix(self, fix_data):
        commit_msg = fix_data.get("commit_message", "fix: auto-fixer patch")
        try:
            subprocess.run(["git", "config", "user.name",  "github-actions[bot]"],
                           check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email",
                            "github-actions[bot]@users.noreply.github.com"],
                           check=True, capture_output=True)
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)

            diff = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                  capture_output=True)
            if diff.returncode == 0:
                print("[INFO] No changes to commit.")
                return True

            subprocess.run(["git", "commit", "-m", commit_msg],
                           check=True, capture_output=True)
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"[COMMITTED] {commit_msg}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Git failed: {exc.stderr.decode()}", file=sys.stderr)
            return False

    def auto_fix(self, fix_data, commit=True):
        if not self.apply_fix(fix_data):
            return False
        if commit:
            return self.commit_fix(fix_data)
        return True


def main():
    parser = argparse.ArgumentParser(description="AI Auto-Fixer for CI/CD failures.")
    parser.add_argument("--input",     "-i", required=True)
    parser.add_argument("--model",           default=DEFAULT_MODEL)
    parser.add_argument("--api-url",         default=DEFAULT_API_URL)
    parser.add_argument("--no-commit",       action="store_true")
    parser.add_argument("--repo",            default=".")
    args = parser.parse_args()

    print("[STEP 1] Reading failure log...")
    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
        return 1
    log_content = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[INFO] Raw log: {len(log_content)} chars")

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