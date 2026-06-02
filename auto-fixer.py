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
import yaml

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
        error_keywords = [
            "error", "failed", "failure", "exception", "traceback",
            "exit code", "no module", "not found", "invalid", "fatal",
            "syntaxerror", "importerror", "cannot", "refused", "rejected",
            "killed", "denied", "timeout", "unavailable", "missing",
        ]

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
            if any(n in low for n in noise_keywords):
                continue
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

        # FIX: greedy match to capture the full JSON object, not the first small {}
        matches = re.findall(r"\{.*\}", raw, re.DOTALL)
        if not matches:
            raise ValueError(f"No JSON found in Ollama response:\n{raw}")

        best = max(matches, key=len)
        return json.loads(best)


class AutoFixer:
    def __init__(self, repo_path="."):
        os.chdir(repo_path)

    def _sanitize_workflow(self, filepath, ai_content):
        """
        Validate and sanitize AI-generated workflow YAML before writing.
        - Must parse as valid YAML
        - Must have 'jobs' key (minimum valid workflow)
        - Always injects workflow_dispatch so re-trigger never 422s again
        - Preserves original workflow name if AI set it to garbage
        """
        # Parse AI content
        try:
            ai_yaml = yaml.safe_load(ai_content)
        except yaml.YAMLError as e:
            print(f"[WARN] AI workflow content is invalid YAML: {e}", file=sys.stderr)
            return None

        if not isinstance(ai_yaml, dict):
            print("[WARN] AI workflow content is not a YAML mapping — rejecting.", file=sys.stderr)
            return None

        if "jobs" not in ai_yaml:
            print("[WARN] AI workflow content has no 'jobs' key — garbage output, rejecting.", file=sys.stderr)
            return None

        # Read original file if it exists so we can preserve key fields
        original_path = Path(filepath)
        original_yaml = {}
        if original_path.exists():
            try:
                original_yaml = yaml.safe_load(original_path.read_text(encoding="utf-8")) or {}
            except Exception:
                original_yaml = {}

        # Always inject workflow_dispatch into the 'on' section
        on_section = ai_yaml.get("on", {})

        if on_section is None:
            on_section = {}

        if isinstance(on_section, str):
            # e.g. on: push  →  expand to dict
            on_section = {on_section: None}
        elif isinstance(on_section, list):
            # e.g. on: [push, pull_request]  →  expand to dict
            on_section = {k: None for k in on_section}

        # Now it's a dict — inject workflow_dispatch
        if "workflow_dispatch" not in on_section:
            on_section["workflow_dispatch"] = None
            print("[INFO] Injected workflow_dispatch trigger into workflow.")

        ai_yaml["on"] = on_section

        # Preserve original workflow name if AI set it to something wrong
        # (e.g. AI set it to the repo full name like "org/repo")
        ai_name = str(ai_yaml.get("name", ""))
        if not ai_yaml.get("name") or "/" in ai_name:
            if original_yaml.get("name"):
                ai_yaml["name"] = original_yaml["name"]
                print(f"[INFO] Restored original workflow name: {original_yaml['name']}")

        return yaml.dump(ai_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def apply_fix(self, fix_data):
        fixed_file    = fix_data.get("fixed_file", "")
        fixed_content = fix_data.get("fixed_content", "")

        if not fixed_file or fixed_file == "unknown":
            print("[WARN] AI could not determine which file to fix.", file=sys.stderr)
            return False
        if not fixed_content:
            print("[WARN] AI returned empty fixed_content.", file=sys.stderr)
            return False

        # Sanitize workflow files before writing — prevents Ollama from
        # overwriting with garbage and stripping workflow_dispatch trigger
        if fixed_file.startswith(".github/workflows/") and fixed_file.endswith(".yml"):
            sanitized = self._sanitize_workflow(fixed_file, fixed_content)
            if sanitized is None:
                print("[WARN] Skipping overwrite — AI content failed validation.", file=sys.stderr)
                return False
            fixed_content = sanitized

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

            # Explicitly push to origin HEAD branch — avoids detached HEAD failure
            # which is common in workflow_run triggered jobs
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True
            ).stdout.strip()

            if branch == "HEAD":
                # Detached HEAD — fall back to main
                branch = "main"

            subprocess.run(["git", "push", "origin", f"HEAD:{branch}"],
                           check=True, capture_output=True)
            print(f"[COMMITTED] {commit_msg} → pushed to {branch}")
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