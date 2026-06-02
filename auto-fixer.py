#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows (Production-Ready)
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
MAX_LOG_CHARS   = 600


# ---------------------------
# 🔧 Extraction Utilities
# ---------------------------

def extract_json(text):
    """Extract valid JSON using bracket matching."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found")

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i+1]

    raise ValueError("Incomplete JSON")


def safe_load_json(text):
    """Attempt to repair and load JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("[WARN] Repairing JSON...")

        text = text.replace('"{}"', '{}')
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        return json.loads(text)


def extract_yaml(text):
    """Extract YAML from messy AI output."""
    text = re.sub(r"```(?:yaml|json)?", "", text)
    text = text.replace("```", "").strip()

    # Heuristic: YAML starts with 'name:' or 'on:'
    match = re.search(r"(name:.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def normalize_path(path):
    """Ensure workflow path is correct."""
    if not path.startswith(".github/workflows"):
        return f".github/workflows/{path}"
    return path


# ---------------------------
# 🧠 Analyzer
# ---------------------------

class WorkflowAnalyzer:
    def __init__(self, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL):
        self.model = model
        self.api_url = api_url

    def _extract_error_lines(self, log_text):
        keywords = ["error", "failed", "exception", "exit code", "invalid"]
        lines = log_text.splitlines()

        relevant = [l.strip() for l in lines if any(k in l.lower() for k in keywords)]
        tail = [l.strip() for l in lines[-15:] if l.strip()]

        merged = list(dict.fromkeys(relevant + tail))
        result = "\n".join(merged)

        print(f"[INFO] Extracted {len(merged)} error lines")
        return result[-MAX_LOG_CHARS:]

    def _find_workflow_file(self):
        p = Path(".github/workflows/ci-local-deploy.yml")
        if p.exists():
            content = p.read_text()
            print(f"[INFO] Found workflow file: {p}")
            return content
        return ""

    def analyze(self, log_text):
        snippet = self._extract_error_lines(log_text)
        workflow = self._find_workflow_file()

        prompt = (
            "Fix this GitHub Actions workflow.\n\n"
            f"Errors:\n{snippet}\n\n"
            f"Workflow:\n{workflow}\n\n"
            "Return ONLY valid JSON.\n"
            "fixed_content MUST be a STRING (YAML).\n"
            "No triple quotes.\n"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": 1500,
        }

        resp = requests.post(self.api_url, json=payload, timeout=180)
        resp.raise_for_status()

        raw = resp.json().get("choices", [{}])[0].get("text", "")
        print(f"[AI RAW]: {raw[:300]}...")

        # ---------------------------
        # ✅ Try JSON Mode
        # ---------------------------
        try:
            extracted = extract_json(raw)
            data = safe_load_json(extracted)

            if "fixed_content" in data:
                data["fixed_file"] = normalize_path(data.get("fixed_file", "ci-local-deploy.yml"))
                return data

        except Exception as e:
            print(f"[WARN] JSON failed: {e}")

        # ---------------------------
        # 🔥 FALLBACK YAML MODE
        # ---------------------------
        print("[INFO] Switching to YAML fallback mode")

        yaml_content = extract_yaml(raw)

        return {
            "root_cause": "fallback_yaml_mode",
            "severity": "critical",
            "fix_type": "config",
            "fixed_file": ".github/workflows/ci-local-deploy.yml",
            "fixed_content": yaml_content,
            "commit_message": "fix: auto-fixer fallback YAML mode",
            "explanation": "Recovered from invalid AI JSON output"
        }


# ---------------------------
# 🛠 Auto Fixer
# ---------------------------

class AutoFixer:
    def __init__(self, repo_path="."):
        os.chdir(repo_path)

    def apply_fix(self, fix_data):
        file = fix_data["fixed_file"]
        content = fix_data["fixed_content"]

        if not file or not content:
            print("[ERROR] Invalid fix data")
            return False

        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Path(file).write_text(content)

        print(f"[FIXED] {file}")
        return True

    def commit_fix(self, fix_data):
        msg = fix_data.get("commit_message", "fix: auto patch")

        subprocess.run(["git", "config", "user.name", "github-actions[bot]"])
        subprocess.run(["git", "config", "user.email", "bot@github.com"])

        subprocess.run(["git", "add", "-A"])

        if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
            print("[INFO] No changes")
            return True

        subprocess.run(["git", "commit", "-m", msg])
        subprocess.run(["git", "push"])

        print(f"[COMMITTED] {msg}")
        return True

    def auto_fix(self, fix_data):
        if not self.apply_fix(fix_data):
            return False
        return self.commit_fix(fix_data)


# ---------------------------
# 🚀 Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    log = Path(args.input).read_text()

    analyzer = WorkflowAnalyzer()

    # ✅ Retry logic
    for attempt in range(3):
        try:
            fix = analyzer.analyze(log)
            break
        except Exception as e:
            print(f"[Retry {attempt+1}] {e}")
    else:
        print("[ERROR] AI failed after retries")
        return 2

    print("\n[ANALYSIS]")
    print(f"Root cause: {fix.get('root_cause')}")
    print(f"File: {fix.get('fixed_file')}")

    fixer = AutoFixer()

    if not fixer.auto_fix(fix):
        return 3

    print("\n✅ AUTO-FIX COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())