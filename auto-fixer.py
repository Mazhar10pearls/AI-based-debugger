#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows (Hardened संस्करण)
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
MAX_LOG_CHARS   = 600


# ---------------------------
# 🔧 JSON Utilities (FIXED)
# ---------------------------

def extract_json(text):
    """Safely extract first valid JSON object using bracket matching."""
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i+1]

    raise ValueError("Incomplete JSON object")


def safe_load_json(text):
    """Try parsing JSON with fallback cleanup."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("[WARN] JSON invalid, attempting repair...")

        text = text.replace('"{}"', '{}')
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)

        return json.loads(text)


def normalize_content(content):
    """Ensure content is string (convert dict → YAML if needed)."""
    if isinstance(content, dict):
        return yaml.dump(content, sort_keys=False)
    return content


# ---------------------------
# 🧠 Analyzer
# ---------------------------

class WorkflowAnalyzer:
    def __init__(self, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL):
        self.model   = model
        self.api_url = api_url

    def _extract_error_lines(self, log_text):
        error_keywords = [
            "error", "failed", "failure", "exception", "traceback",
            "exit code", "no module", "not found", "invalid", "fatal",
            "syntaxerror", "importerror", "cannot", "refused", "rejected",
            "killed", "denied", "unavailable", "missing",
        ]

        noise_keywords = [
            "allow-prereleases", "fetch-depth", "persist-credentials",
            "set-safe-directory", "sparse-checkout", "update-environment",
        ]

        lines = log_text.splitlines()
        relevant = []

        for l in lines:
            s = l.strip()
            low = s.lower()
            if not s:
                continue
            if any(n in low for n in noise_keywords):
                continue
            if any(k in low for k in error_keywords):
                relevant.append(s)

        tail = [l.strip() for l in lines[-15:] if l.strip()]

        seen, out = set(), []
        for l in relevant + tail:
            if l not in seen:
                seen.add(l)
                out.append(l)

        result = "\n".join(out)
        print(f"[INFO] Extracted {len(out)} error lines")
        return result[-MAX_LOG_CHARS:]

    def _find_workflow_file(self):
        workflow_dir = Path(".github/workflows")
        if not workflow_dir.exists():
            return ""

        for name in ["ci-local-deploy.yml", "ci-local-deploy.yaml"]:
            p = workflow_dir / name
            if p.exists():
                content = p.read_text()
                print(f"[INFO] Found workflow file: {p}")
                return content
        return ""

    def analyze(self, log_text):
        snippet  = self._extract_error_lines(log_text)
        workflow = self._find_workflow_file()

        workflow_section = f"\nCurrent workflow:\n{workflow}\n" if workflow else ""

        prompt = (
            "You are a DevOps expert fixing GitHub Actions.\n\n"
            f"Error log:\n{snippet}\n"
            f"{workflow_section}\n"
            "Return ONLY valid JSON.\n\n"
            "Rules:\n"
            "- fixed_content MUST be a STRING (valid YAML)\n"
            "- No trailing commas\n"
            "- No explanations\n\n"
            "Schema:\n"
            "{\n"
            '  "root_cause": "...",\n'
            '  "severity": "critical|high|medium|low",\n'
            '  "fix_type": "...",\n'
            '  "fixed_file": "...",\n'
            '  "fixed_content": "...",\n'
            '  "commit_message": "...",\n'
            '  "explanation": "..."\n'
            "}"
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
        raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        print(f"[AI RAW]: {raw[:300]}...")

        extracted = extract_json(raw)
        return safe_load_json(extracted)


# ---------------------------
# 🛠 Auto Fixer
# ---------------------------

class AutoFixer:
    def __init__(self, repo_path="."):
        os.chdir(repo_path)

    def apply_fix(self, fix_data):
        fixed_file = fix_data.get("fixed_file")
        content    = normalize_content(fix_data.get("fixed_content"))

        if not fixed_file or not content:
            print("[ERROR] Invalid fix data")
            return False

        Path(fixed_file).parent.mkdir(parents=True, exist_ok=True)
        Path(fixed_file).write_text(content)

        print(f"[FIXED] {fixed_file}")
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

    fixer = AutoFixer()

    if not fixer.auto_fix(fix):
        return 3

    print("✅ AUTO-FIX COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())