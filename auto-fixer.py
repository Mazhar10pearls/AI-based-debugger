#!/usr/bin/env python3
"""
AI Auto-Fixer (Stable Version)
Detect → Fix → Commit → Re-run (No Crash)
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
# 🔧 Helpers
# ---------------------------

def extract_json_block(text):
    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i+1]
    return None


def extract_yaml_fallback(text):
    text = re.sub(r"```(?:yaml|yml|json)?", "", text)
    text = text.replace("```", "").strip()

    match = re.search(r"(name:.*)", text, re.DOTALL)
    if not match:
        return text

    yaml_text = match.group(1)

    stop_markers = ["\nChanges made", "\nExplanation", "\nSummary"]
    for m in stop_markers:
        if m in yaml_text:
            yaml_text = yaml_text.split(m)[0]

    return yaml_text.strip()


def is_valid_yaml(content):
    try:
        yaml.safe_load(content)
        return True
    except Exception as e:
        print(f"[WARN] Invalid YAML: {e}")
        return False


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

        relevant = [l for l in lines if any(k in l.lower() for k in keywords)]
        tail = lines[-15:]

        merged = list(dict.fromkeys(relevant + tail))
        print(f"[INFO] Extracted {len(merged)} lines")

        return "\n".join(merged)[-MAX_LOG_CHARS:]

    def _get_workflow(self):
        p = Path(".github/workflows/ci-local-deploy.yml")
        if p.exists():
            print(f"[INFO] Found workflow file")
            return p.read_text()
        return ""

    def analyze(self, log_text):

        snippet = self._extract_error_lines(log_text)
        workflow = self._get_workflow()

        prompt = f"""
Fix this GitHub Actions workflow.

Errors:
{snippet}

Workflow:
{workflow}

Return ONLY JSON.
fixed_content MUST be STRING YAML.
"""

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
        # ✅ Try JSON parsing
        # ---------------------------
        try:
            json_block = extract_json_block(raw)

            if json_block:
                data = json.loads(json_block)

                fc = data.get("fixed_content")

                # 🔥 convert dict → YAML
                if isinstance(fc, dict):
                    data["fixed_content"] = yaml.dump(fc, sort_keys=False)

                return data

        except Exception as e:
            print(f"[WARN] JSON failed: {e}")

        # ---------------------------
        # 🔥 FALLBACK MODE
        # ---------------------------
        print("[INFO] Using YAML fallback")

        yaml_content = extract_yaml_fallback(raw)

        return {
            "root_cause": "fallback_mode",
            "severity": "critical",
            "fix_type": "config",
            "fixed_file": ".github/workflows/ci-local-deploy.yml",
            "fixed_content": yaml_content,
            "commit_message": "fix: auto-fixer fallback",
            "explanation": "Recovered from bad AI output"
        }


# ---------------------------
# 🛠 Fixer
# ---------------------------

class AutoFixer:

    def __init__(self):
        pass

    def apply_fix(self, fix):

        file = fix.get("fixed_file")
        content = fix.get("fixed_content")

        if not file:
            print("[ERROR] No file provided")
            return False

        if not content:
            print("[WARN] Empty fix content")
            return False

        if not is_valid_yaml(content):
            print("[ERROR] YAML invalid — skipping")
            return False

        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Path(file).write_text(content)

        print(f"[FIXED] {file}")
        return True

    def commit_and_push(self, msg):

        try:
            subprocess.run(["git", "config", "user.name", "github-actions[bot]"])
            subprocess.run(["git", "config", "user.email", "bot@github.com"])

            # 🔥 IMPORTANT: sync before push
            subprocess.run(["git", "pull", "--rebase", "origin", "main"])

            subprocess.run(["git", "add", "-A"])

            if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
                print("[INFO] No changes")
                return True

            subprocess.run(["git", "commit", "-m", msg])
            subprocess.run(["git", "push"])

            print("[COMMITTED & PUSHED]")
            return True

        except Exception as e:
            print(f"[ERROR] Git failed: {e}")
            return False


# ---------------------------
# 🚀 Main
# ---------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    log = Path(args.input).read_text()

    analyzer = WorkflowAnalyzer()

    try:
        fix = analyzer.analyze(log)
    except Exception as e:
        print(f"[ERROR] AI failed, using empty fallback: {e}")
        fix = {
            "fixed_file": ".github/workflows/ci-local-deploy.yml",
            "fixed_content": "",
            "commit_message": "fix: fallback safe mode"
        }

    print(f"[INFO] Fixing: {fix.get('fixed_file')}")

    fixer = AutoFixer()

    if fixer.apply_fix(fix):
        fixer.commit_and_push(fix.get("commit_message", "fix: auto-fix"))

    print("\n✅ DONE (pipeline will re-run automatically)")


if __name__ == "__main__":
    main()