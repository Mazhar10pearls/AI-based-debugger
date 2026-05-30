#!/usr/bin/env python3
"""
AI-Powered Auto-Fixer for CI/CD Workflows
Automatically detects, analyzes, and fixes workflow failures with zero manual intervention.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import requests

DEFAULT_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


class WorkflowAnalyzer:
    def __init__(self, model: str = DEFAULT_MODEL, api_url: str = DEFAULT_API_URL):
        self.model = model
        self.api_url = api_url

    def analyze(self, log_text: str, context: str = "") -> dict:
        """Analyze a failure log and return structured fix recommendations."""
        prompt = textwrap.dedent(
            f"""
            You are an expert DevOps engineer and code fixer.
            Analyze this failure log and provide a JSON response with the exact fix.

            Context: {context}

            Failure Log:
            {log_text}

            Respond ONLY with valid JSON in this exact format:
            {{
                "root_cause": "Brief cause",
                "severity": "critical|high|medium|low",
                "fix_type": "dependency|dockerfile|github-action|config|code",
                "fixed_file": "path/to/file",
                "fixed_content": "Complete corrected file content here",
                "commit_message": "Short commit message",
                "explanation": "Brief explanation of the fix"
            }}

            Ensure the fixed_content is the COMPLETE file, not just the diff.
            """
        ).strip()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 2048,
            "temperature": 0.1,
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(
                f"Failed to reach Ollama. Is it running on {self.api_url}?\n{exc}"
            )

        data = response.json()
        text = data["choices"][0].get("text", "")

        # Extract JSON from response
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in AI response:\n{text}")

        return json.loads(json_match.group())


class FailureDetector:
    """Detect failure type from logs and workflows."""

    @staticmethod
    def detect_github_actions_failure(log_path: str) -> Optional[dict]:
        """Detect GitHub Actions workflow failure."""
        if not log_path.endswith(".yml") and not log_path.endswith(".yaml"):
            return None

        with open(log_path, "r") as f:
            content = f.read()

        if "name:" in content and "jobs:" in content:
            return {
                "type": "github-action",
                "file": log_path,
                "context": "GitHub Actions workflow configuration",
            }
        return None

    @staticmethod
    def detect_dockerfile_failure(log_path: str) -> Optional[dict]:
        """Detect Docker build failure."""
        with open(log_path, "r") as f:
            content = f.read().lower()

        if "dockerfile" in log_path.lower() or "docker build" in content:
            if any(
                keyword in content
                for keyword in ["error", "failed", "not found", "permission denied"]
            ):
                return {
                    "type": "dockerfile",
                    "file": log_path,
                    "context": "Docker build failure",
                }
        return None

    @staticmethod
    def detect_dependency_failure(log_path: str) -> Optional[dict]:
        """Detect Python/Node dependency failure."""
        with open(log_path, "r") as f:
            content = f.read().lower()

        if "no module named" in content or "modulenotfounderror" in content:
            return {
                "type": "dependency-python",
                "file": "requirements.txt",
                "context": "Python dependency missing or version mismatch",
            }

        if "npm err" in content or "cannot find module" in content:
            return {
                "type": "dependency-node",
                "file": "package.json",
                "context": "Node.js dependency missing or version mismatch",
            }

        return None

    @staticmethod
    def detect_build_log_failure(log_path: str) -> Optional[dict]:
        """Detect generic CI/CD build log failure."""
        with open(log_path, "r") as f:
            content = f.read()

        failure_indicators = [
            ("FAILED", "test failure"),
            ("ERROR:", "build error"),
            ("fatal error", "compilation error"),
            ("exit code", "process failure"),
        ]

        for keyword, reason in failure_indicators:
            if keyword in content:
                return {
                    "type": "build-log",
                    "file": log_path,
                    "context": reason,
                }

        return None

    @classmethod
    def detect(cls, log_path: str) -> Optional[dict]:
        """Detect failure type from any log or config file."""
        if not os.path.isfile(log_path):
            return None

        # Try each detector
        for detector in [
            cls.detect_github_actions_failure,
            cls.detect_dockerfile_failure,
            cls.detect_dependency_failure,
            cls.detect_build_log_failure,
        ]:
            result = detector(log_path)
            if result:
                return result

        return None


class AutoFixer:
    """Apply fixes and commit to git."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        os.chdir(repo_path)

    def apply_fix(self, fix_data: dict) -> bool:
        """Write fixed file to disk."""
        fixed_file = fix_data["fixed_file"]
        fixed_content = fix_data["fixed_content"]

        # Ensure directory exists
        Path(fixed_file).parent.mkdir(parents=True, exist_ok=True)

        with open(fixed_file, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        print(f"[FIXED] {fixed_file}")
        return True

    def commit_fix(self, fix_data: dict) -> bool:
        """Commit the fix to git."""
        try:
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", fix_data["commit_message"]],
                check=True,
                capture_output=True,
            )
            subprocess.run(["git", "push"], check=True, capture_output=True)
            print(f"[COMMITTED] {fix_data['commit_message']}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] Git operation failed: {exc}", file=sys.stderr)
            return False

    def auto_fix(self, fix_data: dict, commit: bool = True) -> bool:
        """Apply fix and optionally commit."""
        if not self.apply_fix(fix_data):
            return False

        if commit:
            return self.commit_fix(fix_data)

        return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Auto-Fixer: Automatic failure detection and resolution"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to failure log or config file to analyze.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model to use (default: llama3).",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="Ollama API endpoint.",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="Apply fixes but do not commit to git.",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to git repository (default: current directory).",
    )

    args = parser.parse_args()

    # Step 1: Detect failure type
    print("[STEP 1] Detecting failure type...")
    failure = FailureDetector.detect(args.input)
    if not failure:
        print(
            f"[ERROR] Could not detect failure type in {args.input}",
            file=sys.stderr,
        )
        return 1

    print(f"[DETECTED] {failure['type']} - {failure['context']}")

    # Step 2: Read the failure log
    print("[STEP 2] Reading failure log...")
    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Step 3: Analyze with AI
    print("[STEP 3] Analyzing with AI...")
    analyzer = WorkflowAnalyzer(model=args.model, api_url=args.api_url)
    try:
        fix_data = analyzer.analyze(log_content, context=failure["context"])
    except Exception as exc:
        print(f"[ERROR] AI analysis failed: {exc}", file=sys.stderr)
        return 2

    print(f"[ANALYSIS] {fix_data['root_cause']}")
    print(f"[FIX TYPE] {fix_data['fix_type']}")
    print(f"[SEVERITY] {fix_data['severity']}")

    # Step 4: Apply and commit
    print("[STEP 4] Applying fix...")
    fixer = AutoFixer(repo_path=args.repo)
    commit = not args.no_commit

    if not fixer.auto_fix(fix_data, commit=commit):
        return 3

    print("\n=== FIX SUMMARY ===")
    print(f"Root Cause: {fix_data['root_cause']}")
    print(f"Fixed File: {fix_data['fixed_file']}")
    print(f"Explanation: {fix_data['explanation']}")
    print(f"Commit Message: {fix_data['commit_message']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
