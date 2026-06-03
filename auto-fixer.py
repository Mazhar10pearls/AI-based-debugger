#!/usr/bin/env python3
"""
AI Auto-Fixer (Stable Version)
Detect → Fix → Commit  (push triggers re-run naturally — no dispatch needed)
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
MAX_LOG_CHARS   = 1200

# Commit messages written by THIS bot — used to detect loop commits
BOT_COMMIT_PREFIXES = ("fix: auto-fixer", "fix: apply known patch", "fix: fallback")


# ---------------------------
# 🔧 Helpers
# ---------------------------

def extract_json_block(text):
    """Return the longest balanced {...} block found in text."""
    best = None
    for start in [i for i, c in enumerate(text) if c == "{"]:
        stack = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                stack += 1
            elif text[i] == "}":
                stack -= 1
                if stack == 0:
                    candidate = text[start:i + 1]
                    if best is None or len(candidate) > len(best):
                        best = candidate
                    break
    return best


def extract_yaml_fallback(text):
    text = re.sub(r"```(?:yaml|yml|json)?", "", text)
    text = text.replace("```", "").strip()

    match = re.search(r"(name:.*)", text, re.DOTALL)
    if not match:
        return text

    yaml_text = match.group(1)
    for marker in ["\nChanges made", "\nExplanation", "\nSummary"]:
        if marker in yaml_text:
            yaml_text = yaml_text.split(marker)[0]

    return yaml_text.strip()


def is_valid_workflow_yaml(content):
    """YAML must parse cleanly AND contain a 'jobs' key."""
    if not content or not isinstance(content, str):
        return False
    try:
        parsed = yaml.safe_load(content)
        if not isinstance(parsed, dict):
            print("[WARN] YAML root is not a mapping")
            return False
        if "jobs" not in parsed:
            print("[WARN] YAML missing 'jobs' key — likely garbage output")
            return False
        return True
    except Exception as e:
        print(f"[WARN] Invalid YAML: {e}")
        return False


def normalise_fixed_content(data: dict) -> dict:
    """
    Ollama returns fixed_content in three broken shapes — normalise all to a plain string:
      (a) dict          → yaml.dump()
      (b) fenced string → strip ```yaml ... ```
      (c) \\n-escaped   → unescape to real newlines
    """
    fc = data.get("fixed_content")
    if fc is None:
        return data

    if isinstance(fc, dict):
        print("[WARN] fixed_content is a dict — converting to YAML string")
        data["fixed_content"] = yaml.dump(fc, sort_keys=False, allow_unicode=True)
        return data

    if isinstance(fc, str):
        fc = re.sub(r"^```(?:yaml|yml)?\s*\n?", "", fc.strip())
        fc = re.sub(r"\n?```\s*$", "", fc)
        if "\\n" in fc and "\n" not in fc:
            fc = fc.replace("\\n", "\n")
        data["fixed_content"] = fc.strip()

    return data


def last_commit_was_bot() -> bool:
    """
    Returns True if the most recent commit on this branch was already made by
    the auto-fixer bot — used as a local loop guard in addition to the workflow
    'if' condition, in case the script is run outside of Actions.
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=%an|%s"],
            capture_output=True, text=True, check=True
        )
        line = result.stdout.strip()
        author, subject = line.split("|", 1)
        if author == "github-actions[bot]" and any(subject.startswith(p) for p in BOT_COMMIT_PREFIXES):
            print(f"[GUARD] Last commit was already a bot fix: '{subject}' — stopping to prevent loop.")
            return True
    except Exception:
        pass
    return False


# ---------------------------
# 🧠 Analyzer
# ---------------------------

class WorkflowAnalyzer:

    def __init__(self, model=DEFAULT_MODEL, api_url=DEFAULT_API_URL):
        self.model   = model
        self.api_url = api_url

    def _extract_error_lines(self, log_text):
        keywords = [
            "error", "failed", "exception", "exit code", "invalid",
            "fatal", "traceback", "no module", "not found", "syntaxerror",
            "importerror", "cannot", "killed", "denied", "missing",
        ]
        noise = [
            "allow-prereleases", "fetch-depth", "persist-credentials",
            "set-safe-directory", "##[group]", "##[endgroup]",
            "git config", "git submodule", "extraheader",
        ]
        lines = log_text.splitlines()
        relevant = [
            l for l in lines
            if any(k in l.lower() for k in keywords)
            and not any(n in l.lower() for n in noise)
        ]
        tail   = lines[-20:]
        merged = list(dict.fromkeys(relevant + [l for l in tail if l.strip()]))
        result = "\n".join(merged)
        print(f"[DETECT] Extracted {len(merged)} lines ({len(result)} chars)")
        return result[-MAX_LOG_CHARS:]

    def _get_workflow(self) -> tuple[str, str]:
        for name in ["ci-local-deploy.yml", "ci-local-deploy.yaml"]:
            p = Path(".github/workflows") / name
            if p.exists():
                content = p.read_text(encoding="utf-8")
                print(f"[DETECT] Found workflow: {p}")
                return str(p), content
        return "", ""

    def analyze(self, log_text: str) -> dict:
        snippet   = self._extract_error_lines(log_text)
        wf_path, wf_content = self._get_workflow()

        workflow_section = f"\nCurrent workflow file ({wf_path}):\n{wf_content}\n" if wf_content else ""

        prompt = f"""You are a DevOps expert fixing a broken GitHub Actions workflow.

Errors:
{snippet}
{workflow_section}
Return ONLY a JSON object with these exact fields:
  root_cause     - one sentence describing the bug
  severity       - critical | high | medium | low
  fix_type       - dependency | config | code | github-action | dockerfile
  fixed_file     - relative path e.g. .github/workflows/ci-local-deploy.yml
  fixed_content  - THE COMPLETE CORRECTED YAML as a plain JSON string.
                   THIS MUST BE A STRING, NOT A NESTED OBJECT.
                   Escape all newlines as \\n inside the string.
                   Example: "fixed_content": "name: CI\\non:\\n  push:\\n    branches: [main]\\njobs:\\n  build:\\n    runs-on: ubuntu-latest\\n"
  commit_message - short message starting with fix:
  explanation    - one sentence

Return ONLY the JSON. No markdown fences, no extra text."""

        payload = {
            "model":       self.model,
            "prompt":      prompt,
            "temperature": 0.1,
            "max_tokens":  2000,
        }

        try:
            resp = requests.post(self.api_url, json=payload, timeout=180)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama timed out after 180s")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self.api_url}: {exc}")

        raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
        print(f"[AI] Response ({len(raw)} chars): {raw[:300]}{'...' if len(raw) > 300 else ''}")

        # ── Try JSON parse ────────────────────────────────────────────────────
        try:
            block = extract_json_block(raw)
            if block:
                data = json.loads(block)
                data = normalise_fixed_content(data)
                return data
        except Exception as e:
            print(f"[WARN] JSON parse failed: {e}")

        # ── YAML fallback ─────────────────────────────────────────────────────
        print("[WARN] Falling back to raw YAML extraction")
        yaml_content = extract_yaml_fallback(raw)
        return {
            "root_cause":     "fallback_mode",
            "severity":       "critical",
            "fix_type":       "config",
            "fixed_file":     wf_path or ".github/workflows/ci-local-deploy.yml",
            "fixed_content":  yaml_content,
            "commit_message": "fix: auto-fixer fallback",
            "explanation":    "Recovered from bad AI output",
        }


# ---------------------------
# 🛠 Fixer
# ---------------------------

class AutoFixer:

    # Deterministic regex patches applied when AI output is rejected
    KNOWN_PATCHES = [
        ("python-version 3.2 → 3.12",
         re.compile(r"(python-version:\s*['\"]?)3\.2(['\"]?)"),
         r"\g<1>3.12\g<2>"),
    ]

    def apply_fix(self, fix: dict) -> bool:
        file    = fix.get("fixed_file", "").strip()
        content = fix.get("fixed_content")

        if not file:
            print("[ERROR] No fixed_file specified", file=sys.stderr)
            return False

        if ".." in file or file.startswith("/"):
            print(f"[ERROR] Path traversal rejected: {file}", file=sys.stderr)
            return False

        # ── AI content valid? ─────────────────────────────────────────────────
        if is_valid_workflow_yaml(content):
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            Path(file).write_text(content, encoding="utf-8")
            print(f"[FIX] Written AI fix → {file}")
            return True

        # ── Fallback: known patches ───────────────────────────────────────────
        print("[WARN] AI content invalid — trying known patches")
        return self._apply_known_patches(file, fix)

    def _apply_known_patches(self, workflow_file: str, fix: dict) -> bool:
        p = Path(workflow_file)
        if not p.exists():
            # Try to find it ourselves
            for name in ["ci-local-deploy.yml", "ci-local-deploy.yaml"]:
                candidate = Path(".github/workflows") / name
                if candidate.exists():
                    p = candidate
                    break

        if not p.exists():
            print("[ERROR] No workflow file found to patch", file=sys.stderr)
            return False

        content = p.read_text(encoding="utf-8")
        patched = content
        applied = []

        for desc, pattern, replacement in self.KNOWN_PATCHES:
            new = pattern.sub(replacement, patched)
            if new != patched:
                patched = new
                applied.append(desc)

        if applied:
            p.write_text(patched, encoding="utf-8")
            fix["fixed_file"]     = str(p)
            fix["commit_message"] = "fix: apply known patch (" + "; ".join(applied) + ")"
            print(f"[PATCH] Applied to {p}: {', '.join(applied)}")
            return True

        print("[ERROR] No known patches matched", file=sys.stderr)
        return False

    def commit_and_push(self, msg: str) -> bool:
        try:
            subprocess.run(["git", "config", "user.name",  "github-actions[bot]"],
                           check=True, capture_output=True)
            subprocess.run(["git", "config", "user.email",
                            "github-actions[bot]@users.noreply.github.com"],
                           check=True, capture_output=True)

            # Rebase before push to handle concurrent commits cleanly
            subprocess.run(["git", "pull", "--rebase", "origin", "main"],
                           check=True, capture_output=True)

            subprocess.run(["git", "add", "-A"],
                           check=True, capture_output=True)

            if subprocess.run(["git", "diff", "--cached", "--quiet"],
                               capture_output=True).returncode == 0:
                print("[COMMIT] No changes to commit")
                return True

            subprocess.run(["git", "commit", "-m", msg],
                           check=True, capture_output=True)
            subprocess.run(["git", "push"],
                           check=True, capture_output=True)

            print(f"[COMMIT] Pushed: {msg}")
            print("[COMMIT] 'Local CI/CD Deploy' will trigger automatically from this push.")
            return True

        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr)"
            print(f"[ERROR] Git failed: {' '.join(exc.cmd)}\n  {stderr}", file=sys.stderr)
            return False


# ---------------------------
# 🚀 Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Self-healing CI: detect → fix → commit (push re-triggers CI naturally)"
    )
    parser.add_argument("--input", required=True, help="Path to failure.log")
    args = parser.parse_args()

    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Log file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Local loop guard (belt-and-suspenders alongside the workflow 'if' condition)
    if last_commit_was_bot():
        sys.exit(0)

    log = log_path.read_text(encoding="utf-8", errors="replace")
    print(f"[DETECT] Log size: {len(log)} chars")

    analyzer = WorkflowAnalyzer()
    try:
        fix = analyzer.analyze(log)
    except Exception as e:
        print(f"[ERROR] AI analysis failed: {e}", file=sys.stderr)
        fix = {
            "fixed_file":     ".github/workflows/ci-local-deploy.yml",
            "fixed_content":  None,
            "commit_message": "fix: fallback safe mode",
        }

    print(f"\n[INFO] root_cause  : {fix.get('root_cause', 'unknown')}")
    print(f"[INFO] fixed_file  : {fix.get('fixed_file', 'unknown')}")
    print(f"[INFO] explanation : {fix.get('explanation', '')}")

    fixer = AutoFixer()

    if not fixer.apply_fix(fix):
        print("[ERROR] Could not apply any fix", file=sys.stderr)
        sys.exit(3)

    if not fixer.commit_and_push(fix.get("commit_message", "fix: auto-fix")):
        print("[ERROR] Commit/push failed", file=sys.stderr)
        sys.exit(4)

    print("\n✅ Fix committed. CI will re-run from the new push automatically.")


if __name__ == "__main__":
    main()