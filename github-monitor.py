#!/usr/bin/env python3
"""
GitHub Actions Integration
Automatically detect failed workflows and trigger AI auto-fixer.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional

import requests

GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    def __init__(self, token: str, owner: str, repo: str):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_failed_workflow_runs(self, limit: int = 5) -> list:
        """Fetch recent failed workflow runs."""
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/actions/runs"
        params = {"status": "failure", "per_page": limit}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("workflow_runs", [])

    def get_workflow_logs(self, run_id: int) -> str:
        """Download workflow run logs."""
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/logs"

        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        # GitHub returns a zip file; extract to temp location
        import io
        import zipfile

        z = zipfile.ZipFile(io.BytesIO(response.content))
        # Combine all log files
        logs = []
        for name in z.namelist():
            logs.append(z.read(name).decode("utf-8", errors="replace"))

        return "\n".join(logs)

    def create_pr_with_fix(
        self, branch_name: str, commit_hash: str, title: str, body: str
    ) -> Optional[dict]:
        """Create a pull request with the fix."""
        url = f"{GITHUB_API_BASE}/repos/{self.owner}/{self.repo}/pulls"
        payload = {
            "title": title,
            "head": branch_name,
            "base": "main",
            "body": body,
        }

        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 201:
            return response.json()

        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GitHub Actions Integration for Auto-Fixer"
    )
    parser.add_argument(
        "--token",
        required=False,
        help="GitHub token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument("--owner", required=True, help="GitHub repo owner")
    parser.add_argument("--repo", required=True, help="GitHub repo name")
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically create PRs with fixes (requires auto-fixer to work)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of failed runs to check",
    )

    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[ERROR] GitHub token required (--token or GITHUB_TOKEN env var)")
        return 1

    client = GitHubClient(token, args.owner, args.repo)

    print(f"[INFO] Fetching failed workflows for {args.owner}/{args.repo}...")
    try:
        failed_runs = client.get_failed_workflow_runs(limit=args.limit)
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] GitHub API error: {exc}", file=sys.stderr)
        return 2

    if not failed_runs:
        print("[INFO] No failed workflow runs found.")
        return 0

    print(f"[FOUND] {len(failed_runs)} failed run(s)\n")

    for run in failed_runs:
        print(f"Run: {run['name']} (ID: {run['id']})")
        print(f"Conclusion: {run['conclusion']}")
        print(f"Head Branch: {run['head_branch']}")
        print("-" * 60)

    # If auto-fix enabled, trigger the auto-fixer on first failed run
    if args.auto_fix and failed_runs:
        first_run = failed_runs[0]
        print(f"\n[AUTO-FIX] Downloading logs for run {first_run['id']}...")

        try:
            logs = client.get_workflow_logs(first_run["id"])
            log_file = f"github-workflow-{first_run['id']}.log"

            with open(log_file, "w") as f:
                f.write(logs)

            print(f"[SAVED] Logs to {log_file}")
            print("[TODO] Run: python auto-fixer.py --input " + log_file)

        except requests.exceptions.RequestException as exc:
            print(f"[ERROR] Could not download logs: {exc}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
