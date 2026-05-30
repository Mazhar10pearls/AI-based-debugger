#!/usr/bin/env python3
"""
CI/CD Platform Poller
Continuously monitors multiple CI/CD platforms for failures and auto-fixes them.
No manual log dropping required - 100% automatic.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List

import requests


class FailureDetector(ABC):
    """Base class for all CI/CD platform detectors."""

    def __init__(self, config: dict):
        self.config = config
        self.processed_failures = set()

    @abstractmethod
    def detect_failures(self) -> List[dict]:
        """Return list of detected failures."""
        pass

    @abstractmethod
    def get_failure_logs(self, failure: dict) -> str:
        """Get detailed logs for a failure."""
        pass


class GitHubActionsDetector(FailureDetector):
    """Monitor GitHub Actions for workflow failures."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.token = config.get("github_token")
        self.owner = config.get("github_owner")
        self.repo = config.get("github_repo")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def detect_failures(self) -> List[dict]:
        """Fetch failed GitHub Actions runs."""
        if not all([self.token, self.owner, self.repo]):
            return []

        try:
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs"
            response = requests.get(
                url,
                headers=self.headers,
                params={"status": "failure", "per_page": 10},
                timeout=10,
            )
            response.raise_for_status()

            failures = []
            for run in response.json().get("workflow_runs", []):
                failure_id = f"github-{run['id']}"
                if failure_id not in self.processed_failures:
                    failures.append(
                        {
                            "id": failure_id,
                            "platform": "github",
                            "run_id": run["id"],
                            "name": run["name"],
                            "branch": run["head_branch"],
                            "timestamp": run["created_at"],
                        }
                    )
            return failures
        except Exception as exc:
            print(f"[ERROR] GitHub detection failed: {exc}")
            return []

    def get_failure_logs(self, failure: dict) -> str:
        """Download GitHub Actions logs."""
        try:
            import io
            import zipfile

            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/actions/runs/{failure['run_id']}/logs"
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()

            # GitHub returns a zip file
            z = zipfile.ZipFile(io.BytesIO(response.content))
            logs = []
            for name in z.namelist():
                logs.append(z.read(name).decode("utf-8", errors="replace"))

            return (
                f"GitHub Actions Run: {failure['name']}\n"
                f"Branch: {failure['branch']}\n"
                f"Timestamp: {failure['timestamp']}\n\n"
                + "\n".join(logs)
            )
        except Exception as exc:
            print(f"[ERROR] Could not fetch GitHub logs: {exc}")
            return ""


class GitLabDetector(FailureDetector):
    """Monitor GitLab CI for pipeline failures."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.token = config.get("gitlab_token")
        self.gitlab_url = config.get("gitlab_url", "https://gitlab.com")
        self.project_id = config.get("gitlab_project_id")
        self.headers = {"PRIVATE-TOKEN": self.token}

    def detect_failures(self) -> List[dict]:
        """Fetch failed GitLab pipelines."""
        if not all([self.token, self.project_id]):
            return []

        try:
            url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/pipelines"
            response = requests.get(
                url,
                headers=self.headers,
                params={"status": "failed", "per_page": 10},
                timeout=10,
            )
            response.raise_for_status()

            failures = []
            for pipeline in response.json():
                failure_id = f"gitlab-{pipeline['id']}"
                if failure_id not in self.processed_failures:
                    failures.append(
                        {
                            "id": failure_id,
                            "platform": "gitlab",
                            "pipeline_id": pipeline["id"],
                            "branch": pipeline["ref"],
                            "timestamp": pipeline["created_at"],
                            "web_url": pipeline["web_url"],
                        }
                    )
            return failures
        except Exception as exc:
            print(f"[ERROR] GitLab detection failed: {exc}")
            return []

    def get_failure_logs(self, failure: dict) -> str:
        """Get GitLab pipeline logs."""
        try:
            url = f"{self.gitlab_url}/api/v4/projects/{self.project_id}/pipelines/{failure['pipeline_id']}/jobs"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            logs = [
                f"GitLab Pipeline: {failure['web_url']}\n"
                f"Branch: {failure['branch']}\n"
                f"Timestamp: {failure['timestamp']}\n\n"
            ]

            for job in response.json():
                if job["status"] == "failed":
                    logs.append(f"\n=== Job: {job['name']} ===\n")
                    logs.append(job.get("description", "No description"))

            return "".join(logs)
        except Exception as exc:
            print(f"[ERROR] Could not fetch GitLab logs: {exc}")
            return ""


class DockerRegistryDetector(FailureDetector):
    """Monitor Docker Registry builds for failures."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.registry_url = config.get("docker_registry_url", "https://hub.docker.com")
        self.username = config.get("docker_username")
        self.token = config.get("docker_token")

    def detect_failures(self) -> List[dict]:
        """Fetch failed Docker builds."""
        if not all([self.username, self.token]):
            return []

        try:
            # Docker Hub API
            url = f"{self.registry_url}/v2/repositories/{self.username}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            failures = []
            for repo in response.json().get("results", []):
                failure_id = f"docker-{repo['name']}"
                if failure_id not in self.processed_failures:
                    # Check if build is failing
                    if repo.get("last_build_status") == "failure":
                        failures.append(
                            {
                                "id": failure_id,
                                "platform": "docker",
                                "repo": repo["name"],
                                "build_status": repo["last_build_status"],
                            }
                        )
            return failures
        except Exception as exc:
            print(f"[ERROR] Docker detection failed: {exc}")
            return []

    def get_failure_logs(self, failure: dict) -> str:
        """Get Docker build logs."""
        return (
            f"Docker Repository: {failure['repo']}\n"
            f"Last Build Status: {failure['build_status']}\n"
            f"Please check Docker Hub for detailed build logs.\n"
        )


class KubernetesDetector(FailureDetector):
    """Monitor Kubernetes pod failures."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.kubeconfig = config.get("kubeconfig_path")
        self.namespace = config.get("kubernetes_namespace", "default")

    def detect_failures(self) -> List[dict]:
        """Fetch failed Kubernetes pods."""
        try:
            # Use kubectl to get failed pods
            cmd = [
                "kubectl",
                "get",
                "pods",
                "-n",
                self.namespace,
                "-o",
                "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return []

            pods_data = json.loads(result.stdout)
            failures = []

            for pod in pods_data.get("items", []):
                pod_name = pod["metadata"]["name"]
                status = pod["status"]["phase"]

                if status in ["Failed", "Unknown"]:
                    failure_id = f"k8s-{pod_name}"
                    if failure_id not in self.processed_failures:
                        failures.append(
                            {
                                "id": failure_id,
                                "platform": "kubernetes",
                                "pod_name": pod_name,
                                "namespace": self.namespace,
                                "status": status,
                            }
                        )
            return failures
        except Exception as exc:
            print(f"[ERROR] Kubernetes detection failed: {exc}")
            return []

    def get_failure_logs(self, failure: dict) -> str:
        """Get Kubernetes pod logs."""
        try:
            cmd = [
                "kubectl",
                "logs",
                failure["pod_name"],
                "-n",
                failure["namespace"],
                "--tail=100",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return (
                f"Kubernetes Pod: {failure['pod_name']}\n"
                f"Namespace: {failure['namespace']}\n"
                f"Status: {failure['status']}\n\n"
                + result.stdout
            )
        except Exception as exc:
            print(f"[ERROR] Could not fetch K8s logs: {exc}")
            return ""


class SystemLogsDetector(FailureDetector):
    """Monitor system logs for errors."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.log_paths = config.get(
            "log_paths", ["/var/log/syslog", "/var/log/messages"]
        )
        self.lookback_minutes = config.get("lookback_minutes", 5)

    def detect_failures(self) -> List[dict]:
        """Scan system logs for ERROR patterns."""
        failures = []

        for log_path in self.log_paths:
            if not os.path.exists(log_path):
                continue

            try:
                cutoff_time = datetime.now() - timedelta(
                    minutes=self.lookback_minutes
                )

                with open(log_path, "r", errors="ignore") as f:
                    for line in f:
                        if "ERROR" in line or "FAILED" in line:
                            failure_id = f"syslog-{hash(line)}"
                            if failure_id not in self.processed_failures:
                                failures.append(
                                    {
                                        "id": failure_id,
                                        "platform": "syslog",
                                        "log_path": log_path,
                                        "error_line": line.strip(),
                                    }
                                )
            except Exception as exc:
                print(f"[ERROR] Could not read {log_path}: {exc}")

        return failures

    def get_failure_logs(self, failure: dict) -> str:
        """Get context around error in syslog."""
        return (
            f"System Log: {failure['log_path']}\n"
            f"Error: {failure['error_line']}\n"
        )


class CIPlatformPoller:
    """Main polling engine - monitors all platforms and triggers auto-fixes."""

    def __init__(self, config: dict):
        self.config = config
        self.detectors = self._init_detectors()
        self.poll_interval = config.get("poll_interval", 300)  # 5 minutes
        self.auto_fixer_script = config.get("auto_fixer_script", "auto-fixer.py")

    def _init_detectors(self) -> list:
        """Initialize all enabled detectors."""
        detectors = []

        if self.config.get("enable_github"):
            detectors.append(GitHubActionsDetector(self.config))

        if self.config.get("enable_gitlab"):
            detectors.append(GitLabDetector(self.config))

        if self.config.get("enable_docker"):
            detectors.append(DockerRegistryDetector(self.config))

        if self.config.get("enable_kubernetes"):
            detectors.append(KubernetesDetector(self.config))

        if self.config.get("enable_syslog"):
            detectors.append(SystemLogsDetector(self.config))

        return detectors

    def poll_once(self) -> int:
        """Poll all platforms once and fix failures."""
        fixed_count = 0

        for detector in self.detectors:
            try:
                failures = detector.detect_failures()

                for failure in failures:
                    print(
                        f"\n[{datetime.now()}] Detected {failure['platform']} failure: {failure.get('name', failure.get('pod_name', 'Unknown'))}"
                    )

                    # Get failure logs
                    logs = detector.get_failure_logs(failure)

                    if logs:
                        # Save to temp file
                        temp_log = f"/tmp/ci-failure-{failure['id']}.log"
                        with open(temp_log, "w") as f:
                            f.write(logs)

                        # Trigger auto-fixer
                        result = subprocess.run(
                            [
                                "python",
                                self.auto_fixer_script,
                                "--input",
                                temp_log,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )

                        if result.returncode == 0:
                            print(f"[SUCCESS] Fixed {failure['platform']} failure")
                            fixed_count += 1
                            detector.processed_failures.add(failure["id"])
                        else:
                            print(
                                f"[FAILED] Could not fix: {result.stderr[:200]}"
                            )
                    else:
                        print(f"[SKIP] Could not get logs for {failure['id']}")

            except Exception as exc:
                print(f"[ERROR] Detector error: {exc}")

        return fixed_count

    def poll_continuous(self) -> int:
        """Run continuous polling."""
        print(f"[INFO] Starting CI/CD Platform Poller")
        print(f"[INFO] Enabled detectors: {', '.join([d.__class__.__name__ for d in self.detectors])}")
        print(f"[INFO] Poll interval: {self.poll_interval} seconds")
        print(f"[INFO] Auto-fixer: {self.auto_fixer_script}")
        print(f"[INFO] Press Ctrl+C to stop\n")

        try:
            while True:
                print(f"[{datetime.now()}] Polling all platforms...")
                fixed = self.poll_once()

                if fixed == 0:
                    print(f"[{datetime.now()}] No failures detected")

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n[INFO] Poller stopped")
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI/CD Platform Poller - Automatic failure detection from all platforms"
    )
    parser.add_argument(
        "--config",
        "-c",
        help="JSON config file with platform credentials",
    )
    parser.add_argument(
        "--github-token",
        help="GitHub personal access token",
    )
    parser.add_argument(
        "--github-owner",
        help="GitHub repo owner",
    )
    parser.add_argument(
        "--github-repo",
        help="GitHub repo name",
    )
    parser.add_argument(
        "--gitlab-token",
        help="GitLab private token",
    )
    parser.add_argument(
        "--gitlab-url",
        default="https://gitlab.com",
        help="GitLab instance URL",
    )
    parser.add_argument(
        "--gitlab-project-id",
        help="GitLab project ID",
    )
    parser.add_argument(
        "--docker-username",
        help="Docker Hub username",
    )
    parser.add_argument(
        "--docker-token",
        help="Docker Hub API token",
    )
    parser.add_argument(
        "--enable-kubernetes",
        action="store_true",
        help="Monitor Kubernetes pod failures",
    )
    parser.add_argument(
        "--enable-syslog",
        action="store_true",
        help="Monitor system logs",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Poll interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--auto-fixer",
        default="auto-fixer.py",
        help="Path to auto-fixer.py script",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit",
    )

    args = parser.parse_args()

    # Build config
    config = {
        "github_token": args.github_token or os.environ.get("GITHUB_TOKEN"),
        "github_owner": args.github_owner,
        "github_repo": args.github_repo,
        "gitlab_token": args.gitlab_token or os.environ.get("GITLAB_TOKEN"),
        "gitlab_url": args.gitlab_url,
        "gitlab_project_id": args.gitlab_project_id,
        "docker_username": args.docker_username or os.environ.get("DOCKER_USERNAME"),
        "docker_token": args.docker_token or os.environ.get("DOCKER_TOKEN"),
        "enable_github": bool(args.github_token or os.environ.get("GITHUB_TOKEN")),
        "enable_gitlab": bool(args.gitlab_token or os.environ.get("GITLAB_TOKEN")),
        "enable_docker": bool(args.docker_username or os.environ.get("DOCKER_USERNAME")),
        "enable_kubernetes": args.enable_kubernetes,
        "enable_syslog": args.enable_syslog,
        "poll_interval": args.poll_interval,
        "auto_fixer_script": args.auto_fixer,
    }

    poller = CIPlatformPoller(config)

    if args.once:
        return poller.poll_once()
    else:
        return poller.poll_continuous()


if __name__ == "__main__":
    raise SystemExit(main())
