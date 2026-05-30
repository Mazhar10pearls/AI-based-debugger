#!/usr/bin/env python3
"""
Workflow Watcher
Continuously monitor logs and auto-fix failures with zero manual intervention.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import requests


class WorkflowWatcher:
    def __init__(
        self,
        watch_dir: str,
        auto_fixer_script: str = "auto-fixer.py",
        interval: int = 300,
    ):
        self.watch_dir = Path(watch_dir)
        self.auto_fixer_script = auto_fixer_script
        self.interval = interval
        self.processed_files = set()

    def find_new_logs(self) -> list:
        """Find new failure log files in the watch directory."""
        if not self.watch_dir.exists():
            return []

        log_patterns = ["*.log", "*.txt", "requirements.txt", "Dockerfile"]
        new_logs = []

        for pattern in log_patterns:
            for log_file in self.watch_dir.glob(pattern):
                if log_file.is_file() and str(log_file) not in self.processed_files:
                    new_logs.append(log_file)

        return new_logs

    def auto_fix_log(self, log_file: Path) -> bool:
        """Trigger auto-fixer on a log file."""
        print(f"\n[{datetime.now()}] Processing: {log_file}")

        try:
            result = subprocess.run(
                [
                    "python",
                    self.auto_fixer_script,
                    "--input",
                    str(log_file),
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                print(f"[SUCCESS] {log_file} - Fix applied and committed")
                self.processed_files.add(str(log_file))
                return True
            else:
                print(f"[FAILED] {log_file} - Auto-fixer returned code {result.returncode}")
                print(f"  Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {log_file} - Auto-fixer took too long")
            return False
        except Exception as exc:
            print(f"[ERROR] {log_file} - {exc}")
            return False

    def watch(self, once: bool = False) -> int:
        """Start watching for failures."""
        print(f"[INFO] Watching {self.watch_dir} for new failure logs...")
        print(f"[INFO] Checking every {self.interval} seconds")
        print(f"[INFO] Auto-fixer: {self.auto_fixer_script}")

        if not self.auto_fixer_script or not os.path.isfile(self.auto_fixer_script):
            print(
                f"[ERROR] Auto-fixer script not found: {self.auto_fixer_script}",
                file=sys.stderr,
            )
            return 1

        try:
            if once:
                print("[MODE] Running once (no loop)")
                logs = self.find_new_logs()
                for log_file in logs:
                    self.auto_fix_log(log_file)
                return 0

            print("[MODE] Running continuously (Ctrl+C to stop)\n")
            while True:
                logs = self.find_new_logs()

                if logs:
                    print(f"[FOUND] {len(logs)} new log(s)")
                    for log_file in logs:
                        self.auto_fix_log(log_file)
                else:
                    print(f"[{datetime.now()}] No new failures detected")

                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n[INFO] Watcher stopped by user")
            return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Workflow Watcher - Continuous auto-fix for failures"
    )
    parser.add_argument(
        "--watch-dir",
        "-w",
        default="./logs",
        help="Directory to watch for failure logs (default: ./logs)",
    )
    parser.add_argument(
        "--auto-fixer",
        default="auto-fixer.py",
        help="Path to auto-fixer.py script",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Check interval in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (do not loop)",
    )

    args = parser.parse_args()

    watcher = WorkflowWatcher(
        watch_dir=args.watch_dir,
        auto_fixer_script=args.auto_fixer,
        interval=args.interval,
    )

    return watcher.watch(once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
