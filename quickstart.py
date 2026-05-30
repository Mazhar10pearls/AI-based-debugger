#!/usr/bin/env python3
"""
Quick example: Test the auto-fixer on sample-log.txt
Run this after ollama is running in another terminal.
"""

import subprocess
import sys

def main():
    print("=" * 70)
    print("AI Workflow Auto-Fixer - Quick Start Example")
    print("=" * 70)
    print()
    print("[STEP 1] Running auto-fixer on sample-log.txt")
    print("[STEP 2] AI will detect the dependency error")
    print("[STEP 3] Fix will be generated (no commit since it's example)")
    print()
    print("-" * 70)
    
    try:
        result = subprocess.run(
            [sys.executable, "auto-fixer.py", "--input", "sample-log.txt", "--no-commit"],
            check=False
        )
        
        print()
        print("-" * 70)
        if result.returncode == 0:
            print("✅ SUCCESS - Auto-fixer worked!")
            print()
            print("[NEXT] Try these:")
            print("  1. python auto-fixer.py --input sample-log.txt  (with git commit)")
            print("  2. python workflow-watcher.py --watch-dir ./logs  (continuous mode)")
            print("  3. python analyzer.py --input sample-log.txt  (manual analysis)")
        else:
            print("❌ FAILED - Check if ollama is running on http://127.0.0.1:11434")
            print()
            print("[TIP] In another terminal, run:")
            print("  ollama run llama3")
    
    except FileNotFoundError:
        print("❌ auto-fixer.py not found")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
