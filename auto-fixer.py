#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — Generic Pipeline Repair
=========================================================
Works for ANY repository, ANY pipeline type (build, release, deploy, test),
ANY tech stack (Python, Node, Docker, Java, Go, Ruby, etc.)

No hardcoded file paths. The system:
  1. DETECT    — Parse the CI log, identify pipeline type and tech stack
  2. DISCOVER  — Walk the repo dynamically to find files relevant to the failure
  3. ANALYSE   — Ask Llama: what is broken and what should each file look like?
  4. VALIDATE  — Syntax-check every rewritten file before touching disk
  5. WRITE     — Atomically overwrite only the files Llama fixed
  6. TEST      — Run pipeline-appropriate tests; revert on failure
  7. COMMIT    — Push to auto-fix/<timestamp> branch
  8. PR        — Open Pull Request; human reviews before merge

Pipeline types handled:
  - Build pipelines     (Docker build, compile, package)
  - Release pipelines   (publish, deploy, tag, push to registry)
  - Test pipelines      (pytest, jest, go test, maven test, etc.)
  - Lint/check pipelines (flake8, eslint, mypy, etc.)
  - Infrastructure      (terraform, ansible, helm, k8s)
  - Composite           (build + test + release in one workflow)

Exit codes:
  0 — success or loop guard
  1 — log file not found
  2 — AI analysis failed
  3 — all fixes failed validation
  4 — git commit/push failed
  5 — tests failed after fix (reverted)
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# ── Ollama config ──────────────────────────────────────────────────────────────
OLLAMA_API_URL   = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:3b")
AI_TIMEOUT       = 400
MAX_RETRIES      = 3
RETRY_BACKOFF    = [10, 30, 60]

# ── Prompt budget ──────────────────────────────────────────────────────────────
MAX_ERROR_LINES   = 40      # error lines to extract from CI log
MAX_FILE_CHARS    = 2000    # chars per file sent to AI
MAX_TOTAL_CONTEXT = 8000    # total context chars sent to AI
MAX_FILES_FIXED   = 8       # AI cannot claim to fix more than this many files

# ── Safety ────────────────────────────────────────────────────────────────────
CONFIDENCE_MIN   = 0.4
MAX_BOT_ATTEMPTS = 3

BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"

# ── Paths the AI must never touch ─────────────────────────────────────────────
ALWAYS_BLOCKED = {
    ".git",
    "auto-fixer.py",
    "self-healer.py",
}
BLOCKED_PATTERNS = [
    r"\.github/workflows/auto-fix.*\.ya?ml$",
    r"\.github/workflows/self-heal.*\.ya?ml$",
]

# ── Skip these directories when walking the repo ──────────────────────────────
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
    "target", "out", ".gradle", ".idea", ".vscode", "vendor",
    "coverage", ".nyc_output", "tmp", "temp", "logs",
}

MAX_FILE_SIZE_BYTES = 100_000


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════════════

PIPELINE_TYPE_SIGNALS = {
    "build": [
        "docker build", "building image", "dockerfile", "build context",
        "build failed", "compile error", "compilation failed",
        "mvn package", "gradle build", "go build", "npm run build",
        "cargo build", "make ", "cmake", "ant build",
    ],
    "release": [
        "docker push", "pushing image", "publish", "release",
        "deploy", "helm upgrade", "kubectl apply", "terraform apply",
        "ansible-playbook", "aws deploy", "gcloud deploy", "az deploy",
        "npm publish", "gem push", "cargo publish", "pypi", "twine",
        "git tag", "create release",
    ],
    "test": [
        "pytest", "jest", "mocha", "jasmine", "rspec", "go test",
        "mvn test", "gradle test", "unittest", "xunit", "nunit",
        "test failed", "test suite", "assertion", "assert ",
        "failing test", "test error",
    ],
    "lint": [
        "flake8", "pylint", "mypy", "eslint", "tslint", "rubocop",
        "golint", "staticcheck", "shellcheck", "hadolint",
        "checkstyle", "spotbugs", "sonar",
    ],
    "infra": [
        "terraform", "helm", "kubectl", "ansible", "pulumi",
        "cloudformation", "cdk ", "k8s", "kubernetes",
    ],
}

TECH_STACK_SIGNALS = {
    "python":     ["python", "pip", "pytest", "django", "flask", "fastapi",
                   "requirements.txt", "pyproject.toml", "setup.py", ".py"],
    "node":       ["node", "npm", "yarn", "pnpm", "jest", "webpack", "vite",
                   "package.json", "node_modules", ".js", ".ts", ".tsx"],
    "docker":     ["dockerfile", "docker build", "docker push", "containerd",
                   "docker daemon", "image", "container", "registry",
                   "manifest", "layer", "from "],
    "java":       ["java", "maven", "gradle", "mvn", "spring", ".java",
                   "pom.xml", "build.gradle", "jar", "war"],
    "go":         ["go build", "go test", "go mod", "golang", ".go",
                   "go.mod", "go.sum"],
    "ruby":       ["ruby", "rails", "gem", "bundler", "rspec", ".rb",
                   "gemfile", "rakefile"],
    "rust":       ["cargo", "rustc", ".rs", "rust", "cargo.toml"],
    "dotnet":     [".cs", "dotnet", "csproj", "nuget", "msbuild", ".net"],
    "terraform":  ["terraform", ".tf", "tfvars", "tfstate"],
    "kubernetes": ["kubectl", "helm", ".yaml", "k8s", "kube", "pod",
                   "deployment", "service", "ingress"],
    "shell":      ["bash", "sh:", "chmod", "shell", ".sh", "#!/bin"],
}

STACK_FILE_SIGNALS = {
    "python":     [".py", "requirements.txt", "requirements*.txt",
                   "Pipfile", "pyproject.toml", "setup.py", "setup.cfg",
                   "tox.ini", ".flake8", "mypy.ini"],
    "node":       [".js", ".ts", ".tsx", ".jsx", "package.json",
                   "package-lock.json", "yarn.lock", ".eslintrc*",
                   "tsconfig.json", "webpack.config.*", "vite.config.*"],
    "docker":     ["Dockerfile", "Dockerfile.*", "docker-compose*.yml",
                   ".dockerignore"],
    "java":       [".java", "pom.xml", "build.gradle", "settings.gradle",
                   "gradle.properties", "Makefile"],
    "go":         [".go", "go.mod", "go.sum", "Makefile"],
    "ruby":       [".rb", "Gemfile", "Gemfile.lock", "Rakefile", ".ruby-version"],
    "rust":       [".rs", "Cargo.toml", "Cargo.lock"],
    "dotnet":     [".cs", ".csproj", ".sln", "NuGet.Config"],
    "terraform":  [".tf", ".tfvars"],
    "kubernetes": [".yaml", ".yml", "Chart.yaml", "values.yaml"],
    "shell":      [".sh", "Makefile", "GNUmakefile"],
}

WORKFLOW_EXTENSIONS = {".yml", ".yaml"}
WORKFLOW_DIRS       = {".github/workflows", ".gitlab-ci.d", "ci", ".circleci"}


def fingerprint_pipeline(log_text: str) -> tuple[set[str], set[str]]:
    log_low = log_text.lower()

    pipeline_types = set()
    for ptype, signals in PIPELINE_TYPE_SIGNALS.items():
        if any(s in log_low for s in signals):
            pipeline_types.add(ptype)

    tech_stacks = set()
    for stack, signals in TECH_STACK_SIGNALS.items():
        if any(s in log_low for s in signals):
            tech_stacks.add(stack)

    if not pipeline_types:
        pipeline_types.add("build")

    print(f"[DETECT] Pipeline types : {pipeline_types}")
    print(f"[DETECT] Tech stacks    : {tech_stacks or {'unknown'}}")
    return pipeline_types, tech_stacks


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DETECT
# ══════════════════════════════════════════════════════════════════════════════

ERROR_KEYWORDS = [
    "error", "failed", "failure", "exception", "traceback",
    "exit code", "exitcode", "invalid", "fatal", "cannot",
    "refused", "rejected", "killed", "denied", "missing",
    "undefined", "permission denied", "command not found",
    "returned non-zero", "syntaxerror", "importerror",
    "nameerror", "typeerror", "valueerror", "attributeerror",
    "no module named", "not found", "not implemented",
    "media type", "containerd", "docker daemon",
    "no such image", "pull access denied", "manifest",
    "could not find", "no matching", "requirement",
    "assert", "test failed", "compilation", "linker",
    "undefined symbol", "unresolved", "segfault",
    "oom", "out of memory", "timeout", "timed out",
    "step 1/", "step 2/", "step 3/", "step 4/", "step 5/",
    "err!", "panic:", "fatal error",
]

NOISE_KEYWORDS = [
    "##[group]", "##[endgroup]", "extraheader", "sshcommand",
    "safe.directory", "worktreeconfig", "sparse-checkout",
    "set up job", "complete job", "post job", "add mask",
    "removing .pytest_cache", "removing __pycache__",
    "cacheprovider", "rootdir:", "configfile:",
    "no warnings", "warnings summary", "short test summary",
    "passed in", "collecting ", "git config", "git version",
    "persist-credentials", "fetch-depth", "check-latest",
    "allow-prereleases", "freethreaded", "submodule foreach",
]


def extract_error_signal(log_text: str) -> str:
    lines    = log_text.splitlines()
    relevant = [
        l.strip() for l in lines
        if any(k in l.lower() for k in ERROR_KEYWORDS)
        and not any(n in l.lower() for n in NOISE_KEYWORDS)
        and l.strip()
    ]
    relevant = list(dict.fromkeys(relevant))[-MAX_ERROR_LINES:]
    tail = [
        l.strip() for l in lines[-20:]
        if l.strip() and not any(n in l.lower() for n in NOISE_KEYWORDS)
    ]
    merged = list(dict.fromkeys(relevant + tail))
    signal = "\n".join(merged)
    print(f"[DETECT] Error signal: {len(merged)} lines, {len(signal)} chars")
    return signal


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DISCOVER: dynamically find relevant files
# ══════════════════════════════════════════════════════════════════════════════

def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False


def _read_trimmed(path: Path) -> str:
    try:
        raw   = path.read_text(encoding="utf-8", errors="replace")
        lines = [l for l in raw.splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        text  = "\n".join(lines)
        if len(text) > MAX_FILE_CHARS:
            text = text[:MAX_FILE_CHARS] + "\n...(truncated)"
        return text
    except Exception as exc:
        return f"(unreadable: {exc})"


def _score_file(path: Path, error_signal: str,
                tech_stacks: set[str], pipeline_types: set[str]) -> int:
    name    = path.name.lower()
    rel     = str(path).lower()
    err_low = error_signal.lower()
    score   = 0

    # Workflow files that triggered the pipeline
    if any(str(path).startswith(d) for d in WORKFLOW_DIRS) \
            and path.suffix in WORKFLOW_EXTENSIONS:
        score += 30

    # File explicitly mentioned in the error log
    if path.name.lower() in err_low or str(path).lower() in err_low:
        score += 50

    # Tech-stack file patterns
    for stack in tech_stacks:
        for pattern in STACK_FILE_SIGNALS.get(stack, []):
            if pattern.startswith(".") and name.endswith(pattern):
                score += 20
            elif pattern.lower() == name:
                score += 25
            elif "*" in pattern:
                prefix, suffix = pattern.split("*", 1)
                if name.startswith(prefix) and name.endswith(suffix):
                    score += 20

    # High-value config files always worth including
    HIGH_VALUE = {
        "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "requirements.txt", "package.json", "pom.xml", "build.gradle",
        "go.mod", "cargo.toml", "gemfile", "makefile", "gnumakefile",
        "pyproject.toml", "setup.py", "setup.cfg",
    }
    if name in HIGH_VALUE:
        score += 15

    # File content mentions error keywords (only for small files)
    if score > 0 and path.stat().st_size < 5000:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").lower()
            for kw in err_low.split()[:20]:
                if len(kw) > 4 and kw in content:
                    score += 3
        except Exception:
            pass

    return score


def discover_context(error_signal: str,
                     tech_stacks: set[str],
                     pipeline_types: set[str]) -> tuple[str, list[str]]:
    repo_root  = Path(".")
    candidates: list[tuple[int, Path]] = []

    for path in repo_root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not _is_text_file(path):
            continue
        score = _score_file(path, error_signal, tech_stacks, pipeline_types)
        if score > 0:
            candidates.append((score, path))

    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))

    parts, included, total = [], [], 0
    budget = MAX_TOTAL_CONTEXT

    for score, path in candidates:
        rel     = str(path)
        content = _read_trimmed(path)
        block   = f"### {rel}\n```\n{content}\n```"

        if total + len(block) > budget:
            print(f"[DISCOVER] Budget reached — skipping {rel} (score={score})")
            continue

        parts.append(block)
        included.append(rel)
        total += len(block)
        print(f"[DISCOVER] Included {rel} (score={score})")

    context = "\n\n".join(parts)
    print(f"[DISCOVER] {len(included)} files, {len(context)} chars total")
    return context, included


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a CI/CD auto-repair agent. A pipeline has failed.

You will receive:
  1. The pipeline type (build / release / test / lint / infra)
  2. The tech stack detected (Python / Docker / Node / Java / etc.)
  3. The CI error log (filtered to relevant lines only)
  4. The current content of relevant repository files

YOUR JOB:
  - Identify every file that needs to change to fix the failure
  - There may be ONE bug in ONE file, or MULTIPLE bugs across MULTIPLE files
  - Return the COMPLETE corrected content for every file that needs changing
  - Fix ALL bugs you can see — do not fix only one if multiple exist

RULES — follow exactly:
  - Return ONLY valid JSON. No markdown, no explanation, no preamble.
  - "file" must be copied EXACTLY from the ### header (e.g. "sample_app/Dockerfile")
  - "fixed_content" must be the COMPLETE corrected file — not a snippet, not a diff
  - Only include files that actually need changes
  - Do not remove or rewrite logic that is not related to the bug
  - Do not change file paths, imports, or structure unless they are the bug

JSON schema (return this and nothing else):
{
  "pipeline_type": "build|release|test|lint|infra|composite",
  "root_cause": "one sentence listing every bug found across all files",
  "confidence": 0.0,
  "commit_message": "fix: short description under 72 chars",
  "fixes": [
    {
      "file": "exact/path/as/in/### header",
      "reason": "what was wrong in this file and what you changed",
      "fixed_content": "complete corrected file content"
    }
  ]
}"""

USER_PROMPT = """\
## Pipeline type detected
{pipeline_types}

## Tech stack detected
{tech_stacks}

## CI failure log (relevant lines only)
{error_signal}

## Repository files (fix only what is broken)
{repo_context}"""


def build_prompt(error_signal: str, repo_context: str,
                 pipeline_types: set, tech_stacks: set) -> str:
    user = USER_PROMPT.format(
        pipeline_types=", ".join(sorted(pipeline_types)) or "unknown",
        tech_stacks=", ".join(sorted(tech_stacks)) or "unknown",
        error_signal=error_signal,
        repo_context=repo_context,
    )
    total = len(SYSTEM_PROMPT) + len(user)
    if total > MAX_TOTAL_CONTEXT + 2000:
        overhead = len(USER_PROMPT.format(
            pipeline_types="", tech_stacks="",
            error_signal=error_signal, repo_context=""))
        allowed  = (MAX_TOTAL_CONTEXT + 2000) - len(SYSTEM_PROMPT) - overhead - 100
        trimmed  = repo_context[:max(allowed, 500)] + "\n...(trimmed)"
        user = USER_PROMPT.format(
            pipeline_types=", ".join(sorted(pipeline_types)) or "unknown",
            tech_stacks=", ".join(sorted(tech_stacks)) or "unknown",
            error_signal=error_signal,
            repo_context=trimmed,
        )
        print(f"[AI] Prompt trimmed: {len(SYSTEM_PROMPT)+len(user)} chars")
    return user


def call_ai(error_signal: str, repo_context: str,
            pipeline_types: set, tech_stacks: set) -> str:
    user_prompt = build_prompt(error_signal, repo_context, pipeline_types, tech_stacks)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    payload = {
        "model":       OLLAMA_MODEL,
        "prompt":      full_prompt,
        "temperature": 0.1,
        "max_tokens":  2500,
    }
    print(f"[AI] Prompt: {len(full_prompt)} chars (~{len(full_prompt)//4} tokens)")

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[AI] Calling Ollama — attempt {attempt+1}/{MAX_RETRIES}...")
            resp = requests.post(OLLAMA_API_URL, json=payload, timeout=AI_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json().get("choices", [{}])[0].get("text", "").strip()
            print(f"[AI] Response: {len(raw)} chars")
            print(f"[AI] Preview : {raw[:500]}{'...' if len(raw)>500 else ''}")
            return raw
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
            print(f"[AI] Timeout on attempt {attempt+1}. Waiting {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_API_URL}: {exc}")

    raise RuntimeError(f"Ollama timed out after {MAX_RETRIES} attempts: {last_exc}")


def parse_ai_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    best = None
    for start in [i for i, c in enumerate(cleaned) if c == "{"]:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i+1]
                    if best is None or len(candidate) > len(best):
                        best = candidate
                    break

    if best:
        try:
            return json.loads(best)
        except json.JSONDecodeError:
            repaired = best + "}" * (best.count("{") - best.count("}"))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"No valid JSON found in AI response:\n{raw[:600]}")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def _is_blocked(file_path: str) -> bool:
    if any(file_path == b or file_path.startswith(b.rstrip("/") + "/")
           for b in ALWAYS_BLOCKED):
        return True
    return any(re.search(p, file_path) for p in BLOCKED_PATTERNS)


def validate_fix(fix: dict) -> tuple[bool, str]:
    file    = (fix.get("file") or "").strip()
    content = fix.get("fixed_content", "")

    if not file:
        return False, "missing 'file' key"
    if ".." in file or file.startswith("/") or file.startswith("~"):
        return False, f"unsafe path: {file}"
    if _is_blocked(file):
        return False, f"blocked path: {file}"
    if not isinstance(content, str) or not content.strip():
        return False, f"empty fixed_content for {file}"
    if not Path(file).exists():
        return False, f"file does not exist in repo: {file}"
    if len(content) > 500_000:
        return False, f"fixed_content too large ({len(content)} chars)"

    if file.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as exc:
            return False, f"Python syntax error: {exc}"

    elif re.search(r"\.ya?ml$", file):
        try:
            parsed = yaml.safe_load(content)
            if not isinstance(parsed, dict):
                return False, "YAML does not parse to a mapping"
            if ".github/workflows" in file and "jobs" not in parsed:
                return False, "workflow YAML missing 'jobs' key"
        except yaml.YAMLError as exc:
            return False, f"YAML error: {exc}"

    elif file.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            return False, f"JSON error: {exc}"

    elif file.endswith(".toml"):
        if not re.search(r"\[.+\]", content):
            return False, "TOML looks empty or malformed"

    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — WRITE
# ══════════════════════════════════════════════════════════════════════════════

def write_fixes(fixes: list[dict]) -> tuple[list[str], dict[str, str]]:
    written, originals = [], {}

    for fix in fixes[:MAX_FILES_FIXED]:
        ok, reason = validate_fix(fix)
        file = fix.get("file", "").strip()

        if not ok:
            print(f"  ✗ {file or '?'} — {reason}", file=sys.stderr)
            continue

        originals[file] = Path(file).read_text(encoding="utf-8", errors="replace")

        p       = Path(file)
        tmp     = p.with_suffix(p.suffix + ".tmp")
        content = fix["fixed_content"]
        if not content.endswith("\n"):
            content += "\n"

        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)

        print(f"  ✓ {file}")
        print(f"    reason : {fix.get('reason','')[:120]}")
        written.append(file)

    return written, originals


def revert_files(originals: dict[str, str]):
    for file, content in originals.items():
        try:
            Path(file).write_text(content, encoding="utf-8")
            print(f"[REVERT] Restored {file}")
        except Exception as exc:
            print(f"[REVERT] Failed to restore {file}: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — TEST
# ══════════════════════════════════════════════════════════════════════════════

def detect_test_commands(tech_stacks: set[str]) -> list[list[str]]:
    commands = []
    if "python" in tech_stacks:
        commands.append(["python", "-m", "pytest", "--tb=short", "-q", "--no-header"])
    if "node" in tech_stacks and Path("package.json").exists():
        try:
            pkg = json.loads(Path("package.json").read_text())
            if "test" in pkg.get("scripts", {}):
                commands.append(["npm", "test", "--", "--passWithNoTests"])
        except Exception:
            pass
    if "go" in tech_stacks and Path("go.mod").exists():
        commands.append(["go", "test", "./..."])
    if "java" in tech_stacks:
        if Path("pom.xml").exists():
            commands.append(["mvn", "test", "-q"])
        elif Path("build.gradle").exists():
            commands.append(["gradle", "test"])
    return commands


def run_tests(tech_stacks: set[str], written: list[str]) -> bool:
    # Lint any changed Dockerfiles with hadolint if available
    for f in written:
        if Path(f).name.lower().startswith("dockerfile"):
            lint = subprocess.run(
                ["hadolint", "--no-fail", f],
                capture_output=True, text=True, timeout=30
            )
            if lint.returncode not in (0, 127):
                for line in lint.stdout.splitlines()[-10:]:
                    print(f"  {line}")

    commands = detect_test_commands(tech_stacks)
    if not commands:
        print("[TEST] No test runner detected — skipping")
        return True

    all_passed = True
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180
            )
            for line in (result.stdout + result.stderr).splitlines()[-20:]:
                print(f"  {line}")
            ok = result.returncode == 0
            print(f"[TEST] {' '.join(cmd[:2])}: {'✓ Passed' if ok else '✗ Failed'}")
            if not ok:
                all_passed = False
        except FileNotFoundError:
            print(f"[TEST] {cmd[0]} not found — skipping")
        except subprocess.TimeoutExpired:
            print(f"[TEST] {cmd[0]} timed out — skipping")

    return all_passed


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — COMMIT
# ══════════════════════════════════════════════════════════════════════════════

def _git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check,
                          capture_output=True, text=True)


def last_commit_was_bot() -> bool:
    try:
        r = _git("log", "-1", "--pretty=%an|||%s")
        author, subject = r.stdout.strip().split("|||", 1)
        if author == BOT_NAME and subject.startswith(BOT_PREFIX):
            print(f"[GUARD] Last commit was bot: '{subject}' — stopping.")
            return True
    except Exception:
        pass
    return False


def count_recent_bot_commits(n: int = 10) -> int:
    try:
        r = _git("log", f"-{n}", "--pretty=%an")
        return sum(1 for l in r.stdout.strip().splitlines()
                   if l.strip() == BOT_NAME)
    except Exception:
        return 0


def commit_to_branch(commit_msg: str) -> str:
    try:
        _git("config", "user.name",  BOT_NAME)
        _git("config", "user.email", BOT_EMAIL)

        print("[COMMIT] Fetching remote...")
        _git("fetch", "origin")

        merge = subprocess.run(
            ["git", "merge", "-X", "ours", "origin/main",
             "--no-edit", "-m", "merge: sync before fix"],
            capture_output=True, text=True
        )
        if merge.returncode != 0:
            print(f"[WARN] Merge issue: {merge.stderr.strip()}")
            _git("merge", "--abort", check=False)

        branch = f"auto-fix/{int(time.time())}"
        _git("checkout", "-b", branch)
        _git("add", "-A")

        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit.")
            _git("checkout", "main", check=False)
            return ""

        _git("commit", "-m", commit_msg)
        _git("push", "-u", "origin", branch)
        print(f"[COMMIT] ✓ Pushed branch: {branch}")
        _git("checkout", "main", check=False)
        return branch

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Git: {exc.stderr.strip()}", file=sys.stderr)
        _git("checkout", "main", check=False)
        _git("merge", "--abort", check=False)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — GitHub API (PR / Issue)
# ══════════════════════════════════════════════════════════════════════════════

def _gh(token: str) -> dict:
    return {
        "Authorization":        f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def open_pr(token: str, repo: str, branch: str,
            commit_msg: str, root_cause: str,
            pipeline_type: str, tech_stacks: set,
            written: list[str], fixes: list[dict]) -> str:

    fix_details = ""
    for fix in fixes:
        fix_details += (
            f"\n**`{fix.get('file','?')}`**\n"
            f"> {fix.get('reason','')}\n"
        )

    body = f"""## 🤖 AI Auto-Fix — Review Required

| Field | Value |
|---|---|
| **Pipeline type** | `{pipeline_type}` |
| **Tech stack** | `{', '.join(sorted(tech_stacks)) or 'unknown'}` |
| **Root cause** | {root_cause} |
| **Files changed** | {', '.join(f'`{f}`' for f in written)} |

---

## What the AI changed

{fix_details}

---

## Review checklist
1. Open **Files changed** tab and read every diff
2. Does the fix address the root cause shown above?
3. Is any unrelated logic accidentally modified?
4. ✅ Correct → **Merge** — CI re-runs automatically
5. ❌ Wrong → **Close** — fix manually and push to main

> Auto-generated by self-healing CI. No AI changes reach main without human review.
"""
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        json={"title": f"🤖 {commit_msg}", "head": branch,
              "base": "main", "body": body},
        headers=_gh(token),
        timeout=30,
    )
    if resp.status_code in (200, 201):
        url = resp.json().get("html_url", "")
        print(f"[PR] ✓ Opened: {url}")
        return url
    print(f"[PR] Failed {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
    return ""


def open_issue(token: str, repo: str, reason: str,
               pipeline_type: str = "", run_url: str = ""):
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        json={
            "title": f"🚨 Auto-fixer stuck [{pipeline_type or 'unknown'}] — manual fix needed",
            "body":  (
                f"## Auto-fixer could not fix the CI failure\n\n"
                f"**Pipeline type:** `{pipeline_type or 'unknown'}`\n\n"
                f"**Reason:** {reason}\n\n"
                f"**Failed run:** {run_url}\n\n"
                f"Please investigate, fix manually, and close this issue."
            ),
            "labels": ["bug", "needs-manual-fix"],
        },
        headers=_gh(token),
        timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"[ISSUE] ✓ Opened: {resp.json().get('html_url','')}")
    else:
        print(f"[ISSUE] Failed {resp.status_code}: {resp.text[:200]}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generic self-healing CI/CD auto-fixer — any pipeline, any stack"
    )
    parser.add_argument("--input",      required=True,
                        help="Path to CI failure log file")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Validate but do not write files or open PR")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip tests after applying fixes")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "")

    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Log not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ── Loop guards ───────────────────────────────────────────────────────────
    if last_commit_was_bot():
        sys.exit(0)

    attempts = count_recent_bot_commits()
    if attempts >= MAX_BOT_ATTEMPTS:
        print(f"[GUARD] {attempts} bot attempts — escalating to issue.")
        if token and repo:
            open_issue(token, repo,
                       f"Auto-fixer attempted {attempts} fixes without success.",
                       run_url=os.environ.get("GITHUB_SERVER_URL","")
                                + "/" + repo + "/actions")
        sys.exit(0)

    # ══ STAGE 1: DETECT ══════════════════════════════════════════════════════
    print("\n━━━ STAGE 1: DETECT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log_text       = log_path.read_text(encoding="utf-8", errors="replace")
    error_signal   = extract_error_signal(log_text)
    pipeline_types, tech_stacks = fingerprint_pipeline(log_text)

    if not error_signal.strip():
        print("[DETECT] No error signal found — nothing to fix.")
        sys.exit(0)

    # ══ STAGE 2: DISCOVER ════════════════════════════════════════════════════
    print("\n━━━ STAGE 2: DISCOVER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    repo_context, included = discover_context(
        error_signal, tech_stacks, pipeline_types
    )

    # ══ STAGE 3: ANALYSE ═════════════════════════════════════════════════════
    print("\n━━━ STAGE 3: ANALYSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    try:
        raw     = call_ai(error_signal, repo_context, pipeline_types, tech_stacks)
        ai_data = parse_ai_response(raw)
    except Exception as exc:
        print(f"[ERROR] AI failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI analysis failed: {exc}",
                       pipeline_type=", ".join(pipeline_types))
        sys.exit(2)

    root_cause    = ai_data.get("root_cause",    "unknown")
    confidence    = float(ai_data.get("confidence", 1.0))
    commit_msg    = ai_data.get("commit_message", "fix: auto-fixer rewrite")
    fixes         = ai_data.get("fixes",         [])
    detected_type = ai_data.get("pipeline_type", ", ".join(sorted(pipeline_types)))

    print(f"\n  pipeline   : {detected_type}")
    print(f"  root_cause : {root_cause}")
    print(f"  confidence : {confidence:.0%}")
    print(f"  fixes      : {len(fixes)} file(s)")
    for fix in fixes:
        print(f"    → {fix.get('file','?')}  ({len(fix.get('fixed_content',''))} chars)")
        print(f"      {fix.get('reason','')[:120]}")

    if confidence < CONFIDENCE_MIN:
        print(f"[SKIP] Confidence {confidence:.0%} below threshold.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({confidence:.0%}). Root cause: {root_cause}",
                       pipeline_type=detected_type)
        sys.exit(0)

    if not fixes:
        print("[ERROR] AI returned no fixes.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI found root cause but produced no fixes: {root_cause}",
                       pipeline_type=detected_type)
        sys.exit(3)

    # ══ STAGE 4: VALIDATE ════════════════════════════════════════════════════
    print("\n━━━ STAGE 4: VALIDATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    valid_fixes = []
    for fix in fixes[:MAX_FILES_FIXED]:
        ok, reason = validate_fix(fix)
        file = fix.get("file", "?")
        if ok:
            print(f"  ✓ {file}")
            valid_fixes.append(fix)
        else:
            print(f"  ✗ {file} — {reason}", file=sys.stderr)

    if not valid_fixes:
        print("[ERROR] All fixes failed validation.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI rewrites failed validation. Root cause: {root_cause}",
                       pipeline_type=detected_type)
        sys.exit(3)

    # ══ DRY RUN ══════════════════════════════════════════════════════════════
    if args.dry_run:
        print("\n━━━ DRY-RUN (no files written) ━━━━━━━━━━━━━━━━━━━━━━━━━")
        for fix in valid_fixes:
            print(f"  would write {fix['file']} ({len(fix['fixed_content'])} chars)")
        sys.exit(0)

    # ══ STAGE 5: WRITE ═══════════════════════════════════════════════════════
    print("\n━━━ STAGE 5: WRITE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    written, originals = write_fixes(valid_fixes)

    if not written:
        print("[ERROR] No files were written.", file=sys.stderr)
        sys.exit(3)

    print(f"[WRITE] {len(written)} file(s) written: {written}")

    # ══ STAGE 6: TEST ════════════════════════════════════════════════════════
    print("\n━━━ STAGE 6: TEST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not args.skip_tests:
        if not run_tests(tech_stacks, written):
            revert_files(originals)
            if token and repo:
                open_issue(token, repo,
                           f"AI fix applied but tests failed — reverted. Root cause: {root_cause}",
                           pipeline_type=detected_type)
            sys.exit(5)
    else:
        print("[TEST] Skipped (--skip-tests)")

    # ══ STAGE 7: COMMIT ══════════════════════════════════════════════════════
    print("\n━━━ STAGE 7: COMMIT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    branch = commit_to_branch(commit_msg)
    if not branch:
        sys.exit(4)

    # ══ STAGE 8: PR ══════════════════════════════════════════════════════════
    print("\n━━━ STAGE 8: PR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    pr_url = ""
    if token and repo:
        pr_url = open_pr(token, repo, branch, commit_msg, root_cause,
                         detected_type, tech_stacks, written, valid_fixes)
    else:
        print("[PR] No GITHUB_TOKEN — skipping.")
        print(f"[PR] Merge manually: git checkout main && git merge {branch} && git push")

    # ══ DONE ═════════════════════════════════════════════════════════════════
    print("\n━━━ ✅ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  pipeline   : {detected_type}")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  PR         : {pr_url or 'not created'}")
    print()
    print("  Next: open PR → review diff → approve → CI re-runs automatically")


if __name__ == "__main__":
    main()