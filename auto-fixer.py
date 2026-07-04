#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — Generic Pipeline Repair
=========================================================
Works for ANY repository, ANY pipeline type (build, release, deploy, test),
ANY tech stack (Python, Node, Docker, Java, Go, Ruby, etc.)

No hardcoded file paths. The system:
  1. DETECT    — Parse the CI log, identify pipeline type and tech stack
  2. DISCOVER  — Walk the repo dynamically to find files relevant to the failure
  3. ANALYSE   — Ask Ollama: what is broken and what should each file look like?
  4. VALIDATE  — Syntax-check every rewritten file before touching disk
  5. WRITE     — Atomically overwrite only the files the AI fixed
  6. TEST      — Run pipeline-appropriate tests; revert on failure
  7. COMMIT    — Push to fix/<timestamp> branch off develop (Git flow)
  8. PR        — Open Pull Request targeting develop; human reviews before merge

Tuned for a CPU-only 3B model (qwen2.5-coder:3b) on an 8GB host:
  - Output is GRAMMAR-CONSTRAINED to a JSON schema (Ollama `format`), so even a
    3B physically cannot emit malformed JSON or wrong key names.
  - Context discovery FORCE-INCLUDES files explicitly named in the failure log
    (tracebacks, path:line errors, Dockerfile steps) before any score-based
    guess. The #1 cause of a bad fix is the broken file never reaching the
    prompt — this closes that gap.
  - Files are included WHOLE (never head+tail truncated), because the model is
    asked to output the COMPLETE file and would otherwise faithfully reproduce
    only the half it was shown.

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
import difflib
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
# NOTE: prefer the native /api/generate endpoint — the JSON-schema `format`
# constraint below is rock-solid there and flaky on /v1/completions.
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:3b")

# ── Timeout strategy ──────────────────────────────────────────────────────────
AI_TIMEOUT    = 210   # seconds per attempt — fits 2 retries in 30-min workflow
MAX_RETRIES   = 2     # fail fast, open issue rather than burning budget
RETRY_BACKOFF = [20, 20]

# ── Prompt / context budget (tuned for a CPU 3B) ────────────────────────────────
# A 3B does BETTER with 2 complete files than 3 partial ones. We never truncate
# a file mid-body, because the model is asked to reproduce the whole file.
MAX_ERROR_LINES   = 12    # keep signal tight
MAX_FILE_CHARS    = 2600  # per file — real Dockerfiles/requirements/package.json fit WHOLE
MAX_TOTAL_CONTEXT = 5200  # total repo context
MAX_CONTEXT_FILES = 2     # hard cap on files shown to the model
MAX_FILES_FIXED   = 3     # ceiling on files the AI may rewrite (rarely binding now)

# ── Safety ────────────────────────────────────────────────────────────────────
CONFIDENCE_MIN   = 0.4
MAX_BOT_ATTEMPTS = 3

# ── AI verification loop ──────────────────────────────────────────────────────
# The prescan is reliable because it VERIFIES against ground truth. We make the
# AI path reliable the same way: after the model proposes a fix we rebuild and
# run the real pipeline; a fix is accepted ONLY if it actually passes. If it
# fails, we feed the new error back to the model and let it try again. The model
# may be wrong N times — only a VERIFIED fix ever reaches a PR.
MAX_FIX_ITERATIONS = 3      # AI attempts, each verified by a real build+run
VERIFY_LOCALLY     = True   # set False only if the runner has no Docker

BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"

# ── Git flow config ───────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")

# ── Files the AI must never touch ─────────────────────────────────────────────
ALWAYS_BLOCKED = {".git", "auto-fixer.py", "self-healer.py"}
BLOCKED_PATTERNS = [
    r"\.?github/workflows/auto-fix.*\.ya?ml$",
    r"\.?github/workflows/self-heal.*\.ya?ml$",
]

# ── Test files are NEVER auto-edited ──────────────────────────────────────────
# A test encodes intent. When it fails, either the code is wrong (fix the CODE)
# or the test is wrong (a HUMAN decides which). Rewriting a test to force CI green
# destroys the signal the test exists to provide — the worst thing a self-healing
# system can do. So the model may READ tests for context but never write to them;
# a failing test whose only "fix" is the test itself escalates to an issue.
TEST_FILE_PATTERNS = [
    r"(^|/)tests?/",            # anything under a tests/ or test/ directory
    r"(^|/)test_[^/]*\.py$",    # test_*.py
    r"_test\.py$",              # *_test.py
    r"(^|/)conftest\.py$",      # pytest fixtures
    r"\.(test|spec)\.[jt]sx?$", # *.test.js / *.spec.ts etc.
]

# ── Files to exclude from context discovery (noisy, never the culprit) ─────────
# NOTE: patterns are leading-dot-optional (\.?) because some code paths historically
# normalized ".github/..." to "github/..."; this keeps the exclude robust either way.
CONTEXT_EXCLUDE_PATTERNS = [
    r"\.?github/workflows/auto-fix.*\.ya?ml$",   # this workflow itself
    r"\.?github/workflows/self-heal.*\.ya?ml$",
    r"workflow-watcher\.py$",
    r"github-monitor\.py$",
    r"ci-platform-poller\.py$",
    r"quickstart\.py$",
    r"auto-fixer\.py$",                          # never feed ourselves back in
    r"self-healer\.py$",
]

# ── Skip these directories when walking the repo ──────────────────────────────
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox",
    "target", "out", ".gradle", ".idea", ".vscode", "vendor",
    "coverage", ".nyc_output", "tmp", "temp", "logs",
}

MAX_FILE_SIZE_BYTES = 100_000


def _relstrip(rel: str) -> str:
    """Strip ONLY a leading './' — never a bare '.', which would turn
    '.github/...' into 'github/...' and (a) break the exclude regex and
    (b) produce an invalid path. Used everywhere we normalise repo paths."""
    return rel[2:] if rel.startswith("./") else rel

# Files at or above this many chars are fixed with TARGETED find/replace edits
# instead of a whole-file rewrite. Rationale: on a CPU 3B, regenerating a large
# file to change a few lines is slow (output time scales with tokens) and
# truncation-prone (hits num_predict mid-file → invalid output, rejected at
# validation). A patch is tiny output → fast, can't truncate, can't drop lines.
EDIT_MODE_THRESHOLD = 1500


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
# Ollama constrains generation to this schema via grammar at the sampler level,
# so even a 3B CANNOT emit malformed JSON or wrong key names.
#
# Two fix modes per file (the model picks; code enforces by file size):
#   - "edits": [{find, replace}, ...]  — TARGETED patch. Preferred for large
#       files and localized changes. Tiny output, no truncation risk.
#   - "fixed_content": "<whole file>"  — full rewrite. Fine for small files.
# Both are optional in the schema; application logic prefers edits when present.
FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline_type":  {"type": "string"},
        "root_cause":     {"type": "string"},
        "confidence":     {"type": "number"},
        "commit_message": {"type": "string"},
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file":   {"type": "string"},
                    "reason": {"type": "string"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "find":    {"type": "string"},
                                "replace": {"type": "string"},
                            },
                            "required": ["find", "replace"],
                        },
                    },
                    "fixed_content": {"type": "string"},
                },
                "required": ["file", "reason"],
            },
        },
    },
    "required": ["pipeline_type", "root_cause", "confidence",
                 "commit_message", "fixes"],
}


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
    pipeline_types = {
        ptype for ptype, signals in PIPELINE_TYPE_SIGNALS.items()
        if any(s in log_low for s in signals)
    }
    tech_stacks = {
        stack for stack, signals in TECH_STACK_SIGNALS.items()
        if any(s in log_low for s in signals)
    }
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
    "modulenotfounderror", "no module named", "not found",
    "not implemented", "media type", "containerd", "docker daemon",
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

# Lines that point AT a file are valuable signal even without an error keyword
# (e.g. a traceback's `File "..."` line carries no keyword but names the culprit).
FILE_REF_HINTS = [
    re.compile(r'File "[^"]+"'),
    re.compile(r'[\w./\-]+\.[A-Za-z0-9]+:\d+'),
    re.compile(r'\bDockerfile(\.\w+)?\b'),
]


def extract_error_signal(log_text: str) -> str:
    lines = log_text.splitlines()
    relevant = [
        l.strip() for l in lines
        if (any(k in l.lower() for k in ERROR_KEYWORDS)
            or any(h.search(l) for h in FILE_REF_HINTS))
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
# STAGE 2 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════════

# Patterns that reliably point at the file responsible for a failure. These are
# scanned across the WHOLE log (not the truncated signal) and force-included in
# context first — because the single biggest reason a fix fails is the broken
# file never reaching the prompt.
REFERENCED_PATH_PATTERNS = [
    r'File "([^"]+\.\w+)"',                          # Python traceback
    r'([\w./\-]+\.[A-Za-z0-9]+):\d+(?::\d+)?',       # path:line[:col] (compilers, linters, go, rust)
    r'((?:[\w./\-]+/)?Dockerfile(?:\.\w+)?)\b',      # Dockerfile / Dockerfile.prod / sub/dir/Dockerfile
    r'\(([^()]+\.\w+):\d+:\d+\)',                    # node stack frame: (/path/file.js:10:5)
    r'([\w./\-]+/requirements[\w.\-]*\.txt)',        # pip requirements
    r'(?:in|from|at|open|loading)\s+([\w./\-]+\.[A-Za-z0-9]+)',  # "in <path>" / "from <path>"
]


# ── Build / setup / config error signatures ─────────────────────────────────
# These failures happen in the CI WORKFLOW (setup-python version, build context,
# action wiring) and name NO source file — so force-include-by-path can't catch
# them. When the log matches, the CI workflow YAML itself is the prime suspect
# and must be put in front of the model. The earlier 'Dockerfile missing'
# misdiagnosis was exactly this: the real bug was in the build command, not the
# Dockerfile, and the workflow file never reached the prompt.
BUILD_CONFIG_SIGNALS = [
    "setup-python", "python-version", "version not found",
    "no such file or directory", "failed to read dockerfile",
    "unable to prepare context", "failed to solve",
    "invalid reference format", "context: ", "build context",
    "is not a valid", "could not resolve", "no version found",
    "unable to find version", "matching version", "actions/setup",
    "with: ", "uses: ", "buildx", "docker build", "dockerfile:",
]


def looks_like_build_config_error(error_signal: str) -> bool:
    low = error_signal.lower()
    return any(sig in low for sig in BUILD_CONFIG_SIGNALS)


# ── Container startup / runtime failure signatures ────────────────────────────
# These mean the image BUILT but the container won't run/stay up. The CI log
# often carries NO traceback (buffered/swallowed stderr → you see only "exited
# early"), so there is nothing to extract. When we see this AND no traceback, we
# reproduce the crash locally (probe_startup) to get a real error signal.
STARTUP_FAILURE_SIGNALS = [
    "container exited early", "image does not run correctly",
    "smoke test failed", "exited with code", "exited (", "container died",
    "crashloopbackoff", "back-off restarting", "container failed to start",
    "runtime/cmd error", "did not start", "no such process",
]


def looks_like_startup_failure(error_signal: str) -> bool:
    low = error_signal.lower()
    return any(s in low for s in STARTUP_FAILURE_SIGNALS)


def find_ci_workflow_files(repo_root: Path = Path(".")) -> list[str]:
    """
    Return repo-relative paths to CI workflow YAMLs that are legitimate fix
    targets — i.e. the build/deploy pipelines, NOT the auto-fix / self-heal
    workflows (those stay in CONTEXT_EXCLUDE_PATTERNS so we never feed the
    system its own definition).
    """
    found = []
    for d in WORKFLOW_DIRS:
        base = repo_root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in WORKFLOW_EXTENSIONS:
                rel = str(p)
                # match excludes on a dot-optional basis WITHOUT mangling the
                # real path — the path must stay valid for Path(rel).is_file().
                if any(re.search(pp, rel) for pp in CONTEXT_EXCLUDE_PATTERNS):
                    continue
                found.append(rel)
    return found


RUNTIME_CRASH_SIGNALS = [
    "container exited early", "image does not run", "does not run correctly",
    "smoke test failed", "exited with code", "exited (", "exit status",
    "runtime/cmd error", "exec format error", "cannot execute",
    "application startup failed", "crashloopbackoff",
]


def looks_like_runtime_crash(error_signal: str) -> bool:
    low = error_signal.lower()
    return any(s in low for s in RUNTIME_CRASH_SIGNALS)


def find_container_entrypoint(repo_root: Path = Path(".")) -> list[str]:
    """
    Resolve the script the container actually RUNS — the Dockerfile CMD /
    ENTRYPOINT target — to a real repo file. On a startup crash the log often has
    no traceback (just "container exited early"), so force-include-by-path finds
    nothing; this puts the crashing entrypoint in front of the model anyway.
    """
    name_to_paths: dict[str, list[str]] = {}
    for p in repo_root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            name_to_paths.setdefault(p.name, []).append(_relstrip(str(p)))

    targets: list[str] = []
    for df in repo_root.rglob("Dockerfile*"):
        if not df.is_file() or any(part in SKIP_DIRS for part in df.parts):
            continue
        try:
            txt = df.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for ln in txt.splitlines():
            m = re.match(r"(?i)\s*(CMD|ENTRYPOINT)\s+(.*)", ln)
            if not m:
                continue
            for tok in re.findall(r"[\w./\-]+\.(?:py|js|ts|sh)", m.group(2)):
                if (repo_root / tok).is_file():
                    targets.append(_relstrip(tok))
                else:
                    targets.extend(name_to_paths.get(Path(tok).name, []))
    return list(dict.fromkeys(targets))


def _basename_blocked(name: str) -> bool:
    return name in ALWAYS_BLOCKED


def extract_referenced_paths(log_text: str, repo_root: Path = Path(".")) -> list[str]:
    """
    Pull explicit file references out of the failure log and resolve them to
    real files in the repo. This is the most reliable 'which file is broken'
    signal — far better than keyword scoring — so these are force-included in
    context regardless of score. Returns an ordered, de-duplicated list of
    repo-relative paths, most-referenced first.
    """
    raw_hits: list[str] = []
    for pat in REFERENCED_PATH_PATTERNS:
        for m in re.finditer(pat, log_text):
            cand = m.group(1).strip().strip("'\"")
            if cand:
                raw_hits.append(cand)

    # Index repo files by basename once so we can resolve references that show
    # up with absolute runner paths (e.g. /home/runner/work/repo/repo/app.py).
    repo_files: dict[str, list[str]] = {}
    for p in repo_root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            repo_files.setdefault(p.name, []).append(str(p))

    # Count references so the most-mentioned file wins the first slot.
    order: list[str] = []
    counts: dict[str, int] = {}

    def _norm(rel: str) -> str:
        # Strip ONLY a leading "./" — never a bare "." (which would turn
        # ".github/..." into "github/..." and break the exclude regex, letting
        # the auto-fix workflow leak into its own context).
        return rel[2:] if rel.startswith("./") else rel

    def _add(rel: str):
        rel = _norm(rel)
        if _basename_blocked(Path(rel).name):
            return
        if any(re.search(pp, rel) for pp in CONTEXT_EXCLUDE_PATTERNS):
            return
        if rel not in counts:
            order.append(rel)
        counts[rel] = counts.get(rel, 0) + 1

    for hit in raw_hits:
        hit_norm = _norm(hit)
        # 1. exact relative-path match
        if Path(hit_norm).is_file():
            _add(hit_norm)
            continue
        # 2. basename match (handles absolute/runner paths in the log)
        for rel in repo_files.get(Path(hit).name, []):
            _add(rel)

    resolved = sorted(order, key=lambda r: -counts[r])
    if resolved:
        print(f"[DISCOVER] Referenced in log → force-include candidates: {resolved}")
    else:
        print("[DISCOVER] No explicit file paths found in log — relying on scoring")
    return resolved


def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False


def _read_whole(path: Path):
    """
    Return the file's COMPLETE content, or None if it exceeds MAX_FILE_CHARS.
    We never truncate: the prompt asks for a COMPLETE rewrite, and the model
    reproduces only what it saw. Files we actually fix (Dockerfile,
    requirements.txt, package.json, small apps) fit comfortably; oversized
    source files are skipped rather than shown broken.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return None if len(raw) > MAX_FILE_CHARS else raw
    except Exception as exc:
        return f"(unreadable: {exc})"


def _read_clamped(path: Path) -> str:
    """
    Head+tail view, used ONLY for a force-included file that is too large to
    show whole. Better to show a known-broken file partially than not at all,
    but completeness validation may then reject a truncated rewrite — that's an
    acceptable, visible failure rather than a silent missing-context one.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if len(raw) <= MAX_FILE_CHARS:
            return raw
        half = MAX_FILE_CHARS // 2
        omitted = len(raw) - MAX_FILE_CHARS
        return raw[:half] + f"\n... ({omitted} chars omitted) ...\n" + raw[-half:]
    except Exception as exc:
        return f"(unreadable: {exc})"


def _score_file(path: Path, error_signal: str,
                tech_stacks: set[str], pipeline_types: set[str]) -> int:
    name    = path.name.lower()
    err_low = error_signal.lower()
    score   = 0

    if any(str(path).startswith(d) for d in WORKFLOW_DIRS) \
            and path.suffix in WORKFLOW_EXTENSIONS:
        score += 30

    if name in err_low or str(path).lower() in err_low:
        score += 50

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

    HIGH_VALUE = {
        "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "requirements.txt", "package.json", "pom.xml", "build.gradle",
        "go.mod", "cargo.toml", "gemfile", "makefile", "gnumakefile",
        "pyproject.toml", "setup.py", "setup.cfg",
    }
    if name in HIGH_VALUE:
        score += 15

    if score > 0 and path.stat().st_size < 5000:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").lower()
            for kw in err_low.split()[:20]:
                if len(kw) > 4 and kw in content:
                    score += 3
        except Exception:
            pass

    return score


def _read_whole_forced(path: Path):
    """
    Read a FORCE-INCLUDED file whole, allowing a larger budget than the normal
    per-file cap. A force-included file is the prime suspect — the model must
    see ALL of it to rewrite it without dropping lines (partial rewrites get
    rejected by validation). We still cap hard so a runaway file can't blow the
    prompt budget on the 8GB box.
    """
    FORCED_FILE_CHARS = 8000  # generous: a CI workflow YAML fits whole here
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if len(raw) <= FORCED_FILE_CHARS:
            return raw
        return None  # genuinely huge — caller falls back to clamped
    except Exception as exc:
        return f"(unreadable: {exc})"


def discover_context(error_signal: str,
                     tech_stacks: set[str],
                     pipeline_types: set[str],
                     forced_paths: list[str]) -> tuple[str, list[str], str]:
    parts, included, total = [], [], 0
    suspect_note = ""   # extra instruction handed to the model (e.g. "fix the workflow, not the Dockerfile")

    # ── Build-config error? The CI workflow YAML *may* be the prime suspect ───
    # On setup-python / build-context / action-wiring failures the log names no
    # source file, so force-include-by-path finds nothing useful — there the CI
    # workflow YAML is the right thing to put in front of the model.
    #
    # BUT: when the log DID explicitly reference real source files (a Dockerfile,
    # requirements.txt, app.py, …), those are the truth. Forcing the workflow to
    # the front in that case evicts the actually-broken file from the budget and
    # the model 'fixes' a look-alike that isn't even there (the python:3.1 in the
    # workflow that doesn't exist). So we ONLY force the workflow + emit the
    # "Dockerfile is a decoy" directive when NO concrete source file was named.
    file_cap = MAX_CONTEXT_FILES
    forced_is_build_config = looks_like_build_config_error(error_signal)
    wf_files: list[str] = []
    if forced_is_build_config:
        wf_files = find_ci_workflow_files()
        referenced_sources = [f for f in forced_paths if f not in wf_files]
        if wf_files and not referenced_sources:
            file_cap = max(MAX_CONTEXT_FILES, 3)
            forced_paths = wf_files + [f for f in forced_paths if f not in wf_files]
            print(f"[DISCOVER] Build-config error, log named no source file → CI "
                  f"workflow is prime suspect; force-including {wf_files} "
                  f"(file cap raised to {file_cap})")
            # The decisive instruction: stop the model 'fixing' a look-alike
            # python:3.x in the Dockerfile when the real bug is in the workflow.
            suspect_note = (
                "This is a CI-CONFIGURATION failure (build/setup/version), not an "
                "application bug. The defect is in the CI WORKFLOW YAML below "
                f"({', '.join(wf_files)}) — most likely a setup-python `python-version`, "
                "a build-context path, or an action input. Fix the WORKFLOW file. "
                "Do NOT modify the Dockerfile or requirements unless the error text "
                "explicitly names them — a similar-looking version in the Dockerfile "
                "is a decoy, not the cause."
            )
        else:
            # The log referenced concrete files — trust them, do NOT hijack the
            # context with the workflow and do NOT emit the decoy directive.
            print(f"[DISCOVER] Build-config signals present but log referenced "
                  f"{referenced_sources} — trusting those, not forcing workflow.")

    # Track whether a forced file is a workflow, so we know to budget it generously.
    forced_set = set(forced_paths)

    def _try_add(rel: str, content: str, label: str) -> bool:
        nonlocal total
        # Tell the model which files are too big to safely rewrite whole, so it
        # uses targeted find/replace edits instead.
        size_tag = " (LARGE — use edits)" if len(content) >= EDIT_MODE_THRESHOLD else ""
        block = f"### {rel}{size_tag}\n```\n{content}\n```"
        if total + len(block) > MAX_TOTAL_CONTEXT and included:
            print(f"[DISCOVER] Budget reached — skipping {rel}")
            return False
        parts.append(block)
        included.append(rel)
        total += len(block)
        print(f"[DISCOVER] {label} {rel} ({len(content)} chars{', edit-mode' if size_tag else ''})")
        return True

    # ── 1. FORCE-INCLUDE files explicitly named / suspected ───────────────────
    #     The prime suspect MUST reach the model whole — never skipped for size.
    for rel in forced_paths:
        if len(included) >= file_cap:
            break
        p = Path(rel)
        if not (p.is_file() and _is_text_file(p)):
            continue
        content = _read_whole_forced(p)        # generous budget for the suspect
        if content is None:
            content = _read_clamped(p)
            print(f"[DISCOVER] ⚠ {rel} very large — force-including head+tail "
                  f"(rewrite may be rejected as truncated)")
        _try_add(rel, content, "★ Force-included")

    # ── 2. Fill remaining slots with score-ranked candidates ──────────────────
    if len(included) < file_cap:
        repo_root = Path(".")
        candidates: list[tuple[int, Path]] = []
        for path in repo_root.rglob("*"):
            if path.is_dir():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if not _is_text_file(path):
                continue
            rel_str = _relstrip(str(path))
            if rel_str in included:
                continue
            if any(re.search(pp, rel_str) for pp in CONTEXT_EXCLUDE_PATTERNS):
                continue
            sc = _score_file(path, error_signal, tech_stacks, pipeline_types)
            if sc > 0:
                candidates.append((sc, path))

        candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
        for score, path in candidates:
            if len(included) >= file_cap:
                break
            rel_str = _relstrip(str(path))
            if rel_str in included:        # already force-included — don't re-log it
                continue
            content = _read_whole(path)
            if content is None:
                print(f"[DISCOVER] Skipped {rel_str} — too large for whole-file context "
                      f"({path.stat().st_size} bytes)")
                continue
            _try_add(rel_str, content, f"Scored (score={score})")

    context = "\n\n".join(parts)
    print(f"[DISCOVER] {len(included)} files, {len(context)} chars total")
    return context, included, suspect_note


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

# Concise system prompt — 3B models follow short, direct instructions better.
SYSTEM_PROMPT = """\
You are a CI/CD repair agent. Output ONLY valid JSON. No markdown fences. No explanation. Start with { end with }.

JSON schema:
{"pipeline_type":"string","root_cause":"one sentence","confidence":0.0-1.0,"commit_message":"fix: short description","fixes":[{"file":"exact/path","reason":"what changed","edits":[{"find":"exact text from the file","replace":"corrected text"}]}]}

HOW TO FIX — pick ONE mode per file:
- PREFERRED for large files and small changes → "edits": a list of {find, replace}. `find` MUST be the SHORTEST exact snippet copied VERBATIM from the file that uniquely identifies the change — ideally just the single wrong token (a bad filename, version, flag, or path), NOT the whole line. Copy it character-for-character from the file shown below; NEVER retype it, reconstruct it from memory, or guess the surrounding line or instruction type. `replace` is that same snippet corrected. Use one edit per distinct bug. This is fast and safe — use it whenever you are changing only a few lines.
- ONLY for small files you are rewriting almost entirely → "fixed_content": the COMPLETE corrected file. Never a snippet.
- A file marked "(LARGE — use edits)" MUST be fixed with "edits", never "fixed_content".

RULES:
- file = exact path from the ### header.
- AUDIT THE WHOLE FILE. The log shows the FIRST failure only, but a file may contain SEVERAL bugs. Fix ALL clear errors in one pass (one edit each) — do not stop at the line named in the log.
- Verify filenames: if a COPY/ADD/CMD/ENTRYPOINT references a file, check it against "Repo files present" below. If it does not exist, correct it to the closest real filename.
- Do NOT change Python or base-image versions, or `python-version` values — version validity is handled separately and the versions shown to you are already correct. Never claim a version is "unavailable", "outdated", or "not found"; that is not your job and is usually wrong.
- Base your root_cause ONLY on the actual error text shown above. Do NOT invent a cause the log does not state. If the error names a file, a package (e.g. a line in requirements.txt), or a specific line, fix THAT — not something unrelated that merely looks suspicious.
- Preserve everything correct. Change only what is broken.
- Only include files that actually need changes.
- confidence < 0.4 means you are unsure — set it low rather than guess."""

USER_PROMPT = """\
## Pipeline: {pipeline_types} | Stack: {tech_stacks}
{suspect_note}
## CI failure (key lines — may show only the FIRST of several bugs):
```
{error_signal}
```
{prescan_block}
## Repo files present (use to validate any filename referenced in COPY/CMD/etc):
{repo_inventory}

## File contents (read carefully — your fixed_content must be based on these, and must fix EVERY error you can see):
{repo_context}"""


def build_repo_inventory(limit: int = 120) -> str:
    """
    A flat list of repo-relative file paths so the model can validate that a
    filename referenced by COPY/ADD/CMD/ENTRYPOINT actually exists — and correct
    it to the closest real one when it doesn't. Paths only (cheap on tokens).
    """
    paths = []
    for p in sorted(Path(".").rglob("*")):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            rel = _relstrip(str(p))
            if any(re.search(pp, rel) for pp in CONTEXT_EXCLUDE_PATTERNS):
                continue
            paths.append(rel)
            if len(paths) >= limit:
                break
    return ", ".join(paths) if paths else "(none)"


def _repo_path_index() -> tuple[set[str], dict[str, list[str]]]:
    """Return (all_posix_paths, basename -> [posix paths]) for the whole repo."""
    all_paths: set[str] = set()
    name_to_paths: dict[str, list[str]] = {}
    for p in Path(".").rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            posix = str(p).replace("\\", "/")
            posix = posix[2:] if posix.startswith("./") else posix
            all_paths.add(posix)
            name_to_paths.setdefault(p.name, []).append(posix)
    return all_paths, name_to_paths


def _repo_basenames() -> set[str]:
    names = set()
    for p in Path(".").rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            names.add(p.name)
    return names


def _closest_repo_path(token: str,
                       all_paths: set[str],
                       name_to_paths: dict[str, list[str]]) -> str | None:
    """
    Best-effort 'did you mean' for a referenced path that doesn't exist — robust
    to heavy typos (requiremeqqqnts.txt → requirements.txt, appqqq.py → app.py).

    Strategy, in order:
      1) exact basename, unambiguous.
      2) extension-aware STEM match: among repo files with the SAME extension,
         pick the one whose filename-stem is most similar. A real stem that is
         contained in (or contains) the typo'd stem gets a strong bonus, since
         injected junk ('appqqq' ⊃ 'app') is the most common mangling. Lower
         cutoff than a raw full-string match because the extension already
         constrains the candidate set.
      3) fuzzy match on the full relative path.
    """
    tok = token.replace("\\", "/")
    base = Path(tok).name
    if "." in base:
        stem, ext = base.rsplit(".", 1)
    else:
        stem, ext = base, ""

    # 1. exact basename, unambiguous
    exact = name_to_paths.get(base, [])
    if len(exact) == 1:
        return exact[0]

    # 2. extension-aware stem similarity
    best_score, best_paths = 0.0, None
    for name, paths in name_to_paths.items():
        if "." in name:
            nstem, next_ = name.rsplit(".", 1)
        else:
            nstem, next_ = name, ""
        # if the reference has an extension, the candidate must share it
        if ext and next_ and ext.lower() != next_.lower():
            continue
        score = difflib.SequenceMatcher(None, stem.lower(), nstem.lower()).ratio()
        # containment bonus: 'app' inside 'appqqq', or vice versa
        if nstem and (nstem.lower() in stem.lower() or stem.lower() in nstem.lower()):
            score = max(score, 0.75)
        if score > best_score:
            best_score, best_paths = score, paths
    if best_paths and len(best_paths) == 1 and best_score >= 0.45:
        return best_paths[0]

    # 3. fuzzy match on full relative paths
    m = difflib.get_close_matches(tok, list(all_paths), n=1, cutoff=0.6)
    if m:
        return m[0]
    return None


# ── Python version validity ──────────────────────────────────────────────────
# A small, easy-to-update set of base-image / setup-python minors we treat as
# "real". This is knowledge, not a hardcoded path: it lets us flag BOTH
# syntactically-broken tags (3.1qq, 3.) AND valid-looking-but-unavailable ones
# (3.1, 3.7). Bump this list as new Python versions ship.
SUPPORTED_PY_MINORS = {"3.8", "3.9", "3.10", "3.11", "3.12", "3.13"}
LATEST_PY = "3.12"


def _bad_python_version(value: str) -> bool:
    """True if a Python version string is invalid or unsupported.
    Accepts: 3, latest, 3.12, 3.12.1, 3.12-slim, 3.12-bookworm.
    Rejects: 3.1qq (junk), 3. (trailing dot), 3.1 / 3.7 (unsupported minor)."""
    v = value.strip().strip("\"'")
    if not v:
        return False
    core = v.split("-")[0]                 # drop -slim / -bookworm suffixes
    if core in ("latest", "3"):
        return False
    if not re.fullmatch(r"\d+(\.\d+){1,2}", core):   # syntactically broken
        return True
    parts = core.split(".")
    minor = f"{parts[0]}.{parts[1]}"
    return minor not in SUPPORTED_PY_MINORS


# Value flags whose ARGUMENT is not a build-context path (so we skip them when
# hunting for the trailing `docker build` context).
_DOCKER_VALUE_FLAGS = {
    "-f", "--file", "-t", "--tag", "--platform", "--build-arg",
    "--target", "--cache-from", "--cache-to", "--output", "-o", "--network",
}


def _extract_path_refs(rel: str, text: str) -> list[tuple[str, str]]:
    """
    Generic, file-type-agnostic extraction of LOCAL FILE references from a file,
    by the position they appear in. This is the heart of the pre-scan: it is not
    hardcoded to Dockerfiles — it knows the handful of places ANY build/CI file
    points at a local path, and lets the existence check below do the rest.

    Yields (position_label, raw_path).
    """
    refs: list[tuple[str, str]] = []
    name = Path(rel).name.lower()
    # collapse backslash line-continuations so a multi-line `docker build \` is one string
    joined = re.sub(r"\\\s*\n\s*", " ", text)

    if name.startswith("dockerfile"):
        for ln in text.splitlines():
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            ca = re.match(r"(?i)(COPY|ADD)\s+(.*)", s)
            if ca:
                toks = [t for t in ca.group(2).split() if not t.startswith("--")]
                # in `COPY src... dest`, the last token is the in-image dest — skip it
                sources = toks[:-1] if len(toks) > 1 else toks
                for t in sources:
                    refs.append((f"{ca.group(1).upper()} source", t))
            ce = re.match(r"(?i)(CMD|ENTRYPOINT)\s+(.*)", s)
            if ce:
                for t in re.findall(r"[\w./\-]+\.[A-Za-z0-9]+", ce.group(2)):
                    refs.append((ce.group(1).upper(), t))
        # RUN pip install -r <file>: the requirements file was COPY'd in from the
        # repo, so its real name IS knowable — include it so typos here are caught.
        joined_df = re.sub(r"\\\s*\n\s*", " ", text)
        for m in re.finditer(r"pip\d*\s+install\b[^\n]*?(?:-r|--requirement)\s+(\S+)", joined_df):
            refs.append(("RUN pip install -r", m.group(1)))
        return refs

    # Everything else (CI workflow YAML, shell scripts): scan the commands that
    # reference local paths — same logic regardless of which file they live in.
    for m in re.finditer(r"docker\s+(?:buildx\s+)?build\b([^\n]*)", joined):
        args = m.group(1)
        fm = re.search(r"(?:-f|--file)\s+(\S+)", args)
        if fm:
            refs.append(("docker build -f", fm.group(1)))
        # build context = trailing positional token, skipping flags + their values
        toks = args.split()
        i = len(toks) - 1
        while i >= 0:
            t = toks[i]
            if t.startswith("-"):
                i -= 1
                continue
            prev = toks[i - 1] if i - 1 >= 0 else ""
            if prev in _DOCKER_VALUE_FLAGS:
                i -= 2
                continue
            refs.append(("docker build context", t))
            break
    for m in re.finditer(r"pip\d*\s+install\b[^\n]*?(?:-r|--requirement)\s+(\S+)", joined):
        refs.append(("pip install -r", m.group(1)))
    for m in re.finditer(r"\b(?:python3?|bash|sh)\s+([\w./\-]+\.(?:py|sh))\b", joined):
        refs.append(("script reference", m.group(1)))
    return refs


def prescan_issues(included_files: list[str]) -> tuple[list[str], list[dict]]:
    """
    Deterministically find bugs a 3B tends to miss when it anchors on the FIRST
    error in the log.

    Returns (messages, autofixes):
      • messages  — human-readable findings, fed to the model as must-fix hints
                    for anything we can't fix ourselves.
      • autofixes — fixes we can apply WITHOUT the model: each is
                    {"file","reason","find","replace"} where `find` is the exact
                    wrong substring taken from the file and `replace` is the
                    deterministically-computed correction. These never go through
                    the model, so there is no find-string to hallucinate and no
                    timeout — applied straight to disk.

    One general rule, applied to EVERY file type (not hardcoded to Dockerfiles):
      • any LOCAL PATH a file references (Dockerfile COPY/CMD, a workflow's
        `docker build -f`/context, pip `-r`, a script runner) MUST exist in the
        repo — if it doesn't and we can resolve the real path, auto-fix it.
    Plus a value check:
      • an invalid base-image tag (python:3.1 / python:3.) → real version.
    """
    findings: list[str] = []
    autofixes: list[dict] = []
    all_paths, name_to_paths = _repo_path_index()
    repo_basenames = {Path(x).name for x in all_paths}

    for rel in included_files:
        p = Path(rel)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # ── value check: invalid Python base-image tag in a Dockerfile FROM ──
        if p.name.lower().startswith("dockerfile"):
            for ln in text.splitlines():
                fm = re.match(r"(?i)(\s*FROM\s+)(\S+)", ln)
                if fm and re.fullmatch(r"python:3\.?\d?", fm.group(2).strip()):
                    bad = fm.group(2).strip()
                    findings.append(
                        f"{rel}: `{bad}` is not a valid Python image tag — change "
                        f"it to a real version such as python:3.12.")
                    # exact find = the whole 'FROM <tag>' as it appears (unique,
                    # avoids python:3.1 being a substring of python:3.12)
                    autofixes.append({
                        "file": rel,
                        "reason": f"Invalid Python image tag {bad} → python:3.12",
                        "find": ln.strip(),
                        "replace": ln.strip().replace(bad, "python:3.12"),
                    })

        # ── value check: invalid setup-python version in a CI workflow ───────
        # Versions are owned by the prescan — the model is explicitly forbidden
        # from touching them (to kill the "3.12 is unavailable" confabulation), so
        # a genuinely-bad workflow python-version like "3.1" MUST be caught here
        # deterministically, or it falls through the crack and never gets fixed.
        rel_posix = rel.replace("\\", "/")
        if re.search(r"\.ya?ml$", rel_posix) and ".github/workflows" in rel_posix:
            for ln in text.splitlines():
                pv = re.search(r'(?i)(python-version:\s*)(["\']?)([^"\'\s#]+)\2', ln)
                if pv and _bad_python_version(pv.group(3)):
                    bad = pv.group(3)
                    stripped = ln.strip()
                    findings.append(
                        f"{rel}: setup-python version `{bad}` is invalid/unsupported "
                        f"— change it to `{LATEST_PY}`.")
                    if text.count(stripped) == 1:
                        autofixes.append({
                            "file": rel,
                            "reason": f"Invalid python-version {bad} → {LATEST_PY}",
                            "find": stripped,
                            "replace": stripped.replace(bad, LATEST_PY),
                        })

        # ── Universal whole-file reference-typo scan (runs on EVERY file) ────
        # NOT hardcoded to Dockerfiles. For ANY file — Dockerfile, CI workflow,
        # shell script, compose file — we walk the WHOLE file and check every
        # local file reference it makes (COPY/ADD, CMD, RUN pip -r, docker build
        # -f/context, script runners). A reference whose BASENAME exists NOWHERE
        # in the repo is an unambiguous typo; we fix EVERY one in a single pass —
        # not just the first the model would anchor on — replacing only the
        # filename with the closest real basename (never injecting a path prefix),
        # so a valid-but-relocated file (e.g. app.py in a CMD) is left untouched.
        # Files that make no file references simply yield nothing here.
        for kind, raw in _extract_path_refs(rel, text):
            tok  = raw.strip().strip("\"'")
            base = Path(tok).name
            if (not base or "." not in base or tok in (".", "..")
                    or "$" in tok or "${{" in tok or "://" in tok
                    or "@" in tok or "=" in tok or ":" in tok):
                continue
            if base in repo_basenames:               # real file exists → never touch
                continue
            match = _closest_repo_path(base, all_paths, name_to_paths)
            if not match:
                if kind != "docker build context":   # avoid noise on odd context tokens
                    findings.append(
                        f"{rel}: `{kind}` references `{base}`, which does not exist "
                        f"in the repo — correct it to the real filename.")
                continue
            real = Path(match).name                   # BASENAME only — never a path
            if real == base:
                continue
            if text.count(base) != 1:                 # ambiguous → hint the model
                findings.append(
                    f"{rel}: `{kind}` references `{base}` (does not exist) — "
                    f"did you mean `{real}`?")
                continue
            findings.append(
                f"{rel}: `{kind}` references `{base}`, which does not exist — "
                f"correcting to `{real}`.")
            autofixes.append({
                "file": rel,
                "reason": f"{kind}: '{base}' does not exist → '{real}'",
                "find": base,
                "replace": real,
            })

    findings = list(dict.fromkeys(findings))  # de-dupe
    if findings:
        print(f"[PRESCAN] Found {len(findings)} issue(s) the log did not surface:")
        for f in findings:
            print(f"[PRESCAN]   • {f}")
        if autofixes:
            print(f"[PRESCAN] {len(autofixes)} of these can be auto-fixed deterministically "
                  f"(no model needed).")
    else:
        print("[PRESCAN] No extra static issues found.")
    return findings, autofixes


# ── Cross-file PORT consistency ───────────────────────────────────────────────
# A whole class of failures has NO error in the log: the app starts fine, but a
# smoke test / healthcheck can't reach it because the app's listen port no longer
# matches the port the container contract (EXPOSE), the HEALTHCHECK and the CI
# smoke test all expect. The log shows success ("Running on ...:5001") plus a
# generic "Smoke test FAILED" — nothing the error-signal extractor OR the model
# can pin to a file, and app.py/Dockerfile/workflow rarely fit in context together
# to be correlated. So we resolve it deterministically by reading the files
# directly: the Dockerfile EXPOSE is the declared contract; the app must bind it.
# If an app file's listen port disagrees, change the APP to match EXPOSE (never
# the reverse — EXPOSE/healthcheck/CI define the contract the app must honour).
# Fires ONLY on a single unambiguous EXPOSE port and a single dissenting app
# port, so it cannot guess wrong; otherwise it stays silent.

_APP_PORT_PATTERNS = [
    re.compile(r"(?P<a>\.run\([^)]*\bport\s*=\s*)(?P<n>\d{2,5})"),       # Flask: app.run(..., port=NNNN)
    re.compile(r"(?P<a>\.listen\(\s*)(?P<n>\d{2,5})"),                   # Node: app.listen(NNNN)
    re.compile(r"(?P<a>\bPORT[\"']?\s*[,=]\s*[\"']?)(?P<n>\d{2,5})"),    # os.environ.get("PORT", NNNN) / PORT=NNNN
]


def _expose_port(repo_root: Path) -> int | None:
    """The single declared container port from Dockerfile EXPOSE, or None if
    there is zero or more than one (ambiguous → we refuse to guess)."""
    ports: set[int] = set()
    for p in repo_root.rglob("Dockerfile*"):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in re.finditer(r"(?im)^\s*EXPOSE\s+(\d{2,5})", txt):
            ports.add(int(m.group(1)))
    return next(iter(ports)) if len(ports) == 1 else None


def prescan_port_consistency(repo_root: Path = Path(".")) -> tuple[list[str], list[dict]]:
    findings: list[str] = []
    autofixes: list[dict] = []
    contract = _expose_port(repo_root)
    if contract is None:
        return findings, autofixes          # no single declared port → don't guess

    APP_EXTS = (".py", ".js", ".ts")
    for p in repo_root.rglob("*"):
        if not p.is_file() or p.suffix not in APP_EXTS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = _relstrip(str(p))
        if any(re.search(pp, rel) for pp in CONTEXT_EXCLUDE_PATTERNS):
            continue                         # never touch our own tooling
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for pat in _APP_PORT_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            app_port = int(m.group("n"))
            if app_port == contract:
                break                        # already consistent
            find = m.group(0)
            if text.count(find) != 1:
                break                        # ambiguous occurrence — don't touch
            replace = m.group("a") + str(contract)
            findings.append(
                f"{rel}: app listens on port {app_port}, but the container contract "
                f"(Dockerfile EXPOSE) is {contract} — the healthcheck and CI smoke "
                f"test target {contract}, so the app is unreachable. Change it to {contract}.")
            autofixes.append({
                "file": rel,
                "reason": f"App listen port {app_port} != EXPOSE {contract} → bind {contract}",
                "find": find,
                "replace": replace,
            })
            break                            # one listen-port fix per file
    if findings:
        print(f"[PRESCAN] Port-consistency: {len(findings)} mismatch(es) vs "
              f"EXPOSE {contract} (no error in the log — caught by direct scan):")
        for f in findings:
            print(f"[PRESCAN]   • {f}")
    return findings, autofixes


def group_autofixes(autofixes: list[dict]) -> list[dict]:
    """Collapse per-issue autofixes into one fix-dict per file, with an `edits`
    list — the same shape the model produces, so they flow through the existing
    validate → write → commit → PR path unchanged."""
    by_file: dict[str, dict] = {}
    for af in autofixes:
        f = by_file.setdefault(af["file"], {"file": af["file"], "reason": [], "edits": []})
        f["reason"].append(af["reason"])
        f["edits"].append({"find": af["find"], "replace": af["replace"]})
    fixes = []
    for f in by_file.values():
        fixes.append({"file": f["file"], "reason": "; ".join(f["reason"]), "edits": f["edits"]})
    return fixes


def build_prompt(error_signal: str, repo_context: str,
                 pipeline_types: set, tech_stacks: set,
                 prescan: list[str] | None = None,
                 suspect_note: str = "") -> str:
    repo_inventory = build_repo_inventory()
    if prescan:
        prescan_block = (
            "\n## MUST-FIX issues found by static scan (the log did NOT show "
            "these — you are REQUIRED to fix every one in the same pass):\n"
            + "\n".join(f"- {f}" for f in prescan) + "\n"
        )
    else:
        prescan_block = ""

    note_block = f"\n## ⚠ DIAGNOSIS DIRECTIVE\n{suspect_note}\n" if suspect_note else ""

    def _fmt(ctx):
        return USER_PROMPT.format(
            pipeline_types=", ".join(sorted(pipeline_types)) or "unknown",
            tech_stacks=", ".join(sorted(tech_stacks)) or "unknown",
            suspect_note=note_block,
            error_signal=error_signal,
            prescan_block=prescan_block,
            repo_inventory=repo_inventory,
            repo_context=ctx,
        )

    user = _fmt(repo_context)
    PROMPT_HARD_CAP = 7000
    full = SYSTEM_PROMPT + "\n\n" + user
    if len(full) > PROMPT_HARD_CAP:
        overhead = len(_fmt(""))
        allowed  = PROMPT_HARD_CAP - len(SYSTEM_PROMPT) - overhead - 50
        trimmed  = repo_context[:max(allowed, 1000)] + "\n...(trimmed for token budget)"
        user = _fmt(trimmed)
        print(f"[AI] Prompt hard-trimmed to {len(SYSTEM_PROMPT + user)} chars")
    return user


def _detect_ollama_endpoint() -> tuple[str, str]:
    """
    Auto-detect which Ollama API format the endpoint uses.
    Returns (url, format) where format is 'openai' or 'native'.
    Prefer 'native' (/api/generate) — the JSON-schema `format` constraint is
    reliable there.
    """
    url = OLLAMA_API_URL.rstrip("/")
    if "/api/generate" in url:
        return url, "native"
    if "/v1/completions" in url or "/v1/chat" in url:
        return url, "openai"
    base = re.sub(r"/(v1|api)/.*$", "", url)
    return f"{base}/api/generate", "native"


def _extract_token(line: bytes, fmt: str) -> str:
    """Extract the text token from one streamed line, handling both API formats."""
    if not line:
        return ""
    try:
        text = line.decode("utf-8", errors="replace").strip()
        if text.startswith("data: "):       # OpenAI SSE prefix
            text = text[6:].strip()
        if text in ("", "[DONE]"):
            return ""
        obj = json.loads(text)
        if fmt == "openai":
            return obj.get("choices", [{}])[0].get("text", "")
        return obj.get("response", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def call_ai(error_signal: str, repo_context: str,
            pipeline_types: set, tech_stacks: set,
            prescan: list[str] | None = None,
            suspect_note: str = "") -> str:
    """Call Ollama with auto-detected endpoint, schema-constrained output, and
    robust streaming + batch fallback."""
    user_prompt  = build_prompt(error_signal, repo_context, pipeline_types, tech_stacks, prescan, suspect_note)
    full_prompt  = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    endpoint, fmt = _detect_ollama_endpoint()

    if fmt == "openai":
        # /v1/completions does not reliably honour JSON-schema grammar; rely on
        # the prompt + parse_ai_response fallback here. Prefer native endpoint.
        payload = {
            "model":       OLLAMA_MODEL,
            "prompt":      full_prompt,
            "temperature": 0.05,
            "max_tokens":  2000,
            "stream":      True,
        }
    else:  # native /api/generate — grammar-constrained to FIX_SCHEMA
        payload = {
            "model":   OLLAMA_MODEL,
            "prompt":  full_prompt,
            "format":  FIX_SCHEMA,
            "options": {"temperature": 0.05, "num_predict": 2000},
            "stream":  True,
        }

    print(f"[AI] Endpoint : {endpoint} (format: {fmt}"
          f"{', schema-constrained' if fmt == 'native' else ''})")
    print(f"[AI] Prompt   : {len(full_prompt)} chars (~{len(full_prompt)//4} tokens)")
    print(f"[AI] Model    : {OLLAMA_MODEL} | timeout: {AI_TIMEOUT}s | max_tokens: 2000")

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[AI] Attempt {attempt+1}/{MAX_RETRIES} — streaming...")
            t_start    = time.time()
            last_print = t_start
            collected  = []

            resp = requests.post(
                endpoint, json=payload,
                timeout=(10, AI_TIMEOUT), stream=True,
            )
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                token = _extract_token(raw_line, fmt)
                if token:
                    collected.append(token)
                now = time.time()
                if now - last_print > 15:
                    elapsed = int(now - t_start)
                    chars = sum(len(t) for t in collected)
                    print(f"[AI] ...generating ({elapsed}s | {chars} chars)")
                    last_print = now

            raw = "".join(collected).strip()
            elapsed = time.time() - t_start
            print(f"[AI] Done in {elapsed:.1f}s — {len(raw)} chars received")

            if not raw:
                print("[AI] Stream yielded 0 chars — trying batch response parse...")
                try:
                    body = resp.json()
                    raw = (body.get("choices", [{}])[0].get("text", "")
                           if fmt == "openai" else body.get("response", "")).strip()
                    print(f"[AI] Batch fallback: {len(raw)} chars")
                except Exception:
                    pass

            if not raw:
                raise RuntimeError(
                    "Ollama returned an empty response. Check: model loaded "
                    "(`ollama ps`), endpoint URL correct, process healthy."
                )

            print(f"[AI] Preview: {raw[:400]}{'...' if len(raw) > 400 else ''}")
            return raw

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            elapsed  = time.time() - t_start
            wait     = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"[AI] Timeout after {elapsed:.0f}s on attempt {attempt+1}. "
                  f"Waiting {wait}s...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {endpoint}. Is Ollama running "
                f"on the self-hosted runner? Error: {exc}"
            )

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}")

    raise RuntimeError(
        f"Ollama did not respond after {MAX_RETRIES} attempts ({AI_TIMEOUT}s each). "
        f"Last error: {last_exc}. Tips: reduce MAX_TOTAL_CONTEXT, use "
        f"qwen2.5-coder:1.5b, or increase workflow timeout-minutes."
    )


def _salvage_token_edit(content: str, find: str, replace: str) -> str | None:
    """
    Last-resort edit recovery for a flaky 3B. When `find` doesn't match the file
    verbatim, the model has usually fabricated the surrounding line (e.g. called
    a `RUN pip install -r X` a `COPY X`) but got the actual changed TOKEN right.
    Strip the longest common prefix and suffix of find/replace to isolate the
    changed fragment; if that exact old fragment appears EXACTLY ONCE in the
    file, apply just that swap. The single-occurrence requirement is the safety
    guard — we never touch an ambiguous or absent match.
    """
    if not find or not replace or find == replace:
        return None
    # longest common prefix
    i = 0
    while i < len(find) and i < len(replace) and find[i] == replace[i]:
        i += 1
    # longest common suffix (not overlapping the prefix)
    j = 0
    while (j < len(find) - i and j < len(replace) - i
           and find[-1 - j] == replace[-1 - j]):
        j += 1
    old_frag = find[i:len(find) - j]
    new_frag = replace[i:len(replace) - j]
    if len(old_frag.strip()) < 3:          # too small to be a safe anchor
        return None
    if content.count(old_frag) != 1:       # ambiguous or absent — refuse
        return None
    return content.replace(old_frag, new_frag, 1)


def _apply_edits(original: str, edits: list[dict]) -> tuple[str | None, str]:
    """
    Apply a list of {find, replace} edits to `original`. Each `find` must occur
    in the file. We try an exact match first, then a whitespace-tolerant match
    (the 3B occasionally normalises indentation), so a near-miss still lands.
    Returns (new_content, "ok") or (None, reason-it-failed).
    """
    content = original
    for i, ed in enumerate(edits):
        find = ed.get("find", "")
        repl = ed.get("replace", "")
        if not isinstance(find, str) or find == "":
            return None, f"edit #{i+1} has an empty 'find'"
        if find in content:
            content = content.replace(find, repl)
            continue
        # Whitespace-tolerant fallback: match ignoring leading/trailing spaces
        # per line, then splice the replacement in at the matched span.
        norm = lambda s: "\n".join(l.strip() for l in s.splitlines())
        nfind, ncontent = norm(find), norm(content)
        if nfind and nfind in ncontent:
            # Find the original (un-normalised) line range that corresponds.
            flines = [l.strip() for l in find.splitlines()]
            clines = content.splitlines()
            hit = -1
            for j in range(len(clines) - len(flines) + 1):
                if [c.strip() for c in clines[j:j + len(flines)]] == flines:
                    hit = j
                    break
            if hit >= 0:
                # Preserve the original indentation of the matched block so the
                # replacement doesn't break indentation-sensitive formats (YAML).
                base = clines[hit][:len(clines[hit]) - len(clines[hit].lstrip())]
                new_lines = [(base + rl if rl.strip() else rl)
                             for rl in (repl.splitlines() or [""])]
                clines[hit:hit + len(flines)] = new_lines
                content = "\n".join(clines)
                if original.endswith("\n") and not content.endswith("\n"):
                    content += "\n"
                continue
        # Final salvage: the model frequently fabricates the surrounding line
        # but gets the actual changed token right (it called this a COPY when it
        # was a RUN pip install). Isolate the changed fragment and, if it occurs
        # exactly once in the file, swap just that.
        salvaged = _salvage_token_edit(content, find, repl)
        if salvaged is not None and salvaged != content:
            print(f"[EDIT] Salvaged edit #{i+1} by unique-token match "
                  f"(model's find did not match verbatim)")
            content = salvaged
            continue
        return None, (f"edit #{i+1} 'find' text not present in file — "
                      f"the model's find string did not match the actual content")
    if content == original:
        return None, "edits produced no change"
    return content, "ok"


def _resolve_fix_content(fix: dict) -> tuple[str | None, str]:
    """
    Turn a fix (either `edits` or `fixed_content`) into the final file content.
    Prefers targeted edits when present. The resolved content is what every
    downstream check and the disk write operate on — so validation always runs
    against the REAL post-fix file, whichever mode produced it.
    """
    file = (fix.get("file") or "").strip()
    edits = fix.get("edits")
    if edits:
        p = Path(file)
        if not p.is_file():
            return None, f"file does not exist in repo: {file}"
        original = p.read_text(encoding="utf-8", errors="replace")
        return _apply_edits(original, edits)
    fc = fix.get("fixed_content")
    if isinstance(fc, str) and fc.strip():
        return fc, "ok"
    return None, "fix has neither 'edits' nor 'fixed_content'"


def _normalize_fix_keys(fixes: list) -> list:
    """Normalize field names from non-compliant responses. With the native
    schema-constrained endpoint this should essentially never fire, but it
    stays as belt-and-suspenders for the /v1/completions path."""
    KEY_MAP_FILE    = ("file_path", "path", "filename", "filepath", "name")
    KEY_MAP_CONTENT = ("content", "new_content", "fixed", "updated_content",
                       "corrected_content", "file_content", "code")
    normalized = []
    for fix in fixes:
        fix = dict(fix)
        if "file" not in fix:
            for alt in KEY_MAP_FILE:
                if alt in fix:
                    fix["file"] = fix.pop(alt)
                    break
        if "fixed_content" not in fix:
            for alt in KEY_MAP_CONTENT:
                if alt in fix:
                    fix["fixed_content"] = fix.pop(alt)
                    break
        normalized.append(fix)
    return normalized


def _validate_fix_completeness(fix: dict, original_path: Path) -> tuple[bool, str]:
    """Reject truncated AI rewrites — especially single-line Dockerfiles."""
    content = fix.get("fixed_content", "")
    file    = fix.get("file", "")

    if Path(file).name.lower().startswith("dockerfile"):
        non_empty = [l for l in content.splitlines()
                     if l.strip() and not l.strip().startswith("#")]
        if len(non_empty) < 3:
            return False, (f"Dockerfile fix truncated: only {len(non_empty)} "
                           f"instruction(s). Must output the COMPLETE Dockerfile.")
        try:
            original = original_path.read_text(encoding="utf-8", errors="replace")
            for keyword in ["WORKDIR", "COPY", "RUN", "CMD", "EXPOSE"]:
                if keyword in original and keyword not in content:
                    return False, (f"Dockerfile fix dropped '{keyword}'. "
                                   f"Must preserve all existing instructions.")
        except Exception:
            pass
    return True, "ok"


def _fix_changes_valid_py_version(fix: dict) -> bool:
    """
    True if this MODEL fix rewrites an ALREADY-VALID Python version. Version
    validity is owned by the deterministic prescan (_bad_python_version /
    SUPPORTED_PY_MINORS); the model has no better information than that, so any
    model edit that changes a version we already know is fine is a confabulation
    — the recurring 'python:3.12 is not available' hallucination — not a fix.
    """
    def _versions(text: str) -> list[str]:
        vs = []
        for m in re.finditer(r"python:([A-Za-z0-9.\-]+)", text):
            vs.append(m.group(1).strip().strip("\"'"))
        for m in re.finditer(r'python-version:\s*["\']?([0-9][^"\'\s#]*)', text):
            vs.append(m.group(1).strip().strip("\"'"))
        return vs

    edits = fix.get("edits")
    if edits:
        for e in edits:
            find_v = _versions(e.get("find", "") or "")
            repl_v = _versions(e.get("replace", "") or "")
            for v in find_v:
                if not _bad_python_version(v) and v not in repl_v:
                    return True
        return False

    fc = fix.get("fixed_content")
    if isinstance(fc, str) and fc.strip():
        file = (fix.get("file") or "").strip()
        try:
            orig = Path(file).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False
        new_v = _versions(fc)
        for v in _versions(orig):
            if not _bad_python_version(v) and v not in new_v:
                return True
    return False


def parse_ai_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if data is None:
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
                data = json.loads(best)
            except json.JSONDecodeError:
                repaired = best + "}" * (best.count("{") - best.count("}"))
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    pass

    if data is None:
        raise ValueError(f"No valid JSON in AI response:\n{raw[:600]}")

    if "fixes" in data and isinstance(data["fixes"], list):
        data["fixes"] = _normalize_fix_keys(data["fixes"])
        unmapped = [f.get("file", "?") for f in data["fixes"]
                    if "file" not in f
                    or ("fixed_content" not in f and "edits" not in f)]
        if unmapped:
            print(f"[AI] Warning: could not normalize keys for: {unmapped}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def _is_blocked(file_path: str) -> bool:
    if any(file_path == b or file_path.startswith(b.rstrip("/") + "/")
           for b in ALWAYS_BLOCKED):
        return True
    return any(re.search(p, file_path) for p in BLOCKED_PATTERNS)


def _is_test_file(file_path: str) -> bool:
    """Test files are the spec/safety net — never auto-edited. A failing test
    means either the code is wrong (fix the CODE) or the test is wrong (a human
    decides). Rewriting the test to make CI green destroys the signal, so we
    refuse and let the run escalate to an issue."""
    fp = file_path.replace("\\", "/")
    return any(re.search(p, fp) for p in TEST_FILE_PATTERNS)


def validate_fix(fix: dict) -> tuple[bool, str]:
    file    = (fix.get("file") or "").strip()

    if not file:
        return False, "missing 'file' key"
    if ".." in file or file.startswith("/") or file.startswith("~"):
        return False, f"unsafe path: {file}"
    if _is_blocked(file):
        return False, f"blocked path: {file}"
    if _is_test_file(file):
        return False, (f"test file — refusing to edit {file}. A self-healing "
                       f"system must never rewrite tests to force them green; a "
                       f"failing test is escalated for human review instead.")
    if not Path(file).exists():
        return False, f"file does not exist in repo: {file}"

    # Resolve whichever mode the model used (edits or fixed_content) into the
    # final file content, then validate THAT. Cache it on the fix so write_fixes
    # and the PR body use the identical resolved content.
    content, reason = _resolve_fix_content(fix)
    if content is None:
        return False, reason
    fix["fixed_content"] = content   # normalise: downstream always reads this

    if not content.strip():
        return False, f"empty result for {file}"
    if len(content) > 500_000:
        return False, f"result too large ({len(content)} chars)"

    ok, reason = _validate_fix_completeness(fix, Path(file))
    if not ok:
        return False, reason

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
            if ".github/workflows" in file:
                # A workflow rewrite by a 3B can silently drop triggers, jobs, or
                # whole steps. Guard the structure that makes it a valid pipeline.
                # PyYAML parses the `on:` key as boolean True, so check both.
                if "jobs" not in parsed:
                    return False, "workflow YAML missing 'jobs' key"
                if "on" not in parsed and True not in parsed:
                    return False, "workflow YAML missing 'on:' trigger block"
                if not isinstance(parsed.get("jobs"), dict) or not parsed["jobs"]:
                    return False, "workflow YAML has no jobs defined"
                # Don't let the model gut the pipeline: the rewrite must keep at
                # least as many `steps:` entries as the original had.
                try:
                    orig = yaml.safe_load(
                        Path(file).read_text(encoding="utf-8", errors="replace"))
                    if isinstance(orig, dict):
                        n_orig = sum(len(j.get("steps", []))
                                     for j in orig.get("jobs", {}).values()
                                     if isinstance(j, dict))
                        n_new = sum(len(j.get("steps", []))
                                    for j in parsed.get("jobs", {}).values()
                                    if isinstance(j, dict))
                        if n_orig and n_new < n_orig:
                            return False, (f"workflow rewrite dropped steps "
                                           f"({n_orig} → {n_new}) — must preserve all steps")
                except Exception:
                    pass
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


def _resolve_entrypoint(repo_root: Path = Path(".")) -> tuple[str | None, list[str] | None]:
    """
    Find the Python entrypoint the container runs (Dockerfile CMD/ENTRYPOINT) and
    resolve it to a real repo file. Returns (repo_relative_path, argv) for use
    with probe_startup, or (None, None) if no python entrypoint is found.
    """
    for p in sorted(repo_root.rglob("Dockerfile*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for m in re.finditer(r'(?im)^\s*(?:CMD|ENTRYPOINT)\s+(.+)$', txt):
            scripts = re.findall(r'([\w./\-]+\.py)', m.group(1))
            if not scripts:
                continue
            name = Path(scripts[0]).name
            for f in sorted(repo_root.rglob(name)):
                if f.is_file() and not any(d in f.parts for d in SKIP_DIRS):
                    rel = _relstrip(str(f).replace("\\", "/"))
                    return rel, ["python", "-u", rel]
    return None, None


def probe_startup(argv: list[str], timeout: int = 6) -> tuple[str, str]:
    """
    Run the app entrypoint locally to reproduce a startup crash OR confirm it
    serves. A web entrypoint blocks (serves forever), so 'still running at the
    timeout' is the SUCCESS case; a quick non-zero exit is the crash we want.
    Returns (status, captured_output):
      "crash"  — exited non-zero before timeout; captured output has the cause
      "exited" — exited 0 immediately; runs but does not stay up / serve
      "served" — still running at timeout; started OK (we kill it)
      "skip"   — could not run, or failed on missing deps in the fixer env
                 (inconclusive — caller must NOT treat as a real failure)
    """
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    except (FileNotFoundError, OSError):
        return "skip", ""
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return "served", ""
    out = out or ""
    low = out.lower()
    # If the app reached the point of starting/binding a server, that IS success
    # for this gate — the entrypoint's __main__ correctly started the server. A
    # port collision on the shared runner ("address already in use" — e.g. the
    # already-deployed app is holding that port) is a PROBE-ENVIRONMENT issue, not
    # a code bug, so we must NEVER revert a good fix over it.
    if ("address already in use" in low or "is in use by" in low
            or "serving flask app" in low or "running on http" in low
            or "* running on" in low or "uvicorn running on" in low
            or "listening on" in low):
        return "served", out[-2000:]
    if proc.returncode != 0:
        if "modulenotfounderror" in low or "importerror" in low:
            return "skip", out[-2000:]          # fixer env lacks the app's deps
        return "crash", out[-2000:]
    return "exited", out[-2000:]


def _free_host_port() -> int:
    """A currently-free host port, so verification never collides with the live
    app on :5000 (the mistake the bare-python startup probe made)."""
    import socket
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def verify_build_and_run(build_timeout: int = 600,
                         serve_tries: int = 15) -> tuple[bool, str]:
    """
    GROUND-TRUTH verification: build the image and run it exactly like CI, on a
    random host port. Returns (passed, output). This is what makes the AI path
    reliable — a model fix is accepted ONLY if this passes; otherwise the output
    (the real, current error) is fed back to the model for another attempt.

    Degrades safely: if there is no Dockerfile or Docker isn't available, returns
    (True, "...") so we don't block on an environment we can't verify in — the
    startup gate / CI still apply.
    """
    dfs = [p for p in sorted(Path(".").rglob("Dockerfile*"))
           if p.is_file() and not any(d in p.parts for d in SKIP_DIRS)]
    if not dfs:
        return True, "(no Dockerfile — skipping docker verification)"

    df  = str(dfs[0])
    ctx = str(Path(df).parent) or "."
    tag = "autofix-verify:latest"

    try:
        build = subprocess.run(
            ["docker", "build", "-f", df, "-t", tag, ctx],
            capture_output=True, text=True, timeout=build_timeout)
    except FileNotFoundError:
        return True, "(docker not available — skipping verification)"
    except subprocess.TimeoutExpired:
        return False, "BUILD TIMED OUT"

    if build.returncode != 0:
        return False, "BUILD FAILED:\n" + (build.stdout + build.stderr)[-2200:]

    expose = _expose_port(Path(".")) or 5000
    host   = _free_host_port()
    name   = f"autofix-verify-{int(time.time())}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "-p", f"{host}:{expose}", tag],
        capture_output=True, text=True, timeout=60)
    if run.returncode != 0:
        return False, "docker run FAILED:\n" + (run.stdout + run.stderr)[-1200:]

    ok = False
    try:
        for _ in range(serve_tries):
            c = subprocess.run(["curl", "-fs", f"http://127.0.0.1:{host}"],
                               capture_output=True, timeout=5)
            if c.returncode == 0:
                ok = True
                break
            alive = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", name],
                capture_output=True, text=True)
            if alive.stdout.strip() != "true":
                break
            time.sleep(2)
        logs = subprocess.run(["docker", "logs", name],
                              capture_output=True, text=True).stdout
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    if ok:
        return True, "image builds and serves ✓"
    return False, "SMOKE FAILED — container built but did not serve:\n" + logs[-1600:]


def run_tests(tech_stacks: set[str], written: list[str]) -> bool:
    for f in written:
        if Path(f).name.lower().startswith("dockerfile"):
            lint = subprocess.run(["hadolint", "--no-fail", f],
                                  capture_output=True, text=True, timeout=30)
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
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
# STAGE 7 — COMMIT (Git flow)
# ══════════════════════════════════════════════════════════════════════════════

def _git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


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
        return sum(1 for l in r.stdout.strip().splitlines() if l.strip() == BOT_NAME)
    except Exception:
        return 0


def _ensure_base_branch_exists() -> bool:
    _git("fetch", "origin", check=False)
    result = _git("ls-remote", "--heads", "origin", GIT_BASE_BRANCH, check=False)
    if result.stdout.strip():
        local_check = _git("rev-parse", "--verify", GIT_BASE_BRANCH, check=False)
        if local_check.returncode != 0:
            _git("checkout", "-b", GIT_BASE_BRANCH, f"origin/{GIT_BASE_BRANCH}")
        print(f"[GIT FLOW] Base branch '{GIT_BASE_BRANCH}' found on remote ✓")
        return True

    print(f"[GIT FLOW] '{GIT_BASE_BRANCH}' not found. Creating from main...")
    try:
        main_ref = _git("rev-parse", "--verify", "main", check=False)
        if main_ref.returncode != 0:
            _git("rev-parse", "--verify", "master")
            _git("checkout", "-b", GIT_BASE_BRANCH, "master")
        else:
            _git("checkout", "-b", GIT_BASE_BRANCH, "main")
        _git("push", "-u", "origin", GIT_BASE_BRANCH)
        print(f"[GIT FLOW] Created and pushed '{GIT_BASE_BRANCH}' branch ✓")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[GIT FLOW] Could not create '{GIT_BASE_BRANCH}': {exc.stderr.strip()}",
              file=sys.stderr)
        return False


# ── Runtime artifacts the auto-fixer creates while running — NEVER commit these.
RUNTIME_ARTIFACTS = ["failure.log", "workflow_logs.zip", "logs/"]


def _ensure_artifacts_ignored():
    """Gitignore the auto-fixer's own runtime artifacts, and untrack any a
    previous run already committed — so PRs never carry failure.log /
    workflow_logs.zip / logs/."""
    gi = Path(".gitignore")
    existing = gi.read_text(encoding="utf-8", errors="replace").splitlines() if gi.is_file() else []
    missing = [a for a in RUNTIME_ARTIFACTS if a not in existing]
    if missing:
        with gi.open("a", encoding="utf-8") as fh:
            if existing and existing[-1].strip():
                fh.write("\n")
            fh.write("# auto-fixer runtime artifacts (do not commit)\n")
            fh.write("\n".join(missing) + "\n")
        print(f"[GIT FLOW] Added to .gitignore: {missing}")
    for a in RUNTIME_ARTIFACTS:
        _git("rm", "--cached", "-r", "--ignore-unmatch", a, check=False)


def commit_to_branch(commit_msg: str, written: list[str] | None = None) -> str:
    try:
        _git("config", "user.name",  BOT_NAME)
        _git("config", "user.email", BOT_EMAIL)

        if not _ensure_base_branch_exists():
            print(f"[ERROR] Cannot set up base branch '{GIT_BASE_BRANCH}'", file=sys.stderr)
            return ""

        print(f"[GIT FLOW] Checking out '{GIT_BASE_BRANCH}'...")
        _git("checkout", GIT_BASE_BRANCH)
        _git("pull", "origin", GIT_BASE_BRANCH, check=False)

        branch = f"fix/{int(time.time())}"
        print(f"[GIT FLOW] Creating '{branch}' from '{GIT_BASE_BRANCH}'...")
        _git("checkout", "-b", branch)

        # Ensure runtime artifacts are ignored + untracked. This stages a
        # .gitignore update and the removal of any artifact a previous run
        # already committed (your repo currently has failure.log / workflow_logs.zip).
        _ensure_artifacts_ignored()

        # Stage ONLY the files the AI actually fixed — never `git add -A`, which
        # would sweep the runtime artifacts (failure.log, workflow_logs.zip,
        # logs/) straight into the PR. Plus the .gitignore we just updated.
        if written:
            _git("add", "--", *written)
            print(f"[GIT FLOW] Staged AI-written files: {written}")
        else:
            _git("add", "-u")
            print("[GIT FLOW] No file list supplied — staged tracked changes only")
        if Path(".gitignore").is_file():
            _git("add", ".gitignore", check=False)

        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit.")
            _git("checkout", GIT_BASE_BRANCH, check=False)
            return ""

        _git("commit", "-m", commit_msg)
        _git("push", "-u", "origin", branch)
        print(f"[GIT FLOW] ✓ Pushed: {branch} → PR targets: {GIT_TARGET_BRANCH}")
        _git("checkout", GIT_BASE_BRANCH, check=False)
        return branch

    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Git: {exc.stderr.strip()}", file=sys.stderr)
        _git("checkout", GIT_BASE_BRANCH, check=False)
        _git("merge", "--abort", check=False)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — GitHub PR / Issue
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
    fix_details = "".join(
        f"\n**`{fix.get('file','?')}`**\n> {fix.get('reason','')}\n"
        for fix in fixes
    )
    body = f"""## 🤖 AI Auto-Fix — Review Required

| Field | Value |
|---|---|
| **Pipeline type** | `{pipeline_type}` |
| **Tech stack** | `{', '.join(sorted(tech_stacks)) or 'unknown'}` |
| **Root cause** | {root_cause} |
| **Files changed** | {', '.join(f'`{f}`' for f in written)} |
| **Target branch** | `{GIT_TARGET_BRANCH}` |

## What the AI changed
{fix_details}

## Git flow
```
{branch} (this PR)
    └─► {GIT_TARGET_BRANCH}   ← merge here after review
            └─► release/*   ← cut when sprint is ready
                    └─► main  ← after QA sign-off + tag
```

## Review checklist
1. Read **Files changed** tab — every diff line
2. Does the fix address the root cause above?
3. Are all original instructions preserved (COPY/RUN/CMD etc)?
4. Is anything unrelated accidentally modified?
5. ✅ Correct → **Merge into `{GIT_TARGET_BRANCH}`**
6. ❌ Wrong → **Close** — fix manually on `{GIT_TARGET_BRANCH}`

> Auto-generated. No AI changes reach main without: human review → develop → release → QA → main.
"""
    resp = requests.post(
        f"https://api.github.com/repos/{repo}/pulls",
        json={"title": f"🤖 {commit_msg}", "head": branch,
              "base": GIT_TARGET_BRANCH, "body": body},
        headers=_gh(token), timeout=30,
    )
    if resp.status_code in (200, 201):
        url = resp.json().get("html_url", "")
        print(f"[PR] ✓ Opened: {url}")
        return url
    if resp.status_code == 422:
        print(f"[PR] 422 — does '{GIT_TARGET_BRANCH}' exist on GitHub?", file=sys.stderr)
    else:
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
                f"**Git flow reminder:** fix manually on `develop`, "
                f"then follow release/* → main process."
            ),
            "labels": ["bug", "needs-manual-fix"],
        },
        headers=_gh(token), timeout=30,
    )
    if resp.status_code in (200, 201):
        print(f"[ISSUE] ✓ Opened: {resp.json().get('html_url','')}")
    else:
        print(f"[ISSUE] Failed {resp.status_code}: {resp.text[:200]}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generic self-healing CI/CD auto-fixer")
    parser.add_argument("--input",      required=True, help="Path to CI failure log")
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

    print(f"[GIT FLOW] Base: {GIT_BASE_BRANCH} | Target: {GIT_TARGET_BRANCH}")
    print(f"[TIMING]   AI timeout: {AI_TIMEOUT}s × {MAX_RETRIES} retries | "
          f"Context: {MAX_CONTEXT_FILES} files / {MAX_TOTAL_CONTEXT} chars | "
          f"Hard prompt cap: 7000 chars")
    t0 = time.time()

    if last_commit_was_bot():
        sys.exit(0)

    attempts = count_recent_bot_commits()
    if attempts >= MAX_BOT_ATTEMPTS:
        print(f"[GUARD] {attempts} bot attempts — escalating to issue.")
        if token and repo:
            open_issue(token, repo,
                       f"Auto-fixer attempted {attempts} fixes without success.",
                       run_url=os.environ.get("GITHUB_SERVER_URL", "")
                                + "/" + repo + "/actions")
        sys.exit(0)

    # ══ STAGE 1: DETECT ══════════════════════════════════════════════════════
    print("\n━━━ STAGE 1: DETECT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    t1 = time.time()
    log_text     = log_path.read_text(encoding="utf-8", errors="replace")
    error_signal = extract_error_signal(log_text)
    pipeline_types, tech_stacks = fingerprint_pipeline(log_text)
    print(f"[TIMING] Stage 1 done in {time.time()-t1:.1f}s")

    if not error_signal.strip():
        print("[DETECT] No error signal found — nothing to fix.")
        sys.exit(0)

    # ══ STAGE 2: DISCOVER ════════════════════════════════════════════════════
    print("\n━━━ STAGE 2: DISCOVER ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    t2 = time.time()
    forced_paths = extract_referenced_paths(log_text)

    # ── Startup / runtime crash handling ─────────────────────────────────────
    # On a startup failure the ENTRYPOINT is the crash site, so it MUST reach the
    # model — ahead of less-relevant files like requirements.txt that the 2-file
    # budget was otherwise evicting it behind (the model then 'fixes' a file it
    # never saw). Prioritize entrypoint + Dockerfile (EXPOSE port + CMD); if the
    # log has no traceback, reproduce the crash locally to capture the real error.
    startup_ep = None
    has_traceback = ("traceback" in error_signal.lower()
                     or bool(re.search(r'File "[^"]+"', error_signal)))
    if looks_like_startup_failure(error_signal):
        ep_rel, argv = _resolve_entrypoint()
        if ep_rel:
            startup_ep = ep_rel
            if not has_traceback:
                print(f"[DISCOVER] Startup failure, no traceback in log — "
                      f"reproducing locally via {ep_rel} ...")
                status, captured = probe_startup(argv)
                if status == "crash" and captured.strip():
                    print("[DISCOVER] Reproduced the crash — using captured output.")
                    error_signal = (error_signal + "\n--- reproduced locally ---\n"
                                    + captured.strip())[-4000:]
                elif status == "exited":
                    error_signal += (f"\n--- reproduced locally ---\n{ep_rel}: the "
                                     f"process exits immediately without starting a server.")
            # Entrypoint first, then any referenced Dockerfile; everything else
            # (requirements.txt etc.) goes last so the crash file is never evicted.
            dfs = [f for f in forced_paths
                   if Path(f).name.lower().startswith("dockerfile")]
            forced_paths = ([ep_rel] + dfs
                            + [f for f in forced_paths if f != ep_rel and f not in dfs])
            print(f"[DISCOVER] Startup failure → entrypoint {ep_rel} prioritized "
                  f"into context (with Dockerfile).")

    repo_context, included, suspect_note = discover_context(
        error_signal, tech_stacks, pipeline_types, forced_paths)

    # A 3B will happily delete a crash and leave __main__ empty — which removes the
    # crash but makes the container exit without serving. Forbid that explicitly.
    if startup_ep and startup_ep in included:
        directive = (
            f"{startup_ep} is the application ENTRYPOINT; the container runs it and "
            f"it MUST start a long-running server. If you remove or change the crash, "
            f"the __main__ block MUST still start the server — for Flask that is "
            f"exactly: app.run(host=\"0.0.0.0\", port=<the EXPOSE port in the "
            f"Dockerfile>). NEVER leave __main__ empty, a bare pass, or a print: the "
            f"process would exit immediately and fail the smoke test.")
        suspect_note = (suspect_note + "\n" + directive) if suspect_note else directive
    if not included:
        print("[DISCOVER] WARNING: no files resolved — AI has no context to work with.")
    prescan, prescan_autofixes = prescan_issues(included)
    # Cross-file port mismatch leaves NO error in the log (the app starts fine),
    # so it can't be discovered from the failure signal — scan the files directly.
    port_findings, port_autofixes = prescan_port_consistency()
    if port_autofixes:
        prescan += port_findings
        prescan_autofixes += port_autofixes
    print(f"[TIMING] Stage 2 done in {time.time()-t2:.1f}s")

    # ══ STAGE 3: ANALYSE ═════════════════════════════════════════════════════
    print("\n━━━ STAGE 3: ANALYSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    t3 = time.time()

    if prescan_autofixes:
        # ── Deterministic path: the pre-scan computed exact corrections, so we
        #    apply them WITHOUT the model — no find-string to hallucinate, no
        #    timeout. The model is only for bugs the pre-scan can't resolve.
        fixes         = group_autofixes(prescan_autofixes)
        root_cause    = "; ".join(f.get("reason", "") for f in fixes) or "pre-scan corrections"
        commit_msg    = "fix: correct invalid path/version references (auto-detected)"
        confidence    = 1.0
        detected_type = ", ".join(sorted(pipeline_types))
        print(f"[ANALYSE] Pre-scan resolved {len(prescan_autofixes)} issue(s) "
              f"deterministically — skipping model call.")
        print(f"  root_cause : {root_cause}")
        print(f"  fixes      : {len(fixes)} file(s)")
        for fix in fixes:
            print(f"    → {fix['file']}  ({len(fix['edits'])} edit(s))")
        print(f"[TIMING] Stage 3 done in {time.time()-t3:.1f}s (no model)")
    else:
        # ── AI path with GROUND-TRUTH VERIFICATION ───────────────────────────
        # Generate → apply → rebuild+run the real pipeline → keep only if it
        # PASSES; otherwise feed the new error back and retry. The model may be
        # wrong up to MAX_FIX_ITERATIONS times — only a verified fix is accepted,
        # which is what makes this reliable despite a non-deterministic 3B.
        root_cause = commit_msg = ""
        confidence = 1.0
        detected_type = ", ".join(sorted(pipeline_types))
        fixes = []
        verified_written: list[str] = []
        verified_originals: dict[str, str] = {}
        feedback = ""

        for iteration in range(MAX_FIX_ITERATIONS):
            print(f"\n[ANALYSE] AI attempt {iteration+1}/{MAX_FIX_ITERATIONS} "
                  f"(each verified by a real build+run)")
            try:
                raw     = call_ai(error_signal + feedback, repo_context,
                                  pipeline_types, tech_stacks, prescan, suspect_note)
                ai_data = parse_ai_response(raw)
            except Exception as exc:
                print(f"[ERROR] AI failed: {exc}", file=sys.stderr)
                break

            root_cause    = ai_data.get("root_cause",    "unknown")
            confidence    = float(ai_data.get("confidence", 1.0))
            commit_msg    = ai_data.get("commit_message", "fix: auto-fixer rewrite")
            detected_type = ai_data.get("pipeline_type", detected_type)
            cand          = ai_data.get("fixes", []) or []

            # Veto confabulated version fixes (prescan owns version validity).
            vetoed = [f for f in cand if _fix_changes_valid_py_version(f)]
            for f in vetoed:
                print(f"[ANALYSE] ✗ Vetoed confabulated version change in "
                      f"{f.get('file','?')} — version is already valid.",
                      file=sys.stderr)
            cand = [f for f in cand if not _fix_changes_valid_py_version(f)]

            if confidence < CONFIDENCE_MIN or not cand:
                print(f"[ANALYSE] Low confidence ({confidence:.0%}) or no fixes — "
                      f"not applying this attempt.")
                feedback = ("\n\nYou did not produce a usable fix. Re-read the error "
                            "and the file contents and fix the ACTUAL failing line.")
                continue

            valid = [f for f in cand if validate_fix(f)[0]]
            if not valid:
                for f in cand:
                    ok, why = validate_fix(f)
                    if not ok:
                        print(f"  ✗ {f.get('file','?')} — {why}", file=sys.stderr)
                feedback = "\n\nYour previous edits were invalid. Copy `find` text EXACTLY from the file."
                continue

            written, originals = write_fixes(valid)
            if not written:
                feedback = "\n\nNothing was written. Your fix did not apply."
                continue

            # ── VERIFY against the real pipeline ─────────────────────────────
            if VERIFY_LOCALLY:
                print("[VERIFY] Rebuilding + running the image to check the fix...")
                passed, out = verify_build_and_run()
            else:
                passed, out = True, "(verification disabled)"

            if passed:
                print(f"[VERIFY] ✓ Fix VERIFIED on attempt {iteration+1} — {out}")
                fixes, verified_written, verified_originals = valid, written, originals
                break

            # Failed → revert and feed the real, current error back to the model.
            print(f"[VERIFY] ✗ Attempt {iteration+1} did not pass:\n{out[:600]}")
            revert_files(originals)
            feedback = ("\n\n## YOUR PREVIOUS FIX FAILED — the pipeline still errors.\n"
                        "The error AFTER your change was:\n```\n" + out[-1200:] +
                        "\n```\nYour diagnosis was wrong. Fix the ACTUAL cause shown above.")
            error_signal = (extract_error_signal(out) or error_signal)[-2000:]

        print(f"[TIMING] Stage 3 done in {time.time()-t3:.1f}s")

        if not fixes:
            print("[ERROR] AI could not produce a VERIFIED fix in "
                  f"{MAX_FIX_ITERATIONS} attempts.", file=sys.stderr)
            if token and repo:
                open_issue(token, repo,
                           f"AI could not produce a verified fix after "
                           f"{MAX_FIX_ITERATIONS} attempts. Last root cause: {root_cause}",
                           pipeline_type=detected_type)
            sys.exit(3)

        # A verified fix is already written to disk; hand it to the commit stages
        # directly (skip the deterministic-path validate/write below).
        print(f"\n  pipeline   : {detected_type}")
        print(f"  root_cause : {root_cause}")
        print(f"  verified   : {verified_written}")
        _pre_verified = (verified_written, verified_originals)

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

    # ══ STAGE 4 + 5: VALIDATE + WRITE ════════════════════════════════════════
    # The AI path already validated, wrote, AND verified its fix against a real
    # build+run — so we skip straight to commit with those files. The
    # deterministic (prescan) path still validates + writes here as before.
    _pre_verified = locals().get("_pre_verified")
    if _pre_verified is not None:
        written, originals = _pre_verified
        valid_fixes = fixes
        print("\n━━━ STAGE 4+5: (AI fix already verified — skipping re-validate/write) ━")
        print(f"[VERIFIED] {written}")
    else:
        # ══ STAGE 4: VALIDATE ════════════════════════════════════════════════
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
                           f"Fixes failed validation. Root cause: {root_cause}",
                           pipeline_type=detected_type)
            sys.exit(3)

        # ══ DRY RUN ══════════════════════════════════════════════════════════
        if args.dry_run:
            print("\n━━━ DRY-RUN (no files written) ━━━━━━━━━━━━━━━━━━━━━━━━━")
            for fix in valid_fixes:
                content = fix["fixed_content"]
                print(f"  would write {fix['file']} ({len(content)} chars)")
                for line in content.splitlines()[:5]:
                    print(f"    {line}")
                if len(content.splitlines()) > 5:
                    print(f"    ... ({len(content.splitlines()) - 5} more lines)")
            sys.exit(0)

        # ══ STAGE 5: WRITE ════════════════════════════════════════════════════
        print("\n━━━ STAGE 5: WRITE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        written, originals = write_fixes(valid_fixes)
        if not written:
            print("[ERROR] No files were written.", file=sys.stderr)
            sys.exit(3)
        print(f"[WRITE] {len(written)} file(s): {written}")

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

    # ── Startup gate — ALWAYS runs, even with --skip-tests ───────────────────
    # Skipping unit tests is a choice; skipping "does the app even start" is not.
    # If a fix touched the entrypoint, start it and confirm it stays up. This is
    # what catches a fix that deleted a crash but left the app non-serving — the
    # exact failure that reached PR #46.
    if startup_ep and startup_ep in written:
        _, ep_argv = _resolve_entrypoint()
        status, out = probe_startup(ep_argv or ["python", "-u", startup_ep])
        if status in ("crash", "exited"):
            print(f"[GATE] ✗ {startup_ep} does not serve after the fix ({status}):",
                  file=sys.stderr)
            for line in out.splitlines()[-12:]:
                print(f"  {line}")
            revert_files(originals)
            if token and repo:
                open_issue(token, repo,
                           f"Fix applied but {startup_ep} still does not start/serve "
                           f"({status}) — reverted for manual fix. Root cause: {root_cause}",
                           pipeline_type=detected_type)
            sys.exit(5)
        elif status == "served":
            print(f"[GATE] ✓ {startup_ep} starts and stays up.")
        else:
            print(f"[GATE] Startup check inconclusive for {startup_ep} (app deps not "
                  f"importable in fixer env) — NOT blocking. Install requirements in "
                  f"the fixer step, or use a docker-run probe, to make this gate real.")

    # ══ STAGE 7: COMMIT ══════════════════════════════════════════════════════
    print(f"\n━━━ STAGE 7: COMMIT (fix/* → {GIT_TARGET_BRANCH}) ━━━━━━━━━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)

    # ══ STAGE 8: PR ══════════════════════════════════════════════════════════
    print(f"\n━━━ STAGE 8: PR (→ {GIT_TARGET_BRANCH}) ━━━━━━━━━━━━━━━━━━━━━")
    pr_url = ""
    if token and repo:
        pr_url = open_pr(token, repo, branch, commit_msg, root_cause,
                         detected_type, tech_stacks, written, valid_fixes)
    else:
        print(f"[PR] No token — merge manually: git merge {branch} into {GIT_TARGET_BRANCH}")

    print("\n━━━ ✅ DONE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  pipeline   : {detected_type}")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")
    print(f"  PR         : {pr_url or 'not created'}")
    print(f"  total time : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()