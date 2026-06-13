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
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/v1/completions")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:3b")

# ── Timeout strategy ──────────────────────────────────────────────────────────
# Workflow timeout-minutes = 30. Budget breakdown:
#   ~2 min  setup/checkout/logs
#   ~8 min  Ollama (2 attempts × 210s + 20s backoff)
#   ~2 min  git + PR
#   ──────────────────────────────
#   ~12 min expected | 30 min ceiling
AI_TIMEOUT    = 210   # seconds per attempt — fits 2 retries in 30-min workflow
MAX_RETRIES   = 2     # fail fast, open issue rather than burning budget
RETRY_BACKOFF = [20, 20]

# ── Prompt budget ─────────────────────────────────────────────────────────────
# qwen2.5-coder:3b on CPU: inference time ≈ 1s per 10 output tokens.
# 2000 max_tokens output = ~200s at that rate — fits in 210s timeout.
# Prompt size drives prefill time: each 1000 chars ≈ 5-10s prefill on CPU.
# Hard ceiling: total prompt <= 7000 chars (~1750 tokens prefill).
MAX_ERROR_LINES   = 15    # keep signal tight
MAX_FILE_CHARS    = 1800  # per file — enough for Dockerfiles and small configs
MAX_TOTAL_CONTEXT = 4500  # total repo context — primary speed knob
MAX_FILES_FIXED   = 3     # 3B model handles 1-3 file fixes reliably

# ── Safety ────────────────────────────────────────────────────────────────────
CONFIDENCE_MIN   = 0.4
MAX_BOT_ATTEMPTS = 3

BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"

# ── Git flow config ───────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")

# ── Paths the AI must never touch ─────────────────────────────────────────────
ALWAYS_BLOCKED = {".git", "auto-fixer.py", "self-healer.py"}
BLOCKED_PATTERNS = [
    r"\.github/workflows/auto-fix.*\.ya?ml$",
    r"\.github/workflows/self-heal.*\.ya?ml$",
]

# ── Files to exclude from context discovery (not blocked from editing, just noisy) ──
CONTEXT_EXCLUDE_PATTERNS = [
    r"\.github/workflows/auto-fix.*\.ya?ml$",   # this workflow itself — not relevant
    r"\.github/workflows/self-heal.*\.ya?ml$",
    r"workflow-watcher\.py$",                    # monitoring scripts, not pipeline code
    r"github-monitor\.py$",
    r"ci-platform-poller\.py$",
    r"quickstart\.py$",
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
# STAGE 2 — DISCOVER
# ══════════════════════════════════════════════════════════════════════════════

def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False


def _read_smart(path: Path) -> str:
    """
    Read file content intelligently:
    - For small files: full content
    - For large files: head + tail to preserve structure
    Capped at MAX_FILE_CHARS for prompt budget control.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if len(raw) <= MAX_FILE_CHARS:
            return raw
        # Keep head + tail so AI sees both top-level declarations and bottom content
        half = MAX_FILE_CHARS // 2
        head = raw[:half]
        tail = raw[-half:]
        omitted = len(raw) - MAX_FILE_CHARS
        return head + f"\n... ({omitted} chars omitted) ...\n" + tail
    except Exception as exc:
        return f"(unreadable: {exc})"


def _score_file(path: Path, error_signal: str,
                tech_stacks: set[str], pipeline_types: set[str]) -> int:
    name    = path.name.lower()
    rel     = str(path).lower()
    err_low = error_signal.lower()
    score   = 0

    if any(str(path).startswith(d) for d in WORKFLOW_DIRS) \
            and path.suffix in WORKFLOW_EXTENSIONS:
        score += 30

    if path.name.lower() in err_low or str(path).lower() in err_low:
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
        # Skip files that are never useful context (monitoring scripts, this workflow)
        rel_str = str(path)
        if any(re.search(p, rel_str) for p in CONTEXT_EXCLUDE_PATTERNS):
            print(f"[DISCOVER] Excluded (noise filter): {rel_str}")
            continue
        score = _score_file(path, error_signal, tech_stacks, pipeline_types)
        if score > 0:
            candidates.append((score, path))

    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))

    parts, included, total = [], [], 0

    for score, path in candidates:
        rel     = str(path)
        content = _read_smart(path)
        block   = f"### {rel}\n```\n{content}\n```"

        if total + len(block) > MAX_TOTAL_CONTEXT:
            print(f"[DISCOVER] Budget reached — skipping {rel} (score={score})")
            continue

        parts.append(block)
        included.append(rel)
        total += len(block)
        print(f"[DISCOVER] Included {rel} (score={score}, {len(content)} chars)")

    context = "\n\n".join(parts)
    print(f"[DISCOVER] {len(included)} files, {len(context)} chars total")
    return context, included


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

# Concise system prompt — 3B models follow short, direct instructions better
# than long elaborate ones. Every extra sentence costs inference time AND
# reduces compliance accuracy.
SYSTEM_PROMPT = """\
You are a CI/CD repair agent. Output ONLY valid JSON. No markdown fences. No explanation. Start with { end with }.

JSON schema:
{"pipeline_type":"string","root_cause":"one sentence","confidence":0.0-1.0,"commit_message":"fix: short description","fixes":[{"file":"exact/path","reason":"what changed","fixed_content":"COMPLETE file — every line preserved except the broken one"}]}

RULES:
- fixed_content = the COMPLETE corrected file. Never a snippet or diff.
- file = exact path from the ### header.
- For Dockerfiles: keep ALL existing RUN/COPY/EXPOSE/CMD/WORKDIR/HEALTHCHECK lines. Only fix the broken instruction.
- Only include files that actually need changes.
- confidence < 0.4 means you are unsure — set it low rather than guess."""

USER_PROMPT = """\
## Pipeline: {pipeline_types} | Stack: {tech_stacks}

## CI failure (key lines):
```
{error_signal}
```

## Repo files (read carefully — your fixed_content must be based on these):
{repo_context}"""


def build_prompt(error_signal: str, repo_context: str,
                 pipeline_types: set, tech_stacks: set) -> str:
    user = USER_PROMPT.format(
        pipeline_types=", ".join(sorted(pipeline_types)) or "unknown",
        tech_stacks=", ".join(sorted(tech_stacks)) or "unknown",
        error_signal=error_signal,
        repo_context=repo_context,
    )
    # Hard cap: total prompt <= 7000 chars (~1750 tokens).
    # Beyond this, prefill time on CPU exceeds 30s and instruction-following degrades.
    PROMPT_HARD_CAP = 7000
    full = SYSTEM_PROMPT + "\n\n" + user
    if len(full) > PROMPT_HARD_CAP:
        overhead = len(USER_PROMPT.format(
            pipeline_types="", tech_stacks="", error_signal=error_signal, repo_context=""))
        allowed  = PROMPT_HARD_CAP - len(SYSTEM_PROMPT) - overhead - 50
        trimmed  = repo_context[:max(allowed, 1000)] + "\n...(trimmed for token budget)"
        user = USER_PROMPT.format(
            pipeline_types=", ".join(sorted(pipeline_types)) or "unknown",
            tech_stacks=", ".join(sorted(tech_stacks)) or "unknown",
            error_signal=error_signal,
            repo_context=trimmed,
        )
        print(f"[AI] Prompt hard-trimmed to {len(SYSTEM_PROMPT + user)} chars")
    return user


def _detect_ollama_endpoint() -> tuple[str, str]:
    """
    Auto-detect which Ollama API format the endpoint uses.
    Returns (url, format) where format is 'openai' or 'native'.

    Ollama exposes two APIs:
      /v1/completions  — OpenAI-compatible. Streaming chunks look like:
                         data: {"choices":[{"text":"..."}]}
                         (SSE format — lines prefixed with "data: ")
      /api/generate    — Native Ollama. Streaming chunks look like:
                         {"response":"...","done":false}

    The previous code used /v1/completions but parsed it as native JSON lines
    without stripping the "data: " SSE prefix → got 0 chars from stream.
    """
    url = OLLAMA_API_URL.rstrip("/")
    if "/api/generate" in url:
        return url, "native"
    if "/v1/completions" in url or "/v1/chat" in url:
        return url, "openai"
    # Fallback: probe the base
    base = re.sub(r"/(v1|api)/.*$", "", url)
    return f"{base}/api/generate", "native"


def _extract_token(line: bytes, fmt: str) -> str:
    """Extract the text token from one streamed line, handling both API formats."""
    if not line:
        return ""
    try:
        text = line.decode("utf-8", errors="replace").strip()
        # OpenAI SSE format: lines start with "data: "
        if text.startswith("data: "):
            text = text[6:].strip()
        if text in ("", "[DONE]"):
            return ""
        obj = json.loads(text)
        if fmt == "openai":
            # /v1/completions streaming chunk
            return obj.get("choices", [{}])[0].get("text", "")
        else:
            # /api/generate native streaming chunk
            return obj.get("response", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def call_ai(error_signal: str, repo_context: str,
            pipeline_types: set, tech_stacks: set) -> str:
    """
    Call Ollama with auto-detected endpoint format and robust streaming.

    Key fixes vs previous version:
    1. Auto-detects /v1/completions (OpenAI SSE) vs /api/generate (native)
       and strips the "data: " prefix that was silently dropping all tokens.
    2. Falls back to reading full response body if stream yields 0 chars
       (handles Ollama instances that ignore stream=true).
    3. Heartbeat prints every 15s keep GitHub Actions from treating the
       job as hung.
    """
    user_prompt  = build_prompt(error_signal, repo_context, pipeline_types, tech_stacks)
    full_prompt  = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
    endpoint, fmt = _detect_ollama_endpoint()

    # Build payload for whichever API format
    if fmt == "openai":
        payload = {
            "model":       OLLAMA_MODEL,
            "prompt":      full_prompt,
            "temperature": 0.05,
            "max_tokens":  2000,
            "stream":      True,
        }
    else:  # native /api/generate
        payload = {
            "model":   OLLAMA_MODEL,
            "prompt":  full_prompt,
            "options": {"temperature": 0.05, "num_predict": 2000},
            "stream":  True,
        }

    print(f"[AI] Endpoint : {endpoint} (format: {fmt})")
    print(f"[AI] Prompt   : {len(full_prompt)} chars (~{len(full_prompt)//4} tokens)")
    print(f"[AI] Model    : {OLLAMA_MODEL} | timeout: {AI_TIMEOUT}s | max_tokens: 2000")

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"[AI] Attempt {attempt+1}/{MAX_RETRIES} — streaming from {fmt} endpoint...")
            t_start    = time.time()
            last_print = t_start
            collected  = []

            resp = requests.post(
                endpoint,
                json=payload,
                timeout=(10, AI_TIMEOUT),  # (connect_timeout, read_timeout)
                stream=True,
            )
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                token = _extract_token(raw_line, fmt)
                if token:
                    collected.append(token)

                now = time.time()
                if now - last_print > 15:
                    elapsed = int(now - t_start)
                    chars_so_far = sum(len(t) for t in collected)
                    print(f"[AI] ...generating ({elapsed}s | {chars_so_far} chars collected)")
                    last_print = now

            raw = "".join(collected).strip()
            elapsed = time.time() - t_start
            print(f"[AI] Done in {elapsed:.1f}s — {len(raw)} chars received")

            # If streaming yielded nothing, try reading body as a single
            # non-streamed response (some Ollama builds ignore stream=true)
            if not raw:
                print("[AI] Stream yielded 0 chars — attempting batch response parse...")
                try:
                    body = resp.json()
                    if fmt == "openai":
                        raw = body.get("choices", [{}])[0].get("text", "").strip()
                    else:
                        raw = body.get("response", "").strip()
                    print(f"[AI] Batch fallback: {len(raw)} chars")
                except Exception:
                    pass

            if not raw:
                raise RuntimeError(
                    "Ollama returned an empty response. "
                    "Check: model is loaded (`ollama ps`), endpoint URL is correct, "
                    "and Ollama process is healthy."
                )

            print(f"[AI] Preview: {raw[:400]}{'...' if len(raw) > 400 else ''}")
            return raw

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            elapsed  = time.time() - t_start
            wait     = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
            print(f"[AI] Timeout after {elapsed:.0f}s on attempt {attempt+1}. "
                  f"Waiting {wait}s before retry...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {endpoint}. "
                f"Is Ollama running on the self-hosted runner? Error: {exc}"
            )

        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}")

    raise RuntimeError(
        f"Ollama did not respond after {MAX_RETRIES} attempts ({AI_TIMEOUT}s each). "
        f"Last error: {last_exc}. "
        f"Tips: reduce MAX_TOTAL_CONTEXT further, use qwen2.5-coder:1.5b, "
        f"or increase workflow timeout-minutes."
    )


def _normalize_fix_keys(fixes: list) -> list:
    """Normalize field names from non-compliant model responses."""
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
            return False, (
                f"Dockerfile fix truncated: only {len(non_empty)} instruction(s). "
                f"Must output the COMPLETE Dockerfile."
            )
        try:
            original = original_path.read_text(encoding="utf-8", errors="replace")
            for keyword in ["WORKDIR", "COPY", "RUN", "CMD", "EXPOSE"]:
                if keyword in original and keyword not in content:
                    return False, (
                        f"Dockerfile fix dropped '{keyword}' — instruction missing. "
                        f"Must preserve all existing instructions."
                    )
        except Exception:
            pass

    return True, "ok"


def parse_ai_response(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    data = None

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if data is None:
        # Find the largest valid JSON object in the response
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
        unmapped = [
            f.get("file", "?") for f in data["fixes"]
            if "file" not in f or "fixed_content" not in f
        ]
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


def commit_to_branch(commit_msg: str) -> str:
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
        _git("add", "-A")

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
    parser = argparse.ArgumentParser(
        description="Generic self-healing CI/CD auto-fixer"
    )
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
          f"Context budget: {MAX_TOTAL_CONTEXT} chars | Hard prompt cap: 7000 chars")
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
    repo_context, included = discover_context(error_signal, tech_stacks, pipeline_types)
    print(f"[TIMING] Stage 2 done in {time.time()-t2:.1f}s")

    # ══ STAGE 3: ANALYSE ═════════════════════════════════════════════════════
    print("\n━━━ STAGE 3: ANALYSE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    t3 = time.time()
    try:
        raw     = call_ai(error_signal, repo_context, pipeline_types, tech_stacks)
        ai_data = parse_ai_response(raw)
    except Exception as exc:
        print(f"[ERROR] AI failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI analysis failed: {exc}",
                       pipeline_type=", ".join(pipeline_types))
        sys.exit(2)
    print(f"[TIMING] Stage 3 done in {time.time()-t3:.1f}s")

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
        content_len = len(fix.get("fixed_content", ""))
        print(f"    → {fix.get('file','?')}  ({content_len} chars)")
        print(f"      {fix.get('reason','')[:120]}")
        if content_len < 50:
            print(f"      ⚠ WARNING: fixed_content very short ({content_len} chars) "
                  f"— likely truncated")

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
            content = fix["fixed_content"]
            print(f"  would write {fix['file']} ({len(content)} chars)")
            for line in content.splitlines()[:5]:
                print(f"    {line}")
            if len(content.splitlines()) > 5:
                print(f"    ... ({len(content.splitlines()) - 5} more lines)")
        sys.exit(0)

    # ══ STAGE 5: WRITE ═══════════════════════════════════════════════════════
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

    # ══ STAGE 7: COMMIT ══════════════════════════════════════════════════════
    print(f"\n━━━ STAGE 7: COMMIT (fix/* → {GIT_TARGET_BRANCH}) ━━━━━━━━━━━")
    branch = commit_to_branch(commit_msg)
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


if __name__ == "__main__":
    main()