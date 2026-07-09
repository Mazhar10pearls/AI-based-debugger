#!/usr/bin/env python3
"""
Self-Healing CI/CD Auto-Fixer — RAG + staged-AI flow.

Pipeline (matches the diagram):

    Collect logs + repo context (Python)
        → Retrieve similar historical failures  ══ RAG: embed signal, top-k ══
        → AI Stage 1: extract facts
        → AI Stage 2: determine root cause (+ confidence)
        → AI Stage 3: generate fix        ══ flat issues: evidence → corrected ══
        → AI Stage 4: self-verify         ══ AI checks its own patch ══
        → Python validation (locate by evidence, syntax, apply, tests)
        → Confidence gate
        → Apply patch → Commit + Push + PR
        → Store the successful case for future retrieval

Design notes carried over from the single-call version:
  * A small local model (qwen2.5-coder:3b) diagnoses well but mangles nested
    structure. Stage 3's output is a FLAT issues list — quote the offending
    text, quote the corrected text, one entry per bug. Python pairs each quote
    with its correction and LOCATES each fix by searching for the AI's own
    quoted evidence in the shown files. The AI authors every change; Python
    never writes a fix of its own.
  * The reference-typo class (broken COPY/ADD paths) still runs through the
    focused correct-the-line sub-pass inside Stage 3 — that's the part that
    made typos reliable, so it's kept.

Why staged instead of one call:
  Four smaller tasks each stay inside the 3B's reliable instruction-following
  window better than one big task. The cost is latency (four sequential calls
  on CPU). Two escape hatches:
    FAST_MODE=1        → collapse Stage 1+2 into one call
    SKIP_SELF_VERIFY=1 → drop Stage 4 (Python validation is still the hard gate)

Why RAG:
  Priming Stage 2/3 with concrete past (root_cause, evidence→corrected) pairs
  for a similar signal measurably steadies a 3B on failures it has seen before.
  Memory lives OUTSIDE the checked-out repo so it survives across runs on a
  self-hosted runner (default ~/.ai-fixer/memory.jsonl, override AI_FIXER_HOME).
  If the embedding model is unavailable, retrieval degrades to "no context"
  and the pipeline runs exactly as the non-RAG version did.

Exit codes:
  0 success / nothing to do   2 AI failed        4 git failed
  1 log not found             3 no valid fix     5 tests failed (reverted)
"""

import argparse
import ast
import difflib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

# security layer lives beside this script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import security

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL",   "qwen2.5-coder:3b")
EMBED_MODEL    = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
AI_TIMEOUT     = 210
MAX_RETRIES    = 2
RETRY_BACKOFF  = [20, 20]
MAX_AI_ROUNDS  = 3

# staged-flow toggles
FAST_MODE        = os.environ.get("FAST_MODE", "").lower() in ("1", "true", "yes")
SKIP_SELF_VERIFY = os.environ.get("SKIP_SELF_VERIFY", "").lower() in ("1", "true", "yes")

# ── RAG memory ────────────────────────────────────────────────────────────────
AI_FIXER_HOME = Path(os.environ.get("AI_FIXER_HOME", os.path.expanduser("~/.ai-fixer")))
MEMORY_PATH   = AI_FIXER_HOME / "memory.jsonl"
RAG_TOP_K     = int(os.environ.get("RAG_TOP_K", "3"))
RAG_MIN_SIM   = float(os.environ.get("RAG_MIN_SIM", "0.72"))
RAG_MAX_CASES = int(os.environ.get("RAG_MAX_CASES", "500"))   # cap store growth
RAG_ENABLED   = os.environ.get("RAG_ENABLED", "1").lower() not in ("0", "false", "no")

# ── Prompt / context budget ───────────────────────────────────────────────────
MAX_ERROR_LINES   = 14
MAX_FILE_CHARS    = 4000
MAX_TOTAL_CONTEXT = 9000
MAX_CONTEXT_FILES = 3
MAX_FILES_FIXED   = 4

# ── Git flow ──────────────────────────────────────────────────────────────────
GIT_BASE_BRANCH   = os.environ.get("GIT_BASE_BRANCH",   "develop")
GIT_TARGET_BRANCH = os.environ.get("GIT_TARGET_BRANCH", "develop")
BOT_NAME   = "github-actions[bot]"
BOT_EMAIL  = "github-actions[bot]@users.noreply.github.com"
BOT_PREFIX = "fix:"
MAX_BOT_ATTEMPTS = 3

ALWAYS_BLOCKED   = {".git", "auto-fixer.py"}
BLOCKED_PATTERNS = [r"\.?github/workflows/auto-fix.*\.ya?ml$",
                    r"\.?github/workflows/self-heal.*\.ya?ml$"]

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "env",
             "dist", "build", ".pytest_cache", "target", "out", "vendor",
             ".idea", ".vscode", "coverage", "tmp", "temp", "logs"}
MAX_FILE_SIZE_BYTES = 100_000


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — READ LOGS + DETECT TECH STACK  (unchanged, proven)
# ══════════════════════════════════════════════════════════════════════════════

ERROR_KEYWORDS = [
    "error", "failed", "failure", "exception", "traceback", "exit code",
    "invalid", "fatal", "cannot", "refused", "denied", "missing", "undefined",
    "permission denied", "command not found", "returned non-zero", "syntaxerror",
    "importerror", "nameerror", "typeerror", "valueerror", "attributeerror",
    "modulenotfounderror", "no module named", "not found", "could not find",
    "no matching", "requirement", "assert", "test failed", "no such file",
    "failed to solve", "err!", "panic:", "fatal error",
]
NOISE_KEYWORDS = [
    "##[group]", "##[endgroup]", "set up job", "complete job", "post job",
    "add mask", "git config", "git version", "persist-credentials",
    "fetch-depth", "collecting ", "rootdir:", "configfile:",
]
FILE_REF_HINTS = [re.compile(r'File "[^"]+"'),
                  re.compile(r'[\w./\-]+\.[A-Za-z0-9]+:\d+'),
                  re.compile(r'\bDockerfile(\.\w+)?\b')]

TECH_STACK_SIGNALS = {
    "python": ["python", "pip", "pytest", "flask", "django", "requirements.txt",
               "pyproject.toml", ".py"],
    "node":   ["node", "npm", "yarn", "jest", "package.json", ".js", ".ts"],
    "docker": ["dockerfile", "docker build", "docker push", "image", "container"],
    "java":   ["java", "maven", "gradle", "mvn", ".java", "pom.xml"],
    "go":     ["go build", "go test", "go mod", ".go", "go.mod"],
}


def extract_error_signal(log_text: str) -> str:
    lines = log_text.splitlines()
    relevant = [l.strip() for l in lines
                if (any(k in l.lower() for k in ERROR_KEYWORDS)
                    or any(h.search(l) for h in FILE_REF_HINTS))
                and not any(n in l.lower() for n in NOISE_KEYWORDS) and l.strip()]
    relevant = list(dict.fromkeys(relevant))[-MAX_ERROR_LINES:]
    tail = [l.strip() for l in lines[-20:]
            if l.strip() and not any(n in l.lower() for n in NOISE_KEYWORDS)]
    signal = "\n".join(list(dict.fromkeys(relevant + tail)))
    print(f"[DETECT] Error signal: {len(signal)} chars")
    return signal


def fingerprint_stack(log_text: str) -> set:
    low = log_text.lower()
    stacks = {s for s, sig in TECH_STACK_SIGNALS.items() if any(x in low for x in sig)}
    print(f"[DETECT] Tech stacks: {stacks or {'unknown'}}")
    return stacks


def normalize_signal(signal: str) -> str:
    """
    Canonicalize an error signal for RAG: strip volatile bits (timestamps,
    line/column numbers, hex ids, temp paths) so the SAME class of failure
    embeds close together regardless of run-specific noise. Used both as the
    embedding input and as the dedup key when storing a case.
    """
    s = signal.lower()
    s = re.sub(r'\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}\S*', ' ', s)  # timestamps
    s = re.sub(r'0x[0-9a-f]+', ' ', s)                                # hex ids
    s = re.sub(r':\d+(?::\d+)?', ':', s)                              # :line:col
    s = re.sub(r'/tmp/\S+', ' ', s)                                   # temp paths
    s = re.sub(r'\b[0-9a-f]{7,40}\b', ' ', s)                         # sha-ish
    s = re.sub(r'\s+', ' ', s).strip()
    return s[:1200]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DISCOVER FILES  (unchanged, proven)
# ══════════════════════════════════════════════════════════════════════════════

REFERENCED_PATH_PATTERNS = [
    r'File "([^"]+\.\w+)"',
    r'([\w./\-]+\.[A-Za-z0-9]+):\d+(?::\d+)?',
    r'((?:[\w./\-]+/)?Dockerfile(?:\.\w+)?)\b',
    r'([\w./\-]+/requirements[\w.\-]*\.txt)',
    r'(?:in|from|at|open)\s+([\w./\-]+\.[A-Za-z0-9]+)',
]
STACK_FILE_SIGNALS = {
    "python": [".py", "requirements.txt", "pyproject.toml", "setup.py"],
    "node":   [".js", ".ts", "package.json"],
    "docker": ["Dockerfile", "docker-compose"],
    "java":   [".java", "pom.xml", "build.gradle"],
    "go":     [".go", "go.mod"],
}
HIGH_VALUE = {"dockerfile", "requirements.txt", "package.json", "pom.xml",
              "go.mod", "pyproject.toml", "setup.py", "docker-compose.yml"}


def _relstrip(rel): return rel[2:] if rel.startswith("./") else rel
def _is_blocked(fp):
    if any(fp == b or fp.startswith(b.rstrip("/") + "/") for b in ALWAYS_BLOCKED):
        return True
    return any(re.search(p, fp) for p in BLOCKED_PATTERNS)


def _is_text_file(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            return False
        return b"\x00" not in path.read_bytes()[:512]
    except Exception:
        return False


def extract_referenced_paths(log_text: str, root: Path = Path(".")) -> list:
    hits = []
    for pat in REFERENCED_PATH_PATTERNS:
        for m in re.finditer(pat, log_text):
            c = m.group(1).strip().strip("'\"")
            if c:
                hits.append(c)
    by_name = {}
    for p in root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts):
            by_name.setdefault(p.name, []).append(str(p))
    order, counts = [], {}
    def add(rel):
        rel = _relstrip(rel)
        if _is_blocked(rel):
            return
        if rel not in counts:
            order.append(rel)
        counts[rel] = counts.get(rel, 0) + 1
    for h in hits:
        hn = _relstrip(h)
        if Path(hn).is_file():
            add(hn); continue
        for rel in by_name.get(Path(h).name, []):
            add(rel)
    resolved = sorted(order, key=lambda r: -counts[r])
    if resolved:
        print(f"[DISCOVER] Referenced in log: {resolved}")
    return resolved


def _score_file(path: Path, signal: str, stacks: set) -> int:
    name, low, score = path.name.lower(), signal.lower(), 0
    if name in low or str(path).lower() in low:
        score += 50
    for st in stacks:
        for pat in STACK_FILE_SIGNALS.get(st, []):
            if pat.startswith(".") and name.endswith(pat):
                score += 20
            elif pat.lower() == name:
                score += 25
    if name in HIGH_VALUE:
        score += 15
    return score


def find_ci_workflow_files(root: Path = Path(".")) -> list:
    found = []
    wf_dir = root / ".github" / "workflows"
    if not wf_dir.is_dir():
        return found
    for p in sorted(wf_dir.glob("*.y*ml")):
        if not p.is_file():
            continue
        rel = _relstrip(str(p))
        if _is_blocked(rel):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "auto-fixer.py" in text or "auto_fixer.py" in text:
            continue
        found.append(rel)
    return found


def discover_context(signal: str, stacks: set, forced: list) -> tuple:
    parts, included, contents, total = [], [], {}, 0

    def read_whole(path, cap=MAX_FILE_CHARS):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            return raw if len(raw) <= cap else None
        except Exception:
            return None

    def try_add(rel, content):
        nonlocal total
        block = f"### {rel}\n```\n{content}\n```"
        if total + len(block) > MAX_TOTAL_CONTEXT and included:
            return False
        parts.append(block); included.append(rel); contents[rel] = content; total += len(block)
        print(f"[DISCOVER] + {rel} ({len(content)} chars)")
        return True

    for rel in forced:
        if len(included) >= MAX_CONTEXT_FILES:
            break
        if not security.PathPolicy.is_readable_into_context(rel):
            print(f"[SEC] excluding secret-bearing file from context: {rel}", file=sys.stderr)
            continue
        p = Path(rel)
        if p.is_file() and _is_text_file(p):
            c = read_whole(p, cap=8000)
            if c is not None:
                try_add(rel, c)

    if len(included) < MAX_CONTEXT_FILES:
        cand = []
        for p in Path(".").rglob("*"):
            if p.is_dir() or any(part in SKIP_DIRS for part in p.parts):
                continue
            if not _is_text_file(p):
                continue
            rel = _relstrip(str(p))
            if rel in included or _is_blocked(rel):
                continue
            if not security.PathPolicy.is_readable_into_context(rel):
                continue
            sc = _score_file(p, signal, stacks)
            if sc > 0:
                cand.append((sc, p))
        cand.sort(key=lambda x: (-x[0], len(str(x[1]))))
        for _, p in cand:
            if len(included) >= MAX_CONTEXT_FILES:
                break
            rel = _relstrip(str(p))
            c = read_whole(p)
            if c is not None:
                try_add(rel, c)

    context = "\n\n".join(parts)
    print(f"[DISCOVER] {len(included)} files, {len(context)} chars")
    return context, included, contents


# ── static reference scan (hints only) ─────────────────────────────────────────
DOCKERFILE_REF_PATTERNS = [
    (re.compile(r'^\s*COPY\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "COPY"),
    (re.compile(r'^\s*ADD\s+(?:--from=\S+\s+)?(\S+)\s+\S+', re.I | re.M), "ADD"),
    (re.compile(r'-r\s+(\S+\.txt)'), "pip install -r"),
    (re.compile(r'CMD\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "CMD"),
    (re.compile(r'ENTRYPOINT\s*\[\s*"[^"]*"\s*,\s*"([^"]+)"', re.I), "ENTRYPOINT"),
]


def scan_reference_hints(included_contents: dict) -> str:
    return _scan_reference_details(included_contents)[1]


def _scan_reference_details(included_contents: dict) -> tuple:
    all_repo = [_relstrip(str(p)) for p in Path(".").rglob("*")
                if p.is_file() and not any(d in SKIP_DIRS for d in p.parts)]
    basenames = {}
    for f in all_repo:
        basenames.setdefault(Path(f).name.lower(), []).append(f)
    all_repo_set = set(all_repo)

    broken, summary = [], []
    for rel, content in included_contents.items():
        name = Path(rel).name.lower()
        if "dockerfile" not in name and "docker-compose" not in name and "compose.y" not in name:
            continue
        docker_dir = Path(rel).parent
        for pat, label in DOCKERFILE_REF_PATTERNS:
            for m in pat.finditer(content):
                ref = m.group(1).strip().strip("'\"")
                if not ref or ref in (".", "..") or ref.startswith("-") or ref.startswith("$"):
                    continue
                ref_clean = ref.lstrip("./")
                candidates = {ref_clean, _relstrip(str(docker_dir / ref_clean))}
                if any(Path(c).is_file() or c in all_repo_set for c in candidates):
                    continue
                close = difflib.get_close_matches(Path(ref_clean).name.lower(),
                                                   basenames.keys(), n=1, cutoff=0.4)
                closest_full = basenames[close[0]][0] if close else ""
                closest_base = close[0] if close else ""
                line_start = content.rfind("\n", 0, m.start()) + 1
                line_end = content.find("\n", m.end())
                if line_end == -1:
                    line_end = len(content)
                bad_line = content[line_start:line_end]
                broken.append({"file": rel, "line": bad_line,
                               "wrong_token": ref,
                               "closest_basename": closest_base,
                               "closest_full": closest_full})
                summary.append(f"- {rel}: {label} references '{ref}' — NOT found in repo. "
                               f"Closest existing file: "
                               f"{closest_full or 'no close match in repo'}")
    hint = "\n".join(dict.fromkeys(summary))
    if hint:
        print(f"[DISCOVER] Static reference hints:\n{hint}")
    return broken, hint


# ══════════════════════════════════════════════════════════════════════════════
# RAG — embed + memory store  (new)
# ══════════════════════════════════════════════════════════════════════════════

def _embed_endpoint():
    base = re.sub(r"/(v1|api)/.*$", "", OLLAMA_API_URL.rstrip("/"))
    return f"{base}/api/embeddings", f"{base}/api/embed"


def embed_text(text: str):
    """Return an embedding vector for `text`, or None if embeddings are off/unavailable.
    Handles both the legacy /api/embeddings ({'embedding': [...]}) and the newer
    /api/embed ({'embeddings': [[...]]}) Ollama shapes."""
    if not RAG_ENABLED or not text.strip():
        return None
    legacy, modern = _embed_endpoint()
    for url, key in ((legacy, "embedding"), (modern, "embeddings")):
        try:
            r = requests.post(url, json={"model": EMBED_MODEL, "prompt": text,
                                         "input": text}, timeout=(10, 60))
            if r.status_code != 200:
                continue
            body = r.json()
            vec = body.get(key)
            if key == "embeddings" and isinstance(vec, list) and vec and isinstance(vec[0], list):
                vec = vec[0]
            if isinstance(vec, list) and vec and all(isinstance(x, (int, float)) for x in vec):
                return [float(x) for x in vec]
        except Exception:
            continue
    print("[RAG] embedding unavailable — running without retrieved context", file=sys.stderr)
    return None


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    """Append-only JSONL of solved cases with embeddings. Cosine search in pure
    Python — fine for the small corpora a single repo accumulates."""

    def __init__(self, path: Path):
        self.path = path
        self.records = []
        if path.is_file():
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
            except Exception as exc:
                print(f"[RAG] could not read memory ({exc}) — starting empty", file=sys.stderr)
        print(f"[RAG] memory: {len(self.records)} stored case(s) at {path}")

    def search(self, query_vec, k=RAG_TOP_K, threshold=RAG_MIN_SIM):
        if not query_vec:
            return []
        scored = []
        for rec in self.records:
            sim = _cosine(query_vec, rec.get("embedding", []))
            if sim >= threshold:
                scored.append((sim, rec))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def add(self, record: dict):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # dedup: same normalized signature already stored → skip
            sig = record.get("sig", "")
            if sig and any(r.get("sig") == sig for r in self.records):
                print("[RAG] case already stored — not duplicating")
                return
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.records.append(record)
            # trim oldest if we exceed the cap (rewrite tail)
            if len(self.records) > RAG_MAX_CASES:
                self.records = self.records[-RAG_MAX_CASES:]
                self.path.write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in self.records) + "\n",
                    encoding="utf-8")
            print(f"[RAG] stored case (memory now {len(self.records)})")
        except Exception as exc:
            print(f"[RAG] failed to store case: {exc}", file=sys.stderr)


def format_retrieved(matches) -> str:
    """Render top-k matches into a compact prompt block of concrete past fixes."""
    if not matches:
        return ""
    out = []
    for i, (sim, rec) in enumerate(matches, 1):
        rc = (rec.get("root_cause") or "").strip()
        line = f"{i}. (similarity {sim:.2f}) root cause: {rc[:160]}"
        for iss in (rec.get("issues") or [])[:2]:
            ev = (iss.get("evidence") or "")[:70]
            co = (iss.get("corrected") or "")[:70]
            if ev and co:
                line += f"\n   fix: {ev!r} → {co!r}"
        out.append(line)
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
# AI plumbing — shared streaming + JSON extraction  (consolidated)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_endpoint():
    url = OLLAMA_API_URL.rstrip("/")
    if "/api/generate" in url:
        return url, "native"
    if "/v1/completions" in url or "/v1/chat" in url:
        return url, "openai"
    base = re.sub(r"/(v1|api)/.*$", "", url)
    return f"{base}/api/generate", "native"


def _extract_token(line: bytes, fmt: str) -> str:
    if not line:
        return ""
    try:
        text = line.decode("utf-8", errors="replace").strip()
        if text.startswith("data: "):
            text = text[6:].strip()
        if text in ("", "[DONE]"):
            return ""
        obj = json.loads(text)
        return obj.get("choices", [{}])[0].get("text", "") if fmt == "openai" \
            else obj.get("response", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""


def _stream_ollama(prompt, schema=None, num_predict=2000, temperature=0.05,
                   num_ctx=8192, tag="AI", retries=MAX_RETRIES) -> str:
    """Single entry point for every AI stage. Streams the response, returns raw
    text. Retries on timeout only (connection errors are terminal)."""
    endpoint, fmt = _detect_endpoint()
    if fmt == "openai":
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "temperature": temperature,
                   "max_tokens": num_predict, "stream": True}
    else:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt,
                   "options": {"temperature": temperature, "num_predict": num_predict,
                               "num_ctx": num_ctx}, "stream": True}
        if schema:
            payload["format"] = schema
    print(f"[{tag}] {endpoint} ({fmt}) | prompt {len(prompt)} chars | model {OLLAMA_MODEL}")

    last = None
    for attempt in range(retries):
        try:
            t0, collected = time.time(), []
            resp = requests.post(endpoint, json=payload, timeout=(10, AI_TIMEOUT), stream=True)
            resp.raise_for_status()
            for line in resp.iter_lines():
                tok = _extract_token(line, fmt)
                if tok:
                    collected.append(tok)
            raw = "".join(collected).strip()
            print(f"[{tag}] done in {time.time()-t0:.1f}s — {len(raw)} chars")
            if not raw:
                try:
                    body = resp.json()
                    raw = (body.get("choices", [{}])[0].get("text", "")
                           if fmt == "openai" else body.get("response", "")).strip()
                except Exception:
                    pass
            if not raw:
                raise RuntimeError("Ollama returned an empty response.")
            return raw
        except requests.exceptions.Timeout as exc:
            last = exc
            wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF)-1)]
            print(f"[{tag}] timeout; waiting {wait}s...")
            if attempt < retries - 1:
                time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Cannot connect to Ollama at {endpoint}: {exc}")
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}")
    raise RuntimeError(f"Ollama did not respond after {retries} attempts: {last}")


def _json_from(raw: str):
    """Best-effort JSON extraction: direct parse, then widest brace-balanced
    object, then brace-completion."""
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
                    cand = cleaned[start:i+1]
                    best = cand if best is None or len(cand) > len(best) else best
                    break
    if best:
        try:
            return json.loads(best)
        except json.JSONDecodeError:
            try:
                return json.loads(best + "}" * (best.count("{") - best.count("}")))
            except json.JSONDecodeError:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Repo-file listing + shared prompt fragments
# ══════════════════════════════════════════════════════════════════════════════

def repo_file_list(limit: int = 200) -> str:
    files = []
    for p in sorted(Path(".").rglob("*")):
        if p.is_file() and not any(d in SKIP_DIRS for d in p.parts):
            rel = _relstrip(str(p))
            if not _is_blocked(rel):
                files.append(rel)
        if len(files) >= limit:
            break
    return ", ".join(files) if files else "(none found)"


def _context_block(signal, context, stacks, hints, retrieved):
    repo_files = repo_file_list()
    hints_section = (f"## Static reference check (verify each — not authoritative):\n{hints}\n"
                     if hints else "")
    rag_section = (f"## Similar past failures (for reference — verify against THIS repo):\n"
                   f"{retrieved}\n" if retrieved else "")
    return (
        f"## Tech stack: {', '.join(sorted(stacks)) or 'unknown'}\n"
        f"## CI failure (key lines):\n```\n{signal}\n```\n"
        f"## Repo files (these exist — anything referenced but NOT here is a typo):\n{repo_files}\n"
        f"{rag_section}{hints_section}"
        f"## File contents (you may ONLY edit these):\n{context}"
    )


def _cap_prompt(system: str, body: str, context: str, rebuild, cap=9500) -> str:
    """Trim the file-contents portion if the whole prompt is too long."""
    full = f"{system}\n\n{body}"
    if len(full) <= cap:
        return full
    allowed = cap - len(system) - len(rebuild("")) - 100
    trimmed = context[:max(allowed, 1000)] + "\n...(trimmed)"
    return f"{system}\n\n{rebuild(trimmed)}"


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — EXTRACT FACTS  (AI)
# ══════════════════════════════════════════════════════════════════════════════

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "error_type":    {"type": "string"},
        "failing_files": {"type": "array", "items": {"type": "string"}},
        "key_symbols":   {"type": "array", "items": {"type": "string"}},
        "summary":       {"type": "string"},
    },
    "required": ["error_type", "failing_files", "summary"],
}

FACTS_SYSTEM = """\
You are a CI/CD triage engineer. Extract FACTS ONLY — do not diagnose or fix yet.
Output ONLY one JSON object. No markdown fences. Start with { end with }.
{"error_type":"short category e.g. missing_file / bad_version / port_mismatch / import_error","failing_files":["exact/path from a ### header, if any"],"key_symbols":["the literal tokens the error names — filenames, versions, ports"],"summary":"one sentence of what the log shows"}
Copy tokens EXACTLY from the log and file contents. Do not invent files or symbols."""


def ai_extract_facts(signal, context, stacks, hints, retrieved) -> dict:
    def rebuild(ctx):
        return _context_block(signal, ctx, stacks, hints, retrieved) + \
               "\n\nExtract the facts as JSON."
    prompt = _cap_prompt(FACTS_SYSTEM, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, FACTS_SCHEMA, num_predict=800,
                         temperature=0.0, tag="S1-FACTS")
    data = _json_from(raw) or {}
    data.setdefault("error_type", "unknown")
    data.setdefault("failing_files", [])
    data.setdefault("key_symbols", [])
    data.setdefault("summary", "")
    print(f"[S1-FACTS] {data.get('error_type')} | files={data.get('failing_files')} "
          f"| symbols={data.get('key_symbols')}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ROOT CAUSE  (AI)
# ══════════════════════════════════════════════════════════════════════════════

CAUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "root_cause":     {"type": "string"},
        "solution":       {"type": "string"},
        "confidence":     {"type": "number"},
        "commit_message": {"type": "string"},
    },
    "required": ["root_cause", "solution", "confidence", "commit_message"],
}

CAUSE_SYSTEM = """\
You are a senior CI/CD debugging engineer. Given the extracted facts and the file
contents, state the ROOT CAUSE. Do NOT write the fix yet. Output ONLY one JSON object.
{"root_cause":"one sentence: WHY it failed","solution":"plain words: WHAT must change","confidence":0.0-1.0,"commit_message":"fix: short"}
Set confidence below 0.5 if the facts do not clearly pin a single cause."""


def ai_root_cause(facts, signal, context, stacks, hints, retrieved) -> dict:
    facts_line = (f"## Extracted facts:\nerror_type={facts.get('error_type')}; "
                  f"failing_files={facts.get('failing_files')}; "
                  f"key_symbols={facts.get('key_symbols')}; "
                  f"summary={facts.get('summary')}\n")
    def rebuild(ctx):
        return facts_line + _context_block(signal, ctx, stacks, hints, retrieved) + \
               "\n\nGive root cause as JSON."
    prompt = _cap_prompt(CAUSE_SYSTEM, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, CAUSE_SCHEMA, num_predict=700,
                         temperature=0.05, tag="S2-CAUSE")
    data = _json_from(raw) or {}
    data.setdefault("root_cause", "unknown")
    data.setdefault("solution", "")
    data.setdefault("commit_message", "fix: auto-fixer change")
    try:
        data["confidence"] = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        data["confidence"] = 0.5
    print(f"[S2-CAUSE] {data['root_cause']} (confidence {data['confidence']:.0%})")
    return data


def ai_facts_and_cause(signal, context, stacks, hints, retrieved) -> tuple:
    """FAST_MODE: one call producing facts + cause together."""
    schema = {"type": "object", "properties": {
        **FACTS_SCHEMA["properties"], **CAUSE_SCHEMA["properties"]},
        "required": ["error_type", "failing_files", "summary",
                     "root_cause", "solution", "confidence", "commit_message"]}
    system = (FACTS_SYSTEM.split("Output ONLY")[0] +
              "Extract facts AND state the root cause in one JSON object. "
              "Output ONLY one JSON object.\n"
              '{"error_type":"...","failing_files":[...],"key_symbols":[...],'
              '"summary":"...","root_cause":"...","solution":"...",'
              '"confidence":0.0-1.0,"commit_message":"fix: short"}')
    def rebuild(ctx):
        return _context_block(signal, ctx, stacks, hints, retrieved) + \
               "\n\nRespond with the combined JSON."
    prompt = _cap_prompt(system, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, schema, num_predict=1000, temperature=0.05, tag="S1S2")
    data = _json_from(raw) or {}
    facts = {"error_type": data.get("error_type", "unknown"),
             "failing_files": data.get("failing_files", []) or [],
             "key_symbols": data.get("key_symbols", []) or [],
             "summary": data.get("summary", "")}
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    cause = {"root_cause": data.get("root_cause", "unknown"),
             "solution": data.get("solution", ""),
             "confidence": conf,
             "commit_message": data.get("commit_message", "fix: auto-fixer change")}
    print(f"[S1S2] {facts['error_type']} → {cause['root_cause']} "
          f"(confidence {conf:.0%})")
    return facts, cause


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — GENERATE FIX  (AI)  — flat issues contract + focused correct-the-line
# ══════════════════════════════════════════════════════════════════════════════

FIX_SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "file":      {"type": "string"},
                "problem":   {"type": "string"},
                "evidence":  {"type": "string"},
                "corrected": {"type": "string"},
            },
            "required": ["file", "problem", "evidence", "corrected"]}},
    },
    "required": ["issues"],
}

FIX_SYSTEM = """\
You are a senior CI/CD debugging engineer. You know the root cause. Now emit the FIX.
Output ONLY one JSON object. No markdown fences. Start with { end with }.

"issues" = ONE ENTRY PER BUG. Each entry:
  - "file": the ### header path of the file the bug is in
  - "problem": one sentence describing this specific bug
  - "evidence": the offending text COPIED EXACTLY, character-for-character, from the file contents. SHORT — the single wrong token or one wrong line. Never paraphrase; copy it.
  - "corrected": the same text with ONLY the bug fixed. Everything else identical.

Schema:
{"issues":[{"file":"exact/path","problem":"...","evidence":"exact text copied from the file","corrected":"same text, bug fixed"}]}

FINDING EVERY BUG:
A file routinely contains MORE THAN ONE unrelated bug. Finding one and stopping is a FAILURE. Re-read EVERY shown file line by line. For each line naming a file, a version, or a port, check it independently:
- File names: the "## Repo files" list is the truth. A referenced name NOT in it is a typo — corrected = the closest real name. NEVER create a missing file; fix the reference.
- Python versions: valid 3.8–3.13. python:3.1 / python:3.2 / python-version "3.1"/"3.2" are INVALID — corrected uses 3.12.
- Ports: if app.run(port=N) disagrees with Dockerfile EXPOSE M, correct the app to bind M.
- If you change `if __name__ == "__main__":`, it must still start a long-running server.
Add one issues[] entry for EVERY bug — two bugs in one file = two entries with the same "file".

RULES:
- evidence must literally appear in the file contents. If you cannot quote exact offending text, omit that issue.
- evidence and corrected must differ, and be as short as unambiguous allows.
- Only files with a ### header may be fixed."""


def ai_generate_fix(facts, cause, signal, context, stacks, hints, retrieved) -> list:
    ctx_line = (f"## Root cause (already determined): {cause.get('root_cause')}\n"
                f"## Solution direction: {cause.get('solution')}\n"
                f"## Facts: error_type={facts.get('error_type')}, "
                f"symbols={facts.get('key_symbols')}\n")
    def rebuild(ctx):
        return ctx_line + _context_block(signal, ctx, stacks, hints, retrieved) + \
               "\n\nEmit the issues JSON."
    prompt = _cap_prompt(FIX_SYSTEM, rebuild(context), context, rebuild)
    raw = _stream_ollama(prompt, FIX_SCHEMA, num_predict=2800,
                         temperature=0.05, tag="S3-FIX")
    data = _json_from(raw)
    if data is None:
        raise ValueError(f"No valid JSON in Stage 3 response:\n{raw[:400]}")
    issues = data.get("issues", []) or []
    # tolerate a model that drifts to fixes[] shape
    if not issues and isinstance(data.get("fixes"), list):
        issues = data["fixes"]
    return _normalize_issue_keys(issues)


CORRECT_LINE_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id":             {"type": "integer"},
                "corrected_line": {"type": "string"},
            },
            "required": ["id", "corrected_line"]}}
    },
    "required": ["corrections"],
}


def ai_correct_lines(broken: list) -> list:
    """Focused correct-the-line pass for reference typos. Returns
    [{file, line_in_file, corrected_line}, ...]. Never edits anything itself."""
    if not broken:
        return []
    items = []
    for i, b in enumerate(broken, 1):
        items.append(f'{i}. line: "{b["line"]}"\n'
                     f'   contains the wrong filename "{b["wrong_token"]}"; '
                     f'the real file is "{b["closest_full"]}" '
                     f'(basename "{b["closest_basename"]}")')
    system = ("Correct-the-line task. Output ONLY JSON: "
              '{"corrections":[{"id":N,"corrected_line":"..."}]}. '
              "For each numbered line below, return the line with the wrong "
              "filename replaced by the real filename shown. Keep everything "
              "else on the line IDENTICAL — same indentation, directive, quoting, "
              "trailing content. Do not add or remove lines.")
    prompt = f"{system}\n\nLines to correct:\n" + "\n".join(items)
    print(f"[AI-LINES] focused call: {len(broken)} broken line(s)")

    try:
        raw = _stream_ollama(prompt, CORRECT_LINE_SCHEMA, num_predict=1200,
                             temperature=0.0, num_ctx=4096, tag="AI-LINES", retries=1)
    except Exception as exc:
        print(f"[AI-LINES] failed: {exc}", file=sys.stderr)
        return []
    data = _json_from(raw)
    if not data:
        return []

    out = []
    for c in data.get("corrections", []) or []:
        try:
            idx = int(c.get("id", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(broken)):
            continue
        b = broken[idx]
        new_line = c.get("corrected_line", "")
        if not isinstance(new_line, str) or not new_line.strip() or b["line"] == new_line:
            continue
        if b["closest_basename"] and b["closest_basename"] not in new_line:
            print(f"[AI-LINES] discarded id={idx+1}: missing real filename")
            continue
        if b["wrong_token"] in new_line:
            print(f"[AI-LINES] discarded id={idx+1}: still contains wrong token")
            continue
        out.append({"file": b["file"], "line_in_file": b["line"],
                    "corrected_line": new_line})
    return out


def _normalize_issue_keys(issues):
    KF = ("file_path", "path", "filename", "filepath", "name")
    KE = ("find", "wrong", "offending", "original", "bad", "before")
    KC = ("replace", "fix", "replacement", "correct", "fixed", "after")
    out = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        it = dict(it)
        for want, alts in (("file", KF), ("evidence", KE), ("corrected", KC)):
            if want not in it:
                for a in alts:
                    if a in it:
                        it[want] = it.pop(a); break
        out.append(it)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SELF-VERIFY  (AI)
# ══════════════════════════════════════════════════════════════════════════════

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id":     {"type": "integer"},
                "keep":   {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "required": ["id", "keep"]}},
        "confidence": {"type": "number"},
    },
    "required": ["verdicts"],
}

VERIFY_SYSTEM = """\
You are reviewing proposed fixes before they are applied. For each numbered issue, decide keep=true only if:
  - the "evidence" text actually appears in the shown file contents, AND
  - "corrected" is a genuine fix of a real bug (not a no-op, not a new bug).
If evidence is not present, or the change looks wrong or invented, keep=false.
Output ONLY one JSON object:
{"verdicts":[{"id":N,"keep":true,"reason":"short"}],"confidence":0.0-1.0}"""


def ai_self_verify(issues, context) -> tuple:
    """Return (kept_issues, verify_confidence). On any failure, keep all (Python
    validation remains the hard gate)."""
    if SKIP_SELF_VERIFY or not issues:
        return issues, None
    listing = []
    for i, it in enumerate(issues, 1):
        listing.append(f'{i}. file="{it.get("file","")}"\n'
                       f'   evidence: {(it.get("evidence") or "")[:160]!r}\n'
                       f'   corrected: {(it.get("corrected") or "")[:160]!r}')
    prompt = (f"{VERIFY_SYSTEM}\n\n## File contents:\n{context}\n\n"
              f"## Proposed fixes:\n" + "\n".join(listing) +
              "\n\nReturn the verdicts JSON.")
    if len(prompt) > 11000:
        prompt = prompt[:11000]
    try:
        raw = _stream_ollama(prompt, VERIFY_SCHEMA, num_predict=900,
                             temperature=0.0, tag="S4-VERIFY", retries=1)
    except Exception as exc:
        print(f"[S4-VERIFY] failed ({exc}) — keeping all issues", file=sys.stderr)
        return issues, None
    data = _json_from(raw)
    if not data:
        return issues, None
    drop = set()
    for v in data.get("verdicts", []) or []:
        try:
            idx = int(v.get("id", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(issues) and v.get("keep") is False:
            drop.add(idx)
            print(f"[S4-VERIFY] dropped #{idx+1}: {v.get('reason','')[:80]}")
    kept = [it for i, it in enumerate(issues) if i not in drop]
    try:
        vc = float(data.get("confidence")) if data.get("confidence") is not None else None
    except (TypeError, ValueError):
        vc = None
    if not kept and issues:
        # verifier rejected everything — distrust the verifier, not Stage 3
        print("[S4-VERIFY] verifier rejected all issues — keeping Stage 3 output "
              "for Python validation to judge")
        return issues, vc
    return kept, vc


# ══════════════════════════════════════════════════════════════════════════════
# PAIR + LOCATE  (Python turns the AI's quotes into fixes)  — unchanged logic
# ══════════════════════════════════════════════════════════════════════════════

def issues_to_fixes(issues: list, included_contents: dict) -> tuple:
    fixes_by_file, rejects = {}, []
    for n, it in enumerate(issues, 1):
        file = (it.get("file") or "").strip()
        ev   = it.get("evidence")
        cor  = it.get("corrected")
        prob = (it.get("problem") or "").strip()

        if not isinstance(ev, str) or not ev.strip():
            rejects.append(f"issue #{n} ({file or '?'}): empty evidence")
            continue
        if not isinstance(cor, str) or cor == ev:
            rejects.append(f"issue #{n} ({file or '?'}): corrected missing or identical")
            continue

        holders = [f for f, c in included_contents.items() if ev in c]
        if file in included_contents and ev in included_contents[file]:
            target = file
        elif len(holders) == 1:
            target = holders[0]
            print(f"[LOCATE] issue #{n}: filed under '{file or '?'}' but evidence only "
                  f"exists in '{target}' — relocating.")
        elif len(holders) > 1:
            rejects.append(f"issue #{n} ({file or '?'}): evidence in {len(holders)} files — ambiguous")
            continue
        else:
            rejects.append(f"issue #{n} ({file or '?'}): evidence {ev[:80]!r} not found — rejected")
            continue

        entry = fixes_by_file.setdefault(
            target, {"file": target, "reason": prob or "AI fix", "edits": []})
        edit = {"find": ev, "replace": cor}
        if edit not in entry["edits"]:
            entry["edits"].append(edit)
            if prob and prob not in entry["reason"]:
                entry["reason"] = (entry["reason"] + "; " + prob).lstrip("; ")[:300]

    return list(fixes_by_file.values()), rejects


# ══════════════════════════════════════════════════════════════════════════════
# APPLY + VALIDATE  (unchanged, proven)
# ══════════════════════════════════════════════════════════════════════════════

def _salvage_fragment(content: str, find: str, replace: str):
    if not find or not replace or find == replace:
        return None
    i = 0
    while i < len(find) and i < len(replace) and find[i] == replace[i]:
        i += 1
    j = 0
    while (j < len(find) - i and j < len(replace) - i
           and find[-1 - j] == replace[-1 - j]):
        j += 1
    old_frag, new_frag = find[i:len(find) - j], replace[i:len(replace) - j]
    if len(old_frag.strip()) < 3 or content.count(old_frag) != 1:
        return None
    return content.replace(old_frag, new_frag, 1)


def _apply_edits(original: str, edits: list) -> tuple:
    content = original
    for i, ed in enumerate(edits):
        find, repl = ed.get("find", ""), ed.get("replace", "")
        if not isinstance(find, str) or find == "":
            return None, f"edit #{i+1} empty 'find'"
        if find in content:
            content = content.replace(find, repl)
            continue
        nf = "\n".join(l.strip() for l in find.splitlines())
        nc = "\n".join(l.strip() for l in content.splitlines())
        matched = False
        if nf and nf in nc:
            fl = [l.strip() for l in find.splitlines()]
            cl = content.splitlines()
            for j in range(len(cl) - len(fl) + 1):
                if [c.strip() for c in cl[j:j+len(fl)]] == fl:
                    base = cl[j][:len(cl[j]) - len(cl[j].lstrip())]
                    cl[j:j+len(fl)] = [(base + r if r.strip() else r)
                                       for r in (repl.splitlines() or [""])]
                    content = "\n".join(cl)
                    if original.endswith("\n") and not content.endswith("\n"):
                        content += "\n"
                    matched = True
                    break
        if matched:
            continue
        salvaged = _salvage_fragment(content, find, repl)
        if salvaged is not None and salvaged != content:
            print(f"[EDIT] Salvaged edit #{i+1} via unique-fragment match")
            content = salvaged
            continue
        return None, (f"edit #{i+1} 'find' text not present — find: {find[:120]!r}")
    if content == original:
        return None, "edits produced no change"
    return content, "ok"


def _resolve_content(fix: dict) -> tuple:
    file = (fix.get("file") or "").strip()
    if fix.get("edits"):
        p = Path(file)
        if not p.is_file():
            return None, f"file does not exist: {file}"
        return _apply_edits(p.read_text(encoding="utf-8", errors="replace"), fix["edits"])
    fc = fix.get("fixed_content")
    if isinstance(fc, str) and fc.strip():
        return fc, "ok"
    return None, "no 'edits' or 'fixed_content'"


def validate_fix(fix: dict) -> tuple:
    file = (fix.get("file") or "").strip()
    if not file:
        return False, "missing 'file'"
    if ".." in file or file.startswith("/") or file.startswith("~"):
        return False, f"unsafe path: {file}"
    if _is_blocked(file):
        return False, f"blocked path: {file}"
    if not Path(file).exists():
        return False, f"file does not exist: {file}"
    content, reason = _resolve_content(fix)
    if content is None:
        return False, reason
    fix["fixed_content"] = content
    if not content.strip():
        return False, "empty result"
    if file.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:
            return False, f"Python syntax error: {e}"
    elif re.search(r"\.ya?ml$", file):
        try:
            if not isinstance(yaml.safe_load(content), (dict, list)):
                return False, "YAML did not parse"
        except yaml.YAMLError as e:
            return False, f"YAML error: {e}"
    elif file.endswith(".json"):
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return False, f"JSON error: {e}"
    return True, "ok"


def write_fixes(fixes: list) -> tuple:
    written, originals, reasons = [], {}, []
    for fix in fixes[:MAX_FILES_FIXED]:
        ok, reason = validate_fix(fix)
        file = fix.get("file", "").strip()
        if not ok:
            print(f"  ✗ {file or '?'} — {reason}")
            print(f"  ✗ {file or '?'} — {reason}", file=sys.stderr)
            reasons.append(f"{file or '?'}: {reason}")
            continue
        originals[file] = Path(file).read_text(encoding="utf-8", errors="replace")
        content = fix["fixed_content"]
        if not content.endswith("\n"):
            content += "\n"
        p = Path(file)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
        print(f"  ✓ {file} — {fix.get('reason','')[:100]}")
        written.append(file)
    written = list(dict.fromkeys(written))
    return written, originals, reasons


def revert_files(originals: dict):
    for file, content in originals.items():
        try:
            Path(file).write_text(content, encoding="utf-8")
            print(f"[REVERT] {file}")
        except Exception as exc:
            print(f"[REVERT] failed {file}: {exc}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# RUN TESTS  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def detect_test_commands(stacks: set) -> list:
    cmds = []
    if "python" in stacks:
        cmds.append(["python", "-m", "pytest", "--tb=short", "-q"])
    if "node" in stacks and Path("package.json").exists():
        try:
            if "test" in json.loads(Path("package.json").read_text()).get("scripts", {}):
                cmds.append(["npm", "test", "--", "--passWithNoTests"])
        except Exception:
            pass
    if "go" in stacks and Path("go.mod").exists():
        cmds.append(["go", "test", "./..."])
    return cmds


def run_tests(stacks: set) -> bool:
    cmds = detect_test_commands(stacks)
    if not cmds:
        print("[TEST] No test runner detected — skipping")
        return True
    ok_all = True
    safe_env = security.harden_test_env()
    preexec = security.resource_limits_preexec()
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180,
                               env=safe_env, preexec_fn=preexec)
            for line in (r.stdout + r.stderr).splitlines()[-15:]:
                print(f"  {line}")
            passed = r.returncode == 0
            print(f"[TEST] {' '.join(cmd[:2])}: {'✓ Passed' if passed else '✗ Failed'}")
            ok_all = ok_all and passed
        except FileNotFoundError:
            print(f"[TEST] {cmd[0]} not found — skipping")
        except subprocess.TimeoutExpired:
            print(f"[TEST] {cmd[0]} timed out — skipping")
    return ok_all


# ══════════════════════════════════════════════════════════════════════════════
# COMMIT + PUSH + PR  /  CREATE ISSUE  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _git(*args, check=True):
    return subprocess.run(["git", *args], check=check, capture_output=True, text=True)


def last_commit_was_bot() -> bool:
    try:
        author, subject = _git("log", "-1", "--pretty=%an|||%s").stdout.strip().split("|||", 1)
        if author == BOT_NAME and subject.startswith(BOT_PREFIX):
            print("[GUARD] Last commit was bot — stopping.")
            return True
    except Exception:
        pass
    return False


def count_recent_bot_commits(n=10) -> int:
    try:
        lines = _git("log", f"-{n}", "--pretty=%an").stdout.splitlines()
        count = 0
        for author in lines:
            if author.strip() == BOT_NAME:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def _ensure_base_branch() -> bool:
    _git("fetch", "origin", check=False)
    if _git("ls-remote", "--heads", "origin", GIT_BASE_BRANCH, check=False).stdout.strip():
        if _git("rev-parse", "--verify", GIT_BASE_BRANCH, check=False).returncode != 0:
            _git("checkout", "-b", GIT_BASE_BRANCH, f"origin/{GIT_BASE_BRANCH}")
        return True
    try:
        ref = "main" if _git("rev-parse", "--verify", "main", check=False).returncode == 0 else "master"
        _git("checkout", "-b", GIT_BASE_BRANCH, ref)
        _git("push", "-u", "origin", GIT_BASE_BRANCH)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[GIT] cannot create {GIT_BASE_BRANCH}: {exc.stderr.strip()}", file=sys.stderr)
        return False


def commit_to_branch(commit_msg: str, written: list) -> str:
    try:
        _git("config", "user.name", BOT_NAME)
        _git("config", "user.email", BOT_EMAIL)
        if not _ensure_base_branch():
            return ""
        _git("checkout", GIT_BASE_BRANCH)
        _git("pull", "origin", GIT_BASE_BRANCH, check=False)
        branch = f"fix/{int(time.time())}"
        _git("checkout", "-b", branch)
        if written:
            _git("add", "--", *written)
        else:
            _git("add", "-u")
        if _git("diff", "--cached", "--quiet", check=False).returncode == 0:
            print("[COMMIT] Nothing to commit.")
            _git("checkout", GIT_BASE_BRANCH, check=False)
            return ""
        # ── outbound secret gate: never commit a secret the fix introduced ──
        clean, leaks = security.scan_staged_secrets(".")
        if not clean:
            detail = "; ".join(f"{l['file']} [{','.join(l['kinds'])}]" for l in leaks)
            print(f"[SEC] BLOCKING COMMIT — staged changes contain secret-like "
                  f"content: {detail}", file=sys.stderr)
            security.audit({"phase": "secret_commit_blocked", "findings": leaks})
            _git("reset", check=False)                       # unstage everything
            _git("checkout", GIT_BASE_BRANCH, check=False)
            return ""
        _git("commit", "-m", commit_msg)
        _git("push", "-u", "origin", branch)
        print(f"[GIT] Pushed {branch}")
        _git("checkout", GIT_BASE_BRANCH, check=False)
        return branch
    except subprocess.CalledProcessError as exc:
        print(f"[GIT] {exc.stderr.strip()}", file=sys.stderr)
        _git("checkout", GIT_BASE_BRANCH, check=False)
        return ""


def _gh(token):
    return {"Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
            error_analysis="", solution="") -> str:
    details = "".join(f"\n**`{f.get('file','?')}`** — {f.get('reason','')}\n" for f in fixes)
    diag = ""
    if error_analysis or solution:
        diag = (f"### AI diagnosis\n**Error:** {error_analysis or 'n/a'}\n\n"
                f"**Solution:** {solution or 'n/a'}\n\n")
    body = (f"## 🤖 AI Auto-Fix\n\n{diag}"
            f"**Root cause:** {root_cause}\n\n"
            f"**Files changed:** {', '.join(f'`{f}`' for f in written)}\n\n"
            f"## What changed{details}\n\n> Auto-generated — review before merge.")
    r = requests.post(f"https://api.github.com/repos/{repo}/pulls",
                      json={"title": f"🤖 {commit_msg}", "head": branch,
                            "base": GIT_TARGET_BRANCH, "body": body},
                      headers=_gh(token), timeout=30)
    if r.status_code in (200, 201):
        url = r.json().get("html_url", "")
        print(f"[PR] {url}")
        return url
    print(f"[PR] failed {r.status_code}: {r.text[:200]}", file=sys.stderr)
    return ""


def pending_bot_pr(token, repo) -> str:
    try:
        r = requests.get(f"https://api.github.com/repos/{repo}/pulls?state=open&per_page=50",
                         headers=_gh(token), timeout=15)
        if r.status_code == 200:
            for pr in r.json():
                if (pr.get("head", {}).get("ref", "").startswith("fix/")
                        and "🤖" in (pr.get("title") or "")):
                    return pr.get("html_url", "")
    except Exception:
        pass
    return ""


def open_issue(token, repo, reason, run_url=""):
    r = requests.post(f"https://api.github.com/repos/{repo}/issues",
                      json={"title": "🚨 Auto-fixer could not fix CI — manual review",
                            "body": f"**Reason:** {reason}\n\n**Failed run:** {run_url}",
                            "labels": ["bug", "needs-manual-fix"]},
                      headers=_gh(token), timeout=30)
    if r.status_code in (200, 201):
        print(f"[ISSUE] {r.json().get('html_url','')}")
    else:
        print(f"[ISSUE] failed {r.status_code}: {r.text[:200]}", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="AI CI/CD auto-fixer (RAG + staged)")
    ap.add_argument("--input", required=True, help="Path to CI failure log")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_PAT", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "")
    run_url = (os.environ.get("GITHUB_SERVER_URL", "") + "/" + repo + "/actions") if repo else ""

    log_path = Path(args.input)
    if not log_path.is_file():
        print(f"[ERROR] Log not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ── security: kill switch, trust boundary, token hygiene ──
    if security.kill_switch():
        sys.exit(0)
    trust = security.assess_trust()
    print(f"[SEC] origin trust: {trust['level']} ({trust['reason']})")
    for w in security.check_token_scopes(token):
        print(f"[SEC] token warning: {w}", file=sys.stderr)
    security.audit({"phase": "start", "repo": repo, "trust": trust["level"],
                    "trust_reason": trust["reason"], "input": args.input})

    # loop guards
    if last_commit_was_bot():
        sys.exit(0)
    if count_recent_bot_commits() >= MAX_BOT_ATTEMPTS:
        print("[GUARD] Too many bot attempts — escalating.")
        if token and repo:
            open_issue(token, repo, "Auto-fixer attempted too many fixes without success.", run_url)
        sys.exit(0)
    if token and repo:
        url = pending_bot_pr(token, repo)
        if url:
            print(f"[GUARD] A fix PR is already open: {url} — skipping model call.")
            sys.exit(0)

    # ── COLLECT LOGS + CONTEXT ──
    print("\n━━━ COLLECT LOGS + CONTEXT ━━━")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    signal = extract_error_signal(log_text)
    stacks = fingerprint_stack(log_text)
    if not signal.strip():
        print("[DETECT] No error signal — nothing to fix.")
        sys.exit(0)

    forced = extract_referenced_paths(log_text)
    if not forced:
        wf = find_ci_workflow_files()
        if wf:
            print(f"[DISCOVER] No source file referenced — falling back to CI workflow: {wf}")
            forced = wf
    context, included, included_contents = discover_context(signal, stacks, forced)
    if not included:
        print("[DISCOVER] WARNING: no files resolved.")

    # ── security: redact secrets before ANYTHING leaves for the model / store ──
    signal, sig_findings = security.redact_secrets(signal)
    included_contents, ctx_findings = security.scrub_context(included_contents)
    included = [f for f in included if f in included_contents]
    # rebuild the prompt context from the redacted contents (same block format)
    context = "\n\n".join(f"### {rel}\n```\n{c}\n```" for rel, c in included_contents.items())
    if sig_findings or ctx_findings:
        security.audit({"phase": "redaction",
                        "signal_secrets": sorted(set(sig_findings)),
                        "context_secrets": sorted(set(ctx_findings))})

    broken_lines, hints = _scan_reference_details(included_contents)

    # ── RAG RETRIEVE ──
    print("\n━━━ RAG RETRIEVE ━━━")
    norm_sig = normalize_signal(signal)
    query_vec = embed_text(norm_sig)
    store = MemoryStore(MEMORY_PATH) if RAG_ENABLED else None
    matches = store.search(query_vec) if store else []
    retrieved = format_retrieved(matches)
    if retrieved:
        print(f"[RAG] {len(matches)} similar case(s) retrieved")

    # ── AI STAGE 1 + 2 ──
    print("\n━━━ AI STAGE 1: FACTS / STAGE 2: ROOT CAUSE ━━━")
    try:
        if FAST_MODE:
            facts, cause = ai_facts_and_cause(signal, context, stacks, hints, retrieved)
        else:
            facts = ai_extract_facts(signal, context, stacks, hints, retrieved)
            cause = ai_root_cause(facts, signal, context, stacks, hints, retrieved)
    except Exception as exc:
        print(f"[ERROR] AI stage 1/2 failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI diagnosis failed: {exc}", run_url)
        sys.exit(2)

    root_cause = cause["root_cause"]
    solution   = cause["solution"]
    commit_msg = cause["commit_message"]
    confidence = cause["confidence"]

    print("\n  ── diagnosis ──")
    print(f"  CAUSE      : {root_cause}")
    print(f"  SOLUTION   : {solution or '(none)'}")
    print(f"  confidence : {confidence:.0%}")

    # ── AI STAGE 3a: focused correct-the-line for reference typos ──
    prefilled_fixes = []
    if broken_lines:
        print("\n━━━ AI STAGE 3a: CORRECT-THE-LINE ━━━")
        corrections = ai_correct_lines(broken_lines)
        by_file = {}
        for c in corrections:
            entry = by_file.setdefault(c["file"], {"file": c["file"],
                     "reason": "AI-corrected reference typo(s)", "edits": []})
            edit = {"find": c["line_in_file"], "replace": c["corrected_line"]}
            if edit not in entry["edits"]:
                entry["edits"].append(edit)
        prefilled_fixes = list(by_file.values())
        print(f"[AI-LINES] {sum(len(f['edits']) for f in prefilled_fixes)} corrected line(s)")

    # ── AI STAGE 3b: freeform diagnosis fix ──
    print("\n━━━ AI STAGE 3: GENERATE FIX ━━━")
    try:
        issues = ai_generate_fix(facts, cause, signal, context, stacks, hints, retrieved)
    except Exception as exc:
        print(f"[ERROR] AI stage 3 failed: {exc}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo, f"AI fix generation failed: {exc}", run_url)
        sys.exit(2)

    print(f"  issues reported: {len(issues)}")
    for n, it in enumerate(issues, 1):
        print(f"    {n}. {it.get('file','?')}: {it.get('problem','')[:80]}")
        print(f"       {(it.get('evidence') or '')[:60]!r} → {(it.get('corrected') or '')[:60]!r}")

    # ── AI STAGE 4: self-verify ──
    verify_conf = None
    if issues and not SKIP_SELF_VERIFY:
        print("\n━━━ AI STAGE 4: SELF-VERIFY ━━━")
        issues, verify_conf = ai_self_verify(issues, context)
        print(f"  issues after verify: {len(issues)}"
              + (f" | verify confidence {verify_conf:.0%}" if verify_conf is not None else ""))

    # ── CONFIDENCE GATE ──
    combined = confidence if verify_conf is None else (confidence + verify_conf) / 2
    if combined < 0.5:
        print(f"[GATE] Confidence {combined:.0%} too low — escalating instead of guessing.")
        if token and repo:
            open_issue(token, repo,
                       f"AI confidence too low ({combined:.0%}). Root cause: {root_cause}", run_url)
        sys.exit(0)

    # ── PAIR + LOCATE, merge with prefilled ──
    freeform_fixes, pair_rejects = issues_to_fixes(issues, included_contents)
    for rej in pair_rejects:
        print(f"  ✗ {rej}", file=sys.stderr)

    merged = {}
    for f in prefilled_fixes + freeform_fixes:
        entry = merged.setdefault(f["file"], {"file": f["file"],
                 "reason": f.get("reason", ""), "edits": []})
        for e in f.get("edits", []) or []:
            if e not in entry["edits"]:
                entry["edits"].append(e)
        if f.get("reason") and f["reason"] not in entry["reason"]:
            entry["reason"] = (entry["reason"] + "; " + f["reason"]).lstrip("; ")[:300]
    fixes = list(merged.values())

    if not fixes:
        detail = "; ".join(pair_rejects) or "model reported no locatable issues"
        print(f"[ERROR] AI produced no usable fixes. {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI produced no usable fixes. Root cause: {root_cause}\n\nDetail: {detail}",
                       run_url)
        sys.exit(3)

    # ── SECURITY VETTING: path review + supply-chain, before writing anything ──
    print("\n━━━ SECURITY VETTING ━━━")
    vetted, held = [], []
    for fix in fixes:
        file = (fix.get("file") or "").strip()
        ok, reason = validate_fix(fix)                 # populates fix['fixed_content']
        if not ok:
            held.append((file, f"invalid ({reason})")); continue
        try:
            original = Path(file).read_text(encoding="utf-8", errors="replace")
        except Exception:
            original = ""
        sc_verdict, sc_reason = security.vet_supplychain(fix, original)
        if sc_verdict == "reject":
            held.append((file, f"supply-chain reject — {sc_reason}")); continue
        if sc_verdict == "review":
            held.append((file, f"needs review — {sc_reason}")); continue
        needs_review, rr = security.PathPolicy.review_reason(file)
        if needs_review:
            held.append((file, f"needs review — {rr}")); continue
        vetted.append(fix)

    for file, why in held:
        print(f"  ⚠ held: {file} — {why}", file=sys.stderr)
    security.audit({"phase": "vetting",
                    "vetted": [f.get("file") for f in vetted],
                    "held": [{"file": f, "why": w} for f, w in held]})

    if not vetted:
        detail = "; ".join(f"{f}: {w}" for f, w in held) or "all fixes held"
        print("[SEC] no fix passed vetting — escalating to humans.", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI proposed fixes but security policy withheld all of them.\n\n"
                       f"Root cause: {root_cause}\n\nHeld: {detail}", run_url)
        sys.exit(0)
    if held:
        print(f"[SEC] applying {len(vetted)} vetted fix(es); {len(held)} held for review.")
    fixes = vetted

    # ── APPLY + VALIDATE ──
    print("\n━━━ APPLY + VALIDATE ━━━")
    if args.dry_run:
        for fix in fixes:
            ok, reason = validate_fix(fix)
            print(f"  {'would write' if ok else 'reject'} {fix.get('file','?')} — {reason}")
        sys.exit(0)

    written, originals, reject_reasons = write_fixes(fixes)
    if not written:
        detail = "; ".join(reject_reasons) or "no detail captured"
        print(f"[ERROR] No valid fix applied. {detail}", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"AI fix failed validation. Root cause: {root_cause}\n\nDetail: {detail}",
                       run_url)
        sys.exit(3)

    # ── ADDITIONAL ROUNDS (static rescan) ──
    prev_hints = None
    for round_no in range(2, MAX_AI_ROUNDS + 1):
        _, r_included, r_contents = discover_context(signal, stacks, forced)
        r_broken, r_hints = _scan_reference_details(r_contents)
        if not r_hints:
            print(f"[LOOP] Static scan clean after round {round_no-1}.")
            break
        if r_hints == prev_hints:
            print(f"[LOOP] Round {round_no-1} left the same issue(s) — stopping.")
            break
        prev_hints = r_hints
        print(f"\n━━━ AI ROUND {round_no}/{MAX_AI_ROUNDS}: CORRECT-THE-LINE ━━━")
        corrections = ai_correct_lines(r_broken)
        by_file = {}
        for c in corrections:
            entry = by_file.setdefault(c["file"], {"file": c["file"],
                     "reason": "AI-corrected reference typo(s)", "edits": []})
            edit = {"find": c["line_in_file"], "replace": c["corrected_line"]}
            if edit not in entry["edits"]:
                entry["edits"].append(edit)
        fixes_r = list(by_file.values())
        if not fixes_r:
            print(f"[LOOP] Nothing more to apply — stopping.")
            break
        written_r, originals_r, _ = write_fixes(fixes_r)
        if not written_r:
            print(f"[LOOP] Round {round_no} produced no valid fix — stopping.")
            break
        for f in written_r:
            originals.setdefault(f, originals_r[f])
        written = list(dict.fromkeys(written + written_r))
        fixes = fixes + fixes_r

    # ── RUN TESTS (trust-gated; untrusted code is never executed here) ──
    print("\n━━━ RUN TESTS ━━━")
    if not args.skip_tests and security.should_run_tests(trust):
        if not run_tests(stacks):
            revert_files(originals)
            if token and repo:
                open_issue(token, repo,
                           f"Fix applied but tests failed — reverted. Root cause: {root_cause}",
                           run_url)
            sys.exit(5)
    else:
        print("[TEST] Skipped (--skip-tests or untrusted origin)")

    # ── COMMIT + PR (untrusted origins never auto-push) ──
    if not security.should_auto_pr(trust):
        print(f"[SEC] untrusted origin ({trust['reason']}) — withholding auto-PR.")
        revert_files(originals)
        if token and repo:
            open_issue(token, repo,
                       f"Fix computed for an UNTRUSTED origin ({trust['reason']}). "
                       f"Auto-PR withheld pending human review.\n\n"
                       f"Proposed files: {', '.join(written)}\nRoot cause: {root_cause}",
                       run_url)
        security.audit({"phase": "withheld_untrusted", "files": written})
        sys.exit(0)

    print("\n━━━ COMMIT + PR ━━━")
    branch = commit_to_branch(commit_msg, written)
    if not branch:
        sys.exit(4)
    if token and repo:
        open_pr(token, repo, branch, commit_msg, root_cause, written, fixes,
                facts.get("summary", ""), solution)
        security.audit({"phase": "pr_opened", "branch": branch, "files": written})
    else:
        print(f"[PR] No token — merge {branch} manually.")

    # ── STORE SUCCESSFUL CASE (RAG write-back) ──
    # Poisoning guard: only trusted origins may write to the shared memory, so a
    # fork PR can't seed the store with content that primes future prompts.
    # Signal + evidence were already redacted upstream, so nothing secret lands here.
    if store is not None and query_vec is not None and trust.get("level") == "trusted":
        print("\n━━━ STORE CASE IN RAG MEMORY ━━━")
        stored_issues = [{"file": it.get("file", ""),
                          "problem": security.redact_secrets(it.get("problem") or "")[0][:200],
                          "evidence": security.redact_secrets(it.get("evidence") or "")[0][:200],
                          "corrected": security.redact_secrets(it.get("corrected") or "")[0][:200]}
                         for it in issues if it.get("evidence") and it.get("corrected")]
        store.add({
            "sig": norm_sig,
            "embedding": query_vec,
            "stacks": sorted(stacks),
            "error_type": facts.get("error_type", ""),
            "root_cause": root_cause,
            "solution": solution,
            "issues": stored_issues,
            "files_changed": written,
            "confidence": combined,
            "ts": int(time.time()),
        })
        security.secure_file(MEMORY_PATH)
    elif store is not None and trust.get("level") != "trusted":
        print("[SEC] not storing case — origin not trusted (RAG poisoning guard).")

    security.audit({"phase": "done", "root_cause": root_cause, "files": written})

    print("\n━━━ ✅ DONE ━━━")
    print(f"  root cause : {root_cause}")
    print(f"  fixed      : {', '.join(written)}")
    print(f"  branch     : {branch} → {GIT_TARGET_BRANCH}")


if __name__ == "__main__":
    main()