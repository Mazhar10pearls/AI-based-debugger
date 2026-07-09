#!/usr/bin/env python3
"""
security.py — enterprise controls for the AI CI/CD auto-fixer.

This module is the single place where the fixer's trust boundaries live. It is
deliberately import-only (no side effects at import time) and fail-closed: when
a check cannot complete (network down, malformed input), it returns the SAFE
answer (redact / needs-review / don't-run), never the permissive one.

What each control does — and, honestly, does NOT do:

  redact_secrets / scrub_context
      Regex-based scrubbing of well-known secret shapes before any text reaches
      the model, the logs, a PR body, or the RAG store. This REDUCES leakage; it
      is NOT a guarantee. For a hard gate, run gitleaks or trufflehog over the
      checkout first and keep Ollama strictly local. Regexes miss novel formats.

  PathPolicy
      Keeps secret-bearing files (.env, *.pem, .aws/…) out of the model's context
      entirely, and marks security-sensitive files (workflows, CODEOWNERS,
      dependency manifests, auth/…) as review-required so the bot never silently
      rewrites them.

  assess_trust
      Reads the GitHub event payload to tell an internal push from a fork PR.
      This is only meaningful if the WORKFLOW itself is configured not to hand
      secrets to untrusted triggers — the code cannot fix a misconfigured trigger.

  vet_supplychain
      Refuses to hand-edit lockfiles, refuses to auto-ADD dependencies, and
      verifies changed pip/npm package+version against the real registry so a
      hallucinated or typosquatted name is rejected. It CANNOT tell a real-but-
      malicious package from a benign one — pair with a private registry, an
      allowlist, and OSV/Dependabot scanning for actual supply-chain assurance.

  harden_test_env / should_run_tests
      Strips secrets from the test subprocess environment and refuses to execute
      tests from untrusted origins. Python resource limits and env-stripping are
      NOT a sandbox — real isolation needs a container/VM with no secret mounts
      and controlled network egress.

  audit
      Append-only, redacted JSONL trail of every decision, file-mode 0600.
"""

import json
import math
import os
import re
import stat
import sys
import time
from pathlib import Path

import requests

# ── toggles (all default to the safe setting) ─────────────────────────────────
REDACT_ENABLED        = os.environ.get("SEC_REDACT", "1").lower() not in ("0", "false", "no")
ENTROPY_SCAN          = os.environ.get("SEC_ENTROPY", "").lower() in ("1", "true", "yes")
ALLOW_UNTRUSTED_TESTS = os.environ.get("SEC_ALLOW_UNTRUSTED_TESTS", "").lower() in ("1", "true", "yes")
SUPPLYCHAIN_VERIFY    = os.environ.get("SEC_SUPPLYCHAIN", "1").lower() not in ("0", "false", "no")
REGISTRY_TIMEOUT      = int(os.environ.get("SEC_REGISTRY_TIMEOUT", "10"))
AUDIT_PATH            = Path(os.environ.get("SEC_AUDIT_PATH",
                            os.path.expanduser("~/.ai-fixer/audit.jsonl")))
KILL_SWITCH_ENV       = "AI_FIXER_DISABLED"


# ══════════════════════════════════════════════════════════════════════════════
# 1. SECRET REDACTION
# ══════════════════════════════════════════════════════════════════════════════
# Ordered most-specific → most-generic. Each entry redacts either the whole
# match or a named/numbered group. Patterns are the widely-used shapes from
# gitleaks/trufflehog rule sets — high precision, low false-positive.

_PLACEHOLDER_VALUES = {
    "changeme", "example", "test", "password", "secret", "your_password",
    "your-password", "xxxx", "xxxxxxxx", "none", "null", "redacted",
}

# value looks templated / referenced, not a literal secret
_TEMPLATED = re.compile(r'^\s*(\$|\{\{|<|%\(|os\.environ|process\.env)')


def _mask(kind: str) -> str:
    return f"«REDACTED:{kind}»"


_FULL_MATCH_RULES = [
    ("PRIVATE_KEY", re.compile(
        r'-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----'
        r'.*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----',
        re.DOTALL)),
    ("AWS_ACCESS_KEY", re.compile(r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b')),
    ("GITHUB_TOKEN", re.compile(r'\bgh[pousr]_[A-Za-z0-9]{36,}\b')),
    ("GITHUB_PAT_FINEGRAINED", re.compile(r'\bgithub_pat_[A-Za-z0-9_]{22,}\b')),
    ("SLACK_TOKEN", re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{10,}\b')),
    ("GOOGLE_API_KEY", re.compile(r'\bAIza[0-9A-Za-z\-_]{35}\b')),
    ("STRIPE_KEY", re.compile(r'\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}\b')),
    ("JWT", re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b')),
    ("SENDGRID_KEY", re.compile(r'\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b')),
]

# group-2 = the secret value; group-1 = the key name (kept for context)
_ASSIGNMENT_RULE = re.compile(
    r'(?i)\b(pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key|'
    r'secret[_-]?key|client[_-]?secret|auth[_-]?token|private[_-]?key)\b'
    r'\s*[:=]\s*'
    r'(["\']?)([^\s"\'#]{6,})\2')

# credentials embedded in a connection URL
_CONNSTRING_RULE = re.compile(
    r'(?i)\b([a-z][a-z0-9+.\-]*)://([^:/\s@]+):([^@/\s]+)@')


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {c: s.count(c) for c in set(s)}
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def redact_secrets(text: str):
    """Return (scrubbed_text, findings). `findings` is a list of the secret
    *kinds* found (never the values). Safe to call on any string."""
    if not REDACT_ENABLED or not text:
        return text, []
    findings = []
    out = text

    for kind, rule in _FULL_MATCH_RULES:
        def repl(m, k=kind):
            findings.append(k)
            return _mask(k)
        out = rule.sub(repl, out)

    def assign_repl_safe(m):
        key, quote, val = m.group(1), m.group(2), m.group(3)
        if val.lower() in _PLACEHOLDER_VALUES or _TEMPLATED.match(val) or set(val) <= {"*"}:
            return m.group(0)
        findings.append("ASSIGNED_SECRET")
        prefix = m.group(0)[:m.start(3) - m.start(0)]
        return f"{prefix}{_mask('SECRET')}{quote}"
    out = _ASSIGNMENT_RULE.sub(assign_repl_safe, out)

    def conn_repl(m):
        findings.append("CONNSTRING_CREDS")
        return f"{m.group(1)}://{_mask('CREDS')}@"
    out = _CONNSTRING_RULE.sub(conn_repl, out)

    if ENTROPY_SCAN:
        def ent_repl(m):
            tok = m.group(0)
            if len(tok) >= 20 and _shannon_entropy(tok) >= 4.0 and _mask("") not in tok:
                findings.append("HIGH_ENTROPY")
                return _mask("HIGH_ENTROPY")
            return tok
        out = re.sub(r'\b[A-Za-z0-9+/_\-]{20,}\b', ent_repl, out)

    return out, findings


def scrub_context(included_contents: dict):
    """Redact every file body destined for the model/store. Returns
    (scrubbed_contents, total_findings)."""
    scrubbed, total = {}, []
    for rel, content in included_contents.items():
        s, f = redact_secrets(content)
        scrubbed[rel] = s
        total.extend(f)
    if total:
        print(f"[SEC] redacted {len(total)} secret-like value(s) from context: "
              f"{sorted(set(total))}", file=sys.stderr)
    return scrubbed, total


# ══════════════════════════════════════════════════════════════════════════════
# 2. PATH POLICY
# ══════════════════════════════════════════════════════════════════════════════

# Never read these into the model's context — they hold credentials by nature.
_SECRET_FILE_PATTERNS = [
    re.compile(r'(\.env$|(^|/)\.env\.)', re.I),
    re.compile(r'\.pem$', re.I), re.compile(r'\.key$', re.I),
    re.compile(r'\.p12$', re.I), re.compile(r'\.pfx$', re.I),
    re.compile(r'(^|/)id_(rsa|dsa|ecdsa|ed25519)$'),
    re.compile(r'(^|/)\.ssh/'), re.compile(r'(^|/)\.aws/'),
    re.compile(r'(^|/)\.npmrc$'), re.compile(r'(^|/)\.pypirc$'),
    re.compile(r'(^|/)\.git-credentials$'),
    re.compile(r'(^|/)(secrets?|credentials?)([._-][\w]+)?\.(ya?ml|json|txt|env)$', re.I),
    re.compile(r'\.tfstate$'), re.compile(r'(^|/)terraform\.tfvars$'),
    re.compile(r'(^|/)\.htpasswd$'),
]

# May be *proposed* as a fix, but must go to human review, never silent apply.
_REVIEW_REQUIRED_PATTERNS = [
    (re.compile(r'(^|/)\.github/'),                 "CI/workflow definition"),
    (re.compile(r'(^|/)CODEOWNERS$'),               "code-ownership policy"),
    (re.compile(r'(^|/)Dockerfile'),                "container base/build (FROM, RUN)"),
    (re.compile(r'(^|/)docker-compose\.ya?ml$'),    "service composition"),
    (re.compile(r'(^|/)(auth|security|iam|policies?)/'), "security-sensitive code path"),
    (re.compile(r'\.(tf|tfvars)$'),                 "infrastructure-as-code"),
    (re.compile(r'(^|/)requirements[\w.-]*\.txt$'), "python dependency manifest"),
    (re.compile(r'(^|/)package\.json$'),            "npm dependency manifest"),
    (re.compile(r'(^|/)pyproject\.toml$'),          "python project/deps"),
    (re.compile(r'(^|/)go\.mod$'),                  "go module manifest"),
    (re.compile(r'(^|/)pom\.xml$'),                 "maven dependency manifest"),
    (re.compile(r'(^|/)build\.gradle$'),            "gradle dependency manifest"),
]

# Dependency manifests — every one is supply-chain sensitive.
_DEP_MANIFEST_PATTERNS = [
    re.compile(r'(^|/)requirements[\w.-]*\.txt$'),
    re.compile(r'(^|/)package\.json$'),
    re.compile(r'(^|/)pyproject\.toml$'),
    re.compile(r'(^|/)setup\.py$'),
    re.compile(r'(^|/)Pipfile$'),
    re.compile(r'(^|/)go\.mod$'),
    re.compile(r'(^|/)pom\.xml$'),
    re.compile(r'(^|/)build\.gradle$'),
    re.compile(r'(^|/)Cargo\.toml$'),
]

# Never editable at all (regenerated by tooling, not hand-edited by an LLM).
_LOCKFILE_PATTERNS = [
    re.compile(r'(^|/)package-lock\.json$'),
    re.compile(r'(^|/)yarn\.lock$'),
    re.compile(r'(^|/)pnpm-lock\.ya?ml$'),
    re.compile(r'(^|/)poetry\.lock$'),
    re.compile(r'(^|/)Pipfile\.lock$'),
    re.compile(r'(^|/)go\.sum$'),
    re.compile(r'(^|/)Cargo\.lock$'),
]


class PathPolicy:
    @staticmethod
    def is_secret_file(rel: str) -> bool:
        return any(p.search(rel) for p in _SECRET_FILE_PATTERNS)

    @staticmethod
    def is_lockfile(rel: str) -> bool:
        return any(p.search(rel) for p in _LOCKFILE_PATTERNS)

    @staticmethod
    def is_dependency_manifest(rel: str) -> bool:
        return any(p.search(rel) for p in _DEP_MANIFEST_PATTERNS)

    @staticmethod
    def is_readable_into_context(rel: str) -> bool:
        """False for files that must never enter a prompt."""
        return not PathPolicy.is_secret_file(rel)

    @staticmethod
    def review_reason(rel: str):
        """(needs_review, reason). Lockfiles are handled by supply-chain vetting,
        which rejects them outright; here they also count as review-required."""
        if PathPolicy.is_lockfile(rel):
            return True, "lockfile — regenerate with tooling, do not hand-edit"
        for pat, reason in _REVIEW_REQUIRED_PATTERNS:
            if pat.search(rel):
                return True, reason
        return False, ""


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRUST BOUNDARY
# ══════════════════════════════════════════════════════════════════════════════

def assess_trust() -> dict:
    """Classify the run origin from the GitHub event payload.
    Returns {level: 'trusted'|'untrusted'|'unknown', reason, head_repo}.
    Fail-safe: anything ambiguous is 'untrusted'."""
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    base_repo = os.environ.get("GITHUB_REPOSITORY", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")

    payload = {}
    if event_path and Path(event_path).is_file():
        try:
            payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    # workflow_run carries the triggering run's head repository
    run = payload.get("workflow_run") or {}
    head = (run.get("head_repository") or payload.get("pull_request", {})
            .get("head", {}).get("repo") or {})
    head_full = head.get("full_name", "")
    is_fork = bool(head.get("fork")) or (head_full and base_repo and head_full != base_repo)

    if not payload:
        return {"level": "unknown", "reason": "no event payload available",
                "head_repo": head_full}
    if is_fork:
        return {"level": "untrusted", "reason": f"fork origin: {head_full or 'unknown'}",
                "head_repo": head_full}
    return {"level": "trusted", "reason": f"same-repo origin: {head_full or base_repo}",
            "head_repo": head_full or base_repo}


def should_run_tests(trust: dict) -> bool:
    if trust.get("level") == "trusted":
        return True
    if ALLOW_UNTRUSTED_TESTS:
        print("[SEC] running tests for UNTRUSTED origin because "
              "SEC_ALLOW_UNTRUSTED_TESTS=1 — ensure the runner is isolated.",
              file=sys.stderr)
        return True
    print(f"[SEC] skipping test execution: origin is {trust.get('level')} "
          f"({trust.get('reason')}). Set SEC_ALLOW_UNTRUSTED_TESTS=1 only on an "
          f"isolated runner to override.", file=sys.stderr)
    return False


def should_auto_pr(trust: dict) -> bool:
    """Untrusted origins get an issue for humans, not an auto-opened PR."""
    return trust.get("level") == "trusted"


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUPPLY-CHAIN VETTING
# ══════════════════════════════════════════════════════════════════════════════

_PIP_SPEC = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==|>=|<=|~=|!=|>|<)?\s*'
                       r'([A-Za-z0-9][A-Za-z0-9._+!-]*)?')
_NPM_DEP  = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')


def _pip_specs(text: str) -> dict:
    """Parse a requirements-style blob → {package_lower: version_or_None}."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _PIP_SPEC.match(line)
        if m and m.group(1):
            out[m.group(1).lower()] = m.group(3) if m.group(2) == "==" else None
    return out


def _npm_deps(text: str) -> dict:
    """Extract dependency specs from package.json text → {name: range}."""
    out = {}
    try:
        data = json.loads(text)
        for key in ("dependencies", "devDependencies", "optionalDependencies"):
            for name, ver in (data.get(key) or {}).items():
                out[name] = ver
    except Exception:
        for m in _NPM_DEP.finditer(text):
            out[m.group(1)] = m.group(2)
    return out


def _pypi_exists(name: str, version):
    """(exists, verified). Fail-closed: network error → (None, False)."""
    try:
        url = f"https://pypi.org/pypi/{name}/json"
        r = requests.get(url, timeout=REGISTRY_TIMEOUT)
        if r.status_code == 404:
            return False, True
        if r.status_code != 200:
            return None, False
        if version:
            return version in (r.json().get("releases") or {}), True
        return True, True
    except Exception:
        return None, False


def _npm_exists(name: str, version):
    try:
        url = f"https://registry.npmjs.org/{name}"
        r = requests.get(url, timeout=REGISTRY_TIMEOUT)
        if r.status_code == 404:
            return False, True
        if r.status_code != 200:
            return None, False
        if version and re.match(r'^\d', str(version)):   # exact-ish version only
            return version in (r.json().get("versions") or {}), True
        return True, True
    except Exception:
        return None, False


def vet_supplychain(fix: dict, original_content: str):
    """
    Vet a proposed fix that touches a dependency manifest / lockfile.

    Returns (verdict, reason) where verdict is one of:
      "ok"      — not dependency-related, or all changed deps verified to exist
      "review"  — dependency change that must go to a human (added dep, or a
                  manifest type we can't verify against a registry)
      "reject"  — lockfile hand-edit, or a package/version that does NOT exist
                  on the registry (hallucinated or typosquatted)
    Fail-closed: if the registry can't be reached, returns "review", never "ok".
    """
    rel = (fix.get("file") or "").strip()

    if PathPolicy.is_lockfile(rel):
        return "reject", "lockfile edited by AI — lockfiles must be regenerated by tooling"

    name = Path(rel).name.lower()
    is_pip = bool(re.match(r'requirements[\w.-]*\.txt$', name))
    is_npm = name == "package.json"

    if not (is_pip or is_npm):
        # go.mod / pom.xml / pyproject / gradle / cargo: real deps, but no
        # allowlisted registry to verify against here → force human review.
        if PathPolicy.is_dependency_manifest(rel):
            return "review", (f"{Path(rel).name} is a dependency manifest with no "
                              f"registry check available here — verify deps manually")
        return "ok", ""

    if not SUPPLYCHAIN_VERIFY:
        return "review", "supply-chain verification disabled — manual review required"

    new_content = fix.get("fixed_content")
    if not isinstance(new_content, str):
        return "review", "cannot read resulting manifest content"

    if is_pip:
        before, after = _pip_specs(original_content), _pip_specs(new_content)
        checker, ecosystem = _pypi_exists, "PyPI"
    else:
        before, after = _npm_deps(original_content), _npm_deps(new_content)
        checker, ecosystem = _npm_exists, "npm"

    added = {k: v for k, v in after.items() if k not in before}
    changed = {k: v for k, v in after.items() if k in before and before[k] != v}

    if added:
        # An auto-fixer adding a brand-new dependency is a classic injection path.
        return "review", (f"fix ADDS new {ecosystem} dependencies "
                          f"{sorted(added)} — new deps require human approval")

    problems = []
    for pkg, ver in changed.items():
        exists, verified = checker(pkg, ver)
        if exists is False:
            problems.append(f"{pkg}{('=='+str(ver)) if ver else ''} not found on {ecosystem}")
        elif not verified:
            return "review", f"could not verify {pkg} on {ecosystem} (registry unreachable)"
    if problems:
        return "reject", "; ".join(problems)

    if changed:
        return "review", (f"verified {ecosystem} version change(s) "
                          f"{sorted(changed)} — approve the bump before merge")
    return "ok", ""


# ══════════════════════════════════════════════════════════════════════════════
# 5. HARDENED TEST EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

_SECRET_ENV_PATTERNS = [
    re.compile(r'(?i)(token|secret|password|passwd|api[_-]?key|access[_-]?key)'),
    re.compile(r'^AWS_'), re.compile(r'^AZURE_'), re.compile(r'^GCP_'),
    re.compile(r'^GITHUB_TOKEN$'), re.compile(r'^GH_PAT$'), re.compile(r'^GH_TOKEN$'),
    re.compile(r'^NPM_TOKEN$'), re.compile(r'^PYPI_'), re.compile(r'^DOCKER_'),
    re.compile(r'^SSH_'), re.compile(r'(?i)_KEY$'), re.compile(r'(?i)CREDENTIAL'),
]


def harden_test_env() -> dict:
    """A minimal environment for the test subprocess with secrets stripped out,
    so attacker-modified test code cannot read them. Keeps only what a test
    runner needs to function."""
    keep_exact = {"PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR",
                  "PYTHONPATH", "VIRTUAL_ENV", "NODE_PATH", "CI",
                  "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED"}
    env = {}
    for k, v in os.environ.items():
        if k in keep_exact:
            env[k] = v
            continue
        if any(p.search(k) for p in _SECRET_ENV_PATTERNS):
            continue          # drop anything secret-shaped
        # keep other non-secret vars (locale, tooling config) but not GITHUB_*
        if k.startswith("GITHUB_") or k.startswith("RUNNER_"):
            continue
        env[k] = v
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def resource_limits_preexec():
    """preexec_fn for subprocess: cap CPU seconds and address space on POSIX.
    NOT a security sandbox — a defense-in-depth guard against runaway/greedy
    test processes. Returns None (usable directly) on non-POSIX."""
    try:
        import resource
    except ImportError:
        return None

    def _apply():
        cpu = int(os.environ.get("SEC_TEST_CPU_SECONDS", "120"))
        mem_mb = int(os.environ.get("SEC_TEST_MEM_MB", "2048"))
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))
            resource.setrlimit(resource.RLIMIT_AS,
                               (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        except (ValueError, OSError):
            pass
    return _apply


# ══════════════════════════════════════════════════════════════════════════════
# 6. AUDIT + PERMISSIONS + KILL SWITCH + TOKEN SCOPE
# ══════════════════════════════════════════════════════════════════════════════

def secure_file(path: Path):
    """chmod 0600 a sensitive file (memory/audit), best-effort, POSIX only."""
    try:
        Path(path).chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass


def audit(event: dict):
    """Append a redacted event to the audit trail (JSONL, mode 0600)."""
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        safe = {}
        for k, v in event.items():
            if isinstance(v, str):
                safe[k] = redact_secrets(v)[0]
            else:
                safe[k] = v
        safe["ts"] = int(time.time())
        with AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(safe, ensure_ascii=False) + "\n")
        secure_file(AUDIT_PATH)
    except Exception as exc:
        print(f"[SEC] audit write failed: {exc}", file=sys.stderr)


def kill_switch() -> bool:
    if os.environ.get(KILL_SWITCH_ENV, "").lower() in ("1", "true", "yes"):
        print(f"[SEC] {KILL_SWITCH_ENV} is set — auto-fixer disabled.", file=sys.stderr)
        return True
    return False


def check_token_scopes(token: str):
    """Soft check: warn if a CLASSIC PAT carries broad scopes. Fine-grained
    tokens and GITHUB_TOKEN don't expose X-OAuth-Scopes, so this simply no-ops
    for them (returns [])."""
    warnings = []
    if not token:
        return warnings
    try:
        r = requests.get("https://api.github.com/rate_limit",
                         headers={"Authorization": f"Bearer {token}",
                                  "Accept": "application/vnd.github+json"},
                         timeout=10)
        scopes = r.headers.get("X-OAuth-Scopes", "")
        if not scopes:
            return warnings                      # fine-grained / GITHUB_TOKEN
        granted = {s.strip() for s in scopes.split(",") if s.strip()}
        broad = granted & {"repo", "admin:org", "workflow", "delete_repo", "admin:repo_hook"}
        if broad:
            warnings.append(f"token holds broad classic scopes {sorted(broad)}; "
                            f"prefer a fine-grained token with only contents:write "
                            f"and pull-requests:write")
    except Exception:
        pass
    return warnings