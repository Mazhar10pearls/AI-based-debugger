#!/usr/bin/env python3
"""
security.py — breach controls for the AI CI/CD auto-fixer.

This module is the single place where the fixer's trust boundaries live. It is
import-only (no side effects at import time) and FAIL-CLOSED: when a check cannot
complete (network down, git error, malformed input) it returns the SAFE answer
(redact / hold-for-review / don't-run / don't-commit), never the permissive one.

═══════════════════════════════════════════════════════════════════════════════
THREAT MODEL — the breach scenarios this system faces, and the control that
counters each. "This system" = a fixer that reads attacker-influenceable CI logs
and repo files, feeds them to a local model, writes files, runs code, and pushes
to a repo with a token.
═══════════════════════════════════════════════════════════════════════════════

  BREACH 1 — Secret leaks OUT through a prompt, PR, log, or audit line.
    Scenario: a CI log or a config file in context contains an AWS key / DB
    connection string / token. It ends up in the model prompt (and, if Ollama
    is remote or logs, off-box), or gets echoed into a PR body or issue.
    Control: redact_secrets / scrub_context  — scrub known secret shapes from
    every string before it leaves for the model, a PR, an issue, or the audit.
    Limit: regex is best-effort; pair with gitleaks + a local-only Ollama.

  BREACH 2 — The fixer COMMITS a secret it introduced.
    Scenario: the model diagnoses "failing because API key is missing" and its
    "fix" hardcodes a key into a file; the bot commits and pushes it.
    Control: scan_staged_secrets  — scan the STAGED diff right before commit;
    any newly-added secret aborts the commit. This is the outbound gate.
    Limit: same regex caveat; back with GitHub push protection (server-side).

  BREACH 3 — A secret-bearing file is read into the model's context.
    Scenario: discovery pulls .env / id_rsa / .aws/credentials into the prompt.
    Control: PathPolicy.is_readable_into_context  — those files never enter it.

  BREACH 4 — The fixer silently rewrites a security-critical file.
    Scenario: a "fix" edits .github/workflows/*, CODEOWNERS, an auth/ module, or
    IaC — quietly weakening controls or opening a backdoor via an auto-PR.
    Control: PathPolicy.review_reason  — those paths are held for human review,
    never auto-applied.

  BREACH 5 — Supply-chain injection through a dependency "fix".
    Scenario: the fix adds a typosquatted/hallucinated package, edits a lockfile
    by hand, or bumps a pin to a nonexistent version — pulling in attacker code.
    Control: vet_supplychain  — reject lockfile edits, reject nonexistent
    packages/versions (verified against PyPI/npm), hold new deps for review.
    Limit: cannot tell a real-but-malicious package apart; add OSV/Dependabot.

  BREACH 6 — Untrusted code runs on the runner (RCE) with the token in reach.
    Scenario: a fork PR's code (or a fork-triggered failure) reaches the test
    step, executing attacker code on the self-hosted box next to the PAT.
    Control: assess_trust + should_run_tests + should_auto_pr + harden_test_env
    — untrusted origins never run tests or auto-PR; the test subprocess has
    secrets stripped from its env.
    Limit: env-stripping is not a sandbox; isolate the runner (container/VM).

  BREACH 7 — Over-privileged / leaked token → account-wide blast radius.
    Scenario: a broad classic PAT (repo + workflow) leaks; it can touch every
    repo the account can and rewrite CI.
    Control: check_token_scopes  — warns on broad classic scopes so you swap to
    a fine-grained, repo-scoped, short-lived token. (Detection, not prevention.)

  BREACH 8 — Runaway / kill-needed / no forensic trail.
    Control: kill_switch (instant disable), audit (0600 redacted JSONL trail),
    secure_file (lock down sensitive files), resource_limits_preexec (cap a
    test process). Ship the audit off-box and alert on sensitive-path events.

Everything below is grouped by the numbered breach it addresses.
"""

import json
import math
import os
import re
import stat
import subprocess
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


# ══════════════════════════════════════════════════════════════════════════════
# 7. OUTBOUND SECRET GATE — never let the fixer COMMIT a secret
# ══════════════════════════════════════════════════════════════════════════════
# redact_secrets() cleans text on the way IN (logs, prompts, RAG store). This is
# the way OUT: right before the bot commits, scan the STAGED diff for secrets the
# fix would introduce, and abort the commit if any are found.
#
# Only ADDED lines (diff '+' lines) are inspected, so a secret that already lived
# in the file is NOT the bot's doing and does not block its unrelated fix — but
# any secret the fix ADDS aborts the commit. This is the control that prevents a
# "secret commit": the model's diagnosis ("failing because API key is missing")
# must never turn into a hardcoded key pushed to the repo.
#
# This is a fail-CLOSED gate: if the staged diff cannot be read (git error), the
# caller is told the scan did not complete and must not commit. A best-effort
# built-in scanner always runs; if the `gitleaks` binary is present it is used as
# a stronger second pass. For the strongest guarantee, also enable GitHub secret
# scanning with push protection at the repo level as a server-side backstop.

import shutil
import subprocess


def detect_secrets(text: str):
    """Findings-only: the list of secret KINDS present in `text` (never values).
    Shares the exact rule set used by redact_secrets."""
    _, findings = redact_secrets(text)
    return findings


def _staged_diff(repo_root: str = "."):
    """(diff_text, scan_ran). scan_ran is False if git could not produce a diff,
    so the caller can fail closed instead of assuming 'clean'."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--unified=0", "--no-color"],
            cwd=repo_root, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"[SEC] staged-diff read failed (git rc={r.returncode}): "
                  f"{r.stderr.strip()[:160]}", file=sys.stderr)
            return "", False
        return r.stdout, True
    except Exception as exc:
        print(f"[SEC] staged-diff read error: {exc}", file=sys.stderr)
        return "", False


def _gitleaks_findings(repo_root: str):
    """Optional stronger pass. Runs `gitleaks` over staged changes if the binary
    exists. Returns a list of {file, kinds:['GITLEAKS:<rule>']}. Version-tolerant:
    tries the modern `git --staged` subcommand, then the legacy `protect`."""
    if not shutil.which("gitleaks"):
        return []
    import json as _json
    import tempfile
    for argv in (["gitleaks", "git", "--staged"],
                 ["gitleaks", "protect", "--staged"]):
        rep = tempfile.NamedTemporaryFile(prefix="gl_", suffix=".json", delete=False)
        rep.close()
        try:
            r = subprocess.run(
                argv + ["--no-banner", "--redact", "--exit-code", "0",
                        "--report-format", "json", "--report-path", rep.name],
                cwd=repo_root, capture_output=True, text=True, timeout=120)
            # unknown subcommand → try the next invocation form
            if r.returncode != 0 and ("unknown command" in (r.stderr or "").lower()
                                      or "unknown command" in (r.stdout or "").lower()):
                continue
            data = _json.loads(Path(rep.name).read_text() or "[]")
            out = {}
            for f in data:
                path = f.get("File") or f.get("file") or "(unknown)"
                rule = f.get("RuleID") or f.get("Description") or "secret"
                out.setdefault(path, set()).add(f"GITLEAKS:{rule}")
            return [{"file": p, "kinds": sorted(k)} for p, k in out.items()]
        except Exception:
            continue
        finally:
            try:
                os.unlink(rep.name)
            except Exception:
                pass
    return []


def scan_staged_secrets(repo_root: str = "."):
    """
    Scan the git STAGED diff for secrets the pending commit would introduce.

    Returns (clean, findings):
      clean=True,  findings=[]   → staged changes are safe to commit
      clean=False, findings=[…]  → secrets found, OR the scan could not run
                                    (git error). Either way: DO NOT COMMIT.

    findings entries are {file, kinds} — kinds name the detector, never a value.
    Caller must stage its changes (git add) before calling this.
    """
    diff, scan_ran = _staged_diff(repo_root)
    if not scan_ran:
        # fail closed: we could not verify the diff, so we must not vouch for it
        return False, [{"file": "(staged diff)", "kinds": ["SCAN_FAILED"]}]
    if not diff.strip():
        return True, []            # nothing staged → nothing to leak

    by_file = {}
    current = "(unknown)"
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            current = path[2:] if path.startswith("b/") else path
            continue
        if line.startswith("+") and not line.startswith("+++"):
            kinds = detect_secrets(line[1:])
            if kinds:
                by_file.setdefault(current, set()).update(kinds)

    findings = [{"file": f, "kinds": sorted(k)} for f, k in by_file.items()]

    # stronger optional pass; merge any extra findings it surfaces
    for gl in _gitleaks_findings(repo_root):
        match = next((x for x in findings if x["file"] == gl["file"]), None)
        if match:
            match["kinds"] = sorted(set(match["kinds"]) | set(gl["kinds"]))
        else:
            findings.append(gl)

    return (len(findings) == 0), findings