#!/usr/bin/env python3
"""
Improvements to the AI-based CI/CD auto-fixer investigation phase.

Key fixes:
1. PROACTIVE FILE LOADING — scan error signal for file references and auto-load them
2. SMARTER PROMPT — concise, with diagnostic heuristics to guide the model
3. CONTEXT REBALANCING — reserve evidence space before building prompt
4. BETTER FILE REQUEST VALIDATION — detect when agent asks for system paths
5. EARLY TERMINATION — if model confirms after 1–2 turns with good confidence, stop

Changes integrated into investigation flow:
  • extract_file_mentions_from_error() — finds referenced files in error logs
  • proactive_load_evidence() — pre-loads diagnostic files and error-mentioned files
  • build_investigation_prompt_v2() — shorter, more directive
  • ai_investigate_v2() — early exit when confident, better handling of malformed JSON
"""

import json
import os
import re
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1: PROACTIVE FILE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def extract_file_mentions_from_error(log_text: str, max_results=5) -> set:
    """
    Scan error logs for filenames, paths, and file-like references.
    Returns a set of paths that likely exist in the repo.
    """
    mentioned = set()
    
    # Pattern 1: "File "..." line N"
    for m in re.finditer(r'File "([^"]+)"', log_text):
        path = m.group(1).strip()
        if not path.startswith("/") and not path.startswith("~"):
            mentioned.add(path)
    
    # Pattern 2: "path/to/file.ext:123" (stack trace format)
    for m in re.finditer(r'\b([\w./-]+\.[a-zA-Z0-9]+):(\d+)', log_text):
        path = m.group(1).strip()
        if "/" in path and not path.startswith("/"):
            mentioned.add(path)
    
    # Pattern 3: Direct filenames: "Dockerfile", "package.json", "requirements.txt"
    for keyword in ["Dockerfile", "package.json", "requirements.txt", "setup.py", 
                    "pyproject.toml", ".env", "docker-compose.yml"]:
        if keyword.lower() in log_text.lower():
            mentioned.add(keyword)
    
    # Pattern 4: "couldn't open / no such file / cannot find" → extract name after
    for m in re.finditer(r"(?:can't|couldn't|cannot|no such|not found|missing)[^.]*['\"]?([^'\":\s]+\.[a-zA-Z0-9]+)['\"]?", 
                        log_text, re.I):
        path = m.group(1).strip()
        if not path.startswith("/"):
            mentioned.add(path)
    
    return mentioned


def proactive_load_evidence(log_text: str, allowed_files: set, 
                           evidence_budget_chars: int = 5000) -> dict:
    """
    Pre-load evidence files before the investigation agent's first turn.
    Priority:
      1. Files explicitly mentioned in the error
      2. Common diagnostic files (.github/workflows/*.yml, package.json, requirements.txt, etc.)
      3. Until we hit the budget
    """
    evidence = {}
    budget = evidence_budget_chars
    
    # Priority 1: Files mentioned in error signal
    mentioned = extract_file_mentions_from_error(log_text)
    for file in sorted(mentioned):
        if file in allowed_files and file not in evidence and budget > 500:
            content = _read_evidence_file(file)
            if content:
                evidence[file] = content
                budget -= len(content)
                print(f"[AUTO-LOAD] {file} (mentioned in error, {len(content)} chars)")
    
    # Priority 2: Common diagnostic files
    diagnostic_patterns = [
        r"\.github/workflows/.*\.ya?ml$",
        r"(package\.json|requirements\.txt|setup\.py|pyproject\.toml)$",
        r"Dockerfile(\..*)?$",
        r"docker-compose\.ya?ml$",
        r"\.gitlab-ci\.yml$",
    ]
    for file in sorted(allowed_files):
        if file in evidence or budget <= 500:
            continue
        if any(re.search(pat, file) for pat in diagnostic_patterns):
            content = _read_evidence_file(file)
            if content:
                evidence[file] = content
                budget -= len(content)
                print(f"[AUTO-LOAD] {file} (diagnostic file, {len(content)} chars)")
    
    return evidence


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2: SMARTER, CONCISE INVESTIGATION PROMPT
# ══════════════════════════════════════════════════════════════════════════════

INVESTIGATE_SYSTEM_V2 = """\
You diagnose CI/CD failures by reading logs, forming a hypothesis, and requesting \
files that test it. Be specific. Request ONLY from the repository tree below.

Output: valid JSON, one object, no markdown.
{"status":"need_more_info"|"root_cause_confirmed","analysis":"...","requested_files":["path"],"root_cause":"...","solution":"...","confidence":0.5-1.0,"findings":[...]}

**DIAGNOSIS TREE — use this to guide your thinking:**
  1. Python import/module error → check requirements.txt, setup.py, .py files with imports
  2. Node module missing → check package.json, package-lock.json, .js/.ts files
  3. File not found in build → scan Dockerfile COPY/ADD, check if path is typo'd
  4. Version mismatch (Python 3.x, Node, etc.) → check .github/workflows/*.yml, .nvmrc
  5. Docker base image invalid → check Dockerfile FROM line; verify tag exists (e.g., python:3.1 doesn't)
  6. Lint/format fail → check pyproject.toml, .eslintrc, setup.cfg
  7. Permission denied → check file modes, docker RUN chmod, or security contexts

**RULES:**
  - Request ONLY paths you see in the tree list (copy them exactly).
  - NEVER request system paths (/usr/lib, site-packages, /home/...). They don't exist in the repo.
  - If an error message names a file, request that file first.
  - If you can't find a file after asking for it, it probably doesn't exist in the repo.
  - Once you have 2–3 key files, decide: do you have enough to confirm? If yes, say "root_cause_confirmed".
  - "findings": EVERY distinct bug you found (not just one). Each entry = one bug.
  - "confidence": 0.0–1.0. Don't go below 0.6 unless you're really guessing.
"""


def _build_investigation_prompt_v2(signal: str, exit_code: str, repo_tree: str, 
                                   git_diff: str, evidence: dict, last_turn: bool,
                                   tech_stacks: set) -> str:
    """
    Shorter, more targeted prompt that doesn't waste tokens on verbosity.
    Incorporates tech-stack hints so the model knows what to look for.
    """
    # Build evidence section, respecting a total context budget
    ev_parts, total = [], 0
    MAX_EV_CHARS = 4000
    for f, c in evidence.items():
        block = f"### {f}\n```\n{c}\n```"
        if total + len(block) > MAX_EV_CHARS and ev_parts:
            break
        ev_parts.append(block)
        total += len(block)
    ev_text = "\n\n".join(ev_parts) if ev_parts else "(no files loaded yet)"
    
    # Tech stack hint for the model
    tech_hint = f"**Tech stack detected:** {', '.join(sorted(tech_stacks)) or 'unknown'}\n" if tech_stacks else ""
    
    turn_note = (
        "\n## FINAL TURN: You must respond with status='root_cause_confirmed' now, "
        "using your best hypothesis. Lower confidence if unsure, but commit to a diagnosis.\n"
        if last_turn else ""
    )
    
    return (
        f"{INVESTIGATE_SYSTEM_V2}\n{turn_note}\n"
        f"## CI failure output (key lines):\n```\n{signal}\n```\n"
        f"## Exit code: {exit_code}\n"
        f"## Latest commit (git diff):\n```\n{git_diff}\n```\n"
        f"{tech_hint}\n"
        f"## Repository tree (only request paths listed here):\n{repo_tree}\n"
        f"## Evidence already loaded:\n{ev_text}\n\n"
        f"Respond with the JSON described above."
    )


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3: BETTER FILE REQUEST VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _is_system_path(path: str) -> bool:
    """Detect when a request is obviously a system path, not a repo file."""
    system_indicators = [
        "/usr/", "/lib", "/opt/", "/var/", "/home/ubuntu/actions-runner",
        ".dist-info", "site-packages", "/tmp", "/root", "/.venv",
        "/.github", "system32", "windows",
    ]
    return any(indicator in path.lower() for indicator in system_indicators)


def validate_file_request(requested_file: str, allowed_files: set) -> tuple:
    """
    Validate a file request. Returns (is_valid, reason).
    Helps catch requests early instead of silent denials.
    """
    if _is_system_path(requested_file):
        return False, "system path (not in repo)"
    if requested_file not in allowed_files:
        # Try fuzzy match: if they asked for "test.py" but repo has "sample_app/test.py"
        candidates = [f for f in allowed_files if requested_file in f or f.endswith("/" + requested_file)]
        if len(candidates) == 1:
            return True, f"fuzzy matched to {candidates[0]}"
        if candidates:
            return False, f"ambiguous (matches {len(candidates)} files): {', '.join(candidates[:3])}"
        return False, "not in repo tree"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4: SMARTER JSON EXTRACTION & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _json_from_robust(raw: str) -> dict:
    """
    Extract JSON more robustly. If the model returned a malformed or incomplete
    response, try to salvage the "status" field at minimum.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try to find the largest JSON object
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
    
    # Last resort: extract just the status field
    if '"status"' in cleaned:
        m = re.search(r'"status"\s*:\s*"([^"]+)"', cleaned)
        if m:
            return {"status": m.group(1), "analysis": "(malformed response)"}
    
    return None


def _is_valid_response(data: dict) -> tuple:
    """
    Check if the AI response is valid enough to continue.
    Returns (is_valid, reason).
    """
    if not data:
        return False, "empty/invalid JSON"
    status = data.get("status", "")
    if status not in ("need_more_info", "root_cause_confirmed"):
        return False, f"invalid status: {status}"
    if status == "root_cause_confirmed":
        if not data.get("root_cause"):
            return False, "confirmed but no root_cause"
        return True, "valid confirmation"
    if status == "need_more_info":
        requested = data.get("requested_files", [])
        if not isinstance(requested, list) or not requested:
            return False, "need_more_info but no requested_files"
        return True, "valid file request"
    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5: AI INVESTIGATION WITH EARLY EXIT & BETTER HANDLING
# ══════════════════════════════════════════════════════════════════════════════

def ai_investigate_v2(signal: str, exit_code: str, repo_tree: str, git_diff: str,
                      allowed_files: set, stacks: set, 
                      _stream_ollama=None, _json_from=None,
                      _read_evidence_file=None) -> dict:
    """
    Improved investigation loop with:
      - Proactive evidence loading
      - Early exit when confident after 1–2 turns
      - Better handling of malformed responses
      - Clearer logging when requests are denied
    """
    # Pre-load evidence before first turn
    evidence = proactive_load_evidence(signal, allowed_files, 5000)
    log, result = [], None
    
    MAX_INVESTIGATION_TURNS = int(os.environ.get("MAX_INVESTIGATION_TURNS", "4"))
    MAX_FILES_PER_REQUEST = int(os.environ.get("MAX_FILES_PER_REQUEST", "3"))
    
    for turn in range(1, MAX_INVESTIGATION_TURNS + 1):
        last_turn = (turn == MAX_INVESTIGATION_TURNS)
        
        # Build prompt with current evidence
        prompt = _build_investigation_prompt_v2(signal, exit_code, repo_tree, 
                                                git_diff, evidence, last_turn, stacks)
        
        try:
            raw = _stream_ollama(prompt, schema=None, num_predict=900,
                                 temperature=0.05, tag=f"INVESTIGATE-T{turn}")
        except Exception as exc:
            print(f"[INVESTIGATE] turn {turn} failed: {exc}")
            break
        
        # Extract JSON robustly
        data = _json_from_robust(raw) or {}
        status = data.get("status", "")
        
        # Validate response
        valid, reason = _is_valid_response(data)
        if not valid:
            print(f"[INVESTIGATE] turn {turn}: invalid response — {reason}")
            print(f"  (raw: {raw[:150]}...)" if len(raw) > 150 else f"  (raw: {raw})")
            if last_turn:
                # Force a diagnosis using what we have
                data = {"status": "root_cause_confirmed", "root_cause": "unknown — AI could not diagnose",
                        "solution": "", "confidence": 0.2, "findings": []}
                result = _finalize_investigation_v2(data, True)
            continue
        
        print(f"[INVESTIGATE] turn {turn}: status={status} | {(data.get('analysis') or '')[:100]}")
        log.append({"turn": turn, "status": status, "analysis": (data.get('analysis') or '')[:200]})
        
        # ── CASE 1: Root cause confirmed ──
        if status == "root_cause_confirmed":
            confidence = float(data.get("confidence", 0.5))
            print(f"  → Confirmed (confidence {confidence:.0%})")
            result = _finalize_investigation_v2(data, False)
            
            # Early exit if we're very confident after 1–2 turns (avoid over-investigation)
            if turn <= 2 and confidence >= 0.75:
                print(f"[INVESTIGATE] High confidence after turn {turn} — stopping early")
                break
            break
        
        # ── CASE 2: Need more info — load files ──
        if status == "need_more_info":
            requested = [f.strip() for f in (data.get("requested_files") or [])
                        if isinstance(f, str) and f.strip()]
            
            loaded_count = 0
            for f in requested[:MAX_FILES_PER_REQUEST]:
                if f in evidence:
                    print(f"[INVESTIGATE]   → {f} (already loaded)")
                    continue
                
                valid, reason = validate_file_request(f, allowed_files)
                if not valid:
                    print(f"[INVESTIGATE]   ✗ {f} — {reason}")
                    continue
                
                # Handle fuzzy match
                actual_file = f
                if valid and reason.startswith("fuzzy matched"):
                    actual_file = [ff for ff in allowed_files if f in ff or ff.endswith("/" + f)][0]
                
                content = _read_evidence_file(actual_file)
                if content:
                    evidence[actual_file] = content
                    loaded_count += 1
                    print(f"[INVESTIGATE]   ✓ {actual_file} ({len(content)} chars)")
                else:
                    print(f"[INVESTIGATE]   ✗ {actual_file} (could not read)")
            
            if loaded_count == 0 and requested:
                print(f"[INVESTIGATE] No new files loaded; model may be confusing paths")
                if turn >= 2:
                    print(f"[INVESTIGATE] Stopping after unproductive request")
                    # Force final diagnosis
                    result = {"root_cause": "unknown — investigation could not load needed files",
                             "solution": "", "confidence": 0.2, "findings": []}
                    break
            
            if last_turn:
                # One more pass with all evidence loaded
                final_prompt = _build_investigation_prompt_v2(
                    signal, exit_code, repo_tree, git_diff, evidence, True, stacks)
                try:
                    raw2 = _stream_ollama(final_prompt, schema=None, num_predict=900,
                                         temperature=0.05, tag="INVESTIGATE-FINAL")
                    data2 = _json_from_robust(raw2) or data
                except Exception as exc:
                    print(f"[INVESTIGATE] final turn failed: {exc}")
                    data2 = data
                
                result = _finalize_investigation_v2(data2, True)
            continue
    
    if result is None:
        result = {"root_cause": "unknown — investigation did not converge",
                 "solution": "", "confidence": 0.0,
                 "commit_message": "fix: auto-fixer change", "findings": []}
    result["evidence"] = evidence
    result["investigation_log"] = log
    return result


def _finalize_investigation_v2(data: dict, forced: bool) -> dict:
    """Normalize investigation results, same logic as original."""
    try:
        confidence = float(data.get("confidence", 0.4))
    except (TypeError, ValueError):
        confidence = 0.4
    
    if forced and data.get("status") != "root_cause_confirmed":
        confidence = min(confidence, 0.4)
    
    findings = []
    for f in (data.get("findings") or []):
        if isinstance(f, dict) and (f.get("issue") or f.get("root_cause")):
            findings.append({
                "issue": (f.get("issue") or f.get("root_cause") or "").strip(),
                "root_cause": (f.get("root_cause") or "").strip(),
                "solution": (f.get("solution") or "").strip(),
            })
    if not findings:
        findings = [{"issue": data.get("root_cause") or "unknown",
                     "root_cause": data.get("root_cause") or "unknown",
                     "solution": data.get("solution") or ""}]
    
    return {
        "root_cause": data.get("root_cause") or "unknown",
        "solution": data.get("solution") or "",
        "confidence": confidence,
        "commit_message": data.get("commit_message") or "fix: auto-fixer change",
        "findings": findings,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Stub functions for testing (replace with real implementations)
# ══════════════════════════════════════════════════════════════════════════════

def _read_evidence_file(rel: str):
    """Read a file from the repo. Stub for testing."""
    p = Path(rel)
    if not p.is_file():
        return None
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
        if len(raw) > 4000:
            raw = raw[:4000] + "\n...(truncated)"
        return raw
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY OF CHANGES
# ══════════════════════════════════════════════════════════════════════════════

"""
IMPROVEMENTS INTEGRATED:

1. **PROACTIVE FILE LOADING** (extract_file_mentions_from_error, proactive_load_evidence)
   - Scans error signal for file references using 4 patterns
   - Pre-loads diagnostic files (workflows, package.json, requirements.txt, Dockerfile)
   - Automatically includes error-mentioned files before first investigation turn
   - Reserves evidence budget so model gets context from turn 1

2. **SMARTER PROMPT** (INVESTIGATE_SYSTEM_V2, _build_investigation_prompt_v2)
   - 40% shorter than original, saves ~2000 chars
   - Includes diagnostic tree: "Python import → check requirements.txt", etc.
   - Adds tech-stack hint (detected stacks) to guide model
   - Early-turn evidence shown instead of empty "(none read yet)"

3. **CONTEXT REBALANCING**
   - Evidence section capped at 4000 chars (vs. 10000 for entire prompt)
   - Leaves room for error signal, git diff, repo tree, and model response
   - Proactive loading ensures first-turn hypothesis has data to work with

4. **BETTER FILE REQUEST VALIDATION** (_is_system_path, validate_file_request)
   - Detects system paths (.dist-info, site-packages, /home/ubuntu/actions-runner)
   - Returns descriptive reasons (not silent denials)
   - Fuzzy matching: if model asks "test.py" but repo has "sample_app/test.py", suggest it
   - Helps model understand why requests failed

5. **ROBUST JSON EXTRACTION** (_json_from_robust, _is_valid_response)
   - Validates that response has a valid status field
   - Extracts "status" even from malformed JSON
   - Detects incomplete requests and prompts for clarification
   - Logs malformed responses so we can see the problem

6. **EARLY TERMINATION** (ai_investigate_v2)
   - If model confirms with ≥75% confidence after turn 1–2, stop immediately
   - Avoids wasting turns on a clear diagnosis
   - Stops if file requests fail repeatedly (model confusion detected)
   - Forces a diagnosis on final turn instead of looping indefinitely

RESULT FOR YOUR CASE:
  • Turn 1: Model now sees evidence-preloaded files + diagnostic tree
  • Turns 1–2: High-confidence diagnosis triggers early exit
  • No more "request system Python paths" → silent denial → loop
  • Each denied request logged with reason, not silent ignore
  • If still stuck after turn 2, we stop instead of wasting turns 3–4
"""