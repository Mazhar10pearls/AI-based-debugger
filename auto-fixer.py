"""
================================================================================
auto-fixer.py — ROBUSTNESS UPGRADE (drop-in changes)
================================================================================
Apply each numbered section to your existing auto-fixer.py. They are surgical;
unchanged functions stay as-is. Read the one-line rationale above each block.

Design principle behind every change:
  • DETERMINISTIC LAYER (prescan) = fast, can't hallucinate, no model needed.
    This is where "robustness" really lives. Widen it as far as you safely can.
  • MODEL LAYER = the long tail only. Give it the RIGHT files and enough context.
  • The two now run TOGETHER and merge, instead of either/or.
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# (1) CONFIG — replace the matching constants in your file.
#     Bigger context is what lets the model see MULTIPLE broken files at once.
#     NOTE: with the num_ctx fix in section (7), Ollama actually uses this budget;
#     without it Ollama silently truncates to 2048 tokens and your bump does
#     nothing.
# ──────────────────────────────────────────────────────────────────────────────
MAX_ERROR_LINES   = 16     # was 12 — more signal lines reach the model
MAX_FILE_CHARS    = 3500   # was 2600 — most config/app files fit whole
MAX_TOTAL_CONTEXT = 11000  # was 5200 — KEY: lets 3-4 files coexist in the prompt
MAX_CONTEXT_FILES = 4      # was 2  — enables multi-file diagnosis
MAX_FILES_FIXED   = 6      # was 3  — let one pass fix several files
# In build_prompt(), raise the hard cap so the bigger context isn't trimmed away:
#   PROMPT_HARD_CAP = 12000   # was 7000

# Run the model even after a deterministic prescan fix, so a SECOND unrelated
# bug (one prescan can't resolve) still gets caught in the same pass. Set False
# only if CPU time is more precious than coverage.
RUN_MODEL_AFTER_PRESCAN = True


# ──────────────────────────────────────────────────────────────────────────────
# (2) VERSION / IMAGE VALIDITY — add these helpers near your _bad_python_version().
#     Keep _bad_python_version as-is; these generalise it to images + node and
#     are what make "image error" a deterministic, no-model fix.
#     Edit the allow-lists as new versions ship — same philosophy as your
#     existing SUPPORTED_PY_MINORS.
# ──────────────────────────────────────────────────────────────────────────────
import re

SUPPORTED_PY_MINORS   = {"3.8", "3.9", "3.10", "3.11", "3.12", "3.13"}
LATEST_PY             = "3.12"
SUPPORTED_NODE_MAJORS = {"18", "20", "22"}   # current LTS lines
LATEST_NODE           = "20"


def _suggest_image_fix(ref: str) -> str | None:
    """
    Given a Docker `FROM` image reference, return a corrected reference, or None
    if it's valid / we have no opinion (unknown image). Preserves variant
    suffixes like -slim / -alpine / -bookworm so we don't strip the user's intent.

    Catches:  python:3.1, python:3.1-slim, python:3.7, python:3.1qq, python:3.,
              node:17, node:18.x.y.z, node:foo
    Leaves:   python:3.12-slim, node:20-alpine, ubuntu:22.04, myorg/img:tag
    """
    if ":" not in ref:
        return None                              # untagged — let the model judge
    name, tag = ref.split(":", 1)
    name = name.rsplit("/", 1)[-1].lower()       # strip registry/org prefix
    core = tag.split("-")[0]                      # version part, drop -slim etc.
    suffix = tag[len(core):]                      # keep "-slim", "-alpine", ...

    if name == "python":
        if core in ("latest", "3"):
            return None
        if not re.fullmatch(r"\d+\.\d+(\.\d+)?", core):       # syntactically broken
            return f"python:{LATEST_PY}{suffix}"
        minor = ".".join(core.split(".")[:2])
        return f"python:{LATEST_PY}{suffix}" if minor not in SUPPORTED_PY_MINORS else None

    if name == "node":
        if core == "latest":
            return None
        if not re.fullmatch(r"\d+(\.\d+){0,2}", core):
            return f"node:{LATEST_NODE}{suffix}"
        major = core.split(".")[0]
        return f"node:{LATEST_NODE}{suffix}" if major not in SUPPORTED_NODE_MAJORS else None

    return None                                  # unknown image — don't touch


def _suggest_node_version(value: str) -> str | None:
    """Same idea for a workflow setup-node `node-version:` value."""
    v = value.strip().strip("\"'")
    if not v or v == "latest":
        return None
    core = v.split("-")[0]
    if not re.fullmatch(r"\d+(\.\d+){0,2}", core):
        return LATEST_NODE
    major = core.split(".")[0]
    return LATEST_NODE if major not in SUPPORTED_NODE_MAJORS else None


# ──────────────────────────────────────────────────────────────────────────────
# (3) BUILD-CONFIG HEURISTIC — replace the `if forced_is_build_config:` block
#     INSIDE discover_context(). This stops the workflow file from hijacking
#     context when the log already named the real broken file (your bug).
# ──────────────────────────────────────────────────────────────────────────────
def _discover_context_buildconfig_block(error_signal, forced_paths,
                                         looks_like_build_config_error,
                                         find_ci_workflow_files,
                                         MAX_CONTEXT_FILES):
    """
    Illustrative extract — paste the body into discover_context where the old
    block was. `suspect_note`, `file_cap`, `wf_files`, `forced_paths` are the
    same locals you already use there.
    """
    file_cap = MAX_CONTEXT_FILES
    suspect_note = ""
    forced_is_build_config = looks_like_build_config_error(error_signal)
    wf_files: list[str] = []

    if forced_is_build_config:
        wf_files = find_ci_workflow_files()
        # Files the LOG explicitly named (Dockerfile, requirements, app.py …).
        referenced_sources = [f for f in forced_paths if f not in wf_files]

        if wf_files and not referenced_sources:
            # Genuine CI-config failure: log named no source file → workflow is
            # the prime suspect, and the "Dockerfile is a decoy" directive helps.
            file_cap = max(MAX_CONTEXT_FILES, 4)
            forced_paths = wf_files + forced_paths
            print(f"[DISCOVER] Build-config error, log named no source file → CI "
                  f"workflow is prime suspect; force-including {wf_files} "
                  f"(file cap raised to {file_cap})")
            suspect_note = (
                "This is a CI-CONFIGURATION failure (build/setup/version), not an "
                "application bug. The defect is in the CI WORKFLOW YAML below "
                f"({', '.join(wf_files)}) — most likely a setup-python "
                "`python-version`, a build-context path, or an action input. "
                "Fix the WORKFLOW file."
            )
        elif wf_files:
            # The log DID reference concrete files → trust them. Keep the workflow
            # as a SECONDARY suspect and DO NOT emit the decoy directive (that
            # directive is exactly what made the model edit the wrong file).
            file_cap = max(MAX_CONTEXT_FILES, 4)
            forced_paths = forced_paths + [f for f in wf_files if f not in forced_paths]
            print(f"[DISCOVER] Build-config signals present but log referenced "
                  f"{referenced_sources} → trusting those; workflow kept secondary.")

    return file_cap, suspect_note, wf_files, forced_paths


# ──────────────────────────────────────────────────────────────────────────────
# (4) PRESCAN — replace your whole prescan_issues() with this widened version.
#     New coverage: any bad base-image tag (Dockerfile FROM), bad
#     python-version / node-version in workflow YAML, plus your existing
#     path-existence checks. Everything it resolves needs NO model call.
#     Depends on helpers you already have: _repo_path_index, _extract_path_refs,
#     _closest_repo_path, _bad_python_version, and section (2)'s helpers.
# ──────────────────────────────────────────────────────────────────────────────
from pathlib import Path


def prescan_issues(included_files: list[str]) -> tuple[list[str], list[dict]]:
    findings: list[str] = []
    autofixes: list[dict] = []
    all_paths, name_to_paths = _repo_path_index()      # noqa: F821 (your helper)

    for rel in included_files:
        p = Path(rel)
        if not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        low_name  = p.name.lower()
        rel_posix = rel.replace("\\", "/")

        # ── 1. invalid base-image tag in ANY Dockerfile FROM ─────────────────
        if low_name.startswith("dockerfile"):
            for ln in text.splitlines():
                fm = re.match(r"(?i)(\s*FROM\s+)(\S+)", ln)
                if not fm:
                    continue
                ref = fm.group(2).strip()
                suggestion = _suggest_image_fix(ref)
                if suggestion and suggestion != ref:
                    findings.append(
                        f"{rel}: base image `{ref}` is invalid/unsupported — "
                        f"change it to `{suggestion}`.")
                    stripped = ln.strip()
                    autofixes.append({
                        "file": rel,
                        "reason": f"Invalid image tag {ref} -> {suggestion}",
                        "find":  stripped,
                        "replace": stripped.replace(ref, suggestion),
                    })

        # ── 2. invalid setup-python / setup-node version in workflow YAML ────
        if re.search(r"\.ya?ml$", rel_posix) and ".github/workflows" in rel_posix:
            for ln in text.splitlines():
                stripped = ln.strip()

                pv = re.search(r'(?i)python-version:\s*(["\']?)([^"\'\s#]+)\1', ln)
                if pv and _bad_python_version(pv.group(2)):     # noqa: F821
                    bad = pv.group(2)
                    findings.append(
                        f"{rel}: setup-python `{bad}` is invalid/unsupported — "
                        f"change it to `{LATEST_PY}`.")
                    autofixes.append({
                        "file": rel,
                        "reason": f"Invalid python-version {bad} -> {LATEST_PY}",
                        "find":  stripped,
                        "replace": stripped.replace(bad, LATEST_PY),
                    })

                nv = re.search(r'(?i)node-version:\s*(["\']?)([^"\'\s#]+)\1', ln)
                if nv:
                    sug = _suggest_node_version(nv.group(2))
                    if sug:
                        bad = nv.group(2)
                        findings.append(
                            f"{rel}: setup-node `{bad}` is invalid/unsupported — "
                            f"change it to `{sug}`.")
                        autofixes.append({
                            "file": rel,
                            "reason": f"Invalid node-version {bad} -> {sug}",
                            "find":  stripped,
                            "replace": stripped.replace(bad, sug),
                        })

        # ── 3. general local-path existence check (ALL file types) ───────────
        for kind, raw in _extract_path_refs(rel, text):        # noqa: F821
            tok = raw.strip().strip("\"'")
            if not tok or tok in (".", ".."):
                continue
            if ("://" in tok or "$" in tok or "${{" in tok
                    or ":" in tok or "@" in tok or "=" in tok):
                continue
            if not re.fullmatch(r"[\w./\-]+", tok):
                continue
            norm = tok[2:] if tok.startswith("./") else tok
            if Path(norm).exists():
                continue
            suggestion = _closest_repo_path(norm, all_paths, name_to_paths)  # noqa: F821
            if kind == "docker build context" and not suggestion:
                continue
            if suggestion:
                findings.append(
                    f"{rel}: `{kind}` points to `{tok}`, which does not exist. "
                    f"The real path is `{suggestion}` — change it to exactly that.")
                if tok in text:
                    autofixes.append({
                        "file": rel,
                        "reason": f"{kind}: '{tok}' does not exist -> '{suggestion}'",
                        "find":  tok,
                        "replace": suggestion,
                    })
            else:
                findings.append(
                    f"{rel}: `{kind}` points to `{tok}`, which does not exist in "
                    f"the repo — correct it to the real path.")

    findings = list(dict.fromkeys(findings))
    if findings:
        print(f"[PRESCAN] Found {len(findings)} issue(s) the log did not surface:")
        for f in findings:
            print(f"[PRESCAN]   - {f}")
        if autofixes:
            print(f"[PRESCAN] {len(autofixes)} can be auto-fixed deterministically "
                  f"(no model needed).")
    else:
        print("[PRESCAN] No extra static issues found.")
    return findings, autofixes


# ──────────────────────────────────────────────────────────────────────────────
# (5) MERGE — add this near group_autofixes(). Lets deterministic fixes and
#     model fixes coexist: prescan wins on files it touched (it can't
#     hallucinate); the model covers every OTHER file. This is what makes
#     "multiple errors across files" actually work in one pass.
# ──────────────────────────────────────────────────────────────────────────────
def merge_fixes(prescan_fixes: list[dict], model_fixes: list[dict]) -> list[dict]:
    by_file = {f["file"]: f for f in prescan_fixes}     # deterministic first
    for mf in model_fixes:
        f = (mf.get("file") or "").strip()
        if f and f not in by_file:
            by_file[f] = mf
    return list(by_file.values())


# ──────────────────────────────────────────────────────────────────────────────
# (6) MAIN — Stage 2 + Stage 3 rewrite.
#     • Stage 2: prescan the REFERENCED files too, not just the ones that fit
#       in context (so a deterministic fix fires even when budget evicted the
#       broken file).
#     • Stage 3: run prescan AND model, then merge. Confidence gates the MODEL
#       fixes only — deterministic fixes always proceed. Replace your existing
#       Stage 3 block (everything between the STAGE 3 banner and the STAGE 4
#       banner), and delete the old standalone `if confidence < CONFIDENCE_MIN`
#       and `if not fixes` blocks (handled inline below).
# ──────────────────────────────────────────────────────────────────────────────
def _main_stage2_and_3_extract(
        # locals from your main():
        error_signal, log_text, tech_stacks, pipeline_types, token, repo,
        # your functions:
        extract_referenced_paths, discover_context, prescan_issues,
        group_autofixes, merge_fixes, call_ai, parse_ai_response, open_issue,
        CONFIDENCE_MIN):
    import sys, time

    # ── STAGE 2: DISCOVER ──────────────────────────────────────────────────
    t2 = time.time()
    forced_paths = extract_referenced_paths(log_text)
    repo_context, included, suspect_note = discover_context(
        error_signal, tech_stacks, pipeline_types, forced_paths)
    if not included:
        print("[DISCOVER] WARNING: no files resolved — model has no context.")
    # KEY: prescan referenced files too — a deterministic fix must fire even if
    # the broken file got evicted from the model prompt by the budget.
    prescan_targets = list(dict.fromkeys(included + forced_paths))
    prescan, prescan_autofixes = prescan_issues(prescan_targets)
    print(f"[TIMING] Stage 2 done in {time.time()-t2:.1f}s")

    # ── STAGE 3: ANALYSE (deterministic + model, merged) ───────────────────
    t3 = time.time()
    detected_type = ", ".join(sorted(pipeline_types))

    prescan_fixes = group_autofixes(prescan_autofixes) if prescan_autofixes else []
    if prescan_fixes:
        print(f"[ANALYSE] Pre-scan resolved {len(prescan_autofixes)} issue(s) "
              f"deterministically across {len(prescan_fixes)} file(s):")
        for fx in prescan_fixes:
            print(f"    OK (deterministic) {fx['file']}  ({len(fx['edits'])} edit(s))")

    model_fixes, model_root_cause, model_commit = [], "", ""
    model_confidence = 1.0
    run_model = RUN_MODEL_AFTER_PRESCAN or not prescan_fixes

    if run_model:
        try:
            raw = call_ai(error_signal, repo_context, pipeline_types, tech_stacks,
                          prescan, suspect_note)
            ai_data = parse_ai_response(raw)
            model_root_cause = ai_data.get("root_cause", "")
            model_confidence = float(ai_data.get("confidence", 1.0))
            model_commit     = ai_data.get("commit_message", "")
            detected_type    = ai_data.get("pipeline_type", detected_type)
            raw_model_fixes  = ai_data.get("fixes", []) or []
            if model_confidence >= CONFIDENCE_MIN:
                model_fixes = raw_model_fixes
            else:
                print(f"[ANALYSE] Model confidence {model_confidence:.0%} < "
                      f"{CONFIDENCE_MIN:.0%} — dropping model fixes, keeping "
                      f"deterministic ones only.")
        except Exception as exc:
            print(f"[ERROR] AI step failed: {exc}", file=sys.stderr)
            if not prescan_fixes:                 # nothing else to fall back on
                if token and repo:
                    open_issue(token, repo, f"AI analysis failed: {exc}",
                               pipeline_type=detected_type)
                sys.exit(2)
            print("[ANALYSE] Proceeding with deterministic fixes despite model failure.")
    else:
        print("[ANALYSE] Fast path — prescan covered the failure; skipping model.")

    fixes = merge_fixes(prescan_fixes, model_fixes)

    root_cause = (model_root_cause
                  or "; ".join(f.get("reason", "") for f in prescan_fixes)
                  or "unknown")
    commit_msg = (model_commit if (model_fixes and model_commit)
                  else "fix: correct detected CI/build errors (auto-detected)")
    confidence = model_confidence if model_fixes else 1.0

    if not fixes:
        print("[ERROR] No fixes produced (deterministic or model).", file=sys.stderr)
        if token and repo:
            open_issue(token, repo,
                       f"No fixable issues identified. Root cause: {root_cause}",
                       pipeline_type=detected_type)
        sys.exit(3)

    print(f"\n  pipeline   : {detected_type}")
    print(f"  root_cause : {root_cause}")
    print(f"  confidence : {confidence:.0%}")
    print(f"  fixes      : {len(fixes)} file(s) "
          f"({len(prescan_fixes)} deterministic, {len(model_fixes)} model)")
    for fx in fixes:
        n = len(fx.get("edits", []) or [])
        tag = f"{n} edit(s)" if n else f"{len(fx.get('fixed_content','') or '')} chars"
        print(f"    -> {fx.get('file','?')}  ({tag})")
    print(f"[TIMING] Stage 3 done in {time.time()-t3:.1f}s")

    # hand these to your existing Stage 4 (VALIDATE) unchanged:
    return fixes, root_cause, commit_msg, detected_type, confidence


# ──────────────────────────────────────────────────────────────────────────────
# (7) CALL_AI — the single most important one-line fix. In call_ai(), the native
#     /api/generate payload MUST set num_ctx, or Ollama caps context at 2048
#     tokens and silently drops most of your (now larger) prompt. Replace the
#     native "options" dict:
#
#         "options": {"temperature": 0.05, "num_predict": 2000, "num_ctx": 8192},
#
#     And in build_prompt(), raise the hard cap so the bigger context survives:
#
#         PROMPT_HARD_CAP = 12000   # was 7000
#
#     CPU note (8GB box): a larger prompt slows prefill. If runs creep past your
#     210s AI_TIMEOUT, either raise AI_TIMEOUT (and the workflow timeout-minutes)
#     to ~300s, OR lean on the prescan — most common errors (image, version,
#     path) now never reach the model, so they're fixed in <1s regardless.
# ──────────────────────────────────────────────────────────────────────────────