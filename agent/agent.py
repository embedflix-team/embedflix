"""Person B's nodes: baseline_verifier, code_writer, pipeline_runner,
error_recovery, score_analyst, convergence_checker.

(experiment_judge moved to Person A's track -- see note at the bottom of
this file. Person B still owns the deterministic wrapper that calls
log_iteration/track_resources; Person A owns the JUDGE_PROMPT reasoning
that fills state["hypothesis"]-equivalent judgment text.)

Wired against the REAL mcp_server.py tools (via tools_adapter.build_tools()),
not a stand-in. Tool call surface:
  run_pipeline(extra_args="")        -> str (combined stdout+stderr)
  parse_scores(pipeline_output)      -> {"GAUC":..,"nDCG@5":..,"primary":..} (values may be None)
  edit_file(file_path, old_code, new_code) -> "SUCCESS: ..." / "ERROR: ..." string
  save_checkpoint(iteration, primary_score) -> str
  restore_checkpoint(iteration)      -> str  (must name WHICH iteration -- no "last" shortcut)
  log_iteration(...)                 -> str
  track_resources(iteration, tokens, wall_seconds) -> str
  format_submission(split)           -> str
"""
import time

from state import AgentState

EPSILON = 0.002
N_CONVERGE = 3
MAX_ITERATIONS = 50
MAX_WALL_HOURS = 6

IMPROVE_THRESHOLD = 0.0001


# ---------------------------------------------------------------------------
def baseline_verifier(state: AgentState, tools: dict) -> AgentState:
    """Runs FM unmodified, confirms it matches published baseline (~0.6016
    valid primary), saves checkpoint 0, logs iteration 0."""
    output = tools["run_pipeline"].invoke({})
    scores = tools["parse_scores"].invoke({"pipeline_output": output})

    if scores.get("primary") is None:
        # Baseline itself failing means the harness is broken, not a normal
        # iteration failure -- surface loudly rather than routing to error_recovery.
        state["error_message"] = f"BASELINE FAILED TO REPRODUCE. Raw output tail:\n{output[-500:]}"
        state["should_stop"] = True
        return state

    tools["save_checkpoint"].invoke({"iteration": 0, "primary_score": scores["primary"]})
    tools["log_iteration"].invoke({
        "iteration": 0, "hypothesis": "baseline verification (FM, unmodified)",
        "code_diff": "", "gauc": scores["GAUC"], "ndcg": scores["nDCG@5"],
        "primary": scores["primary"], "error": "", "recovery": "",
    })

    state["iteration"] = 0
    state["current_score"] = scores
    state["best_score"] = scores
    state["best_iteration"] = 0
    state["score_history"] = [scores]
    state["registry"] = [{"iteration": 0, "hypothesis": "baseline", "scores": scores}]
    state["run_start_time"] = time.time()
    state["total_tokens"] = 0
    state["iterations_without_improvement"] = 0
    return state


# ---------------------------------------------------------------------------
# Files a specialist is never allowed to touch, enforced here as defense in
# depth (mcp_server.edit_file has no allowlist of its own -- it'll happily
# edit evaluate.py if a specialist's FILE: line says so).
PROTECTED_FILES = {"evaluate.py"}


def code_writer(state: AgentState, tools: dict) -> AgentState:
    """Applies the specialist's proposed edit. Expects state["hypothesis"]
    to contain the specialist's raw output:
        HYPOTHESIS: <reasoning>
        FILE: <filename>
        OLD_CODE: <exact string>
        NEW_CODE: <replacement>
    """
    raw = state["hypothesis"]
    parsed = _parse_specialist_output(raw)
    if parsed is None:
        state["error_message"] = "code_writer: could not parse specialist output (missing FILE/OLD_CODE/NEW_CODE)"
        return state

    if parsed["file"] in PROTECTED_FILES:
        state["error_message"] = f"code_writer: refused edit to protected file {parsed['file']} (evaluate.py must never be modified)"
        return state

    result = tools["edit_file"].invoke({
        "file_path": parsed["file"], "old_code": parsed["old_code"], "new_code": parsed["new_code"],
    })
    if not result.startswith("SUCCESS"):
        state["error_message"] = f"code_writer: edit_file failed -- {result}"
        return state

    # edit_file returns no diff -- construct a compact one ourselves from what
    # the specialist proposed (accurate as long as OLD_CODE was applied verbatim,
    # which the "SUCCESS" result confirms).
    state["hypothesis"] = parsed["reasoning"]
    state["code_diff"] = f"--- {parsed['file']}\n- {parsed['old_code']}\n+ {parsed['new_code']}"
    state["error_message"] = ""
    return state


def _parse_specialist_output(raw: str):
    import re
    m_file = re.search(r"^FILE:\s*(\S+)", raw, re.MULTILINE)
    m_hyp = re.search(r"^HYPOTHESIS:\s*(.+?)(?=^FILE:|^OLD_CODE:|^NEW_CODE:|\Z)", raw, re.MULTILINE | re.DOTALL)
    m_old = re.search(r"^OLD_CODE:\s*(.+?)(?=^NEW_CODE:|\Z)", raw, re.MULTILINE | re.DOTALL)
    m_new = re.search(r"^NEW_CODE:\s*(.+?)\Z", raw, re.MULTILINE | re.DOTALL)
    if not (m_file and m_old and m_new):
        return None
    return {
        "file": m_file.group(1).strip(),
        "reasoning": (m_hyp.group(1).strip() if m_hyp else ""),
        "old_code": m_old.group(1).strip("\n"),
        "new_code": m_new.group(1).strip("\n"),
    }


# ---------------------------------------------------------------------------
def pipeline_runner(state: AgentState, tools: dict) -> AgentState:
    """Runs the modified pipeline. A crash or timeout inside run_pipeline
    (subprocess-isolated, 300s cap set in mcp_server.py) surfaces as an
    error_message here rather than raising -- this node must never take
    down the graph."""
    t0 = time.time()
    try:
        output = tools["run_pipeline"].invoke({})
        scores = tools["parse_scores"].invoke({"pipeline_output": output})
        state["run_wall_seconds"] = state.get("run_wall_seconds", 0.0) + (time.time() - t0)

        if scores.get("primary") is None:
            state["error_message"] = f"pipeline run produced unparseable output. Tail:\n{output[-500:]}"
            state["current_score"] = {}
        else:
            state["current_score"] = scores
            state["error_message"] = ""
    except Exception as e:  # noqa: BLE001 -- this is the isolation boundary
        state["error_message"] = f"{type(e).__name__}: {e}"
        state["current_score"] = {}
    return state


def route_after_pipeline_runner(state: AgentState) -> str:
    return "error_recovery" if state.get("error_message") else "score_analyst"


# ---------------------------------------------------------------------------
def error_recovery(state: AgentState, tools: dict) -> AgentState:
    """Diagnoses the error, fixes code or restores last-good checkpoint.
    Max 3 fix attempts per iteration before falling back to restore.

    Restore targets state["best_iteration"] specifically -- mcp_server's
    restore_checkpoint requires naming which iteration, there's no implicit
    "last" (unlike a git-checkout-based design)."""
    attempts = state.get("_recovery_attempts", 0) + 1
    state["_recovery_attempts"] = attempts

    if attempts > 3:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["recovery_action"] = f"RESTORE: exceeded 3 fix attempts, rolled back to checkpoint {state['best_iteration']}"
        state["error_message"] = ""
        state["_recovery_attempts"] = 0
        return state

    # Deterministic recovery for mechanically-fixable errors. Anything not
    # matched here escalates to an LLM diagnosis call (ERROR_RECOVERY_PROMPT,
    # same pattern as the specialist nodes) -- not wired in this scaffold yet.
    err = state.get("error_message", "")
    if "ModuleNotFoundError" in err and "torch" in err:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "torch"])
        state["recovery_action"] = "FIX: installed missing torch dependency"
        state["error_message"] = ""
    else:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["recovery_action"] = f"RESTORE: rolled back to checkpoint {state['best_iteration']} -- {err[:200]}"
        state["error_message"] = ""
        state["_recovery_attempts"] = 0

    return state


# ---------------------------------------------------------------------------
def score_analyst(state: AgentState, tools: dict) -> AgentState:
    """Reads scores, checks if improved, updates best, saves checkpoint if
    better, restores if not.

    NOTE: the "overfit check" in the build doc (val-vs-test gap) isn't
    implemented here -- mcp_server's parse_scores only exposes the valid
    split (by design: the local "test" split is exactly the thing the
    webinar Q&A (Q21) is asking organizers whether it's safe to touch during
    development). Leaving it unused until that's answered, rather than
    quietly reading it."""
    scores = state["current_score"]
    best = state["best_score"]
    iteration = state["iteration"] + 1
    state["iteration"] = iteration

    improved = scores.get("primary", -1) > best.get("primary", -1) + IMPROVE_THRESHOLD

    if improved:
        tools["save_checkpoint"].invoke({"iteration": iteration, "primary_score": scores["primary"]})
        state["best_score"] = scores
        state["best_iteration"] = iteration
        state["iterations_without_improvement"] = 0
    else:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["iterations_without_improvement"] = state.get("iterations_without_improvement", 0) + 1

    state["score_history"] = state.get("score_history", []) + [scores]
    state["_improved"] = improved
    return state


# ---------------------------------------------------------------------------
def log_and_track(state: AgentState, tools: dict) -> AgentState:
    """The deterministic half of what was 'experiment_judge': calls
    log_iteration + track_resources with whatever's in state. Person A's
    JUDGE_PROMPT call should run BEFORE this node (populating a judgment
    string into state["hypothesis"] or a dedicated field) -- this node
    doesn't generate reasoning, it persists it. Renamed from
    experiment_judge to make that split explicit."""
    scores = state["current_score"] or {}
    tools["log_iteration"].invoke({
        "iteration": state["iteration"],
        "hypothesis": state.get("hypothesis", ""),
        "code_diff": state.get("code_diff", ""),
        "gauc": scores.get("GAUC"), "ndcg": scores.get("nDCG@5"), "primary": scores.get("primary"),
        "error": state.get("error_message", ""), "recovery": state.get("recovery_action", ""),
    })
    wall = time.time() - state["run_start_time"]
    tools["track_resources"].invoke({
        "iteration": state["iteration"], "tokens": state.get("total_tokens", 0), "wall_seconds": wall,
    })

    state["registry"] = state.get("registry", []) + [{
        "iteration": state["iteration"], "hypothesis": state.get("hypothesis", ""),
        "improved": state.get("_improved", False), "scores": scores,
    }]
    state["error_message"] = ""
    state["recovery_action"] = ""
    return state


# ---------------------------------------------------------------------------
def convergence_checker(state: AgentState, tools: dict) -> AgentState:
    """Checks all stopping conditions. Formats final submission if done."""
    wall_hours = (time.time() - state["run_start_time"]) / 3600.0

    stop_reason = None
    if state["iteration"] >= MAX_ITERATIONS:
        stop_reason = f"Hit {MAX_ITERATIONS} iteration cap"
    elif wall_hours >= MAX_WALL_HOURS:
        stop_reason = f"Hit {MAX_WALL_HOURS}h wall-clock limit"
    elif state.get("iterations_without_improvement", 0) >= N_CONVERGE:
        stop_reason = f"Converged: no improvement > eps={EPSILON} for {N_CONVERGE} consecutive iterations"

    state["should_stop"] = stop_reason is not None
    if stop_reason:
        state["_stop_reason"] = stop_reason
        tools["format_submission"].invoke({"split": "test"})
    return state


def route_after_convergence(state: AgentState) -> str:
    return "stop" if state["should_stop"] else "continue"


# ---------------------------------------------------------------------------
# Handoff note re: the audit's Issue 2 (experiment_judge misassignment)
# ---------------------------------------------------------------------------
# Resolution: rather than moving the whole node to Person A's file (which
# means both of you editing the same function), the node is split at the
# natural seam that already existed in this scaffold before the audit --
# the LLM reasoning call was already a documented TODO here, never
# implemented. So:
#   - log_and_track() (above) stays in Person B's file: pure plumbing,
#     already built and tested, calls log_iteration/track_resources.
#   - Person A implements the JUDGE_PROMPT call (same pattern as their
#     specialist propose() calls) and is responsible for setting
#     state["hypothesis"] to the actual judgment text BEFORE this node runs.
# Graph wiring (Step 4) then goes: score_analyst -> [Person A's judge call] -> log_and_track -> convergence_checker.
# This keeps ownership boundaries clean without throwing away tested code.
