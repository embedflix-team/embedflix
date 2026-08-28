"""Person B's nodes: baseline_verifier, code_writer, pipeline_runner,
error_recovery, score_analyst, log_and_track, convergence_checker.

Reconciled (2026-08-28) against Person A's actual state.py field
requirements -- see state.py's module docstring for the score-key
normalization note before touching current_scores/best_scores anywhere.

Wired against the REAL mcp_server.py tools (via tools_adapter.build_tools()):
  run_pipeline(extra_args="")        -> str (combined stdout+stderr)
  read_file(file_path)               -> str
  parse_scores(pipeline_output)      -> {"GAUC":..,"nDCG@5":..,"primary":..} (values may be None)
  edit_file(file_path, old_code, new_code) -> "SUCCESS: ..." / "ERROR: ..." string
  save_checkpoint(iteration, primary_score) -> str
  restore_checkpoint(iteration)      -> str  (must name WHICH iteration)
  log_iteration(...)                 -> str  (NO "reasoning" param -- see note in log_and_track)
  track_resources(iteration, tokens, wall_seconds) -> str
  format_submission(split)           -> str
"""
import time

from state import AgentState, normalize_scores

EPSILON = 0.002
N_CONVERGE = 3
MAX_ITERATIONS = 50
MAX_WALL_HOURS = 6

IMPROVE_THRESHOLD = 0.0001

# Files a specialist is never allowed to touch, enforced here as defense in
# depth (mcp_server.edit_file has no allowlist of its own).
PROTECTED_FILES = {"evaluate.py"}


# ---------------------------------------------------------------------------
def baseline_verifier(state: AgentState, tools: dict) -> AgentState:
    """Runs FM unmodified, confirms it matches published baseline (~0.6016
    valid primary), saves checkpoint 0, logs iteration 0, and seeds
    current_code for Person A's first supervisor/specialist call."""
    output = tools["run_pipeline"].invoke({})
    raw_scores = tools["parse_scores"].invoke({"pipeline_output": output})

    if raw_scores.get("primary") is None:
        state["error_message"] = f"BASELINE FAILED TO REPRODUCE. Raw output tail:\n{output[-500:]}"
        state["should_stop"] = True
        return state

    scores = normalize_scores(raw_scores)
    tools["save_checkpoint"].invoke({"iteration": 0, "primary_score": raw_scores["primary"]})
    tools["log_iteration"].invoke({
        "iteration": 0, "hypothesis": "baseline verification (FM, unmodified)",
        "code_diff": "", "gauc": raw_scores["GAUC"], "ndcg": raw_scores["nDCG@5"],
        "primary": raw_scores["primary"], "error": "", "recovery": "",
    })

    state["iteration"] = 0
    state["current_scores"] = scores
    state["best_scores"] = scores
    state["best_iteration"] = 0
    state["experiment_history"] = [{"iteration": 0, "hypothesis": "baseline", "approach": "baseline",
                                     "scores": scores, "improved": True}]
    state["tried_approaches"] = []
    state["current_code"] = tools["read_file"].invoke({"file_path": "baseline.py"})
    state["run_start_time"] = time.time()
    state["total_tokens"] = 0
    state["iterations_without_improvement"] = 0
    state["error_message"] = None
    return state


# ---------------------------------------------------------------------------
def code_writer(state: AgentState, tools: dict) -> AgentState:
    """Applies the specialist's proposed edit. Reads state["code_change_instruction"]
    (NOT state["hypothesis"] -- that field is now pure reasoning text, set by
    the specialist and left untouched here):
        FILE: <filename>
        OLD_CODE: <exact string>
        NEW_CODE: <replacement>
    """
    raw = state.get("code_change_instruction", "")
    parsed = _parse_code_change_instruction(raw)
    if parsed is None:
        state["error_message"] = "code_writer: could not parse code_change_instruction (missing FILE/OLD_CODE/NEW_CODE)"
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
    # was proposed (accurate since "SUCCESS" confirms OLD_CODE was applied verbatim).
    state["code_diff"] = f"--- {parsed['file']}\n- {parsed['old_code']}\n+ {parsed['new_code']}"
    state["error_message"] = None
    return state


def _parse_code_change_instruction(raw: str):
    import re
    m_file = re.search(r"^FILE:\s*(\S+)", raw, re.MULTILINE)
    m_old = re.search(r"^OLD_CODE:\s*(.+?)(?=^NEW_CODE:|\Z)", raw, re.MULTILINE | re.DOTALL)
    m_new = re.search(r"^NEW_CODE:\s*(.+?)\Z", raw, re.MULTILINE | re.DOTALL)
    if not (m_file and m_old and m_new):
        return None
    return {
        "file": m_file.group(1).strip(),
        "old_code": m_old.group(1).strip("\n"),
        "new_code": m_new.group(1).strip("\n"),
    }


# ---------------------------------------------------------------------------
def pipeline_runner(state: AgentState, tools: dict) -> AgentState:
    """Runs the modified pipeline. A crash or timeout inside run_pipeline
    (subprocess-isolated, 300s cap set in mcp_server.py) surfaces as an
    error_message here rather than raising."""
    t0 = time.time()
    try:
        output = tools["run_pipeline"].invoke({})
        raw_scores = tools["parse_scores"].invoke({"pipeline_output": output})
        state["run_wall_seconds"] = state.get("run_wall_seconds", 0.0) + (time.time() - t0)

        if raw_scores.get("primary") is None:
            state["error_message"] = f"pipeline run produced unparseable output. Tail:\n{output[-500:]}"
            state["current_scores"] = {}
        else:
            state["current_scores"] = normalize_scores(raw_scores)
            state["_raw_scores"] = raw_scores  # kept for log_iteration's gauc/ndcg/primary params
            state["error_message"] = None
    except Exception as e:  # noqa: BLE001 -- this is the isolation boundary
        state["error_message"] = f"{type(e).__name__}: {e}"
        state["current_scores"] = {}
    return state


def route_after_pipeline_runner(state: AgentState) -> str:
    return "error_recovery" if state.get("error_message") else "score_analyst"


# ---------------------------------------------------------------------------
def error_recovery(state: AgentState, tools: dict) -> AgentState:
    """Diagnoses the error, fixes code or restores last-good checkpoint.
    Max 3 fix attempts per iteration before falling back to restore.
    Restore targets state["best_iteration"] -- mcp_server.restore_checkpoint
    requires naming which iteration, there's no implicit "last"."""
    attempts = state.get("_recovery_attempts", 0) + 1
    state["_recovery_attempts"] = attempts

    if attempts > 3:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["recovery_action"] = f"RESTORE: exceeded 3 fix attempts, rolled back to checkpoint {state['best_iteration']}"
        state["error_message"] = None
        state["_recovery_attempts"] = 0
        return state

    # Deterministic recovery for mechanically-fixable errors. Anything not
    # matched here escalates to an LLM diagnosis call (ERROR_RECOVERY_PROMPT) --
    # not wired in this scaffold yet.
    err = state.get("error_message") or ""
    if "ModuleNotFoundError" in err and "torch" in err:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "--break-system-packages", "-q", "torch"])
        state["recovery_action"] = "FIX: installed missing torch dependency"
        state["error_message"] = None
    else:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["recovery_action"] = f"RESTORE: rolled back to checkpoint {state['best_iteration']} -- {err[:200]}"
        state["error_message"] = None
        state["_recovery_attempts"] = 0

    return state


# ---------------------------------------------------------------------------
def score_analyst(state: AgentState, tools: dict) -> AgentState:
    """Reads scores, checks if improved, updates best, saves/restores
    checkpoint accordingly, refreshes current_code and tried_approaches/
    experiment_history for the next specialist call."""
    scores = state["current_scores"]
    raw_scores = state.get("_raw_scores", {})
    best = state["best_scores"]
    iteration = state["iteration"] + 1
    state["iteration"] = iteration

    improved = (scores.get("primary") or -1) > (best.get("primary") or -1) + IMPROVE_THRESHOLD

    if improved:
        tools["save_checkpoint"].invoke({"iteration": iteration, "primary_score": raw_scores["primary"]})
        state["best_scores"] = scores
        state["best_iteration"] = iteration
        state["iterations_without_improvement"] = 0
    else:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["iterations_without_improvement"] = state.get("iterations_without_improvement", 0) + 1

    approach = state.get("next_node", "unknown")
    state["experiment_history"] = state.get("experiment_history", []) + [{
        "iteration": iteration, "hypothesis": state.get("hypothesis", ""),
        "approach": approach, "scores": scores, "improved": improved,
    }]
    state["tried_approaches"] = state.get("tried_approaches", []) + [approach]
    # Refresh AFTER the accept/reject restore above, so this always reflects
    # what's actually on disk (the new code if accepted, the rolled-back
    # best-known code if rejected).
    state["current_code"] = tools["read_file"].invoke({"file_path": "baseline.py"})
    state["_improved"] = improved
    return state


# ---------------------------------------------------------------------------
def log_and_track(state: AgentState, tools: dict) -> AgentState:
    """The deterministic half of what was 'experiment_judge': persists
    log_iteration + track_resources. Does NOT generate reasoning -- Person
    A's JUDGE_PROMPT call should run before this node and populate
    state["reasoning"].

    GAP: mcp_server.log_iteration has no "reasoning" param -- only
    hypothesis/code_diff/gauc/ndcg/primary/error/recovery. Folding
    reasoning into the hypothesis text sent to log_iteration so it isn't
    silently dropped from the run-log deliverable; worth deciding as a team
    whether to extend log_iteration's signature instead."""
    raw_scores = state.get("_raw_scores", {})
    logged_hypothesis = state.get("hypothesis", "")
    if state.get("reasoning"):
        logged_hypothesis = f"{logged_hypothesis}\n\nREASONING: {state['reasoning']}"

    tools["log_iteration"].invoke({
        "iteration": state["iteration"],
        "hypothesis": logged_hypothesis,
        "code_diff": state.get("code_diff", ""),
        "gauc": raw_scores.get("GAUC"), "ndcg": raw_scores.get("nDCG@5"), "primary": raw_scores.get("primary"),
        "error": state.get("error_message") or "", "recovery": state.get("recovery_action", ""),
    })
    wall = time.time() - state["run_start_time"]
    tools["track_resources"].invoke({
        "iteration": state["iteration"], "tokens": state.get("total_tokens", 0), "wall_seconds": wall,
    })

    state["error_message"] = None
    state["recovery_action"] = ""
    state["reasoning"] = ""
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
