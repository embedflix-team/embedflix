"""Person B's nodes: baseline_verifier, code_writer, pipeline_runner,
error_recovery, score_analyst, log_and_track, convergence_checker.

Reconciled (2026-08-28) against Person A's ACTUAL committed specialist code
(agent/specialists/*.py, read directly -- not a paraphrased spec) --
see state.py's module docstring for the field-naming and score-key notes
before touching current_scores/best_scores/experiment_history anywhere.

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
import os
import re
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

CODE_WRITER_MODEL = "claude-sonnet-4-6"
CODE_WRITER_MAX_RETRIES = 2  # total attempts, not extra retries


# ---------------------------------------------------------------------------
def baseline_verifier(state: AgentState, tools: dict) -> AgentState:
    """Runs FM unmodified, confirms it matches published baseline (~0.6016
    valid primary), saves checkpoint 0, logs iteration 0, and seeds
    current_code for the first supervisor/specialist call."""
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
    state["experiment_history"] = [{
        "iteration": 0, "specialist": "baseline", "hypothesis": "baseline (unmodified FM)",
        "gauc": scores["gauc"], "ndcg5": scores["ndcg5"], "primary": scores["primary"],
        "improved": True,
    }]
    state["tried_approaches"] = []
    state["current_code"] = tools["read_file"].invoke({"file_path": "baseline.py"})
    state["run_start_time"] = time.time()
    state["total_tokens"] = 0
    state["iterations_without_improvement"] = 0
    state["error_message"] = None
    return state


# ---------------------------------------------------------------------------
def code_writer(state: AgentState, tools: dict) -> AgentState:
    """LLM-based editor. Specialists (loss_function_changer, sequence_modeller,
    etc.) produce a free-form English instruction in
    state["code_change_instruction"] -- e.g. "Replace the log-loss computation
    in run_fm with a BPR pairwise loss: sample one negative per positive,
    compute sigmoid(pos_score - neg_score)..." -- NOT a structured
    FILE/OLD_CODE/NEW_CODE block.

    This node asks an LLM to turn that instruction + the actual current file
    contents into an exact, verbatim OLD_CODE/NEW_CODE pair (OLD_CODE must be
    a literal substring of current_code so tools["edit_file"] can apply it),
    retries once with corrective feedback if the proposed OLD_CODE doesn't
    match verbatim, and applies the edit via edit_file.
    """
    instruction = (state.get("code_change_instruction") or "").strip()
    current_code = state.get("current_code") or ""

    if not instruction:
        state["error_message"] = "code_writer: code_change_instruction is empty -- nothing to implement"
        return state
    if not current_code:
        state["error_message"] = "code_writer: current_code is empty -- can't ground an edit against nothing"
        return state

    prior_feedback = None
    parsed = None
    tokens_used = 0

    for attempt in range(CODE_WRITER_MAX_RETRIES):
        text, usage = _propose_edit(current_code, instruction, prior_feedback)
        tokens_used += usage
        candidate = _parse_file_old_new_block(text)

        if candidate is None:
            prior_feedback = (
                "Your last response didn't match the required FILE: / OLD_CODE: / NEW_CODE: "
                "format exactly. Respond with EXACTLY those three fields, nothing else."
            )
            continue

        if candidate["file"] in PROTECTED_FILES:
            state["total_tokens"] = state.get("total_tokens", 0) + tokens_used
            state["error_message"] = (
                f"code_writer: refused edit to protected file {candidate['file']} "
                "(evaluate.py must never be modified)"
            )
            return state

        if candidate["old_code"] not in current_code:
            prior_feedback = (
                "Your OLD_CODE did not appear verbatim in the file contents shown to you. "
                "Copy the exact snippet character-for-character from CURRENT FILE CONTENTS "
                "(matching whitespace and indentation), do not paraphrase or reformat it.\n\n"
                f"OLD_CODE you sent:\n{candidate['old_code']}"
            )
            parsed = None
            continue

        parsed = candidate
        break

    state["total_tokens"] = state.get("total_tokens", 0) + tokens_used

    if parsed is None:
        state["error_message"] = (
            f"code_writer: could not derive an applicable edit from the instruction after "
            f"{CODE_WRITER_MAX_RETRIES} attempts -- instruction was: {instruction[:300]}"
        )
        return state

    result = tools["edit_file"].invoke({
        "file_path": parsed["file"], "old_code": parsed["old_code"], "new_code": parsed["new_code"],
    })
    if not result.startswith("SUCCESS"):
        state["error_message"] = f"code_writer: edit_file failed -- {result}"
        return state

    state["code_diff"] = f"--- {parsed['file']}\n- {parsed['old_code']}\n+ {parsed['new_code']}"
    state["error_message"] = None
    return state


_code_writer_client = None


def _get_client():
    global _code_writer_client
    if _code_writer_client is None:
        from anthropic import Anthropic
        _code_writer_client = Anthropic()
    return _code_writer_client


def _propose_edit(current_code: str, instruction: str, prior_feedback: str = None):
    """One LLM call: (instruction, current file) -> FILE/OLD_CODE/NEW_CODE text.
    Returns (response_text, tokens_used)."""
    prompt = f"""You are a precise code-editing assistant for an autonomous ML research agent.
You will be given the full current contents of baseline.py and a natural-language
instruction from an ML specialist describing a code change to make.

Produce an EXACT, surgical edit as a FILE / OLD_CODE / NEW_CODE block that can be
mechanically applied via verbatim string replacement.
- OLD_CODE must be an exact substring of CURRENT FILE CONTENTS below -- copy it
  character-for-character (matching whitespace and indentation exactly). Do not
  paraphrase, reformat, or reindent it.
- Keep OLD_CODE as small as it can be while still uniquely identifying the edit
  location and fully covering what needs to change.
- NEW_CODE is the replacement text implementing the instruction.
- Target file is baseline.py unless the instruction unambiguously names a
  different file.

CURRENT FILE CONTENTS (baseline.py):
```python
{current_code}
```

INSTRUCTION FROM SPECIALIST:
{instruction}
"""
    if prior_feedback:
        prompt += f"\nYOUR PREVIOUS ATTEMPT WAS REJECTED:\n{prior_feedback}\n"

    prompt += """
Respond in EXACTLY this format and nothing else:
FILE: baseline.py
OLD_CODE:
<exact verbatim snippet from CURRENT FILE CONTENTS above>
NEW_CODE:
<replacement code>
"""

    client = _get_client()
    response = client.messages.create(
        model=CODE_WRITER_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text
    usage = getattr(response, "usage", None)
    tokens = (usage.input_tokens + usage.output_tokens) if usage else 0
    return text, tokens


def _parse_file_old_new_block(raw: str):
    m_file = re.search(r"^FILE:\s*(\S+)", raw, re.MULTILINE)
    m_old = re.search(r"^OLD_CODE:\s*\n?(.+?)(?=^NEW_CODE:)", raw, re.MULTILINE | re.DOTALL)
    m_new = re.search(r"^NEW_CODE:\s*\n?(.+?)\Z", raw, re.MULTILINE | re.DOTALL)
    if not (m_file and m_old and m_new):
        return None
    old_code = m_old.group(1).strip("\n")
    new_code = m_new.group(1).strip("\n")
    # Strip a stray ``` fence if the model wrapped its snippet in one.
    old_code = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", old_code).strip("\n")
    new_code = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", new_code).strip("\n")
    if not old_code or not new_code:
        return None
    return {"file": m_file.group(1).strip(), "old_code": old_code, "new_code": new_code}


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
    # matched here escalates straight to a restore.
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
    checkpoint accordingly, refreshes current_code, and appends the flat
    experiment_history entry every specialist's _summarize_history expects:
    {iteration, specialist, hypothesis, gauc, ndcg5, primary, improved}.

    Owns experiment_history and checkpointing outright -- experiment_judge.py
    (as pushed) still appends its own entry and checkpoints too, but per
    Person A's confirmation the rewritten reasoning-only version drops both,
    so this stays the single source of truth for both."""
    scores = state["current_scores"]
    raw_scores = state.get("_raw_scores", {})
    best = state["best_scores"]
    iteration = state["iteration"] + 1
    state["iteration"] = iteration

    improved = (scores.get("primary") or -1) > (best.get("primary") or -1) + IMPROVE_THRESHOLD

    if improved:
        tools["save_checkpoint"].invoke({"iteration": iteration, "primary_score": scores.get("primary", 0.0)})
        state["best_scores"] = scores
        state["best_iteration"] = iteration
        state["iterations_without_improvement"] = 0
    else:
        tools["restore_checkpoint"].invoke({"iteration": state["best_iteration"]})
        state["iterations_without_improvement"] = state.get("iterations_without_improvement", 0) + 1

    specialist = state.get("next_specialist", "unknown")
    state["experiment_history"] = state.get("experiment_history", []) + [{
        "iteration": iteration, "specialist": specialist, "hypothesis": state.get("hypothesis", ""),
        "gauc": scores.get("gauc"), "ndcg5": scores.get("ndcg5"), "primary": scores.get("primary"),
        "improved": improved,
    }]
    state["tried_approaches"] = state.get("tried_approaches", [])  # specialists append their own label already
    # Refresh AFTER the accept/reject restore above, so this always reflects
    # what's actually on disk (the new code if accepted, the rolled-back
    # best-known code if rejected).
    state["current_code"] = tools["read_file"].invoke({"file_path": "baseline.py"})
    state["_improved"] = improved
    return state


# ---------------------------------------------------------------------------
def log_and_track(state: AgentState, tools: dict) -> AgentState:
    """The deterministic half of what was 'experiment_judge': persists
    log_iteration + track_resources. Folds in whatever reasoning/verdict/
    analysis/learning/next_priority is already in state -- set by supervisor's
    "reasoning" field and, once the rewritten reasoning-only judge is wired
    in ahead of this node, by verdict/analysis/learning/next_priority too.

    GAP: mcp_server.log_iteration has no dedicated params for those fields --
    folding them into the hypothesis text sent to log_iteration so they
    aren't silently dropped from the run-log deliverable."""
    raw_scores = state.get("_raw_scores", {})
    logged_hypothesis = state.get("hypothesis", "")

    extra_bits = []
    if state.get("reasoning"):
        extra_bits.append(f"REASONING: {state['reasoning']}")
    if state.get("verdict"):
        extra_bits.append(f"VERDICT: {state['verdict']}")
    if state.get("analysis"):
        extra_bits.append(f"ANALYSIS: {state['analysis']}")
    if state.get("learning"):
        extra_bits.append(f"LEARNING: {state['learning']}")
    if state.get("next_priority"):
        extra_bits.append(f"NEXT_PRIORITY: {state['next_priority']}")
    if extra_bits:
        logged_hypothesis = logged_hypothesis + "\n\n" + "\n".join(extra_bits)

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
    state["verdict"] = ""
    state["analysis"] = ""
    state["learning"] = ""
    state["next_priority"] = ""
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
