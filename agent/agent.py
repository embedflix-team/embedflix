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
N_CONVERGE = 5
MAX_ITERATIONS = 50
MAX_WALL_HOURS = 6

IMPROVE_THRESHOLD = 0.0001

# Files a specialist is never allowed to touch, enforced here as defense in
# depth (mcp_server.edit_file has no allowlist of its own).
PROTECTED_FILES = {"evaluate.py"}

CODE_WRITER_MODEL = "claude-haiku-4-5-20251001"
CODE_WRITER_MAX_RETRIES = 3  # total attempts, not extra retries


# ---------------------------------------------------------------------------
def baseline_verifier(state: AgentState, tools: dict) -> AgentState:
    """Runs FM unmodified, confirms it matches published baseline (~0.6016
    valid primary), saves checkpoint 0, logs iteration 0, and seeds
    current_code for the first supervisor/specialist call."""
    output = tools["run_pipeline"]({})
    raw_scores = tools["parse_scores"]({"pipeline_output": output})

    if raw_scores.get("primary") is None:
        state["error_message"] = f"BASELINE FAILED TO REPRODUCE. Raw output tail:\n{output[-500:]}"
        state["should_stop"] = True
        return state

    scores = normalize_scores(raw_scores)
    tools["save_checkpoint"]({"iteration": 0, "primary_score": raw_scores["primary"]})
    tools["log_iteration"]({
        "iteration": 0, "hypothesis": "baseline verification (FM, unmodified)",
        "code_diff": "", "gauc": raw_scores.get("GAUC") or raw_scores.get("gauc"), "ndcg": raw_scores.get("nDCG@5") or raw_scores.get("ndcg5"),
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
    state["current_code"] = tools["read_file"]({"file_path": "baseline.py"})
    state["run_start_time"] = time.time()
    state["total_tokens"] = 0
    state["iterations_without_improvement"] = 0
    state["error_message"] = None
    return state


# ---------------------------------------------------------------------------
def code_writer(state: AgentState, tools: dict) -> AgentState:
    """Applies the current iteration's code change. Two paths:

    1. DETERMINISTIC (preferred whenever possible): a specialist that already
       knows the exact edit it wants (training_optimizer's fixed hyperparameter
       menu, model_swapper's k=16->32 option) sets state["_deterministic_edit"]
       = {"file", "old_code", "new_code"}. Applied directly via edit_file --
       zero LLM calls, zero translation risk. This is checked first.

    2. LLM-based (fallback for genuinely novel code -- new loss functions,
       new model architectures, anything that can't be a fixed template):
       specialists produce a free-form English instruction in
       state["code_change_instruction"] -- e.g. "Replace the log-loss
       computation in run_fm with a BPR pairwise loss..." -- NOT a structured
       FILE/OLD_CODE/NEW_CODE block. This node asks an LLM to turn that
       instruction + the actual current file contents into an exact, verbatim
       OLD_CODE/NEW_CODE pair, retries with corrective feedback if it doesn't
       match verbatim or doesn't parse, and applies it via edit_file.
    """
    det = state.get("_deterministic_edit")
    # Clear unconditionally, whether used or not -- otherwise a stale value
    # from this iteration could leak into a later iteration whose specialist
    # doesn't set one at all (same class of bug _raw_scores had).
    state["_deterministic_edit"] = None

    if det:
        file_path = det.get("file", "baseline.py")
        old_code = det.get("old_code", "")
        new_code = det.get("new_code", "")
        if file_path in PROTECTED_FILES:
            state["error_message"] = (
                f"code_writer: refused deterministic edit to protected file {file_path} "
                "(evaluate.py must never be modified)"
            )
            return state
        if not old_code or not new_code:
            state["error_message"] = (
                "code_writer: _deterministic_edit was set but missing old_code/new_code"
            )
            return state
        result = tools["edit_file"]({
            "file_path": file_path, "old_code": old_code, "new_code": new_code,
        })
        if not result.startswith("SUCCESS"):
            state["error_message"] = f"code_writer: deterministic edit_file failed -- {result}"
            return state
        state["code_diff"] = f"--- {file_path}\n- {old_code}\n+ {new_code}"
        state["error_message"] = None
        return state

    instruction = (state.get("code_change_instruction") or "").strip()
    # Always read fresh from disk — state current_code can be stale
    try:
        from mcp_server import STARTER_KIT
        import os
        fresh_path = os.path.join(STARTER_KIT, "baseline.py")
        with open(fresh_path) as f:
            current_code = f.read()
    except Exception:
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

        # Post-substitution syntax gate. edit_file also checks and reverts, but
        # by then this retry loop is over -- the edit gets silently dropped and
        # the iteration runs on unchanged code. Catch it here so the model gets
        # a corrective retry instead.
        new_code, _applied, syntax_err = _salvage_new_code(
            current_code, candidate["old_code"], candidate["new_code"]
        )
        if syntax_err is not None:
            prior_feedback = (
                f"Applying your edit makes baseline.py fail to parse: {syntax_err}. "
                "Return ONLY valid Python in NEW_CODE -- no ``` fences, no <CODE>/<NEW_CODE> tags, "
                "no second FILE:/OLD_CODE:/NEW_CODE: block, and nothing after the code."
            )
            parsed = None
            continue

        candidate["new_code"] = new_code
        parsed = candidate
        break

    state["total_tokens"] = state.get("total_tokens", 0) + tokens_used

    if parsed is None:
        state["error_message"] = (
            f"code_writer: could not derive an applicable edit from the instruction after "
            f"{CODE_WRITER_MAX_RETRIES} attempts -- instruction was: {instruction[:300]}"
        )
        return state

    result = tools["edit_file"]({
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
- Keep OLD_CODE as small as it can be while still uniquely identifying the edit location.
- CRITICAL: Keep NEW_CODE under 20 lines maximum. Simple targeted changes only.
- CRITICAL: If NEW_CODE adds a new parameter to any function, you MUST also update ALL callers of that function in the same edit. Never add a parameter that isn't passed through the full call chain.
- NEVER rewrite entire classes or functions. Change only what is necessary.
- NEVER add new classes. Only modify existing code.
- If the instruction requires more than 20 lines of new code, implement only the simplest version that still improves the metric.
- NEW_CODE is the replacement text implementing the instruction.
- Target file is baseline.py unless the instruction unambiguously names a different file.

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
Respond with EXACTLY three sections in this order and NOTHING ELSE:

FILE: baseline.py
OLD_CODE:
(the verbatim snippet copied from CURRENT FILE CONTENTS)
NEW_CODE:
(the replacement Python, and nothing after it)

Hard rules for the output:
- Put raw Python under OLD_CODE: and NEW_CODE:. No ``` fences, no <code>/<pre>
  tags, no XML, no placeholder text in angle brackets.
- After the last line of NEW_CODE, STOP. Do not add an explanation, a summary,
  a closing tag, a "Human:"/"Assistant:" line, or a second FILE/OLD_CODE/
  NEW_CODE block.
- NEW_CODE on its own must be syntactically valid Python at the same
  indentation level as OLD_CODE.
"""

    client = _get_client()
    response = client.messages.create(
        model=CODE_WRITER_MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
        stop_sequences=["\nHuman:", "\nAssistant:", "\n\nHuman:"],
    )
    text = response.content[0].text
    usage = getattr(response, "usage", None)
    tokens = (usage.input_tokens + usage.output_tokens) if usage else 0
    return text, tokens


# A leftover marker line inside a captured payload means the split failed.
_MARKER_LINE_RE = re.compile(r"^(?:FILE|OLD_CODE|NEW_CODE):", re.MULTILINE)
# Wrapper lines the model intermittently adds around the payload -- ``` fences
# and pseudo-tags like <CODE>..</CODE>, </NEW_CODE>, <python>. Written verbatim
# into a .py file every one of these is an instant SyntaxError.
_WRAPPER_LINE_RE = re.compile(
    r"^[ \t]*(?:```[a-zA-Z]*|</?(?:code|new_code|old_code|python|pre)>)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_code_wrappers(s: str) -> str:
    return _WRAPPER_LINE_RE.sub("", s).strip("\n")


def _salvage_new_code(current_code: str, old_code: str, new_code: str):
    """Apply new_code and confirm the result parses. code_writer's LLM
    (haiku) intermittently tacks prose, a 'Human:' turn, or a stray closing
    tag onto the end of NEW_CODE; that trailing junk is always *after* the
    real code, so drop trailing lines one at a time until the file parses.
    Returns (effective_new_code, applied_source, error_str_or_None)."""
    def _apply(nc: str) -> str:
        return current_code.replace(old_code, nc, 1)

    try:
        applied = _apply(new_code)
        compile(applied, "baseline.py", "exec")
        return new_code, applied, None
    except SyntaxError as first_err:
        lines = new_code.split("\n")
        # Only ever trim trailing lines -- never touch the body.
        for cut in range(1, min(len(lines), 60)):
            trimmed = "\n".join(lines[:-cut]).rstrip()
            if not trimmed:
                break
            applied = _apply(trimmed)
            try:
                compile(applied, "baseline.py", "exec")
                return trimmed, applied, None
            except SyntaxError:
                continue
        return new_code, _apply(new_code), f"{first_err.msg} at line {first_err.lineno}"


def _parse_file_old_new_block(raw: str):
    m_file = re.search(r"^FILE:\s*(\S+)", raw, re.MULTILINE)
    m_old = re.search(r"^OLD_CODE:\s*\n?(.+?)(?=^NEW_CODE:)", raw, re.MULTILINE | re.DOTALL)
    # Stop NEW_CODE at the next marker line, not \Z: a second FILE:/OLD_CODE:/
    # NEW_CODE: block from a chatty model would otherwise be pulled into
    # new_code verbatim -> SyntaxError on write -> edit_file silently reverts,
    # so baseline.py never changes and the run no-ops for every iteration.
    m_new = re.search(r"^NEW_CODE:\s*\n?(.+?)(?=^(?:FILE|OLD_CODE|NEW_CODE):|\Z)",
                      raw, re.MULTILINE | re.DOTALL)
    if not (m_file and m_old and m_new):
        return None
    old_code = _strip_code_wrappers(m_old.group(1).strip("\n"))
    new_code = _strip_code_wrappers(m_new.group(1).strip("\n"))
    # If a marker line still survives inside either payload the split is wrong;
    # bail so code_writer retries with corrective feedback instead of applying
    # something that can't parse.
    if _MARKER_LINE_RE.search(old_code) or _MARKER_LINE_RE.search(new_code):
        return None
    if not old_code or not new_code:
        return None
    return {"file": m_file.group(1).strip(), "old_code": old_code, "new_code": new_code}


# ---------------------------------------------------------------------------
def pipeline_runner(state: AgentState, tools: dict) -> AgentState:
    """Runs the modified pipeline. A crash or timeout inside run_pipeline
    (subprocess-isolated, 300s cap set in mcp_server.py) surfaces as an
    error_message here rather than raising.

    code_writer sets error_message and returns early when it can never derive
    or apply a valid edit (OLD_CODE never matched verbatim after retries, a
    protected-file refusal, or edit_file itself failed/reverted a bad edit).
    This node used to run anyway -- since baseline.py is unchanged in that
    case, the pipeline silently re-scored the OLD code as if it were a real
    experiment, and success here reset error_message to None, wiping out the
    fact that no edit had actually applied. That made a completely-unapplied
    edit indistinguishable in the log from 'applied but did not help' --
    confirmed 2026-08-31 from a real run where sequence_modeller's edit never
    landed and the iteration silently logged the exact baseline score with an
    empty code_diff and no visible error.

    Fix: check for an already-set error_message FIRST and skip the run
    entirely -- no point spending ~90s training on code that never changed.
    Keep the failure visible (prefixed SKIPPED) so route_after_pipeline_runner
    sends it to error_recovery, same as a real pipeline crash would.
    """
    if state.get("error_message"):
        state["current_scores"] = {}
        state["_raw_scores"] = {}  # clear -- do not let a prior iteration's real scores leak into this one's log
        state["error_message"] = f"SKIPPED (no edit applied): {state['error_message']}"
        return state

    t0 = time.time()
    try:
        output = tools["run_pipeline"]({})
        raw_scores = tools["parse_scores"]({"pipeline_output": output})
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
        tools["restore_checkpoint"]({"iteration": state["best_iteration"]})
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
        tools["restore_checkpoint"]({"iteration": state["best_iteration"]})
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
        tools["save_checkpoint"]({"iteration": iteration, "primary_score": scores.get("primary", 0.0)})
        state["best_scores"] = scores
        state["best_iteration"] = iteration
        state["iterations_without_improvement"] = 0
    else:
        tools["restore_checkpoint"]({"iteration": state["best_iteration"]})
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
    state["current_code"] = tools["read_file"]({"file_path": "baseline.py"})
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

    tools["log_iteration"]({
    "iteration": state["iteration"],
    "hypothesis": logged_hypothesis,
    "code_diff": state.get("code_diff", ""),
    "gauc": raw_scores.get("GAUC") or raw_scores.get("gauc"),
    "ndcg": raw_scores.get("nDCG@5") or raw_scores.get("ndcg5"),
    "primary": raw_scores.get("primary"),
    "error": state.get("error_message") or "",
    "recovery": state.get("recovery_action", ""),
    })
    wall = time.time() - state["run_start_time"]
    tools["track_resources"]({
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
        tools["format_submission"]({"split": "test"})
    return state


def route_after_convergence(state: AgentState) -> str:
    return "stop" if state["should_stop"] else "continue"
