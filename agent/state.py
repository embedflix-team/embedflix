"""AgentState: the single object every LangGraph node reads and writes.

Reconciled against Person A's ACTUAL committed specialist code (2026-08-28,
read directly from agent/specialists/*.py rather than a paraphrased spec --
prior drift between spec versions is why). This supersedes both earlier
drafts.

Field names below match what supervisor.py / loss_function_changer.py /
sequence_modeller.py / multitask_trainer.py / model_swapper.py /
training_optimizer.py actually read and write. Where that differs from what
the real mcp_server.py tools return, the mismatch is normalized inside
Person B's nodes (agent.py), not here -- see NOTE below.

NOTE on score key casing: mcp_server.parse_scores() returns
{"GAUC": .., "nDCG@5": .., "primary": ..}. Every specialist reads
state["current_scores"]/state["best_scores"] as {"gauc":.., "ndcg5":..,
"primary":..}. agent.py's pipeline_runner/baseline_verifier/score_analyst
normalize on the way in -- by the time a node reads state["current_scores"],
it's always in the gauc/ndcg5/primary shape. Don't bypass that
normalization by reading parse_scores' output directly into state elsewhere.

NOTE on experiment_history shape: every specialist's _summarize_history
reads h.get('iteration'), h.get('specialist'), h.get('hypothesis'),
h.get('primary') from a FLAT dict -- not the old nested {"scores": {...}}
shape. score_analyst builds entries in that flat shape. experiment_judge.py
(as pushed) still builds its own richer entry and appends to history itself
-- per Person A's confirmation ("reasoning-only, no tool calls, no
checkpointing, returns only verdict/analysis/learning/next_priority") that
file is due for a rewrite that drops the append/checkpoint side effects.
Until the rewritten version lands, judge stays stubbed in graph.py and
history stays exclusively owned by score_analyst -- see agent.py.

NOTE on next_specialist: supervisor.py sets state["next_specialist"] (not
next_node -- that was an earlier, wrong field name from an older spec).
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    # --- Loop control ---
    iteration: int
    should_stop: bool
    next_specialist: str               # supervisor's routing decision
    run_start_time: float

    # --- Current iteration proposal (Person A's nodes write, Person B reads) ---
    hypothesis: str                    # human-readable reasoning, e.g. "BPR loss will improve ranking alignment"
    code_change_instruction: str       # free-form English instruction for code_writer (LLM-parsed, not regex-parsed)
    reasoning: str                     # short reasoning string, set by supervisor and each specialist

    # --- Supervisor output (routing context, for logs/judges) ---
    routing_reason: str
    strategy: str

    # --- Judge output (once experiment_judge.py is trimmed to reasoning-only) ---
    verdict: str                       # "improved" | "no_change" | "regression"
    analysis: str
    learning: str
    next_priority: str

    # --- Current iteration execution (Person B writes) ---
    code_diff: str                     # computed by code_writer after a successful edit
    error_message: Optional[str]       # None when clean
    recovery_action: str

    # --- Scores (gauc/ndcg5/primary keys -- see module docstring) ---
    current_scores: Dict[str, Any]
    best_scores: Dict[str, Any]
    best_iteration: int

    # --- History / context for specialists ---
    experiment_history: List[Dict[str, Any]]   # [{iteration, specialist, hypothesis, gauc, ndcg5, primary, improved}, ...]
    tried_approaches: List[str]                # short labels, e.g. ["loss:bpr", "sequence:attention"]
    current_code: str                          # current baseline.py content, refreshed every iteration

    # --- Resources ---
    total_tokens: int
    run_wall_seconds: float
    iterations_without_improvement: int


def initial_state() -> AgentState:
    import time
    return AgentState(
        iteration=0,
        should_stop=False,
        next_specialist="",
        run_start_time=time.time(),
        hypothesis="",
        code_change_instruction="",
        reasoning="",
        routing_reason="",
        strategy="",
        verdict="",
        analysis="",
        learning="",
        next_priority="",
        code_diff="",
        error_message=None,
        recovery_action="",
        current_scores={},
        best_scores={"primary": -1.0},
        best_iteration=-1,
        experiment_history=[],
        tried_approaches=[],
        current_code="",
        total_tokens=0,
        run_wall_seconds=0.0,
        iterations_without_improvement=0,
    )


def normalize_scores(raw: dict) -> dict:
    """mcp_server.parse_scores() keys -> the gauc/ndcg5/primary keys every
    specialist reads. raw values may be None (parse failure) -- passed
    through unchanged."""
    return {
        "gauc": raw.get("GAUC"),
        "ndcg5": raw.get("nDCG@5"),
        "primary": raw.get("primary"),
    }
