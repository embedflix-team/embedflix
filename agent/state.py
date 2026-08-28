"""AgentState: the single object every LangGraph node reads and writes.

Reconciled against Person A's actual field requirements (2026-08-28) --
this supersedes the first draft. Field names below match Person A's spec
exactly; where that spec's naming differs from what the real mcp_server.py
tools return, the mismatch is normalized inside Person B's nodes (agent.py),
not here -- see NOTE below.

NOTE on score key casing: mcp_server.parse_scores() returns
{"GAUC": .., "nDCG@5": .., "primary": ..}. Person A's spec wants
{"gauc": .., "ndcg5": .., "primary": ..} inside current_scores/best_scores.
agent.py's pipeline_runner/baseline_verifier/score_analyst normalize on the
way in -- by the time a node reads state["current_scores"], it's always in
the gauc/ndcg5/primary shape. Don't bypass that normalization by reading
parse_scores' output directly into state elsewhere.
"""
from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    # --- Loop control ---
    iteration: int
    should_stop: bool
    next_node: str                    # supervisor's routing decision
    run_start_time: float

    # --- Current iteration proposal (Person A writes, Person B reads) ---
    hypothesis: str                   # human-readable reasoning, e.g. "BPR loss will improve ranking alignment"
    code_change_instruction: str      # FILE: / OLD_CODE: / NEW_CODE: block for code_writer to parse
    reasoning: str                    # post-hoc judge reasoning (JUDGE_PROMPT output) -- for logs/judges

    # --- Current iteration execution (Person B writes) ---
    code_diff: str                    # computed by code_writer after a successful edit
    error_message: Optional[str]      # None when clean -- was "" in the first draft, changed to match spec
    recovery_action: str

    # --- Scores (gauc/ndcg5/primary keys -- see module docstring) ---
    current_scores: Dict[str, Any]
    best_scores: Dict[str, Any]
    best_iteration: int

    # --- History / context for specialists ---
    experiment_history: List[Dict[str, Any]]   # [{iteration, hypothesis, approach, scores, improved}, ...]
    tried_approaches: List[str]                # short labels, e.g. ["loss_function_changer", "sequence_modeller"]
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
        next_node="",
        run_start_time=time.time(),
        hypothesis="",
        code_change_instruction="",
        reasoning="",
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
    """mcp_server.parse_scores() keys -> Person A's expected keys.
    raw values may be None (parse failure) -- passed through unchanged."""
    return {
        "gauc": raw.get("GAUC"),
        "ndcg5": raw.get("nDCG@5"),
        "primary": raw.get("primary"),
    }
