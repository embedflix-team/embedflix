"""AgentState: the single object every LangGraph node reads and writes.

Field groups mirror the build plan doc exactly (Day 1, Step 2).
"""
from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict):
    # --- Loop control ---
    iteration: int
    should_stop: bool
    next_specialist: str
    run_start_time: float

    # --- Current iteration ---
    hypothesis: str
    code_diff: str
    error_message: str
    recovery_action: str

    # --- Scores ---
    current_score: Dict[str, Any]      # e.g. {"GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.6015}
    best_score: Dict[str, Any]
    best_iteration: int
    score_history: List[Dict[str, Any]]

    # --- Registry & resources ---
    registry: List[Dict[str, Any]]     # full log of every iteration (mirrors runs/<id>/iterations/*.json)
    total_tokens: int
    run_wall_seconds: float

    # --- Convergence ---
    iterations_without_improvement: int


def initial_state() -> AgentState:
    """A freshly-initialized state, ready for baseline_verifier."""
    import time
    return AgentState(
        iteration=0,
        should_stop=False,
        next_specialist="",
        run_start_time=time.time(),
        hypothesis="",
        code_diff="",
        error_message="",
        recovery_action="",
        current_score={},
        best_score={"primary": -1.0},
        best_iteration=-1,
        score_history=[],
        registry=[],
        total_tokens=0,
        run_wall_seconds=0.0,
        iterations_without_improvement=0,
    )
