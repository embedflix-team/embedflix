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
(2026-08-28 rewrite, commit 9e85ac5) is reasoning-only -- no tool calls, no
checkpointing, no experiment_history append -- and is wired into graph.py
for real as of the "Wire real experiment_judge into graph.py" commit.
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

    # --- Internal bookkeeping, passed between separate node CALLS (not just
    # read within one node), so these MUST be declared TypedDict keys.
    # LangGraph's Pregel executor only creates a channel -- and therefore only
    # actually propagates a value between nodes -- for keys present in this
    # schema. An undeclared key set with state["_x"] = ... inside a node is
    # visible to code running later IN THE SAME node call (it's just a dict),
    # but is silently dropped when that node returns and the next node reads
    # state.get("_x", ...) -- confirmed 2026-08-29 from a real run_log.jsonl:
    # log_and_track's raw_scores (set two nodes earlier, in pipeline_runner)
    # came back {} there, logging null gauc/ndcg/primary even though the
    # pipeline had computed real scores and score_analyst (one node closer)
    # had already used them correctly.
    _raw_scores: Dict[str, Any]        # set by pipeline_runner, read by score_analyst + log_and_track
    _improved: bool                    # set by score_analyst
    _recovery_attempts: int            # read+written across separate error_recovery calls
    _stop_reason: Optional[str]        # set by convergence_checker

    # Web-search telemetry: each two-phase specialist writes what it searched
    # (phase 1 = concept discovery, phase 2 = code blueprint) so otel_tracer's
    # on_chain_end can log it. Declared here or Pregel drops them before the
    # tracer sees the node's return -- see the note above.
    _phase1_results: str
    _phase1_query: str
    _phase2_results: str
    _phase2_query: str
    # Optional per-node overrides for the span's search.phase label. Empty ->
    # otel_tracer falls back to "concept_discovery" / "code_blueprint". The
    # judge sets _phase2_label="specialist_context" because its phase-2 search
    # is another concept lookup, not a code-blueprint search.
    _phase1_label: str
    _phase2_label: str


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
        tried_approaches=["loss:bpr", "loss:softmax", "loss:focal", "loss:warp"],
        current_code="",
        total_tokens=0,
        run_wall_seconds=0.0,
        iterations_without_improvement=0,
        _raw_scores={},
        _improved=False,
        _recovery_attempts=0,
        _stop_reason=None,
        _phase1_results="",
        _phase1_query="",
        _phase2_results="",
        _phase2_query="",
        _phase1_label="",
        _phase2_label="",
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
