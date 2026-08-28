"""Step 4: wire the LangGraph. Real supervisor + 5 specialists are wired in
from agent/specialists/*.py. `judge` stays a placeholder -- see NOTE below.

Edge structure (from the build plan doc):
  baseline_verifier -> supervisor
  supervisor -> [one specialist, conditional on next_specialist]
  specialist -> code_writer
  code_writer -> pipeline_runner
  pipeline_runner -> [error_recovery | score_analyst, conditional]
  error_recovery -> score_analyst
  score_analyst -> judge  (reasoning-only node, fills verdict/analysis/learning/next_priority/reasoning)
  judge -> log_and_track
  log_and_track -> convergence_checker
  convergence_checker -> [END | supervisor, conditional]

NOTE on `judge`: agent/specialists/experiment_judge.py is pushed but, as of
2026-08-28, still the OLD interface -- it calls tools["log_iteration"](...)
and tools["save_checkpoint"](...) as direct dict calls (wrong shape vs the
real mcp_server.py signatures) and does its own checkpointing + history
append. Person A confirmed the intended rewrite makes it reasoning-only (no
tool calls, no checkpointing, returns only verdict/analysis/learning/
next_priority/reasoning) -- score_analyst in agent.py already owns
experiment_history + checkpointing on that assumption. Until the rewritten
file lands, `judge` stays `stub_judge` here so the graph doesn't call
mismatched tools. Swap it in by replacing the `g.add_node("judge", stub_judge)`
line below with:
    from specialists.experiment_judge import experiment_judge
    g.add_node("judge", lambda s: experiment_judge(s, tools))
-- once experiment_judge.py no longer touches tools/checkpoints/history directly.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from state import AgentState
import agent

from specialists.supervisor import supervisor
from specialists.loss_function_changer import loss_function_changer
from specialists.sequence_modeller import sequence_modeller
from specialists.multitask_trainer import multitask_trainer
from specialists.model_swapper import model_swapper
from specialists.training_optimizer import training_optimizer

SPECIALIST_FNS = {
    "loss_function_changer": loss_function_changer,
    "sequence_modeller": sequence_modeller,
    "multitask_trainer": multitask_trainer,
    "model_swapper": model_swapper,
    "training_optimizer": training_optimizer,
}
SPECIALISTS = list(SPECIALIST_FNS)


def build_graph(tools: dict):
    g = StateGraph(AgentState)

    # --- Person B's real nodes ---
    g.add_node("baseline_verifier", lambda s: agent.baseline_verifier(s, tools))
    g.add_node("code_writer", lambda s: agent.code_writer(s, tools))
    g.add_node("pipeline_runner", lambda s: agent.pipeline_runner(s, tools))
    g.add_node("error_recovery", lambda s: agent.error_recovery(s, tools))
    g.add_node("score_analyst", lambda s: agent.score_analyst(s, tools))
    g.add_node("log_and_track", lambda s: agent.log_and_track(s, tools))
    g.add_node("convergence_checker", lambda s: agent.convergence_checker(s, tools))

    # --- Person A's real nodes ---
    g.add_node("supervisor", lambda s: supervisor(s, tools))
    for name, fn in SPECIALIST_FNS.items():
        g.add_node(name, lambda s, fn=fn: fn(s, tools))

    # --- Judge stays stubbed -- see module docstring ---
    g.add_node("judge", stub_judge)

    g.set_entry_point("baseline_verifier")

    g.add_edge("baseline_verifier", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_specialist"], {name: name for name in SPECIALISTS})
    for name in SPECIALISTS:
        g.add_edge(name, "code_writer")
    g.add_edge("code_writer", "pipeline_runner")
    g.add_conditional_edges("pipeline_runner", agent.route_after_pipeline_runner,
                             {"error_recovery": "error_recovery", "score_analyst": "score_analyst"})
    g.add_edge("error_recovery", "score_analyst")
    g.add_edge("score_analyst", "judge")
    g.add_edge("judge", "log_and_track")
    g.add_edge("log_and_track", "convergence_checker")
    g.add_conditional_edges("convergence_checker", agent.route_after_convergence,
                             {"stop": END, "continue": "supervisor"})

    return g.compile()


# ---------------------------------------------------------------------------
def stub_judge(state: AgentState) -> AgentState:
    """PLACEHOLDER -- see module docstring for why and how to swap it out."""
    improved = state.get("_improved", False)
    state["verdict"] = "improved" if improved else "no_change"
    state["analysis"] = (
        f"[STUB JUDGE] iteration {state['iteration']}: "
        f"{'improved' if improved else 'did not improve'} over best "
        f"({state['best_scores'].get('primary')})."
    )
    state["learning"] = ""
    state["next_priority"] = ""
    return state
