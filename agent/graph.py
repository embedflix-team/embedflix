"""Step 4: wire the LangGraph. Uses PLACEHOLDER specialist/judge nodes until
Person A delivers her real ones -- swap them at the bottom of this file
(the STUB_* functions) without touching anything else here.

Edge structure (from the build plan doc):
  baseline_verifier -> supervisor
  supervisor -> [one specialist, conditional]
  specialist -> code_writer
  code_writer -> pipeline_runner
  pipeline_runner -> [error_recovery | score_analyst, conditional]
  error_recovery -> score_analyst
  score_analyst -> judge  (Person A's JUDGE_PROMPT node, fills state["reasoning"])
  judge -> log_and_track
  log_and_track -> convergence_checker
  convergence_checker -> [END | supervisor, conditional]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from state import AgentState
import agent

SPECIALISTS = [
    "loss_function_changer", "sequence_modeller", "multitask_trainer",
    "model_swapper", "training_optimizer",
]


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

    # --- Placeholder nodes (Person A's real ones swap in here) ---
    g.add_node("supervisor", stub_supervisor)
    for name in SPECIALISTS:
        g.add_node(name, _make_stub_specialist(name))
    g.add_node("judge", stub_judge)

    g.set_entry_point("baseline_verifier")

    g.add_edge("baseline_verifier", "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next_node"], {name: name for name in SPECIALISTS})
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
# PLACEHOLDER nodes -- delete this section once Person A's real specialist/
# supervisor/judge nodes exist. Kept deliberately dumb (round-robin, no LLM
# call) so the full graph is runnable and testable today.
# ---------------------------------------------------------------------------
def stub_supervisor(state: AgentState) -> AgentState:
    tried = len(state.get("tried_approaches", []))
    state["next_node"] = SPECIALISTS[tried % len(SPECIALISTS)]
    return state


def _make_stub_specialist(name: str):
    def stub_specialist(state: AgentState) -> AgentState:
        state["hypothesis"] = f"[STUB-{name}] placeholder hypothesis, not real ML reasoning"
        # A no-op-ish but syntactically valid edit: nudge lr slightly so the
        # pipeline actually re-runs with a real (if trivial) change each time.
        import random
        lr = round(0.001 * random.uniform(0.5, 1.5), 5)
        state["code_change_instruction"] = (
            "FILE: baseline.py\n"
            "OLD_CODE: def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):\n"
            f"NEW_CODE: def run_fm(splits, k=16, lr={lr}, epochs=40, bs=8192, patience=4, seed=0, verbose=True):"
        )
        return state
    return stub_specialist


def stub_judge(state: AgentState) -> AgentState:
    improved = state.get("_improved", False)
    state["reasoning"] = (
        f"[STUB JUDGE] iteration {state['iteration']}: "
        f"{'improved' if improved else 'did not improve'} over best "
        f"({state['best_scores'].get('primary')})."
    )
    return state
