"""Step 4: wire the LangGraph. Real supervisor + 5 specialists + judge, all
from agent/specialists/*.py.

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

`judge` is now agent/specialists/experiment_judge.py (2026-08-28 rewrite,
commit 9e85ac5, "pure reasoning only, remove tools calls and checkpoint
logic"): it takes no tools argument use, does not touch experiment_history or
checkpoints, and returns only verdict/analysis/learning/next_priority/
reasoning -- matching what score_analyst in agent.py already assumed it owns
(experiment_history + checkpointing stay exclusively there).
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
from specialists.feature_engineer import feature_engineer
from specialists.experiment_judge import experiment_judge

SPECIALIST_FNS = {
    "loss_function_changer": loss_function_changer,
    "sequence_modeller": sequence_modeller,
    "multitask_trainer": multitask_trainer,
    "model_swapper": model_swapper,
    "training_optimizer": training_optimizer,
    "feature_engineer": feature_engineer,
}
SPECIALISTS = list(SPECIALIST_FNS)

def _extract_specialist_output(result: dict) -> dict:
    """Extract only keys specialists are allowed to update.
    Prevents stale full-state snapshots from overwriting good LangGraph state."""
    if not isinstance(result, dict):
        raise TypeError(f"Specialist returned {type(result).__name__}, expected dict")
    keys = [
        "hypothesis", "code_change_instruction", "reasoning",
        "tried_approaches", "next_specialist", "routing_reason",
        "strategy", "verdict", "analysis", "learning", "next_priority",
        "_phase1_results", "_phase1_query", "_phase2_results", "_phase2_query",
        "_phase1_label", "_phase2_label", "_deterministic_edit",
    ]
    return {k: result[k] for k in keys if k in result}

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
        g.add_node(name, lambda s, fn=fn: _extract_specialist_output(fn(s, tools)))

    # --- Person A's real judge (reasoning-only, no tools/checkpointing) ---
    g.add_node("judge", lambda s: experiment_judge(s, tools))

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
