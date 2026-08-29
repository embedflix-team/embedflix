# agent/specialists/supervisor.py
from anthropic import Anthropic
import os
from agent.logger import log_node_start, log_node_result
import time

client = Anthropic()

# All specialist node names the supervisor can route to
SPECIALISTS = [
    "loss_function_changer",
    "sequence_modeller", 
    "multitask_trainer",
    "model_swapper",
    "training_optimizer"
]

def supervisor(state: dict, tools: dict) -> dict:
    """
    The brain of the agent. Looks at the full experiment history,
    current scores, what's been tried, and decides what to try next.
    This is what judges evaluate for autonomy and innovation quality.
    """

    node_start = time.time()
    log_node_start("multitask_trainer", state.get("iteration", 1))
    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    current = state.get("current_scores", {}).get("primary", 0.6016)
    best = state.get("best_scores", {}).get("primary", 0.6016)
    iteration = state.get("iteration", 1)

    prompt = f"""You are the supervisor of an autonomous ML research agent improving 
a KuaiRand-Pure video recommender system. Your job is to decide which 
specialist agent to call next to maximally improve the ranking score.

CURRENT STATE:
- Iteration: {iteration} / 50 max
- Current primary score: {current:.4f}
- Best primary score: {best:.4f}  
- Official baseline to beat: 0.6016
- Gap to baseline: {current - 0.6016:+.4f}
- Metric: primary = mean(GAUC, nDCG@5) — both are RANKING metrics

WHAT HAS BEEN TRIED:
{tried if tried else "Nothing yet — this is iteration 1"}

FULL EXPERIMENT HISTORY:
{history_summary}

AVAILABLE SPECIALISTS:
1. loss_function_changer — switches from log-loss to BPR/softmax/focal loss
   Best when: model trains well but ranking quality is poor (low nDCG vs GAUC)
   
2. sequence_modeller — adds user history / recent watch features  
   Best when: model ignores temporal patterns, users have rich history
   
3. multitask_trainer — adds auxiliary tasks (click, like, play_time)
   Best when: long_view signal is sparse, other signals are abundant
   
4. model_swapper — upgrades FM to DeepFM/DCN/higher-k
   Best when: feature interactions are not well captured, score plateaus
   
5. training_optimizer — tunes lr, batch size, regularization, patience
   Best when: training is unstable, overfitting, or converging too early

ROUTING RULES:
- Never repeat a specialist that already failed to improve the score
- If score improved last iteration, try building on that approach first
- If score dropped last iteration, try a different specialist
- If stuck for 2+ iterations, escalate to model_swapper
- Prioritize loss_function_changer early — it's the highest-leverage change
- Always explain your routing decision clearly for the judge logs

Decide which specialist to call next and why.

Respond in this exact format:
ROUTE_TO: [loss_function_changer | sequence_modeller | multitask_trainer | model_swapper | training_optimizer]
ROUTING_REASON: [2-3 sentences explaining why this specialist now, what you expect it to improve, and what signal from history led to this decision]
STRATEGY: [one sentence — overall strategy for this phase of the run]
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    parsed = _parse_response(text)
    log_node_result(
        node_name="multitask_trainer",
        iteration=state.get("iteration", 1),
        hypothesis=parsed["hypothesis"],
        issue_found="Only long_view label used — 11 signals wasted",
        proposed_fix=parsed["code_instruction"],
        reasoning=parsed["reasoning"],
        duration_seconds=time.time() - node_start,
        tokens_in=response.usage.input_tokens,
        tokens_out=response.usage.output_tokens,
    )

    # Validate route — fall back to loss_function_changer if invalid
    if parsed["route_to"] not in SPECIALISTS:
        parsed["route_to"] = "loss_function_changer"

    return {
        **state,
        "next_specialist": parsed["route_to"],
        "routing_reason": parsed["routing_reason"],
        "strategy": parsed["strategy"],
        "reasoning": f"Supervisor routed to {parsed['route_to']}: {parsed['routing_reason']}"
    }


def _summarize_history(history: list) -> str:
    if not history:
        return "No experiments yet. This is iteration 1."
    lines = []
    for h in history[-10:]:  # last 10 for supervisor — needs more context
        delta = h.get("primary", 0) - 0.6016
        lines.append(
            f"- Iter {h.get('iteration')}: [{h.get('specialist', '?')}] "
            f"{h.get('hypothesis', '?')} "
            f"→ primary {h.get('primary', '?')} ({delta:+.4f} vs baseline)"
        )
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    result = {
        "route_to": "loss_function_changer",
        "routing_reason": "",
        "strategy": ""
    }
    for line in text.split("\n"):
        if line.startswith("ROUTE_TO:"):
            result["route_to"] = line.replace("ROUTE_TO:", "").strip()
        elif line.startswith("ROUTING_REASON:"):
            result["routing_reason"] = line.replace("ROUTING_REASON:", "").strip()
        elif line.startswith("STRATEGY:"):
            result["strategy"] = line.replace("STRATEGY:", "").strip()
    return result