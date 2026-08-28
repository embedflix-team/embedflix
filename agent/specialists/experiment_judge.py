# agent/specialists/experiment_judge.py
from anthropic import Anthropic
import os

client = Anthropic()

def experiment_judge(state: dict, tools: dict) -> dict:
    """
    Runs AFTER score_analyst gets the new scores.
    Judges whether the experiment worked, why, and what to learn from it.
    This node produces the rich reasoning logs that judges evaluate for
    Innovation and Autonomy scoring.
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    current = state.get("current_scores", {}).get("primary", 0.6016)
    best = state.get("best_scores", {}).get("primary", 0.6016)
    gauc = state.get("current_scores", {}).get("gauc", 0.6674)
    ndcg5 = state.get("current_scores", {}).get("ndcg5", 0.5357)
    hypothesis = state.get("hypothesis", "unknown")
    specialist = state.get("next_specialist", "unknown")
    iteration = state.get("iteration", 1)
    error = state.get("error_message", None)

    prompt = f"""You are an ML research judge reviewing an experiment run by an 
autonomous recommender system agent.

EXPERIMENT JUST RUN:
- Iteration: {iteration}
- Specialist used: {specialist}
- Hypothesis: {hypothesis}
- Result: GAUC={gauc:.4f}, nDCG@5={ndcg5:.4f}, primary={current:.4f}
- Previous best: {best:.4f}
- Delta vs best: {current - best:+.4f}
- Delta vs baseline (0.6016): {current - 0.6016:+.4f}
- Error during run: {error if error else "None"}

FULL EXPERIMENT HISTORY:
{history_summary}

YOUR JOB:
Analyze this experiment result deeply. Explain:
1. Did it work? Why or why not?
2. What does the GAUC vs nDCG@5 split tell us? 
   (If GAUC improved but nDCG didn't, ranking order improved but top-5 precision didn't)
3. What should the agent learn from this for future iterations?
4. Is there a pattern emerging across experiments?

This analysis will be read by hackathon judges to assess the agent's 
research quality and autonomous reasoning ability.

Respond in this exact format:
VERDICT: [improved | no_change | regression]
ANALYSIS: [3-4 sentences — what worked, what didn't, why, what the metric split reveals]
LEARNING: [1-2 sentences — what the agent should remember for future iterations]
NEXT_PRIORITY: [one sentence — what area looks most promising to try next]
KEEP_CHECKPOINT: [yes | no — whether to save this as the new best]
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    parsed = _parse_response(text)

    # Build the iteration log entry — this is what judges read
    log_entry = {
        "iteration": iteration,
        "specialist": specialist,
        "hypothesis": hypothesis,
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": current,
        "delta_vs_baseline": round(current - 0.6016, 4),
        "verdict": parsed["verdict"],
        "analysis": parsed["analysis"],
        "learning": parsed["learning"],
        "next_priority": parsed["next_priority"],
        "error": error
    }

    # Log it via MCP tool
    tools["log_iteration"]({"iteration": iteration, "log": log_entry})

    # Save checkpoint if improved
    if parsed["keep_checkpoint"] == "yes" and current > best:
        tools["save_checkpoint"]({"iteration": iteration, "primary": current})
        new_best = current
    else:
        new_best = best

    # Add to experiment history
    history = state.get("experiment_history", [])
    history.append(log_entry)

    return {
        **state,
        "experiment_history": history,
        "best_scores": {
            **state.get("best_scores", {}),
            "primary": new_best
        },
        "verdict": parsed["verdict"],
        "analysis": parsed["analysis"],
        "learning": parsed["learning"],
    }


def _summarize_history(history: list) -> str:
    if not history:
        return "No experiments yet. This is iteration 1."
    lines = []
    for h in history[-10:]:
        lines.append(
            f"- Iter {h.get('iteration')}: [{h.get('specialist', '?')}] "
            f"{h.get('hypothesis', '?')} "
            f"→ primary {h.get('primary', '?')} "
            f"(verdict: {h.get('verdict', '?')})"
        )
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    result = {
        "verdict": "no_change",
        "analysis": "",
        "learning": "",
        "next_priority": "",
        "keep_checkpoint": "no"
    }
    for line in text.split("\n"):
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("ANALYSIS:"):
            result["analysis"] = line.replace("ANALYSIS:", "").strip()
        elif line.startswith("LEARNING:"):
            result["learning"] = line.replace("LEARNING:", "").strip()
        elif line.startswith("NEXT_PRIORITY:"):
            result["next_priority"] = line.replace("NEXT_PRIORITY:", "").strip()
        elif line.startswith("KEEP_CHECKPOINT:"):
            result["keep_checkpoint"] = line.replace("KEEP_CHECKPOINT:", "").strip()
    return result