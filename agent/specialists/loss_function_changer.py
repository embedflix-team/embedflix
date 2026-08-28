# agent/specialists/loss_function_changer.py
from anthropic import Anthropic
import os

client = Anthropic(
    default_headers={
        "anthropic-workspace-id": os.environ.get("ANTHROPIC_WORKSPACE_ID", "")
    }
)

def loss_function_changer(state: dict, tools: dict) -> dict:
    """
    Proposes a change to the loss function.
    The baseline uses log-loss (pointwise), but we're evaluated on ranking.
    BPR (pairwise) or softmax (listwise) align better with GAUC/nDCG.
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline model: Factorization Machine trained with log-loss (binary cross-entropy)
- Current validation primary score: {state.get("current_scores", {}).get("primary", 0.6016)}
- Best score seen: {state.get("best_scores", {}).get("primary", 0.6016)}
- Metric being optimized: primary = mean(GAUC, nDCG@5) — both are RANKING metrics
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: The baseline is trained with log-loss (predicts click probability) 
but evaluated on ranking quality. This is a mismatch. Ranking-aware losses fix this.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

LOSS FUNCTION OPTIONS (choose the best one not yet tried):
1. BPR loss (Bayesian Personalized Ranking): pairwise — trains model to rank positives 
   above negatives directly. Best for GAUC improvement.
2. Softmax/listwise loss: treats all items per user as a group, maximizes probability 
   of positive at top. Best for nDCG improvement.
3. Focal loss: down-weights easy negatives, focuses on hard cases. Helps with 
   class imbalance (95% negatives).
4. WARP loss: Weighted Approximate-Rank Pairwise — samples negatives until 
   a violation is found, very efficient.

Decide which loss function to try next and explain exactly why it will improve 
GAUC or nDCG@5 for this dataset.

Respond in this exact format:
HYPOTHESIS: [one sentence — what you're trying and why]
LOSS_CHOICE: [bpr | softmax | focal | warp]
CODE_INSTRUCTION: [exact, specific instruction for a code writer to implement this 
in baseline.py — include function signatures, where to add it, what to replace]
REASONING: [2-3 sentences of ML reasoning for the judge logs]
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    parsed = _parse_response(text)

    return {
        **state,
        "hypothesis": parsed["hypothesis"],
        "code_change_instruction": parsed["code_instruction"],
        "reasoning": parsed["reasoning"],
        "tried_approaches": tried + [f"loss:{parsed['loss_choice']}"]
    }


def _summarize_history(history: list) -> str:
    if not history:
        return "No experiments yet. This is iteration 1."
    lines = []
    for h in history[-5:]:  # last 5 only
        lines.append(
            f"- Iter {h.get('iteration')}: {h.get('hypothesis', '?')} "
            f"→ primary {h.get('primary', '?')}"
        )
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    result = {
        "hypothesis": "",
        "loss_choice": "bpr",
        "code_instruction": "",
        "reasoning": ""
    }
    for line in text.split("\n"):
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("LOSS_CHOICE:"):
            result["loss_choice"] = line.replace("LOSS_CHOICE:", "").strip()
        elif line.startswith("CODE_INSTRUCTION:"):
            result["code_instruction"] = line.replace("CODE_INSTRUCTION:", "").strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    return result