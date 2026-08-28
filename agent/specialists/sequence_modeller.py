# agent/specialists/sequence_modeller.py
from anthropic import Anthropic
import os

client = Anthropic()

def sequence_modeller(state: dict, tools: dict) -> dict:
    """
    Proposes adding user history / sequence features.
    The baseline uses only static features (user_id, item_id, etc).
    Adding 'what did this user watch recently' dramatically improves ranking.
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline model: Factorization Machine with 5 static features only
  (user_id, video_id, author_id, music_id, tag)
- Current validation primary score: {state.get("current_scores", {}).get("primary", 0.6016)}
- Best score seen: {state.get("best_scores", {}).get("primary", 0.6016)}
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: The baseline ignores user history entirely. 
Adding sequence features — what the user watched/clicked recently — 
lets the model capture short-term intent, which is the strongest signal 
in short-video recommendation.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

SEQUENCE FEATURE OPTIONS (choose best not yet tried):
1. mean_pooling: average embeddings of last N watched videos. Simple, effective baseline.
2. last_k_items: just concatenate IDs of last 3-5 watched items as new features.
3. category_history: count of how many times user watched each category recently.
4. recency_weighted: more recent interactions weighted higher (exponential decay).
5. click_vs_skip_ratio: ratio of clicks to impressions in user's recent history.

Decide which sequence feature to add and explain exactly why it helps GAUC/nDCG@5.

Respond in this exact format:
HYPOTHESIS: [one sentence — what you're adding and why]
SEQUENCE_CHOICE: [mean_pooling | last_k_items | category_history | recency_weighted | click_vs_skip_ratio]
CODE_INSTRUCTION: [exact instruction for code_writer — where in data.py or baseline.py to add this, what arrays to compute from train data sorted by date, how to join to X matrix]
REASONING: [2-3 sentences of ML reasoning for judge logs]
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
        "tried_approaches": tried + [f"sequence:{parsed['sequence_choice']}"]
    }


def _summarize_history(history: list) -> str:
    if not history:
        return "No experiments yet. This is iteration 1."
    lines = []
    for h in history[-5:]:
        lines.append(
            f"- Iter {h.get('iteration')}: {h.get('hypothesis', '?')} "
            f"→ primary {h.get('primary', '?')}"
        )
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    result = {
        "hypothesis": "",
        "sequence_choice": "mean_pooling",
        "code_instruction": "",
        "reasoning": ""
    }
    for line in text.split("\n"):
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("SEQUENCE_CHOICE:"):
            result["sequence_choice"] = line.replace("SEQUENCE_CHOICE:", "").strip()
        elif line.startswith("CODE_INSTRUCTION:"):
            result["code_instruction"] = line.replace("CODE_INSTRUCTION:", "").strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    return result