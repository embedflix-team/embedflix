# agent/specialists/multitask_trainer.py
from anthropic import Anthropic
import os

client = Anthropic()

def multitask_trainer(state: dict, tools: dict) -> dict:
    """
    Proposes joint training on multiple feedback signals.
    KuaiRand has 12 signals (click, like, follow, play_time, etc).
    The baseline only uses long_view. Using others as auxiliary tasks
    improves the shared representation and boosts ranking quality.
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline model: FM trained on long_view label only (single task)
- Current validation primary score: {state.get("current_scores", {}).get("primary", 0.6016)}
- Best score seen: {state.get("best_scores", {}).get("primary", 0.6016)}
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: KuaiRand provides 12 feedback signals beyond long_view:
is_click, is_like, is_follow, is_comment, is_forward, is_hate, 
play_time_ms, duration_ms, is_profile_enter, is_share, is_collect.
Training jointly on related signals (especially is_click and play_time_ms) 
improves the shared user/item embeddings even though only long_view is scored.
This is called multi-task learning (MTL).

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

MULTI-TASK OPTIONS (choose best not yet tried):
1. click_auxiliary: add is_click as auxiliary task with shared embeddings, 
   separate output head. Weight: 0.5 main + 0.5 auxiliary.
2. playtime_auxiliary: add play_time_ms (normalized) as regression auxiliary task.
   Strong signal for engagement quality.
3. like_follow_auxiliary: add is_like + is_follow jointly. Both signal deep engagement.
4. full_mtl: train on long_view + click + like + play_time together with 
   learned task weights (uncertainty weighting).
5. hard_sharing: share bottom 2 embedding layers, separate top layers per task.

Decide which multi-task approach to try and explain why it helps long_view ranking.

Respond in this exact format:
HYPOTHESIS: [one sentence — what auxiliary tasks you're adding and why]
MTL_CHOICE: [click_auxiliary | playtime_auxiliary | like_follow_auxiliary | full_mtl | hard_sharing]
CODE_INSTRUCTION: [exact instruction for code_writer — which labels to load from the CSV, how to add auxiliary loss to baseline.py training loop, what weight to use]
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
        "tried_approaches": tried + [f"mtl:{parsed['mtl_choice']}"]
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
        "mtl_choice": "click_auxiliary",
        "code_instruction": "",
        "reasoning": ""
    }
    for line in text.split("\n"):
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("MTL_CHOICE:"):
            result["mtl_choice"] = line.replace("MTL_CHOICE:", "").strip()
        elif line.startswith("CODE_INSTRUCTION:"):
            result["code_instruction"] = line.replace("CODE_INSTRUCTION:", "").strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    return result