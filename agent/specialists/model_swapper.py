# agent/specialists/model_swapper.py
from anthropic import Anthropic
import os

client = Anthropic()

def model_swapper(state: dict, tools: dict) -> dict:
    """
    Proposes switching the model architecture entirely.
    Baseline is FM (Factorization Machine) — good but basic.
    DeepFM, DCN, or DIN can capture higher-order feature interactions
    which FM misses, especially with user history features.
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline model: Factorization Machine (FM, k=16, lr=0.001, 5 categorical fields)
- Current validation primary score: {state.get("current_scores", {}).get("primary", 0.6016)}
- Best score seen: {state.get("best_scores", {}).get("primary", 0.6016)}
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}
- Constraint: numpy only preferred (no torch/pandas), must run on CPU in ~40s

KEY INSIGHT: FM captures only pairwise feature interactions (A×B).
Deeper models capture higher-order interactions (A×B×C) which matter
for complex user behaviour patterns. However complexity must be balanced
against the CPU-only constraint and 40s runtime budget.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

MODEL OPTIONS (choose best not yet tried):
1. deeper_fm: add a 2-layer MLP on top of FM embeddings (hidden: 64→32).
   Captures non-linear interactions. Still numpy-implementable.
2. deepfm: combine FM + deep MLP sharing same embeddings.
   Industry standard, best balance of accuracy vs complexity.
3. dcn: Deep & Cross Network — explicit cross layers + deep layers.
   Very efficient at capturing bounded-degree interactions.
4. fm_higher_k: increase FM embedding dim from k=16 to k=32 or k=64.
   Simple change, often underexplored, can significantly help.
5. field_aware_fm: FFM — each feature has separate embedding per field.
   More expressive than FM, same paradigm.

Decide which architecture change to try given the CPU/numpy constraint.

Respond in this exact format:
HYPOTHESIS: [one sentence — which model and why it beats FM here]
MODEL_CHOICE: [deeper_fm | deepfm | dcn | fm_higher_k | field_aware_fm]
CODE_INSTRUCTION: [exact instruction for code_writer — what class to add or modify in baseline.py, key hyperparameters, what to keep the same from FM]
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
        "tried_approaches": tried + [f"model:{parsed['model_choice']}"]
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
        "model_choice": "deepfm",
        "code_instruction": "",
        "reasoning": ""
    }
    for line in text.split("\n"):
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("MODEL_CHOICE:"):
            result["model_choice"] = line.replace("MODEL_CHOICE:", "").strip()
        elif line.startswith("CODE_INSTRUCTION:"):
            result["code_instruction"] = line.replace("CODE_INSTRUCTION:", "").strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    return result