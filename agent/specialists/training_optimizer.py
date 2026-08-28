# agent/specialists/training_optimizer.py
from anthropic import Anthropic
import os

client = Anthropic()

def training_optimizer(state: dict, tools: dict) -> dict:
    """
    Proposes changes to training settings — learning rate, batch size,
    regularization, early stopping, embedding size.
    These are often overlooked but can squeeze significant gains.
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline training settings: lr=0.001, batch_size=8192, Adam optimizer,
  early stopping patience=4, embedding k=16, no regularization
- Current validation primary score: {state.get("current_scores", {}).get("primary", 0.6016)}
- Best score seen: {state.get("best_scores", {}).get("primary", 0.6016)}
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}
- Dataset: 1.14M train rows, 124K validation rows, 5 categorical features

KEY INSIGHT: Training hyperparameters are often the easiest wins.
The baseline uses default settings that were not tuned for this dataset.
Small changes to lr, regularization, or batch size can improve both
convergence speed and final score.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

TRAINING OPTIONS (choose best not yet tried):
1. lr_decay: add learning rate decay (multiply lr by 0.5 every 2 epochs).
   Helps model converge to a better minimum instead of oscillating.
2. l2_regularization: add L2 penalty (lambda=1e-5) on embeddings.
   Prevents overfitting on high-frequency users/items.
3. larger_batch: increase batch_size from 8192 to 16384 or 32768.
   More stable gradients, faster convergence on large dataset.
4. smaller_lr: reduce lr from 0.001 to 0.0005.
   More careful updates, often helps when model is close to convergence.
5. increased_patience: increase early stopping patience from 4 to 8.
   Gives model more time to escape local minima.
6. embedding_regularization: add dropout on embeddings (rate=0.1).
   Reduces co-adaptation between embedding dimensions.

Decide which training change to try next based on the experiment history.

Respond in this exact format:
HYPOTHESIS: [one sentence — what setting to change and why it helps]
TRAINING_CHOICE: [lr_decay | l2_regularization | larger_batch | smaller_lr | increased_patience | embedding_regularization]
CODE_INSTRUCTION: [exact instruction for code_writer — which variable in baseline.py to change, what value to set, where to add the new logic]
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
        "tried_approaches": tried + [f"training:{parsed['training_choice']}"]
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
        "training_choice": "l2_regularization",
        "code_instruction": "",
        "reasoning": ""
    }
    for line in text.split("\n"):
        if line.startswith("HYPOTHESIS:"):
            result["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
        elif line.startswith("TRAINING_CHOICE:"):
            result["training_choice"] = line.replace("TRAINING_CHOICE:", "").strip()
        elif line.startswith("CODE_INSTRUCTION:"):
            result["code_instruction"] = line.replace("CODE_INSTRUCTION:", "").strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.replace("REASONING:", "").strip()
    return result