# agent/specialists/training_optimizer.py
from anthropic import Anthropic
import os

client = Anthropic()

def training_optimizer(state: dict, tools: dict) -> dict:
    """
    Proposes changes to training settings.
    Phase 1: web_search for concept
    Phase 2: web_search for code blueprint
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    # PHASE 1 — discover concept
    concept_results = tools["web_search"]({
        "query": "training hyperparameter optimization recommender system learning rate regularization",
        "search_type": "concept",
        "n_results": 3
    })

    technique = _decide_technique(concept_results.get("results", ""), tried)

    # PHASE 2 — get code blueprint
    code_results = tools["web_search"]({
        "query": f"{technique} numpy implementation training loop",
        "search_type": "code",
        "n_results": 2
    })

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline: lr=0.001, batch_size=8192, Adam, early stopping patience=4, k=16, no regularization
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}
- Dataset: 1.14M train rows, 124K validation rows, 5 categorical features

KEY INSIGHT: Default training settings are rarely optimal. Small targeted changes
to lr, regularization, or batch size can improve both convergence and final score.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

CONCEPT RESEARCH (what you found on the web):
{concept_results.get("results", "No results")}

CODE BLUEPRINT (real implementation reference):
{code_results.get("results", "No results")}

Based on the research above, propose ONE specific training change to baseline.py.
Adapt the code blueprint to fit the existing FM baseline structure.
Do not invent math — adapt what you found.

Respond in this exact format:
HYPOTHESIS: [one sentence — what setting to change and why it helps]
TRAINING_CHOICE: [lr_decay | l2_regularization | larger_batch | smaller_lr | increased_patience | embedding_regularization]
CODE_INSTRUCTION: [exact instruction for code_writer — which variable to change, what value, where to add logic]
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


def _decide_technique(search_results: str, tried: list) -> str:
    candidates = [
        ("l2_regularization", "L2 regularization embedding training"),
        ("lr_decay", "learning rate decay schedule training"),
        ("larger_batch", "large batch size training stability"),
        ("smaller_lr", "small learning rate fine tuning"),
        ("increased_patience", "early stopping patience tuning"),
        ("embedding_regularization", "embedding dropout regularization"),
    ]
    for key, technique in candidates:
        if f"training:{key}" not in tried:
            return technique
    return "L2 regularization embedding training"


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