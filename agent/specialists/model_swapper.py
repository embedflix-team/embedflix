# agent/specialists/model_swapper.py
from anthropic import Anthropic
import os
from agent.logger import log_node_start, log_node_result
import time

client = Anthropic()

def model_swapper(state: dict, tools: dict) -> dict:
    """
    Proposes upgrading the model architecture.
    Phase 1: web_search for concept
    Phase 2: web_search for code blueprint
    """

    node_start = time.time()
    log_node_start("multitask_trainer", state.get("iteration", 1))
    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary", 0.6016)

    # PHASE 1 — discover concept
    concept_results = tools["web_search"]({
        "query": "DeepFM DCN feature interaction recommender system ranking improvement",
        "search_type": "concept",
        "n_results": 3
    })

    technique = _decide_technique(concept_results.get("results", ""), tried)

    # PHASE 2 — get code blueprint
    code_results = tools["web_search"]({
        "query": f"{technique} numpy implementation from scratch",
        "search_type": "code",
        "n_results": 2
    })

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline: Factorization Machine (FM) with k=16 embeddings, 5 features
- FM captures pairwise feature interactions but misses higher-order interactions
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}
- Constraint: must stay numpy-only (no torch/tensorflow) for fast CPU training

KEY INSIGHT: FM only models second-order interactions. Architectures like DeepFM
add a neural component that captures arbitrary-order interactions, which 
significantly improves ranking on sparse categorical data.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

CONCEPT RESEARCH (what you found on the web):
{concept_results.get("results", "No results")}

CODE BLUEPRINT (real implementation reference):
{code_results.get("results", "No results")}

Based on the research above, propose ONE specific model upgrade for baseline.py.
Adapt the code blueprint to fit the existing numpy-only FM baseline structure.
Do not invent math — adapt what you found.
Remember: numpy only, no torch.

Respond in this exact format:
HYPOTHESIS: [one sentence — which architecture to try and why it captures more]
MODEL_CHOICE: [deepfm | higher_k | dcn | wider_fm | field_aware_fm]
CODE_INSTRUCTION: [exact instruction for code_writer — what class to add/modify, which forward pass to change, how to keep numpy-only]
REASONING: [2-3 sentences of ML reasoning for judge logs]
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
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

    return {
        **state,
        "hypothesis": parsed["hypothesis"],
        "code_change_instruction": parsed["code_instruction"],
        "reasoning": parsed["reasoning"],
        "tried_approaches": tried + [f"model:{parsed['model_choice']}"]
    }


def _decide_technique(search_results: str, tried: list) -> str:
    candidates = [
        ("higher_k", "FM higher embedding dimension k=32 k=64"),
        ("deepfm", "DeepFM deep component MLP feature interaction"),
        ("field_aware_fm", "Field-aware Factorization Machine FFM"),
        ("dcn", "Deep Cross Network feature crossing"),
        ("wider_fm", "FM wider embeddings feature interactions"),
    ]
    for key, technique in candidates:
        if f"model:{key}" not in tried:
            return technique
    return "FM higher embedding dimension k=32 k=64"


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
        "model_choice": "higher_k",
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