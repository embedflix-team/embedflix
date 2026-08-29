# agent/specialists/sequence_modeller.py
from anthropic import Anthropic
import os
from agent.logger import log_node_start, log_node_result
import time

client = Anthropic()

def sequence_modeller(state: dict, tools: dict) -> dict:
    """
    Proposes adding user history / sequence features.
    Phase 1: web_search for concept
    Phase 2: web_search for code blueprint
    """
 
    node_start = time.time()
    log_node_start("loss_function_changer", state.get("iteration", 1))
    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary", 0.6016)

    # PHASE 1 — discover concept
    concept_results = tools["web_search"]({
        "query": "user history sequence features recommender system GAUC nDCG improvement",
        "search_type": "concept",
        "n_results": 3
    })

    technique = _decide_technique(concept_results.get("results", ""), tried)

    # PHASE 2 — get code blueprint
    code_results = tools["web_search"]({
        "query": f"{technique} numpy implementation user history recommender",
        "search_type": "code",
        "n_results": 2
    })

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline: FM with 5 static features only (user_id, video_id, author_id, music_id, tag)
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: The baseline ignores user history entirely. Sequence features capture
short-term intent — the strongest signal in short-video recommendation.
CRITICAL: sequence features must be computed from training data sorted by date ONLY —
never leak future interactions into past feature windows.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

CONCEPT RESEARCH (what you found on the web):
{concept_results.get("results", "No results")}

CODE BLUEPRINT (real implementation reference):
{code_results.get("results", "No results")}

Based on the research above, propose ONE specific sequence feature to add to baseline.py.
Adapt the code blueprint to fit the existing FM baseline structure.
Do not invent math — adapt what you found.

Respond in this exact format:
HYPOTHESIS: [one sentence — what you're adding and why]
SEQUENCE_CHOICE: [mean_pooling | last_k_items | category_history | recency_weighted | click_vs_skip_ratio]
CODE_INSTRUCTION: [exact instruction for code_writer — where to add this, what arrays to compute, how to join to X matrix]
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
        node_name="loss_function_changer",
        iteration=state.get("iteration", 1),
        hypothesis=parsed["hypothesis"],
        issue_found="Log-loss vs ranking metric mismatch",
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
        "tried_approaches": tried + [f"sequence:{parsed['sequence_choice']}"]
    }


def _decide_technique(search_results: str, tried: list) -> str:
    candidates = [
        ("mean_pooling", "mean pooling user watch history embeddings"),
        ("last_k_items", "last K items user interaction history features"),
        ("category_history", "user category watch count history features"),
        ("recency_weighted", "recency weighted exponential decay user history"),
        ("click_vs_skip_ratio", "click skip ratio user behaviour features"),
    ]
    for key, technique in candidates:
        if f"sequence:{key}" not in tried:
            return technique
    return "mean pooling user watch history embeddings"


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