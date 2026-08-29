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
    Phase 1: web_search for concept (what technique to try)
    Phase 2: web_search for code blueprint (how to implement it)
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary", 0.6016)

    # PHASE 1 — discover concept
    concept_results = tools["web_search"]({
        "query": "improve GAUC nDCG ranking loss function recommender system pairwise listwise",
        "search_type": "concept",
        "n_results": 3
    })

    # decide technique from phase 1 (simple keyword pick — Claude will reason properly)
    technique = _decide_technique(concept_results.get("results", ""), tried)

    # PHASE 2 — get real code blueprint for that technique
    code_results = tools["web_search"]({
        "query": f"{technique} numpy implementation recommender system",
        "search_type": "code",
        "n_results": 2
    })

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline: Factorization Machine trained with log-loss (binary cross-entropy)
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5) — both are RANKING metrics
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: Log-loss trains for click probability but we are evaluated on ranking 
quality. Ranking-aware losses fix this mismatch directly.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

CONCEPT RESEARCH (what you found on the web):
{concept_results.get("results", "No results")}

CODE BLUEPRINT (real implementation reference):
{code_results.get("results", "No results")}

Based on the research above, propose ONE specific loss function change to baseline.py.
Adapt the code blueprint to fit the existing FM baseline structure.
Do not invent math — adapt what you found.

Respond in this exact format:
HYPOTHESIS: [one sentence — what you're trying and why]
LOSS_CHOICE: [bpr | softmax | focal | warp]
CODE_INSTRUCTION: [exact instruction for a code writer to implement this in baseline.py]
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


def _decide_technique(search_results: str, tried: list) -> str:
    """Pick a technique from phase 1 results that hasn't been tried yet."""
    candidates = [
        ("bpr", "Bayesian Personalised Ranking BPR loss"),
        ("softmax", "softmax listwise loss recommender"),
        ("focal", "focal loss class imbalance"),
        ("warp", "WARP loss pairwise ranking"),
    ]
    for key, technique in candidates:
        if f"loss:{key}" not in tried:
            return technique
    return "BPR loss pairwise ranking"  # fallback


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