# agent/specialists/multitask_trainer.py
from anthropic import Anthropic
import os
import re

client = Anthropic()

def multitask_trainer(state: dict, tools: dict) -> dict:
    """
    Proposes adding auxiliary tasks using extra feedback signals.
    Phase 1: web_search for concept
    Phase 2: web_search for code blueprint
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    # PHASE 1 — discover concept
    concept_results = tools["web_search"]({
        "query": "multi-task learning auxiliary signals recommender system GAUC improvement",
        "search_type": "concept",
        "n_results": 3
    })

    technique = _decide_technique(concept_results.get("results", ""), tried)

    # PHASE 2 — get code blueprint
    code_results = tools["web_search"]({
        "query": f"{technique} numpy implementation multi-task recommender",
        "search_type": "code",
        "n_results": 2
    })

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline: FM trained only on long_view label (binary)
- KuaiRand provides 12 feedback signals: click, like, follow, comment, 
  forward, long_view, play_time, and more
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5) — scored on long_view only
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: Training only on long_view wastes 11 other signals. Multi-task learning
shares representations across tasks — click and like are correlated with long_view,
so joint training improves the shared embedding quality even though only long_view is scored.

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

CONCEPT RESEARCH (what you found on the web):
{concept_results.get("results", "No results")}

CODE BLUEPRINT (real implementation reference):
{code_results.get("results", "No results")}

Based on the research above, propose ONE specific multi-task approach for baseline.py.
Adapt the code blueprint to fit the existing FM baseline structure.
Do not invent math — adapt what you found.

Respond in this exact format:
HYPOTHESIS: [one sentence — which auxiliary signals to add and why]
MULTITASK_CHOICE: [shared_bottom | click_auxiliary | like_auxiliary | playtime_auxiliary | esmm_style]
CODE_INSTRUCTION: [exact instruction for code_writer — which label columns to use, how to combine losses, where to modify training loop]
REASONING: [2-3 sentences of ML reasoning for judge logs]
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
        "tried_approaches": tried + [f"multitask:{parsed['multitask_choice']}"],
        "_last_call_tokens": response.usage.input_tokens + response.usage.output_tokens,
    }


def _decide_technique(search_results: str, tried: list) -> str:
    candidates = [
        ("click_auxiliary", "click auxiliary task multi-task learning"),
        ("like_auxiliary", "like signal auxiliary loss joint training"),
        ("shared_bottom", "shared bottom multi-task neural network"),
        ("playtime_auxiliary", "play time regression auxiliary task"),
        ("esmm_style", "ESMM entire space multi-task model"),
    ]
    for key, technique in candidates:
        if f"multitask:{key}" not in tried:
            return technique
    return "click auxiliary task multi-task learning"


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
        "multitask_choice": "click_auxiliary",
        "code_instruction": "",
        "reasoning": ""
    }
    # Values can span multiple lines (numbered lists, multi-sentence text) and
    # Claude sometimes decorates the label ("**CODE_INSTRUCTION:**",
    # "CODE_INSTRUCTION :"). Capture each field from its label to the next
    # known label (or end of text) instead of only its first line -- the
    # first-line scan left code_writer with an empty code_change_instruction
    # whenever the model answered with a list.
    labels = ["HYPOTHESIS", "MULTITASK_CHOICE", "CODE_INSTRUCTION", "REASONING"]
    label_re = re.compile(r"^[ \t>#*_-]*(" + "|".join(labels) + r")[ \t]*\**[ \t]*:",
                          re.MULTILINE)
    matches = list(label_re.finditer(text))
    for i, m in enumerate(matches):
        key = m.group(1).lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[m.end():end].strip().strip("*").strip()
        if key in result:
            result[key] = value
    return result