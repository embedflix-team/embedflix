# agent/specialists/model_swapper.py
from anthropic import Anthropic
import os
import re

client = Anthropic()

def model_swapper(state: dict, tools: dict) -> dict:
    """
    Proposes upgrading the model architecture.
    Phase 1: web_search for concept
    Phase 2: web_search for code blueprint
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

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

CRITICAL EXECUTION CONSTRAINT: the pipeline always runs
`python3 baseline.py --model fm` -- there is no other entry point, no CLI flag
to select a different model, and nothing else ever calls a new class. If you
propose a new class (e.g. a DeepFM-style class) WITHOUT ALSO making the
existing --model fm path actually use it, that class is permanently dead code
that never runs and never affects the score -- this has happened before.

ANOTHER HARD CONSTRAINT: code_writer applies your instruction as exactly ONE
find-and-replace edit at ONE location in the file. It CANNOT make two edits
in two different places (e.g. add a class near the top AND separately change
the MODELS dispatch dict near the bottom) -- only the first part would ever
land, and the model would stay dead code exactly as before.

Because of that, STRONGLY PREFER modifying the existing FM class / run_fm
function IN PLACE (add the new logic directly inside FM's __init__/step/
logits/predict methods) -- that is a single edit location and is what
--model fm already executes, so it is guaranteed reachable. Only propose a
brand-new separate class if the entire upgrade -- new class definition AND
the MODELS['fm'] dispatch line that calls it -- can be captured within one
single contiguous OLD_CODE snippet (i.e. they are close enough together in
the file to select as one block). If they are not contiguous, do the in-place
version instead. Never propose a new class with a separate, disconnected
wiring instruction.

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
CODE_INSTRUCTION: [exact instruction for code_writer — prefer modifying FM/run_fm IN PLACE at one location; how to keep numpy-only; must describe a change code_writer can apply as ONE single contiguous edit]
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
    # Values can span multiple lines (numbered lists, multi-sentence text) and
    # Claude sometimes decorates the label ("**CODE_INSTRUCTION:**",
    # "CODE_INSTRUCTION :"). Capture each field from its label to the next
    # known label (or end of text) instead of only its first line -- the
    # first-line scan left code_writer with an empty code_change_instruction
    # whenever the model answered with a list.
    labels = ["HYPOTHESIS", "MODEL_CHOICE", "CODE_INSTRUCTION", "REASONING"]
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