# agent/specialists/loss_function_changer.py
from anthropic import Anthropic
import os
import re

from specialists._insight import WITHIN_USER_INVARIANCE

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
    primary = state.get("current_scores", {}).get("primary") or 0.6016

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

{WITHIN_USER_INVARIANCE}

CURRENT SITUATION:
- Baseline: Factorization Machine trained with POINTWISE LOG-LOSS / binary
  cross-entropy -- `g = (sigmoid(z) - y) / B`. This is the shipped official
  baseline (Phase 0 restored it; an earlier unvalidated BPR edit was reverted).
  FM.step()'s entire loss computation is this one block; there is no dispatch
  on a loss-function choice anywhere in FM.__init__, FM.step, or run_fm.
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5) — both are RANKING metrics
- Iteration: {state.get("iteration", 1)}

KEY INSIGHT: log-loss trains for calibrated click probability, but we are
scored on WITHIN-USER ranking. A pairwise (BPR) or listwise (softmax) loss
optimises relative order directly. BPR must sample its negative from the SAME
USER's impressions to match the metric -- a cross-user pair teaches nothing the
metric rewards. Note: pure loss changes on this FM plateau near 0.602 (the FM
already encodes the ID-level signal); a listwise loss is still worth one try.

CRITICAL EXECUTION CONSTRAINT: code_writer applies your instruction as exactly
ONE find-and-replace edit at ONE location in the file. It CANNOT make two edits
in two different places (e.g. define a new loss function elsewhere in the file
AND separately change what FM.step() calls) -- only the first part would ever
land, and FM.step()'s existing hardcoded log-loss computation would keep running
completely unchanged, silently. The resulting score would look like a real
experiment but would reflect nothing about your proposed loss -- this exact
failure mode has happened before with a different specialist's edits.

Because of that, your CODE_INSTRUCTION must describe a change made ENTIRELY
INSIDE FM.step()'s existing body, replacing its current log-loss computation
in place with your proposed alternative -- one single contiguous edit, not a
new function defined elsewhere plus a separate call-site change. Never propose
a new loss function with a separate, disconnected wiring instruction.

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
HYPOTHESIS: [one sentence — what you're trying and why, relative to the current log-loss]
LOSS_CHOICE: [bpr | softmax | focal | warp]
CODE_INSTRUCTION: [exact instruction for a code writer to implement this IN PLACE inside FM.step(), replacing its current log-loss computation]
REASONING: [2-3 sentences of ML reasoning for the judge logs]
"""

    response = client.messages.create(
        model="claude-haiku-4-5",
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
        "tried_approaches": tried + [f"loss:{parsed['loss_choice']}"],
        "_last_call_tokens": response.usage.input_tokens + response.usage.output_tokens,
    }


def _decide_technique(search_results: str, tried: list) -> str:
    """Pick a technique from phase 1 results that hasn't been tried yet.
    Baseline is pointwise log-loss (Phase 0 restored the shipped FM), so all
    four of these -- including BPR -- are genuine moves away from it."""
    candidates = [
        ("bpr", "Bayesian Personalised Ranking BPR per-user pairwise loss"),
        ("softmax", "softmax listwise loss recommender"),
        ("focal", "focal loss class imbalance"),
        ("warp", "WARP loss pairwise ranking hard negative mining"),
    ]
    for key, technique in candidates:
        if f"loss:{key}" not in tried:
            return technique
    return "Bayesian Personalised Ranking BPR per-user pairwise loss"  # fallback


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
    # Values can span multiple lines (numbered lists, multi-sentence text) and
    # Claude sometimes decorates the label ("**CODE_INSTRUCTION:**",
    # "CODE_INSTRUCTION :"). Capture each field from its label to the next
    # known label (or end of text) instead of only its first line -- the
    # first-line scan left code_writer with an empty code_change_instruction
    # whenever the model answered with a list.
    labels = ["HYPOTHESIS", "LOSS_CHOICE", "CODE_INSTRUCTION", "REASONING"]
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
