# agent/specialists/training_optimizer.py
from anthropic import Anthropic
import re

client = Anthropic()

# (label, OLD_CODE, NEW_CODE, one-line description) -- each OLD_CODE is an
# exact substring of starter-kit/baseline.py's run_fm/FM defaults, verified
# unique in the file. "lr=0.001" alone also matches FM.__init__'s unrelated
# default, so lr's OLD_CODE includes "epochs=40" to disambiguate down to the
# run_fm signature (the one that actually reaches training).
CHANGES = [
    ("training:l2", "l2=1e-6", "l2=1e-5",
     "increase L2 weight regularization to reduce overfitting"),
    ("training:lr", "lr=0.001, epochs=40", "lr=0.0005, epochs=40",
     "halve the learning rate for more stable convergence"),
    ("training:patience", "patience=4", "patience=8",
     "double early-stopping patience so training doesn't stop before converging"),
    ("training:bs", "bs=8192", "bs=4096",
     "halve the batch size for noisier, potentially better-generalizing updates"),
]


def training_optimizer(state: dict, tools: dict) -> dict:
    """
    Proposes ONE of the four fixed, pre-verified single-line hyperparameter
    edits above to starter-kit/baseline.py -- deliberately not open-ended.
    The previous version let the LLM invent its own training tweak in free
    text, and code_writer's resulting edits sometimes never actually touched
    the real training call (dead code that didn't affect the score). Every
    OLD_CODE/NEW_CODE pair here is verified to exist verbatim and uniquely
    in baseline.py, so code_writer has nothing left to invent -- it's a
    guaranteed single-hunk edit that provably affects training. No web
    search: these are fixed, known-good edits, not a research question.
    """
    history_summary = _summarize_history(state.get("experiment_history", []))
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    untried = [c for c in CHANGES if c[0] not in tried]
    menu = untried or CHANGES  # all 4 exhausted -> allow re-picking, still single-line-safe

    menu_text = "\n".join(
        f"{i + 1}. {label} -- change `{old}` to `{new}` ({desc})"
        for i, (label, old, new, desc) in enumerate(menu)
    )

    prompt = f"""You are an ML expert improving a KuaiRand-Pure recommender system.

CURRENT SITUATION:
- Baseline: lr=0.001, batch_size=8192, Adam, early stopping patience=4, k=16, l2=1e-6
- Current primary score: {primary:.4f} (baseline to beat: 0.6016)
- Metric: primary = mean(GAUC, nDCG@5)
- Iteration: {state.get("iteration", 1)}

EXPERIMENT HISTORY:
{history_summary}

ALREADY TRIED: {tried}

You may ONLY choose ONE of the following exact, pre-verified single-line
changes to starter-kit/baseline.py -- do not propose anything else, do not
combine multiple changes, do not invent new code or new parameters:

{menu_text}

Pick the one most likely to help given the history above.

Respond in this exact format:
CHOICE: [just the number from the list above]
REASONING: [2-3 sentences of ML reasoning for judge logs -- why this change, given the history]
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    idx = _parse_choice(text, len(menu))
    reasoning = _parse_reasoning(text)
    label, old_code, new_code, desc = menu[idx]

    hypothesis = f"Training optimizer: {desc} ({label})."
    # code_change_instruction kept for human-readable logging only -- the
    # actual edit is applied deterministically via _deterministic_edit below,
    # which code_writer checks first and applies with zero LLM involvement.
    # No translation risk: the exact OLD_CODE/NEW_CODE is already known.
    code_instruction = (
        f"(deterministic) find the exact substring `{old_code}` and replace "
        f"it with `{new_code}` in starter-kit/baseline.py."
    )

    return {
        **state,
        "hypothesis": hypothesis,
        "code_change_instruction": code_instruction,
        "reasoning": reasoning or hypothesis,
        "tried_approaches": tried + [label],
        "_deterministic_edit": {
            "file": "baseline.py",
            "old_code": old_code,
            "new_code": new_code,
        },
    }


def _parse_choice(text: str, n: int) -> int:
    m = re.search(r"CHOICE[ \t]*\**[ \t]*:[ \t]*\**[ \t]*(\d+)", text, re.IGNORECASE)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n:
            return idx
    return 0


def _parse_reasoning(text: str) -> str:
    # Same multi-line-safe pattern as the other specialists: capture from the
    # label to the next known label (or end of text), not just its first line.
    m = re.search(r"REASONING[ \t]*\**[ \t]*:[ \t]*\**[ \t]*(.+)\Z", text,
                   re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().strip("*").strip() if m else ""


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
