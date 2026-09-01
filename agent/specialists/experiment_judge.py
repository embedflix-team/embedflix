# agent/specialists/experiment_judge.py
from anthropic import Anthropic
import re

client = Anthropic()

def experiment_judge(state: dict, tools: dict) -> dict:
    """
    Analyzes experiment results and produces verdict/analysis for judge logs.
    Phase 1: web_search for context on what the metrics mean
    Phase 2: web_search for what the literature says about this result pattern
    """

    history_summary = _summarize_history(state.get("experiment_history", []))
    current = state.get("current_scores", {}).get("primary") or 0.6016
    best = state.get("best_scores", {}).get("primary") or 0.6016
    gauc = state.get("current_scores", {}).get("gauc") or 0.6674
    ndcg5 = state.get("current_scores", {}).get("ndcg5") or 0.5357
    hypothesis = state.get("hypothesis", "unknown")
    specialist = state.get("next_specialist", "unknown")
    iteration = state.get("iteration", 1)
    error = state.get("error_message", None)

    # PHASE 1 — look up what this metric pattern means
    concept_results = tools["web_search"]({
        "query": f"GAUC nDCG gap analysis recommender system ranking metric interpretation",
        "search_type": "concept",
        "n_results": 3
    })

    # PHASE 2 — look up what literature says about this experiment's approach.
    # Query off the hypothesis text, not the internal node name ("model_swapper"
    # etc.) which means nothing to a web search.
    approach = hypothesis if hypothesis and hypothesis != "unknown" else "recommender system ranking"
    code_results = tools["web_search"]({
        "query": f"{approach} recommender system when it works when it fails",
        "search_type": "concept",
        "n_results": 2
    })

    prompt = f"""You are an ML research judge reviewing an experiment run by an
autonomous recommender system agent.

EXPERIMENT JUST RUN:
- Iteration: {iteration}
- Specialist used: {specialist}
- Hypothesis: {hypothesis}
- Result: GAUC={gauc:.4f}, nDCG@5={ndcg5:.4f}, primary={current:.4f}
- Previous best: {best:.4f}
- Delta vs best: {current - best:+.4f}
- Delta vs baseline (0.6016): {current - 0.6016:+.4f}
- Error during run: {error if error else "None"}

FULL EXPERIMENT HISTORY:
{history_summary}

METRIC CONTEXT (from research):
{concept_results.get("results", "No results")}

SPECIALIST CONTEXT (from research):
{code_results.get("results", "No results")}

YOUR JOB:
Analyze this experiment result deeply using the research context above. Explain:
1. Did it work? Why or why not — grounded in the research?
2. What does the GAUC vs nDCG@5 split tell us?
   (GAUC improved but nDCG didn't = ranking order improved but top-5 precision didn't)
3. What should the agent learn from this for future iterations?
4. Is there a pattern emerging across experiments?

This analysis will be read by hackathon judges to assess the agent's
research quality and autonomous reasoning ability.

Respond in this exact format:
VERDICT: [improved | no_change | regression]
ANALYSIS: [3-4 sentences — what worked, what didn't, why, what the metric split reveals]
LEARNING: [1-2 sentences — what the agent should remember for future iterations]
NEXT_PRIORITY: [one sentence — what area looks most promising to try next]
REASONING: [one sentence summary for the run log]
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
        "verdict": parsed["verdict"],
        "analysis": parsed["analysis"],
        "learning": parsed["learning"],
        "next_priority": parsed["next_priority"],
        "reasoning": parsed["reasoning"],
        # judge bypasses _extract_specialist_output (returns full **state), so
        # it accumulates its own reasoning-call tokens directly.
        "total_tokens": state.get("total_tokens", 0)
                        + response.usage.input_tokens + response.usage.output_tokens,
    }


def _summarize_history(history: list) -> str:
    if not history:
        return "No experiments yet. This is iteration 1."
    lines = []
    for h in history[-10:]:
        lines.append(
            f"- Iter {h.get('iteration')}: [{h.get('specialist', '?')}] "
            f"{h.get('hypothesis', '?')} "
            f"→ primary {h.get('primary', '?')} "
            f"(verdict: {h.get('verdict', '?')})"
        )
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    result = {
        "verdict": "no_change",
        "analysis": "",
        "learning": "",
        "next_priority": "",
        "reasoning": ""
    }
    # Values can span multiple lines (ANALYSIS is 3-4 sentences) and Claude
    # sometimes decorates the label ("**ANALYSIS:**", "ANALYSIS :"). Capture
    # each field from its label to the next known label (or end of text)
    # instead of only its first line.
    labels = ["VERDICT", "ANALYSIS", "LEARNING", "NEXT_PRIORITY", "REASONING"]
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