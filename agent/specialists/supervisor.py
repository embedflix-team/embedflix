# agent/specialists/supervisor.py
from anthropic import Anthropic
import os

from specialists.feature_engineer import CANDIDATES as FEATURE_CANDIDATES

client = Anthropic()

# All specialist node names the supervisor can route to
SPECIALISTS = [
    "loss_function_changer",
    "sequence_modeller", 
    "multitask_trainer",
    "model_swapper",
    "training_optimizer",
    "feature_engineer",
]

def supervisor(state: dict, tools: dict) -> dict:
    """
    The brain of the agent. Looks at the full experiment history,
    current scores, what's been tried, and decides what to try next.
    This is what judges evaluate for autonomy and innovation quality.
    """
    tried = state.get("tried_approaches", [])

    # Forced control-flow override, not a prompt-level suggestion: per the
    # 2026-08-31 experiment design, try feature_engineer's full deterministic
    # menu before any other specialist, every time, deterministically -- no
    # LLM call for this decision. This file's own ROUTE_TO parser below is a
    # fragile single-line scanner (see _parse_response) that silently falls
    # back to loss_function_changer on a miss, so a prompt-only "always route
    # to feature_engineer first" rule isn't reliable enough for something
    # this run's outcome hinges on. Once all FEATURE_CANDIDATES labels appear
    # in tried_approaches, this no longer fires and normal LLM routing below
    # resumes (feature_engineer stays a normal, re-pickable ROUTE_TO option).
    feature_labels = [c[0] for c in FEATURE_CANDIDATES]
    untried_features = [l for l in feature_labels if l not in tried]
    if untried_features:
        reason = (
            f"Forced: feature_engineer menu not yet exhausted "
            f"({len(feature_labels) - len(untried_features)}/{len(feature_labels)} tried) -- "
            "trying the untested numeric engagement-statistics features before any other "
            "specialist, per the experiment design: if none of them improve the score, the "
            "run stops early instead of spending budget on loss/model/sequence changes."
        )
        return {
            **state,
            "next_specialist": "feature_engineer",
            "routing_reason": reason,
            "strategy": "Exhaust feature_engineer's deterministic menu first, then resume normal routing.",
            "reasoning": f"Supervisor routed to feature_engineer (forced): {reason}",
        }

    history_summary = _summarize_history(state.get("experiment_history", []))
    current = state.get("current_scores", {}).get("primary") or 0.6016
    best = state.get("best_scores", {}).get("primary") or 0.6016
    iteration = state.get("iteration", 1)

    prompt = f"""You are the supervisor of an autonomous ML research agent improving 
a KuaiRand-Pure video recommender system. Your job is to decide which 
specialist agent to call next to maximally improve the ranking score.

CURRENT STATE:
- Iteration: {iteration} / 50 max
- Current primary score: {current:.4f}
- Best primary score: {best:.4f}  
- Official baseline to beat: 0.6016
- Gap to baseline: {current - 0.6016:+.4f}
- Metric: primary = mean(GAUC, nDCG@5) — both are RANKING metrics

WHAT HAS BEEN TRIED:
{tried if tried else "Nothing yet — this is iteration 1"}

FULL EXPERIMENT HISTORY:
{history_summary}

DATASET CONTEXT (use this to guide specialist selection):
- Baseline uses only 5 fields: user_id, video_id, author_id, tab, dur_bucket
- KuaiRand-Pure has two feature files not yet used by baseline (real column
  names, via data.get_feature_info()):
  * user_features_pure.csv (30 columns) — user_active_degree,
    is_lowactive_period, is_live_streamer, is_video_author, follow_user_num(_range),
    fans_user_num(_range), friend_user_num(_range), register_days(_range),
    onehot_feat0..17
  * video_features_statistic_pure.csv (51 columns) — show_cnt, play_cnt,
    play_duration, complete/valid/long_time/short_time_play_cnt, play_progress,
    like_cnt, comment_cnt, follow_cnt, share_cnt, download_cnt, collect_cnt,
    report_cnt, reduce_similar_cnt, and *_user_num variants of most of these
- data.py handles data loading and can be modified alongside baseline.py
- IMPORTANT — already tested empirically (ablation_features.py against the
  real held-out split): adding the CATEGORICAL columns from these files
  (onehot_feat*, user_active_degree, video_type, etc.) as new FM fields makes
  the score WORSE (0.5953 -> 0.5936 / 0.5933), not better. The untested,
  promising direction is the NUMERIC engagement-statistics columns
  (play_cnt/like_cnt/show_cnt/comment_cnt/share_cnt/download_cnt/...),
  bucketed the same way dur_bucket already is — this is exactly what
  feature_engineer's menu does. Do not route to a specialist expecting a
  categorical feature dump to help; it already didn't.
- NEVER use log-file per-interaction columns (is_click, is_like, is_follow,
  is_comment, is_forward, play_time_ms) as FM input fields — LABEL
  ('long_view') is derived from play_time_ms/duration_ms on the same row, and
  the other is_* columns are simultaneous outcomes of the same impression, so
  using them as inputs would leak the label rather than genuinely improve
  ranking. (multitask_trainer's auxiliary-task use of click/like/play_time as
  a secondary training *target*, not an input feature, is a different and
  legitimate technique — that's not affected by this rule.)

AVAILABLE SPECIALISTS:
1. loss_function_changer — baseline already trains with BPR pairwise loss;
   this switches to softmax/focal/warp/pointwise instead
   Best when: model trains well but ranking quality is poor (low nDCG vs GAUC)
   
2. sequence_modeller — adds user history / recent watch features  
   Best when: model ignores temporal patterns, users have rich history
   
3. multitask_trainer — adds auxiliary tasks (click, like, play_time)
   Best when: long_view signal is sparse, other signals are abundant
   
4. model_swapper — upgrades FM to DeepFM/DCN/higher-k
   Best when: feature interactions are not well captured, score plateaus
   
5. training_optimizer — tunes lr, batch size, regularization, patience
   Best when: training is unstable, overfitting, or converging too early

6. feature_engineer — adds bucketed numeric video engagement stats
   (play_cnt/like_cnt/show_cnt/etc. from video_features_statistic_pure.csv,
   never used before) as new FM field domains
   Best when: you want to revisit the feature menu after its forced first
   pass -- note its 3-candidate menu already runs automatically before your
   first routing decision each run, so you'll rarely need to pick it
   yourself unless you want to re-try a candidate

ROUTING RULES:
- Never repeat a specialist that already failed to improve the score
- If score improved last iteration, try building on that approach first
- If score dropped last iteration, try a different specialist
- Check ALREADY TRIED / FULL EXPERIMENT HISTORY above before ruling anything
  out -- do not assume a specialist has been tried unless it actually
  appears there for this run
- If stuck for 2+ iterations, escalate to model_swapper
- Always explain your routing decision clearly for the judge logs

Decide which specialist to call next and why.

Respond in this exact format:
ROUTE_TO: [loss_function_changer | sequence_modeller | multitask_trainer | model_swapper | training_optimizer | feature_engineer]
ROUTING_REASON: [2-3 sentences explaining why this specialist now, what you expect it to improve, and what signal from history led to this decision]
STRATEGY: [one sentence — overall strategy for this phase of the run]
"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    parsed = _parse_response(text)

    # Validate route — fall back to loss_function_changer if invalid
    if parsed["route_to"] not in SPECIALISTS:
        parsed["route_to"] = "loss_function_changer"

    return {
        **state,
        "next_specialist": parsed["route_to"],
        "routing_reason": parsed["routing_reason"],
        "strategy": parsed["strategy"],
        "reasoning": f"Supervisor routed to {parsed['route_to']}: {parsed['routing_reason']}"
    }


def _summarize_history(history: list) -> str:
    if not history:
        return "No experiments yet. This is iteration 1."
    lines = []
    for h in history[-10:]:  # last 10 for supervisor — needs more context
        primary_val = h.get("primary")
        # primary_val can be None (e.g. a recovered iteration whose retry
        # never produced a real score) -- don't do arithmetic on None.
        delta_str = f"({primary_val - 0.6016:+.4f} vs baseline)" if primary_val is not None else "(no score -- recovery/error iteration)"
        lines.append(
            f"- Iter {h.get('iteration')}: [{h.get('specialist', '?')}] "
            f"{h.get('hypothesis', '?')} "
            f"→ primary {h.get('primary', '?')} {delta_str}"
        )
    return "\n".join(lines)


def _parse_response(text: str) -> dict:
    result = {
        "route_to": "loss_function_changer",
        "routing_reason": "",
        "strategy": ""
    }
    for line in text.split("\n"):
        if line.startswith("ROUTE_TO:"):
            result["route_to"] = line.replace("ROUTE_TO:", "").strip()
        elif line.startswith("ROUTING_REASON:"):
            result["routing_reason"] = line.replace("ROUTING_REASON:", "").strip()
        elif line.startswith("STRATEGY:"):
            result["strategy"] = line.replace("STRATEGY:", "").strip()
    return result