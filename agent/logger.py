# agent/logger.py
# Central logger for the full agent run.
# Tracks: node activity, timing, tokens, API calls, human interventions, metrics.

import json
import time
import os
from datetime import datetime

LOG_PATH = "logs/run_log.jsonl"
SUMMARY_PATH = "logs/run_summary.txt"

# Global counters for the full run
_run_stats = {
    "total_api_calls": 0,
    "total_tokens_in": 0,
    "total_tokens_out": 0,
    "human_interventions": 0,
    "start_time": None,
}

def init_run():
    """Call once at the start of the agent run."""
    os.makedirs("logs", exist_ok=True)
    _run_stats["start_time"] = time.time()
    _entry = {
        "event": "run_start",
        "timestamp": datetime.now().isoformat(),
        "message": "Agent run started"
    }
    _write(_entry)
    _print_live("🚀 AGENT RUN STARTED", "")


def log_node_start(node_name: str, iteration: int):
    """Call at the start of every node."""
    _entry = {
        "event": "node_start",
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "node": node_name,
        "wall_clock_seconds": _elapsed(),
    }
    _write(_entry)
    _print_live(f"▶ [{iteration}] {node_name.upper()}", "starting...")


def log_node_result(
    node_name: str,
    iteration: int,
    hypothesis: str = "",
    issue_found: str = "",
    proposed_fix: str = "",
    reasoning: str = "",
    duration_seconds: float = 0,
    tokens_in: int = 0,
    tokens_out: int = 0,
):
    """Call at the end of every node with what it found and decided."""
    _run_stats["total_api_calls"] += 1
    _run_stats["total_tokens_in"] += tokens_in
    _run_stats["total_tokens_out"] += tokens_out

    _entry = {
        "event": "node_result",
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "node": node_name,
        "hypothesis": hypothesis,
        "issue_found": issue_found,
        "proposed_fix": proposed_fix,
        "reasoning": reasoning,
        "duration_seconds": round(duration_seconds, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_clock_seconds": _elapsed(),
    }
    _write(_entry)

    # Print live summary
    _print_live(f"✅ [{iteration}] {node_name.upper()} DONE ({duration_seconds:.1f}s)", "")
    if issue_found:
        _print_live("  🔍 ISSUE FOUND", issue_found)
    if hypothesis:
        _print_live("  💡 HYPOTHESIS", hypothesis)
    if proposed_fix:
        _print_live("  🔧 PROPOSED FIX", proposed_fix[:120] + "..." if len(proposed_fix) > 120 else proposed_fix)
    _print_live(f"  🪙 TOKENS", f"in={tokens_in} out={tokens_out} | total so far: in={_run_stats['total_tokens_in']} out={_run_stats['total_tokens_out']}")


def log_scores(iteration: int, gauc: float, ndcg5: float, primary: float, best_so_far: float):
    """Call after every pipeline run with the new scores."""
    delta = primary - 0.6016
    is_best = primary > best_so_far

    _entry = {
        "event": "scores",
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "gauc": gauc,
        "ndcg5": ndcg5,
        "primary": primary,
        "delta_vs_baseline": round(delta, 4),
        "is_best": is_best,
        "wall_clock_seconds": _elapsed(),
    }
    _write(_entry)

    flag = "🏆 NEW BEST!" if is_best else ""
    _print_live(
        f"📊 [{iteration}] SCORES",
        f"GAUC={gauc:.4f} | nDCG@5={ndcg5:.4f} | primary={primary:.4f} | delta={delta:+.4f} vs baseline {flag}"
    )


def log_error(iteration: int, node_name: str, error: str, recovery_action: str = ""):
    """Call when a node hits an error."""
    _entry = {
        "event": "error",
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "node": node_name,
        "error": error,
        "recovery_action": recovery_action,
        "wall_clock_seconds": _elapsed(),
    }
    _write(_entry)
    _print_live(f"❌ [{iteration}] ERROR in {node_name}", error)
    if recovery_action:
        _print_live(f"  🔄 RECOVERY", recovery_action)


def log_human_intervention(iteration: int, reason: str, action_taken: str):
    """Call whenever a human manually intervenes."""
    _run_stats["human_interventions"] += 1
    _entry = {
        "event": "human_intervention",
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "reason": reason,
        "action_taken": action_taken,
        "intervention_count": _run_stats["human_interventions"],
        "wall_clock_seconds": _elapsed(),
    }
    _write(_entry)
    _print_live(
        f"🙋 HUMAN INTERVENTION #{_run_stats['human_interventions']}",
        f"Reason: {reason} | Action: {action_taken}"
    )


def log_convergence(iteration: int, reason: str):
    """Call when the agent decides to stop."""
    _entry = {
        "event": "convergence",
        "timestamp": datetime.now().isoformat(),
        "iteration": iteration,
        "reason": reason,
        "wall_clock_seconds": _elapsed(),
    }
    _write(_entry)
    _print_live(f"🏁 CONVERGED at iteration {iteration}", reason)
    write_summary(iteration)


def write_summary(final_iteration: int):
    """Write a human-readable summary at the end of the run."""
    elapsed = _elapsed()
    total_tokens = _run_stats["total_tokens_in"] + _run_stats["total_tokens_out"]

    summary = f"""
========================================
EMBEDFLIX AGENT RUN SUMMARY
========================================
Total iterations:        {final_iteration}
Total wall clock:        {elapsed:.1f}s ({elapsed/60:.1f} min)
Total API calls:         {_run_stats['total_api_calls']}
Total tokens (in):       {_run_stats['total_tokens_in']}
Total tokens (out):      {_run_stats['total_tokens_out']}
Total tokens combined:   {total_tokens}
Human interventions:     {_run_stats['human_interventions']}
========================================
Full log: {LOG_PATH}
========================================
"""
    with open(SUMMARY_PATH, "w") as f:
        f.write(summary)
    print(summary)


# ── internal helpers ──────────────────────────────────────────────────────────

def _elapsed() -> float:
    if _run_stats["start_time"] is None:
        return 0.0
    return round(time.time() - _run_stats["start_time"], 2)


def _write(entry: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _print_live(label: str, value: str):
    print(f"{label}: {value}" if value else label)
