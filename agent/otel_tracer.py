# agent/otel_tracer.py
# LangGraph callback handler that auto-captures all node activity.
# Attach to graph — zero changes needed to individual nodes.

import json
import time
import uuid
from datetime import datetime

LOG_PATH = "logs/otel_trace.jsonl"
SUMMARY_PATH = "logs/run_summary.txt"

# OTel-style run-level IDs
_trace_id = str(uuid.uuid4()).replace("-", "")
_run_stats = {
    "start_time": None,
    "total_tokens_in": 0,
    "total_tokens_out": 0,
    "total_api_calls": 0,
    "human_interventions": 0,
    "node_timings": {},  # node_name -> [durations]
}


class EmbedflixTracer:
    """
    Drop-in LangGraph callback handler.
    Captures every node start/end and every LLM call automatically.
    Writes OTel-structured spans to JSONL.
    Person B attaches this to the graph in graph.py:
        graph.invoke(state, config={"callbacks": [EmbedflixTracer()]})
    """

    def __init__(self):
        _run_stats["start_time"] = time.time()
        self._span_stack = {}  # node_name -> {span_id, start_time}
        self._iteration = 0
        _write_span({
            "name": "embedflix.run.start",
            "trace_id": _trace_id,
            "span_id": _new_id(),
            "timestamp": _now(),
            "attributes": {
                "agent.name": "embedflix",
                "agent.baseline": 0.6016,
            }
        })
        _print(f"🚀 EMBEDFLIX AGENT STARTED | trace_id={_trace_id[:8]}...")

    # ── LangGraph hooks ──────────────────────────────────────────

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Fires when any node starts."""
        node_name = serialized.get("name", "unknown")
        span_id = _new_id()
        self._span_stack[node_name] = {
            "span_id": span_id,
            "start_time": time.time()
        }
        iteration = inputs.get("iteration", self._iteration)
        self._iteration = iteration

        _write_span({
            "name": f"embedflix.node.{node_name}",
            "trace_id": _trace_id,
            "span_id": span_id,
            "parent_span_id": _trace_id[:32],
            "timestamp": _now(),
            "attributes": {
                "agent.node": node_name,
                "agent.iteration": iteration,
                "gen_ai.system": "anthropic",
            }
        })
        _print(f"▶  [{iteration}] {node_name.upper()} starting...")

    def on_chain_end(self, outputs, **kwargs):
        """Fires when any node finishes — captures hypothesis, fix, reasoning."""
        node_name = outputs.get("_node_name", "unknown")
        span_info = self._span_stack.pop(node_name, {})
        duration = round(time.time() - span_info.get("start_time", time.time()), 2)

        # Track per-node timing
        if node_name not in _run_stats["node_timings"]:
            _run_stats["node_timings"][node_name] = []
        _run_stats["node_timings"][node_name].append(duration)

        _write_span({
            "name": f"embedflix.node.{node_name}.end",
            "trace_id": _trace_id,
            "span_id": _new_id(),
            "parent_span_id": span_info.get("span_id", ""),
            "timestamp": _now(),
            "duration_seconds": duration,
            "attributes": {
                "agent.node": node_name,
                "agent.iteration": self._iteration,
                "agent.hypothesis": outputs.get("hypothesis", ""),
                "agent.reasoning": outputs.get("reasoning", ""),
                "agent.code_change": outputs.get("code_change_instruction", "")[:300],
                "agent.verdict": outputs.get("verdict", ""),
                "agent.next_specialist": outputs.get("next_specialist", ""),
            }
        })
        _print(f"✅ [{self._iteration}] {node_name.upper()} done ({duration}s)")
        if outputs.get("hypothesis"):
            _print(f"   💡 {outputs['hypothesis'][:100]}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Fires on every Claude API call."""
        _run_stats["total_api_calls"] += 1

    def on_llm_end(self, response, **kwargs):
        """Fires when Claude responds — captures token usage."""
        try:
            usage = response.llm_output.get("token_usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            _run_stats["total_tokens_in"] += tokens_in
            _run_stats["total_tokens_out"] += tokens_out

            _write_span({
                "name": "gen_ai.completion",
                "trace_id": _trace_id,
                "span_id": _new_id(),
                "timestamp": _now(),
                "attributes": {
                    "gen_ai.system": "anthropic",
                    "gen_ai.usage.input_tokens": tokens_in,
                    "gen_ai.usage.output_tokens": tokens_out,
                    "gen_ai.usage.total_tokens_so_far":
                        _run_stats["total_tokens_in"] + _run_stats["total_tokens_out"],
                }
            })
            _print(f"   🪙 tokens: in={tokens_in} out={tokens_out} | "
                   f"total={_run_stats['total_tokens_in']+_run_stats['total_tokens_out']}")
        except Exception:
            pass

    def on_chain_error(self, error, **kwargs):
        """Fires on any node error."""
        _write_span({
            "name": "embedflix.error",
            "trace_id": _trace_id,
            "span_id": _new_id(),
            "timestamp": _now(),
            "attributes": {
                "error.type": type(error).__name__,
                "error.message": str(error)[:500],
                "agent.iteration": self._iteration,
            }
        })
        _print(f"❌ ERROR: {error}")

    # ── Score + convergence logging (called manually from score_analyst) ──

    def log_scores(self, iteration: int, gauc: float, ndcg5: float, primary: float, best: float):
        delta = round(primary - 0.6016, 4)
        is_best = primary > best
        _write_span({
            "name": "embedflix.scores",
            "trace_id": _trace_id,
            "span_id": _new_id(),
            "timestamp": _now(),
            "attributes": {
                "agent.iteration": iteration,
                "metrics.gauc": gauc,
                "metrics.ndcg5": ndcg5,
                "metrics.primary": primary,
                "metrics.delta_vs_baseline": delta,
                "metrics.is_best": is_best,
                "agent.wall_clock_seconds": _elapsed(),
            }
        })
        flag = "🏆 NEW BEST!" if is_best else ""
        _print(f"📊 [{iteration}] GAUC={gauc:.4f} | nDCG@5={ndcg5:.4f} | "
               f"primary={primary:.4f} | Δ={delta:+.4f} {flag}")

    def log_human_intervention(self, iteration: int, reason: str, action: str):
        _run_stats["human_interventions"] += 1
        _write_span({
            "name": "embedflix.human_intervention",
            "trace_id": _trace_id,
            "span_id": _new_id(),
            "timestamp": _now(),
            "attributes": {
                "agent.iteration": iteration,
                "intervention.count": _run_stats["human_interventions"],
                "intervention.reason": reason,
                "intervention.action": action,
            }
        })
        _print(f"🙋 HUMAN INTERVENTION #{_run_stats['human_interventions']}: {reason}")

    def write_summary(self, final_iteration: int, best_primary: float):
        elapsed = _elapsed()
        total_tokens = _run_stats["total_tokens_in"] + _run_stats["total_tokens_out"]
        avg_timings = {k: round(sum(v)/len(v), 2)
                       for k, v in _run_stats["node_timings"].items()}

        summary = f"""
╔══════════════════════════════════════════════╗
║         EMBEDFLIX AGENT RUN SUMMARY          ║
╚══════════════════════════════════════════════╝
Trace ID:              {_trace_id[:16]}...
Total iterations:      {final_iteration}
Wall clock:            {elapsed:.1f}s  ({elapsed/60:.1f} min)
Best primary score:    {best_primary:.4f}  (Δ {best_primary-0.6016:+.4f} vs baseline)

TOKEN USAGE
  Input tokens:        {_run_stats['total_tokens_in']}
  Output tokens:       {_run_stats['total_tokens_out']}
  Total tokens:        {total_tokens}
  Total API calls:     {_run_stats['total_api_calls']}

AUTONOMY
  Human interventions: {_run_stats['human_interventions']}

NODE TIMING (avg seconds per call)
{chr(10).join(f"  {k}: {v}s" for k, v in avg_timings.items())}

Trace log: {LOG_PATH}
"""
        import os
        os.makedirs("logs", exist_ok=True)
        with open(SUMMARY_PATH, "w") as f:
            f.write(summary)
        print(summary)


# ── helpers ───────────────────────────────────────────────────────────────────

def _new_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:16]

def _now() -> str:
    return datetime.now().isoformat()

def _elapsed() -> float:
    if _run_stats["start_time"] is None:
        return 0.0
    return round(time.time() - _run_stats["start_time"], 2)

def _write_span(span: dict):
    import os
    os.makedirs("logs", exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(span) + "\n")

def _print(msg: str):
    print(msg, flush=True)