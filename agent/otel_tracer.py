# agent/otel_tracer.py
# LangGraph callback handler that auto-captures all node activity.
# Attach to graph -- zero changes needed to individual nodes.

import json
import os
import time
import uuid
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

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


class EmbedflixTracer(BaseCallbackHandler):
    """
    Drop-in LangGraph callback handler.
    Captures every node start/end and every LLM call automatically.
    Writes OTel-structured spans to JSONL.
    Person B attaches this to the graph in graph.py:
        graph.invoke(state, config={"callbacks": [EmbedflixTracer()]})

    2026-08-28 fixes (verified empirically against langgraph==1.2.11 /
    langchain-core==1.6.1 -- this is not a guess, it's what a real
    StateGraph.invoke(..., config={"callbacks": [...]}) call actually sends):

    1. MUST subclass BaseCallbackHandler. A bare duck-typed class crashes
       immediately on the first node -- langchain-core's callback manager
       reads handler.ignore_chain / handler.raise_error etc. before
       dispatching, and those only exist on BaseCallbackHandler subclasses.

    2. on_chain_start's `serialized` arg is always None for plain-function
       StateGraph nodes on this version -- serialized.get("name", ...) would
       crash with AttributeError. The real node name is in kwargs["name"]
       (also kwargs["metadata"]["langgraph_node"]).

    3. on_chain_end never receives an "_node_name" key in outputs -- nothing
       in agent.py or the specialists sets one, and there'd be no way to
       thread it through StateGraph's return values even if they did. The
       correct correlation key across on_chain_start/on_chain_end for the
       SAME node execution is kwargs["run_id"] (a stable UUID per node call),
       not the node name -- important because a node like "supervisor" runs
       many times across iterations, so keying by name alone reused across
       loop iterations would just be wrong.

    Known limitation, not fixed here (would need a design decision with
    Person B, not just a bugfix): on_llm_start/on_llm_end only fire for
    LangChain-wrapped LLM calls (e.g. langchain_anthropic.ChatAnthropic).
    Every specialist, the supervisor, and the judge call the raw
    `anthropic.Anthropic().messages.create(...)` SDK directly -- that
    bypasses LangChain's callback system entirely. So as long as that's true,
    these two hooks never fire, and the summary's TOKEN USAGE / API CALLS
    section will always read 0. Fixing that means switching those calls to
    go through a LangChain-callback-aware path, which is a bigger change
    than this file should make unilaterally.
    """

    def __init__(self):
        super().__init__()
        _run_stats["start_time"] = time.time()
        self._span_stack = {}  # run_id -> {node_name, span_id, start_time}
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
        """Fires when any node (and the graph itself) starts."""
        run_id = kwargs.get("run_id")
        metadata = kwargs.get("metadata") or {}
        node_name = metadata.get("langgraph_node") or kwargs.get("name", "unknown")

        # The outermost "LangGraph" span wraps the whole run, not one node --
        # track it (for the run-level start marker) but don't count it as a
        # node timing entry.
        if node_name == "LangGraph":
            return

        span_id = _new_id()
        self._span_stack[run_id] = {
            "node_name": node_name,
            "span_id": span_id,
            "start_time": time.time(),
        }
        iteration = inputs.get("iteration", self._iteration) if isinstance(inputs, dict) else self._iteration
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
        """Fires when any node (and the graph itself) finishes -- captures
        hypothesis, fix, reasoning. Correlated to its on_chain_start via
        run_id, not node name (outputs never carries a node-name key)."""
        run_id = kwargs.get("run_id")
        span_info = self._span_stack.pop(run_id, None)
        if span_info is None:
            # The outermost "LangGraph" run_id, or a run_id we never saw a
            # start for -- nothing to report.
            return

        node_name = span_info["node_name"]
        duration = round(time.time() - span_info["start_time"], 2)

        if node_name not in _run_stats["node_timings"]:
            _run_stats["node_timings"][node_name] = []
        _run_stats["node_timings"][node_name].append(duration)

        outputs = outputs if isinstance(outputs, dict) else {}
        _write_span({
            "name": f"embedflix.node.{node_name}.end",
            "trace_id": _trace_id,
            "span_id": _new_id(),
            "parent_span_id": span_info["span_id"],
            "timestamp": _now(),
            "duration_seconds": duration,
            "attributes": {
                "agent.node": node_name,
                "agent.iteration": self._iteration,
                "agent.hypothesis": outputs.get("hypothesis", ""),
                "agent.reasoning": outputs.get("reasoning", ""),
                "agent.code_change": (outputs.get("code_change_instruction") or "")[:300],
                "agent.verdict": outputs.get("verdict", ""),
                "agent.next_specialist": outputs.get("next_specialist", ""),
            }
        })
        _print(f"✅ [{self._iteration}] {node_name.upper()} done ({duration}s)")
        if outputs.get("hypothesis"):
            _print(f"   💡 {outputs['hypothesis'][:100]}")

        # Log web search results if present in state
        if outputs.get("_phase1_results"):
            _write_span({
                "name": "embedflix.web_search.phase1",
                "trace_id": _trace_id,
                "span_id": _new_id(),
                "timestamp": _now(),
                "attributes": {
                    "agent.node": node_name,
                    "agent.iteration": self._iteration,
                    "search.phase": str(outputs.get("_phase1_label") or "concept_discovery"),
                    "search.results": str(outputs.get("_phase1_results", ""))[:500],
                    "search.query": str(outputs.get("_phase1_query", "")),
                }
            })
        if outputs.get("_phase2_results"):
            _write_span({
                "name": "embedflix.web_search.phase2",
                "trace_id": _trace_id,
                "span_id": _new_id(),
                "timestamp": _now(),
                "attributes": {
                    "agent.node": node_name,
                    "agent.iteration": self._iteration,
                    "search.phase": str(outputs.get("_phase2_label") or "code_blueprint"),
                    "search.results": str(outputs.get("_phase2_results", ""))[:500],
                    "search.query": str(outputs.get("_phase2_query", "")),
                }
            })

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Fires on every LangChain-wrapped LLM call. See the class
        docstring's Known limitation -- this never fires for the raw
        anthropic.Anthropic() SDK calls this codebase actually uses."""
        _run_stats["total_api_calls"] += 1

    def on_llm_end(self, response, **kwargs):
        """Fires when a LangChain-wrapped LLM responds. Same limitation as
        on_llm_start -- dead code path until specialist LLM calls go through
        a LangChain-callback-aware client."""
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
        os.makedirs(os.path.dirname(_abs(SUMMARY_PATH)), exist_ok=True)
        with open(_abs(SUMMARY_PATH), "w") as f:
            f.write(summary)
        print(summary)


# ── helpers ───────────────────────────────────────────────────────────────────

def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _abs(relative_path: str) -> str:
    """Resolve log/summary paths relative to this file, not the process's
    current working directory -- keeps output in the same place regardless
    of where the agent is launched from (matches mcp_server.py's pattern)."""
    return os.path.join(_here(), "..", relative_path)


def _new_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:16]

def _now() -> str:
    return datetime.now().isoformat()

def _elapsed() -> float:
    if _run_stats["start_time"] is None:
        return 0.0
    return round(time.time() - _run_stats["start_time"], 2)

def _write_span(span: dict):
    os.makedirs(os.path.dirname(_abs(LOG_PATH)), exist_ok=True)
    with open(_abs(LOG_PATH), "a") as f:
        f.write(json.dumps(span) + "\n")

def _print(msg: str):
    print(msg, flush=True)
