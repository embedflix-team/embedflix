"""Real entrypoint: builds the graph, runs it, wires in the OTel tracer.

Nothing else in this repo actually calls .invoke() -- graph.py only defines
build_graph(tools); agent.py's/graph.py's own tests all invoke it inline.
This is the first checked-in place that runs the whole thing for real.

Usage:
    cd agent/
    python3 main.py

Requires (see README / .env.example):
    ANTHROPIC_API_KEY   -- supervisor, every specialist, judge, code_writer
    TAVILY_API_KEY      -- web_search (optional: degrades gracefully without it)

Kept synchronous (.invoke(), not .ainvoke()) deliberately -- every node
function in agent.py and every specialist is a plain sync function making
sync anthropic.Anthropic().messages.create() calls. Switching to ainvoke()
would need every node converted to async first; that's a bigger change than
this entrypoint should make on its own.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Must run BEFORE importing graph -- graph.py imports the specialists, and
# every specialist does `client = Anthropic()` at module level, which reads
# ANTHROPIC_API_KEY from the environment at import time. Load .env first or
# that import crashes even with a real .env file sitting right next to it.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from graph import build_graph
from tools_adapter import build_tools
from state import initial_state
from otel_tracer import EmbedflixTracer


def run_agent():
    tools = build_tools()
    compiled = build_graph(tools)
    tracer = EmbedflixTracer()

    state = initial_state()
    result = compiled.invoke(
        state,
        config={"callbacks": [tracer], "recursion_limit": 200},
    )

    tracer.write_summary(
        final_iteration=result["iteration"],
        best_primary=result["best_scores"].get("primary", 0.0),
    )
    return result


if __name__ == "__main__":
    run_agent()
