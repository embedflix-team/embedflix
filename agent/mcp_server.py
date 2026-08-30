"""
KuaiRand Hackathon MCP Server — built with FastMCP 4
Install:  pip install fastmcp
Run:      python mcp_server.py        (stdio, for Claude Desktop / Claude Code)
          fastmcp dev mcp_server.py   (dev mode with inspector UI)
Every tool is a plain Python function — FastMCP auto-generates the JSON schema
from type annotations and docstrings, validates inputs, and handles transport.
"""
import json
import os
import re
import shutil
import subprocess
import time
from typing import Annotated
from fastmcp import FastMCP
# ── Server setup ─────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="KuaiRandAgentTools",
    instructions=(
        "Tools for an autonomous ML research agent iterating on the "
        "KuaiRand-Pure recommender-system benchmark. "
        "Each tool corresponds to one action the agent can take: "
        "run the pipeline, read/edit code, checkpoint, log, and submit."
    ),
)
# Resolve paths relative to this file so the server works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
STARTER_KIT = os.path.join(_HERE, "../starter-kit")
CHECKPOINTS = os.path.join(_HERE, "../checkpoints")
LOG_PATH = os.path.join(_HERE, "../run_log.jsonl")
RESOURCE_LOG = os.path.join(_HERE, "../resource_log.json")
# ── Tool 1: Run the FM pipeline ───────────────────────────────────────────────
@mcp.tool
def run_pipeline(
    extra_args: Annotated[str, "Extra CLI flags passed to baseline.py (e.g. '--model fm --seed 1')"] = "",
) -> str:
    """
    Runs baseline.py --model fm in the starter-kit directory.
    Returns the combined stdout + stderr so the agent can parse scores
    and errors. Timeout is 300 s (covers a full FM training run).
    """
    cmd = f"python3 baseline.py --model fm {extra_args}".strip()
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,
        cwd=STARTER_KIT,
    )
    return result.stdout + result.stderr
# ── Tool 2: Read a file ───────────────────────────────────────────────────────
@mcp.tool
def read_file(
    file_path: Annotated[str, "Path relative to the starter-kit directory (e.g. 'data.py')"],
) -> str:
    """
    Reads any text file inside the starter-kit directory.
    The agent uses this to inspect code before editing it — always read
    before you write.
    """
    full_path = os.path.join(STARTER_KIT, file_path)
    try:
        with open(full_path) as f:
            return f.read()
    except Exception as e:
        return f"ERROR: {e}"
# ── Tool 3: Edit a file ───────────────────────────────────────────────────────
@mcp.tool
def edit_file(
    file_path: Annotated[str, "Path relative to the starter-kit directory"],
    old_code: Annotated[str, "Exact string to find in the file (must appear exactly once)"],
    new_code: Annotated[str, "Replacement string"],
) -> str:
    """
    Replaces old_code with new_code in the specified file.
    Fails safely if the exact string is not found — no changes are written.
    Always call read_file first to confirm the string exists verbatim.
    """
    full_path = os.path.join(STARTER_KIT, file_path)
    try:
        with open(full_path) as f:
            original_content = f.read()
        if old_code not in original_content:
            return f"ERROR: exact string not found in {file_path}. No changes made."
        new_content = original_content.replace(old_code, new_code, 1)
        with open(full_path, "w") as f:
            f.write(new_content)

        # code_writer's edits are LLM-generated and can come back truncated
        # (e.g. a big multitask/DeepFM rewrite hitting max_tokens mid-write),
        # which silently lands as a syntax error on disk otherwise. Validate
        # before accepting the edit and revert if it doesn't parse.
        if full_path.endswith(".py"):
            import py_compile
            try:
                py_compile.compile(full_path, doraise=True)
            except py_compile.PyCompileError as e:
                with open(full_path, "w") as f:
                    f.write(original_content)
                return f"ERROR: syntax check failed after edit, reverted -- {e}"

        return f"SUCCESS: {file_path} updated."
    except Exception as e:
        return f"ERROR: {e}"
# ── Tool 4: Save checkpoint ───────────────────────────────────────────────────
@mcp.tool
def save_checkpoint(
    iteration: Annotated[int, "Current iteration number (1-indexed)"],
    primary_score: Annotated[float, "Validation primary score achieved at this checkpoint"],
) -> str:
    """
    Saves the current baseline.py and data.py as a numbered checkpoint.
    Call this whenever the validation primary score improves so that the
    best-performing code version is never lost.
    """
    folder = os.path.join(CHECKPOINTS, f"iter_{iteration}")
    os.makedirs(folder, exist_ok=True)
    for fname in ("baseline.py", "data.py"):
        shutil.copy(os.path.join(STARTER_KIT, fname), os.path.join(folder, fname))
    with open(os.path.join(folder, "score.json"), "w") as f:
        json.dump({"primary": primary_score, "iteration": iteration}, f)
    return f"Checkpoint saved: iter {iteration}, primary={primary_score:.4f}"
# ── Tool 5: Restore checkpoint ────────────────────────────────────────────────
@mcp.tool
def restore_checkpoint(
    iteration: Annotated[int, "Iteration number to restore from"],
) -> str:
    """
    Restores baseline.py and data.py from a previously saved checkpoint.
    Use this for error recovery when the current changes have made things
    worse and you want to roll back to the last good state.
    """
    folder = os.path.join(CHECKPOINTS, f"iter_{iteration}")
    if not os.path.exists(folder):
        return f"ERROR: No checkpoint found at iter {iteration}"
    for fname in ("baseline.py", "data.py"):
        shutil.copy(os.path.join(folder, fname), os.path.join(STARTER_KIT, fname))
    return f"Restored from iter {iteration}"
# ── Tool 6: Log one iteration ─────────────────────────────────────────────────
@mcp.tool
def log_iteration(
    iteration: Annotated[int, "Iteration number"],
    hypothesis: Annotated[str, "What the agent intended to try and why"],
    code_diff: Annotated[str, "Summary of the code change applied (not a full diff, just a description)"],
    gauc: Annotated[float, "Validation GAUC score"],
    ndcg: Annotated[float, "Validation nDCG@5 score"],
    primary: Annotated[float, "Validation primary score (mean of GAUC and nDCG@5)"],
    error: Annotated[str, "Any error message encountered during this iteration"] = "",
    recovery: Annotated[str, "How the agent recovered from the error, if any"] = "",
) -> str:
    """
    Appends one complete iteration record to run_log.jsonl.
    This log is a judging deliverable — it demonstrates hypothesis-driven
    reasoning and autonomous error recovery. Call after every iteration,
    including failed ones.
    """
    entry = {
        "iteration": iteration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis": hypothesis,
        "code_diff": code_diff,
        "score": {"GAUC": gauc, "nDCG@5": ndcg, "primary": primary},
        "error": error,
        "recovery": recovery,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return "Logged successfully"
# ── Tool 7: Track resource usage ──────────────────────────────────────────────
@mcp.tool
def track_resources(
    iteration: Annotated[int, "Iteration number"],
    tokens: Annotated[int, "Total LLM tokens used so far (input + output)"],
    wall_seconds: Annotated[float, "Wall-clock seconds elapsed for this iteration"],
) -> str:
    """
    Appends token + wall-clock usage for one iteration to resource_log.json.
    Feasibility scoring (15% of total) is based on total tokens and wall-clock
    time — track these every iteration so the final report is accurate.
    """
    data: list[dict] = []
    if os.path.exists(RESOURCE_LOG):
        with open(RESOURCE_LOG) as f:
            data = json.load(f)
    data.append({"iteration": iteration, "tokens": tokens, "wall_seconds": wall_seconds})
    with open(RESOURCE_LOG, "w") as f:
        json.dump(data, f, indent=2)
    return f"Tracked iter {iteration}: {tokens} tokens, {wall_seconds:.1f}s"
# ── Tool 8: Format and validate submission ────────────────────────────────────
@mcp.tool
def format_submission(
    split: Annotated[str, "Which split to submit: 'test' (for final) or 'valid' (for local scoring)"] = "test",
) -> str:
    """
    Generates a submission CSV with the FM baseline scores, then validates it.
    Returns the combined output of `submit.py --make` and `submit.py --check`.
    Run this before any final submission to confirm the file passes all checks.
    """
    make = subprocess.run(
        f"python3 submit.py --make --split {split} submission.csv",
        shell=True,
        capture_output=True,
        text=True,
        cwd=STARTER_KIT,
    )
    check = subprocess.run(
        f"python3 submit.py --check --split {split} submission.csv",
        shell=True,
        capture_output=True,
        text=True,
        cwd=STARTER_KIT,
    )
    return make.stdout + "\n" + check.stdout + check.stderr
# ── Tool 9: Parse scores from pipeline output ─────────────────────────────────
@mcp.tool
def parse_scores(
    pipeline_output: Annotated[str, "Raw stdout from a run_pipeline call"],
) -> dict:
    """
    Extracts GAUC, nDCG@5, and primary from a baseline.py output string.
    Returns a dict with keys GAUC, nDCG@5, primary (all float | None).
    Values are None when the pattern isn't found — this signals a parse
    failure and the agent should inspect pipeline_output for errors.
    """
    result: dict[str, float | None] = {"GAUC": None, "nDCG@5": None, "primary": None}
    pattern = (
        r"valid\s+GAUC\s+([\d.]+)\s+\|\s+nDCG@5\s+([\d.]+)\s+\|\s+primary\s+([\d.]+)"
    )
    match = re.search(pattern, pipeline_output)
    if match:
        result["GAUC"] = float(match.group(1))
        result["nDCG@5"] = float(match.group(2))
        result["primary"] = float(match.group(3))
    return result
# ── Tool 10: Web search (live research) ───────────────────────────────────────
@mcp.tool
def web_search(
    query: Annotated[str, "Search query"],
    search_type: Annotated[str, "'concept' for papers/ideas, 'code' for implementations"] = "concept",
    n_results: Annotated[int, "Number of results"] = 3,
) -> dict:
    """
    Searches the web for ML research. Use search_type='concept' first to
    discover techniques (e.g. "improve GAUC ranking loss" -> discovers "BPR
    loss"), then search_type='code' with the technique name to find a real
    implementation to adapt (e.g. "BPR loss numpy implementation").

    Returns {"results": <cleaned text>, "query": .., "search_type": ..} --
    dict shape matches what every specialist actually reads
    (concept_results.get("results", ...)), not a bare string.

    Requires TAVILY_API_KEY in the environment. If the key is missing or the
    search fails for any reason, "results" carries a message saying so
    instead of raising -- the agent should proceed on its own knowledge
    rather than stall the run over a flaky network call.
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        search_query = query
        if search_type == "code":
            # Code search targets GitHub / Papers with Code specifically.
            search_query = f"site:github.com {query} OR site:paperswithcode.com {query}"
        raw_results = client.search(search_query, max_results=n_results)
        cleaned = _clean_results(raw_results["results"], search_type)
    except Exception as e:
        cleaned = f"Search failed: {e}. Proceed with your existing knowledge."
    return {"results": cleaned, "query": query, "search_type": search_type}


_CODE_FENCE_RE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


def _clean_results(results: list, search_type: str) -> str:
    """Strips noise from search results before prompt injection."""
    cleaned = []
    for r in results:
        title = r.get("title", "untitled")
        content = r.get("content", "")
        content = re.sub(r"https?://\S+", "", content)
        if search_type == "code":
            # Prefer actual fenced code blocks if the page has any; fall back
            # to the raw (URL-stripped) content otherwise.
            fences = _CODE_FENCE_RE.findall(content)
            body = "\n\n".join(f.strip() for f in fences) if fences else content
            body = body[:800]
            cleaned.append(f"SOURCE: {title}\n{body}")
        else:
            cleaned.append(f"CONCEPT: {title}\n{content[:500]}")
    return "\n\n---\n\n".join(cleaned)


# ── Tool 11: Read local knowledge base (offline fallback for web_search) ──────
@mcp.tool
def read_papers(
    query: Annotated[str, "What to look up (e.g. 'BPR loss implementation')"],
    domain: Annotated[
        str,
        "Optional: scope to one specialist's area -- 'loss_function_changer', "
        "'sequence_modeller', 'multitask_trainer', 'model_swapper', "
        "'training_optimizer', or omit/'' for general dataset + organizer-hint content",
    ] = "",
    n_results: Annotated[int, "Number of reference chunks to return"] = 3,
) -> dict:
    """
    Retrieves curated ML reference material from the local knowledge base
    (agent/knowledge_base.py) -- BPR/softmax/focal loss math, DIN sequence
    modeling, multi-task learning, dataset facts, and organizer hints on
    what's already been tried with no gain. Works fully offline, no API key
    required (unlike web_search's Tavily dependency) -- use this when
    web_search fails, or any time as a cheap grounding check alongside it.

    Returns {"results": <formatted text>, "query": .., "domain": ..} -- same
    dict shape as web_search, so specialists can read either the same way
    (result.get("results", ...)).
    """
    from knowledge_base import query_knowledge_base

    domain_arg = domain if domain else None
    text = query_knowledge_base(query, domain=domain_arg, n_results=n_results)
    return {"results": text, "query": query, "domain": domain or "general"}


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # fastmcp dev mcp_server.py  →  hot-reload + inspector UI
    # python mcp_server.py       →  stdio for Claude Desktop / Claude Code
    mcp.run()
