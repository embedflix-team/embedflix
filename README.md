# embedflix — Person B scaffold (execution & control flow)

Wired against the **real** `agent/mcp_server.py` (not a stand-in) — `agent/tools_adapter.py`
imports its functions directly and wraps them in `.invoke({...})`, so there's exactly one
copy of the tool logic and nothing can drift.

## Layout

```
starter-kit/              organizer files, edited IN PLACE by the agent (baseline.py, data.py)
  KuaiRand-Pure/data/        currently holds SYNTHETIC test fixture data (same schema, fake
                              values) -- swap for the real download before a real run:
                              wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz

agent/
  mcp_server.py            your MCP server, verbatim
  tools_adapter.py         build_tools() -> dict of .invoke()-able wrappers around mcp_server's
                            functions, for in-process testing without stdio transport
  state.py                 AgentState TypedDict -- reconciled against Person A's ACTUAL
                            committed specialist code, not a paraphrased spec
  agent.py                 baseline_verifier, code_writer, pipeline_runner, error_recovery,
                            score_analyst, log_and_track, convergence_checker
  graph.py                 build_graph(tools) -- wires all 15 nodes into a compiled StateGraph
  specialists/             Person A's real nodes: supervisor.py, loss_function_changer.py,
                            sequence_modeller.py, multitask_trainer.py, model_swapper.py,
                            training_optimizer.py, experiment_judge.py (not wired yet, see below)

checkpoints/, run_log.jsonl, resource_log.json   created by the tools as the agent runs
```

## Verified working (against real mcp_server.py + real specialist code, synthetic fixture data)

- Full graph compiles and runs end-to-end: `baseline_verifier` → `supervisor` → `[specialist]`
  → `code_writer` → `pipeline_runner` → `score_analyst` → `judge` → `log_and_track` →
  `convergence_checker` → loops back to `supervisor` or stops. Tested with every Anthropic call
  stubbed (no live API key in this environment) to isolate control flow / state shape from LLM
  output quality — real prose in, real prose out is untested until someone runs it with a key.
- **`code_writer`** is now an LLM-based editor, not a regex parser: it takes the specialist's
  free-form English `code_change_instruction` + the actual current file contents, asks an LLM to
  produce an exact `FILE:`/`OLD_CODE:`/`NEW_CODE:` block, verifies `OLD_CODE` is a verbatim
  substring of the file before applying it, and retries once with corrective feedback if the LLM's
  first attempt doesn't match. Unit-tested: clean parse, fenced-code-block stripping, malformed
  response, protected-file refusal (never calls `edit_file` for a protected file), and the
  retry-then-succeed path against the real `baseline.py` (edit applied, then reverted).
- Crash path: `pipeline_runner` catches a broken pipeline, routes to `error_recovery`,
  which restores from `best_iteration`'s checkpoint and clears the error
- **Safety check**: `code_writer` refuses any edit targeting `evaluate.py`, even though
  `mcp_server.edit_file` itself has no such allowlist — this is a deliberate defense-in-depth
  addition since the "never modify evaluate.py" rule has no other enforcement point
- **`judge`** (`specialists/experiment_judge.py`, 2026-08-28 rewrite) is wired in for real — it's
  reasoning-only, touches no tools, no checkpoints, no `experiment_history`, and returns exactly
  `verdict`/`analysis`/`learning`/`next_priority`/`reasoning`, which `log_and_track` folds into the
  `log_iteration` call. All 15 nodes (supervisor, 5 specialists, judge, all of Person B's execution
  nodes) are the real thing now, nothing stubbed.

## Research grounding: web_search + read_papers (2026-08-29)

Every specialist does two-phase live research before proposing a change: `web_search`
(Tavily-backed, `search_type="concept"` then `search_type="code"`) discovers a technique and
finds a real implementation to adapt. `read_papers` (`agent/knowledge_base.py`, ChromaDB) is
the offline fallback/grounding layer — domain-tagged curated chunks (BPR/softmax/focal loss,
DIN sequence modeling, MTL, dataset facts, organizer hints) that specialists can query scoped
to their own area. Both tools return `{"results": .., ...}` and never raise — `web_search`
degrades to "proceed with your existing knowledge" on failure, `read_papers` degrades to plain
keyword matching over the same chunks if ChromaDB/its embedding model isn't available.

**Setup note:** `read_papers`'s real vector search needs ChromaDB's embedding model
(`all-MiniLM-L6-v2`, ONNX) downloaded once, which needs actual unrestricted internet — it
failed with a 403 from both this repo's own cloud sandbox and the device-bridge VM used to
build this (both have restricted network egress), so **run once in a real terminal on a
normal machine** to warm the cache before relying on it — after that it works fully offline.
Until then it silently uses the keyword fallback, which still works, just less precisely.

`agent/main.py` is the actual entrypoint — nothing else in the repo calls `.invoke()`. It
wires `build_tools()` + `build_graph()` + the OTel tracer (`agent/otel_tracer.py`) together and
runs the whole loop. Run `python3 main.py` from inside `agent/` (needs `ANTHROPIC_API_KEY` at
minimum — see `.env.example`).

## Known gaps

- `error_recovery`'s LLM diagnosis call isn't wired — current version does deterministic
  pattern-matching (torch missing → install; anything else → restore).
- Overfit checking (val-vs-test gap) is deliberately **not** implemented: `mcp_server.parse_scores`
  only exposes the valid split, and whether the local "test" split is safe to touch during
  development is one of the open webinar questions (Q21).
- `otel_tracer.py`'s `on_llm_start`/`on_llm_end` hooks never fire — every specialist/supervisor/
  judge calls `anthropic.Anthropic()` directly rather than a LangChain-wrapped LLM, which is what
  LangChain's callback system actually requires to see an LLM call. `run_summary.txt`'s TOKEN
  USAGE section will read 0 until specialist LLM calls go through a callback-aware path — a
  bigger design decision than a bugfix, left as-is for now.
