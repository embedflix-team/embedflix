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
  state.py                 AgentState TypedDict
  agent.py                 baseline_verifier, code_writer, pipeline_runner, error_recovery,
                            score_analyst, log_and_track, convergence_checker -- all smoke-tested
                            against the real tools below

checkpoints/, run_log.jsonl, resource_log.json   created by the tools as the agent runs
```

## Verified working (against real mcp_server.py, synthetic fixture data)

- `baseline_verifier` → `code_writer` → `pipeline_runner` → `score_analyst` → `log_and_track`
  → `convergence_checker`, full success chain
- Crash path: `pipeline_runner` catches a broken pipeline, routes to `error_recovery`,
  which restores from `best_iteration`'s checkpoint and clears the error
- **Safety check**: `code_writer` refuses any edit targeting `evaluate.py`, even though
  `mcp_server.edit_file` itself has no such allowlist — this is a deliberate defense-in-depth
  addition since the "never modify evaluate.py" rule has no other enforcement point

## Known gaps (by design, not yet Person B's job)

- **`log_and_track`** (renamed from `experiment_judge`) is pure plumbing — it persists whatever
  reasoning is already in `state["hypothesis"]`. It does **not** generate the "why did this
  help" judgment text. That's Person A's `JUDGE_PROMPT` LLM call, which should run immediately
  before this node in the graph and populate that field. See the note at the bottom of `agent.py`.
- `error_recovery`'s LLM diagnosis call (`ERROR_RECOVERY_PROMPT`) isn't wired — current version
  does deterministic pattern-matching (torch missing → install; anything else → restore).
- Overfit checking (val-vs-test gap) is deliberately **not** implemented: `mcp_server.parse_scores`
  only exposes the valid split, and whether the local "test" split is safe to touch during
  development is one of the open webinar questions (Q21).
- Graph wiring (Step 4) not done yet — nodes are tested individually, not yet assembled into
  a `StateGraph`.
