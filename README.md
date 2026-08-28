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

## Known gaps

- **`judge` is still a stub in `graph.py`.** `specialists/experiment_judge.py` is pushed, but as
  committed it's still the OLD interface: it calls `tools["log_iteration"]({...})` and
  `tools["save_checkpoint"]({...})` as direct dict calls (wrong shape vs. the real
  `mcp_server.py` signatures — those need positional/keyword args, not one dict), and it does its
  own checkpointing + `experiment_history` append. Per the confirmed plan, the rewrite should make
  it reasoning-only (no tool calls, no checkpointing, returns only
  `verdict`/`analysis`/`learning`/`next_priority`/`reasoning`) — `score_analyst` already owns
  checkpointing and `experiment_history` on that assumption. Once the rewritten file lands, swap
  the stub for the real import — one line, see the comment at the top of `graph.py`.
- `error_recovery`'s LLM diagnosis call isn't wired — current version does deterministic
  pattern-matching (torch missing → install; anything else → restore).
- Overfit checking (val-vs-test gap) is deliberately **not** implemented: `mcp_server.parse_scores`
  only exposes the valid split, and whether the local "test" split is safe to touch during
  development is one of the open webinar questions (Q21).
- `requirements.txt` (Person A's `pip freeze`) is missing `langgraph`, `fastmcp`, and
  `langchain-mcp-adapters` — needs consolidating before anyone else sets up a fresh environment.
