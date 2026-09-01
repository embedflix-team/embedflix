# Embedflix — Devpost project description

*(Paste target for the Devpost form. Diagrams referenced here live in `deliverables/diagrams.md` — screenshot D1 and D2 into the Devpost gallery.)*

---

## Why this wins — the 5 points, up front

| # | Point | Criterion |
|---|---|---|
| **1** | **The insight.** GAUC + nDCG@5 are computed *strictly within each user's own ~5-impression list*, so any score term constant across a user's impressions is **mathematically invisible** to the metric. This predicts the organizers' own ablation (user features made the score *worse*) and redirects every agent proposal to item-side signal + user×item crosses. | Innovation (20%) |
| **2** | **Measurement integrity.** The repo's "baseline" was not the shipped FM — an unvalidated loss edit (its run-log entry scored `primary: None`) had been committed over it. We restored + pinned the official FM (reproduction test, 0.6016 ± 0.0003 / 5 seeds), and replaced a noise-accepting accept threshold with a **two-stage seed-confirmed gate + a no-op detector**. Every number is real. | Technical Execution (35%) |
| **3** | **The architecture.** FM logit → one feature inside a **LightGBM `lambdarank`** re-ranker that optimises nDCG@5 directly + train-only target encodings + engagement ratios, 5-seed rank-ensembled. We *document a rejected variant* (OOF stacking → overfit). | Technical Execution + Innovation |
| **4** | **Autonomy.** Deterministic-edit fast paths (0 LLM), a forced feature→model-swap curriculum, `error_recovery` with checkpoint/restore, a `code_writer` syntax-gate + salvage. Manual interventions counted + reported. | Impact & Relevance (20%) |
| **5** | **Feasibility.** NumPy + LightGBM, **0 GPU-hours**, single CPU core, ~1–3 min/iteration. Haiku 4.5 for cheap specialists, Opus 5 only for code-writing, prompt caching on the resent baseline prefix. Clears the Feasibility quality gate. | Feasibility (15%) |

---

## Inspiration & problem

Machine-learning engineers spend most of their time in one loop: take a dataset and a metric, change something, train, evaluate, reflect, repeat. That loop is *code* — so it can be automated. Track 2 asks for an agent that runs it on the KuaiRand-Pure short-video ranking benchmark and beats the official Factorization-Machine baseline. The hard part isn't running training — it's the agent **knowing what to change**, and knowing whether a change actually helped.

## What it does

**Embedflix** is a LangGraph agent — a supervisor + 6 specialist proposers + a reasoning judge + 7 execution nodes — that iterates on KuaiRand-Pure autonomously. Its headline contribution is a *problem insight*:

> The metric ranks **each user's own ~5 impressions**. A score term that shifts all of one user's impressions by the same amount cannot reorder them — it is **invisible** to GAUC and nDCG@5. Only item-side signal, or an explicit user×item cross, can move the score.

This is *why* the organizers' `ablation_features.py` found that user/categorical features make the score **worse** (0.5953 → 0.5936). Embedflix puts this insight in the supervisor's routing prompt and every specialist prompt, so the agent stops proposing changes that *cannot* work.

Acting on it, the agent:
1. **restored the official baseline** — the repo's had been silently replaced by an unvalidated loss that never produced a valid score;
2. **made its own measurement trustworthy** — a reproduction test + a two-stage seed-confirmed accept gate + a no-op detector;
3. **swapped the model** to a **LightGBM LambdaRank re-ranker stacked on the FM's logit**, with leakage-safe train-only target encodings and video engagement ratios, 5-seed rank-ensembled.

**Result:** converged **validation primary 0.6045 (+0.0029)** over the 0.6016 official baseline; test 0.5975 (+0.0029). NumPy + LightGBM, single CPU core, **no GPU**.

*(Embed diagram **D1** — architecture — and **D2** — the within-user insight — here.)*

## How we built it

- **Orchestration:** LangGraph state graph. Specialists propose; `code_writer` applies a **deterministic edit** (0 LLM) when the exact change is known, otherwise asks Claude Opus 5 for a verbatim `OLD_CODE`/`NEW_CODE` block and guards it with a syntax-gate + trailing-junk salvage + corrective retries.
- **Measurement integrity (Phase 0):** `git`-restore the shipped FM; freeze it as `baseline_official.py`; `tests/test_baseline_reproduces.py` asserts byte-identity + 0.6016 ± 0.0003 over 5 seeds. `score_analyst` now screens at seed 0 and confirms the 3-seed mean beats best by ε/2; a bit-identical score is flagged as a *no-op*, not a failed technique.
- **The model (Phase 3):** `starter-kit/model_lgbm.py` — `objective="lambdarank"`, `eval_at=[5]`; features = `fm_score` (personalisation), `te_video`/`te_author` (train-only smoothed long_view rate), 5 engagement ratios, user×item crosses; predictions produced in `data.load()` row order for `row_id` alignment (`tests/test_lgbm_alignment.py`, 19 checks). 5-seed rank-average keeps variance below ε.
- **Reflected + rejected:** 4-fold OOF stacking of `fm_score` — tried, measured (0.5905, overfit), reverted, and the *reason* documented in the source.

*(Embed diagram **D3** — measurement-integrity timeline — here.)*

## Challenges we ran into

- **A contaminated baseline.** The repo's "0.6039 baseline" was produced by no code we could point to — an edit whose own log entry read `primary: None`. Catching this changed the whole framing.
- **Single-hunk edits landing as dead code.** `code_writer` applies one find-and-replace; specialists sometimes want multi-site changes, so the new code was unreferenced and the deterministic pipeline reproduced the old score *exactly* — which the harness recorded as "technique tried, no improvement". Fixed with the no-op detector + a syntax-gate + salvage.
- **Seed noise vs a `1e-4` accept threshold** — it accepted pure noise. Replaced with the two-stage seed-confirmed gate.

## Accomplishments we're proud of

- A problem insight that **predicts** the organizers' own ablation result.
- A **positive, honest, seed-confirmed** delta — not a cherry-picked peak.
- **0 GPU-hours**; the whole thing runs on one CPU core.
- A run log that shows real hypothesis-driven reasoning and autonomous error recovery.

## What we learned

Most of the value on a benchmark like this is upstream of the model: understanding *exactly* what the metric rewards, and making sure your measurement can tell a real gain from noise. A weaker model with trustworthy evaluation beats a stronger model you can't trust.

## What's next

- **Causal user-session / sequence features** (what this user just watched) — the one signal the FM+IDs cannot encode; spec'd, not yet shipped.
- **Matched-strength OOF stacking** for `fm_score`.
- The **bonus benchmarks** (KuaiRand-1k / 27k).

---

## Built with

- **Development tools:** VS Code, terminal, **Claude Code** (the agent-development environment), git. macOS, single CPU core, **no GPU**.
- **APIs:** Anthropic Claude API — **Claude Opus 5** (`code_writer`), **Claude Haiku 4.5** (supervisor, all specialists, judge); **Tavily Search API** (two-phase live research per specialist); Anthropic **prompt caching** on the resent `baseline.py` prefix.
- **Libraries & frameworks:** **LangGraph** (orchestration), **FastMCP** (tool server), `anthropic` SDK, **NumPy** (the baseline FM + data pipeline), **LightGBM** (the LambdaRank re-ranker), **ChromaDB** (offline knowledge-base fallback for `read_papers`), `python-dotenv`.
- **Datasets & assets:** **KuaiRand-Pure** only — `log_standard_4_08_to_4_21_pure.csv` + `log_standard_4_22_to_5_08_pure.csv` (train 1.14M / valid 125K / test 171K rows), `user_features_pure.csv`, `video_features_basic_pure.csv`, `video_features_statistic_pure.csv`; the organizer **starter kit** (`data.py`, `baseline.py`, `evaluate.py`, `submit.py`). **No external training data** anywhere in the project.
