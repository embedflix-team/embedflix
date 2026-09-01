"""Shared problem-insight text injected into every specialist + supervisor
prompt. This is the project's core 'Innovation & Problem Insight' claim, and
before Phase 5 it appeared in none of the prompts."""

WITHIN_USER_INVARIANCE = """KEY INSIGHT (applies to every change you propose):
evaluate.py scores GAUC and nDCG@5 STRICTLY WITHIN each user's own logged
impression list (~5 impressions/user). A score component that is constant
across one user's impressions cannot reorder that list, so it is
mathematically invisible to both metrics:
  - the FM's W[user_id] linear term        -> exactly zero effect
  - any user_features_pure.csv column used as a plain linear feature -> zero
    (this is why the organizers' own ablation_features.py found user features
     make things WORSE: 0.5953 -> 0.5936 -- parameters added, no within-list
     signal)
  - user features DO help through a user x item CROSS (the cross varies within
    the list); item-side signal helps directly.
Spend capacity on item-side signal and user x item interactions, never on
per-user constants."""

STACK_STATUS = """CURRENT BEST MODEL: a LightGBM LambdaRank stack
(starter-kit/model_lgbm.py, reachable via model_swapper's 'model:lgbm') that
uses the official FM's logit as one feature and re-ranks with train-only
target encodings + video engagement ratios, optimising nDCG@5 directly.
Standalone-validated, 5-seed ensemble: valid 0.6045 (+0.0029 vs the
0.6016 baseline) / test 0.5975 (+0.0029).

The plain numpy FM's per-ID embeddings already saturate the item-side signal
that IDs alone can carry, so pure FM feature/loss tweaks plateau near 0.602
(Phase 1 tried five leakage-free item-feature sets; all landed within seed
noise). Propose changes that add signal the FM + IDs cannot already encode:
user session/sequence state, user x item crosses, or a stronger re-ranker on
top of the stack."""

DATASET_FACTS = """DATASET FACTS (KuaiRand-Pure, from data.py / baseline_scores.json):
  - label: long_view (binary); ~ 33% positive overall
  - splits by date: train 1,141,112 rows / valid 124,909 / test 170,588
  - the FM baseline uses 5 categorical fields:
    user_id, video_id, author_id, tab, dur_bucket
    (music_id and tag exist in video_features_basic_pure.csv but are UNUSED)
  - test-set user mix: 27.1% all-negative (nDCG=0 for any model),
    9.2% all-positive, 63.7% discriminative -> oracle ceiling is
    test primary 0.8645, not 1.0; judge progress against that.
  - official baseline: valid primary 0.6016 (GAUC 0.6674 / nDCG@5 0.5357),
    test primary 0.5946; 5-seed std ~0.0008 (test), ~0.0003 (valid)."""
