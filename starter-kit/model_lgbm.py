"""LightGBM LambdaRank re-ranker for KuaiRand-Pure, stacked on the FM.

WHY THIS ARCHITECTURE. evaluate.py scores GAUC + nDCG@5 strictly within each
user's own logged impression list, so there are exactly two useful kinds of
signal:

  1. personalisation -- does THIS user's taste match this video. The shipped
     FM's V[user_id] . V[video_id] interaction captures this; it is worth
     ~0.02 primary over a pure item-quality prior (FM 0.6016 vs te_video-alone
     0.5807). A tree model built only on item features cannot reconstruct it.
  2. item quality -- how long-viewable is this video in general. The FM's
     per-video embedding half-captures this; video engagement statistics
     (long_time_play/show, play_progress, ...) add a bit more.

So: train the official FM, take its raw logit as ONE feature, and let
LightGBM LambdaRank re-rank using that plus the item-quality signal it can't
otherwise see. LambdaRank optimises nDCG@5 directly and uses continuous
features as raw splits (no bucketisation).

LEAKAGE DISCIPLINE. Target-encoding statistics: TRAIN split only,
Bayesian-smoothed (k + a*p_bar)/(n + a), a=20. The FM feature is the same FM
the baseline reproduces (early-stopped on valid, like the official). The row's
own per-interaction log columns (is_click, is_like, play_time_ms, long_view)
are never inputs.
CAVEAT (disclosed): video_features_statistic_pure.csv aggregates span the whole
dataset period, so the ratio features are mildly forward-looking vs the train
window -- organizer-provided data, rules ban only *external* data.

SUBMISSION ALIGNMENT. Feature matrices are built in data.load() row order;
predictions come out in that order and line up 1:1 with submit.py's row_id.
Data is sorted by user only to build LightGBM's `group` array.
"""
import argparse
import csv
import os
import time

import numpy as np

try:
    from data import load, encode
    from evaluate import evaluate
except ImportError:
    from starter_kit.data import load, encode  # type: ignore
    from starter_kit.evaluate import evaluate  # type: ignore

import lightgbm as lgb

TE_ALPHA = 20.0
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "KuaiRand-Pure", "data")

# item-quality ratios over the video-stat CSV that showed real within-user
# ranking signal in isolation (probe: r_ltplay_show 0.5804, play_progress
# 0.5586; dropped r_like_show / r_follow_show / r_share_show -- all ~random).
RATIO_COLS = [
    ("r_ltplay_show", "long_time_play_cnt", "show_cnt"),
    ("r_validplay_show", "valid_play_cnt", "show_cnt"),
    ("r_complete_play", "complete_play_cnt", "play_cnt"),
    ("play_progress", "play_progress", None),
    ("r_playdur_show", "play_duration", "show_cnt"),
]
LOG_COUNT_COLS = ["show_cnt", "play_cnt"]
# low-cardinality item categoricals only -- author_id / music_id memorised and
# hurt generalisation in the first run.
CAT_COLS = ["tab", "video_type", "upload_type"]
# user-side numeric columns. A user column is constant within a user's
# impression list, so on its own it is invisible to within-user ranking
# (probe: te_user ~ random). It earns its place ONLY through interaction:
# a shallow split on the user column partitions users, and deeper splits on
# item features then differ per partition. log1p'd; missing/'UNKNOWN' -> 0.
USER_NUM_COLS = ["follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
USER_CAT_COLS = ["user_active_degree", "is_video_author", "is_live_streamer"]

FEATURE_NAMES = (
    ["fm_score", "fm_rank_in_user", "duration_ms", "te_video", "te_author"]
    + [name for name, _n, _d in RATIO_COLS]
    + [f"log_{c}" for c in LOG_COUNT_COLS]
    + [f"u_{c}" for c in USER_NUM_COLS]
    + CAT_COLS
    + [f"u_{c}" for c in USER_CAT_COLS]
)


def _safe_div(a, b):
    return a / b if b else 0.0


def _video_side(data_dir):
    basic, stat = {}, {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            basic[r["video_id"]] = r
    with open(os.path.join(data_dir, "video_features_statistic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            stat[r["video_id"]] = r
    vinfo = {}
    for vid in set(basic) | set(stat):
        b, s = basic.get(vid, {}), stat.get(vid, {})
        ratios = []
        for _name, num, den in RATIO_COLS:
            a = float(s.get(num) or 0.0)
            ratios.append(a if den is None else _safe_div(a, float(s.get(den) or 0.0)))
        vinfo[vid] = {
            "ratios": ratios,
            "logc": [np.log1p(float(s.get(c) or 0.0)) for c in LOG_COUNT_COLS],
            "video_type": b.get("video_type", "UNK"),
            "upload_type": b.get("upload_type", "UNK"),
        }
    return vinfo


def _user_side(data_dir):
    """user_id -> {numeric col -> float, cat col -> str}. Non-numeric range
    strings ('[100,1k)', '500+', 'UNKNOWN') collapse to 0.0 for the numeric
    columns -- the raw integer columns (follow_user_num etc.) are the ones
    used here and are always present."""
    uinfo = {}
    with open(os.path.join(data_dir, "user_features_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            row = {}
            for c in USER_NUM_COLS:
                v = r.get(c, "")
                try:
                    row[c] = float(v)
                except (TypeError, ValueError):
                    row[c] = 0.0
            for c in USER_CAT_COLS:
                row[c] = r.get(c, "UNK") or "UNK"
            uinfo[r["user_id"]] = row
    return uinfo


def _rank_in_user(users, scores):
    """dense rank (0..k-1, higher score -> higher rank) of `scores` within each
    contiguous-or-not user group, returned in the original row order."""
    out = np.zeros(len(scores), dtype=np.float32)
    idx_by_u = {}
    for i, u in enumerate(users):
        idx_by_u.setdefault(u, []).append(i)
    for u, idxs in idx_by_u.items():
        s = np.asarray([scores[i] for i in idxs])
        order = np.argsort(s, kind="stable")
        ranks = np.empty(len(s), dtype=np.float32)
        ranks[order] = np.arange(len(s), dtype=np.float32)
        for j, i in enumerate(idxs):
            out[i] = ranks[j]
    return out


def _smoothed_rate(train_rows, key_idx):
    gp = sum(x[6] for x in train_rows) / max(len(train_rows), 1)
    num, pos = {}, {}
    for x in train_rows:
        k = x[key_idx]
        num[k] = num.get(k, 0) + 1
        pos[k] = pos.get(k, 0) + x[6]

    def rate(k):
        n = num.get(k, 0)
        return (pos.get(k, 0) + TE_ALPHA * gp) / (n + TE_ALPHA) if n else gp

    return rate


def _train_fm(FM, dim, Xtr, ytr, Xstop, ystop, ustop, seed):
    """One FM fit over the global `dim` index space, early-stopped on
    (Xstop, ystop) by valid primary. Xtr already carries global offset indices
    from encode(), so `dim` must be the full value even when Xtr is a subset."""
    m = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1.0, None, 0
    for _ep in range(40):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            m.step(Xtr[idx[i:i + 8192]], ytr[idx[i:i + 8192]])
        p = evaluate(ustop, list(ystop), list(m.predict(Xstop)))["primary"]
        if p > best + 1e-5:
            best, bad, best_state = p, 0, (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4:
                break
    m.V, m.W, m.b = best_state
    return m, best


def _fm_scores(splits, seed=0, verbose=True):
    """FM raw logit for every row of every split, in data.load() order, from a
    single FM fit on all of train (early-stopped on valid, exactly like the
    official baseline).

    Train scores are in-sample. Tried 4-fold OOF for the train scores instead
    (2026-09-01): it was strictly worse -- 0.5905 valid / 0.5831 test, LightGBM
    overfit and early-stopped at iteration 3. Cause: the fold FMs (3/4 data) are
    weaker than the full-train FM used for valid/test, so fm_score's
    distribution shifts between train and serving. The shipped FM is k=16 and
    heavily early-stopped -- it does not memorise individual rows -- so the
    in-sample feature transfers cleanly (+0.0019 test, confirmed)."""
    import baseline_official as BO

    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, _ = enc["test"]

    m, best = _train_fm(BO.FM, dim, Xtr, ytr, Xva, yva, uva, seed)
    if verbose:
        print(f"  [fm feature] valid primary {best:.4f}")
    return {"train": m.predict(Xtr), "valid": m.predict(Xva), "test": m.predict(Xte)}


def build_features(splits, fm_seed=0, verbose=True):
    """feats[split] = (X float32 (N,F), y, users) in data.load() order; cat_index."""
    tr = splits["train"]
    vinfo = _video_side(_DATA)
    uinfo = _user_side(_DATA)
    te_v = _smoothed_rate(tr, 2)
    te_a = _smoothed_rate(tr, 3)
    fm = _fm_scores(splits, seed=fm_seed, verbose=verbose)

    all_cat = CAT_COLS + [f"u_{c}" for c in USER_CAT_COLS]
    codebooks = {c: {} for c in all_cat}

    def _cat(row, col):
        if col == "tab":
            return row[4]
        if col.startswith("u_"):
            return uinfo.get(row[1], {}).get(col[2:], "UNK")
        return vinfo.get(row[2], {}).get(col, "UNK")

    for x in tr:
        for c in all_cat:
            cb = codebooks[c]
            v = _cat(x, c)
            if v not in cb:
                cb[v] = len(cb) + 1  # 0 = unseen

    n_num = len(FEATURE_NAMES) - len(all_cat)
    cat_index = list(range(n_num, len(FEATURE_NAMES)))
    zero_unum = {c: 0.0 for c in USER_NUM_COLS}

    feats = {}
    for name, rows in splits.items():
        N = len(rows)
        X = np.empty((N, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.empty(N, dtype=np.float32)
        users = [x[1] for x in rows]
        fmv = fm[name]
        fm_rank = _rank_in_user(users, fmv)
        for i, x in enumerate(rows):
            vi = vinfo.get(x[2], {})
            ui = uinfo.get(x[1], zero_unum)
            c = 0
            X[i, c] = fmv[i]; c += 1
            X[i, c] = fm_rank[i]; c += 1
            X[i, c] = x[5]; c += 1
            X[i, c] = te_v(x[2]); c += 1
            X[i, c] = te_a(x[3]); c += 1
            for rv in vi.get("ratios", [0.0] * len(RATIO_COLS)):
                X[i, c] = rv; c += 1
            for lv in vi.get("logc", [0.0] * len(LOG_COUNT_COLS)):
                X[i, c] = lv; c += 1
            for uc in USER_NUM_COLS:
                X[i, c] = np.log1p(ui.get(uc, 0.0)); c += 1
            for cc in all_cat:
                X[i, c] = codebooks[cc].get(_cat(x, cc), 0); c += 1
            y[i] = x[6]
        feats[name] = (X, y, users)
    return feats, cat_index


def _group_sort(users):
    order = np.argsort(np.asarray(users, dtype=object), kind="stable")
    su = [users[i] for i in order]
    groups, cur, cnt = [], None, 0
    for u in su:
        if u != cur:
            if cnt:
                groups.append(cnt)
            cur, cnt = u, 1
        else:
            cnt += 1
    if cnt:
        groups.append(cnt)
    return order, groups


PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[5],
    label_gain=[0, 1],
    lambdarank_truncation_level=5,
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.02,
    min_data_in_leaf=150,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    min_gain_to_split=1e-4,
    verbosity=-1,
)


def run_lgbm(splits, seed=0, num_boost_round=3000, verbose=True):
    feats, cat_index = build_features(splits, fm_seed=seed, verbose=verbose)
    Xtr, ytr, utr = feats["train"]
    Xva, yva, uva = feats["valid"]
    Xte, yte, ute = feats["test"]

    tr_ord, tr_grp = _group_sort(utr)
    va_ord, va_grp = _group_sort(uva)

    dtrain = lgb.Dataset(Xtr[tr_ord], label=ytr[tr_ord], group=tr_grp,
                         categorical_feature=cat_index, free_raw_data=False)
    dvalid = lgb.Dataset(Xva[va_ord], label=yva[va_ord], group=va_grp,
                         categorical_feature=cat_index, reference=dtrain, free_raw_data=False)

    params = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    model = lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=[dvalid], valid_names=["valid"],
        callbacks=[lgb.early_stopping(120, verbose=verbose),
                   lgb.log_evaluation(100 if verbose else 0)],
    )

    def _sc(X, y, u, split):
        p = model.predict(X, num_iteration=model.best_iteration)
        r = evaluate(u, list(y), list(p))
        if verbose:
            print(f"  {split:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
        return r, p

    va_r, _ = _sc(Xva, yva, uva, "valid")
    te_r, te_p = _sc(Xte, yte, ute, "test")
    if verbose:
        imp = sorted(zip(FEATURE_NAMES, model.feature_importance("gain")), key=lambda t: -t[1])
        print("  top features (gain): " + ", ".join(f"{n}={v:.0f}" for n, v in imp[:8]))
        print(f"  best_iteration={model.best_iteration}")
    return {"valid": va_r, "test": te_r, "model": model, "test_pred": te_p,
            "test_users": ute, "feats": feats}


def _rank_avg(users, pred_list):
    """Average of per-user rank positions across models (models aren't
    calibrated to each other, so rank-average, not score-average)."""
    ranks = [_rank_in_user(users, p) for p in pred_list]
    return np.mean(ranks, axis=0)


def run_ensemble(splits, seeds=(0, 1, 2), verbose=True):
    """Full FM+LGBM stack per seed; rank-average valid/test predictions.
    Reduces the ~0.0008 seed variance below the eps=0.002 convergence rule."""
    va_preds, te_preds = [], []
    uva = ute = yva = yte = None
    for s in seeds:
        r = run_lgbm(splits, seed=s, verbose=verbose)
        m = r["model"]
        # each seed has its own feature matrix (fm_score depends on fm_seed),
        # so predict each model on the matrix it was trained against
        Xva, yva, uva = r["feats"]["valid"]
        Xte, yte, ute = r["feats"]["test"]
        va_preds.append(m.predict(Xva, num_iteration=m.best_iteration))
        te_preds.append(m.predict(Xte, num_iteration=m.best_iteration))

    va_rank = _rank_avg(uva, va_preds)
    te_rank = _rank_avg(ute, te_preds)
    va = evaluate(uva, list(yva), list(va_rank))
    te = evaluate(ute, list(yte), list(te_rank))
    if verbose:
        print(f"\n=== ENSEMBLE ({len(seeds)} seeds, rank-averaged) ===")
        print(f"  valid  GAUC {va['GAUC']:.4f} | nDCG@5 {va['nDCG@5']:.4f} | primary {va['primary']:.4f}")
        print(f"  test   GAUC {te['GAUC']:.4f} | nDCG@5 {te['nDCG@5']:.4f} | primary {te['primary']:.4f}")
    return {"valid": va, "test": te,
            "valid_pred": va_rank, "valid_users": uva,
            "test_pred": te_rank, "test_users": ute}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="", help="comma list -> rank-averaged ensemble")
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k: len(v) for k, v in splits.items()})
    t0 = time.time()
    if a.seeds:
        run_ensemble(splits, seeds=tuple(int(s) for s in a.seeds.split(",")))
        print(f"[{time.time() - t0:.1f}s]")
    else:
        res = run_lgbm(splits, seed=a.seed)
        print(f"\n=== lgbm+fm (seed={a.seed}) ===  [{time.time() - t0:.1f}s]")
        for sp in ("valid", "test"):
            r = res[sp]
            print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
