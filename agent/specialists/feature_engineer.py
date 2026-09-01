# agent/specialists/feature_engineer.py
"""feature_engineer -- item-side feature specialist.

WHY ITEM-SIDE ONLY. evaluate.py scores GAUC and nDCG@5 *within each user's own
logged impression list*. A score term that is constant across one user's list
therefore cannot reorder it and is mathematically invisible to both metrics:

  - the FM's W[user_id] linear term  -> exactly zero effect
  - any user_features_pure.csv column used as a linear feature -> exactly zero
    (this is why the organizers' own ablation_features.py found user features
     make things *worse*: 0.5953 -> 0.5936 / 0.5933 -- pure added noise)
  - only item-side signal (or a user x item cross) varies within the list

So every candidate here adds an ITEM-side field domain, ranked by how much it
varies within one user's impressions and how directly it predicts long_view.
baseline.py's FM is purely categorical-embedding based (every FIELDS entry is a
vocab lookup, no continuous path), so each numeric signal is quantile-bucketed
-- the same trick data.py already uses for dur_bucket -- and the bucket id is
treated as a categorical domain. No FM architecture change.

Menu (priority order; the forced supervisor override runs them all before any
other specialist):

  1. features:target_enc    -- per-video and per-author historical long_view
                               rate, computed from the TRAIN LOG ONLY,
                               Bayesian-smoothed (k + a*p_bar)/(n + a), a=20.
                               A leakage-free direct prior on the label.
  2. features:longview_prior-- ratios from video_features_statistic_pure.csv
                               that are near-proxies for long_view:
                               long_time_play_cnt/show_cnt,
                               complete_play_cnt/play_cnt,
                               valid_play_cnt/show_cnt, play_progress.
  3. features:item_cat      -- the item categoricals baseline.py never uses:
                               music_id, tag, video_type, upload_type.
  4. features:play_quality  -- engagement-quality ratios:
                               play_duration/play_cnt, like_cnt/show_cnt,
                               follow_cnt/show_cnt, share_cnt/show_cnt.
  5. features:combined      -- union of whichever of 1-4 actually improved the
                               score (read from experiment_history at runtime);
                               falls back to target_enc + longview_prior if
                               none has improved yet.

LEAKAGE DISCIPLINE. Target-encoding statistics and every quantile bucket edge
are fit on the TRAIN split only. The row's own per-interaction log columns
(is_click, is_like, play_time_ms, long_view) are never used as inputs.
CAVEAT (disclosed): video_features_statistic_pure.csv aggregates span the whole
dataset period, so candidates 2 and 4 are mildly forward-looking relative to
the train window. It is organizer-provided data (the rules ban only *external*
data); candidate 1 is the fully clean alternative and is tried first.

Each candidate is a FULL replacement of data.py via _deterministic_edit --
zero LLM calls. The generated data.py keeps the exact load()/encode() contract
baseline.py and submit.py depend on: row indices 0-6 are unchanged
(date, user_id, video_id, author_id, tab, duration_ms, label), extra per-video
data is appended at indices 7-8, and encode() still returns (enc, dim) with
enc[split] = (X int32 (N, len(FIELDS)), y, users).
"""

N_BUCKETS = 24
TE_ALPHA = 20.0

# spec shape: {"te": bool, "ratios": [(field_name, num_col, den_col_or_None)],
#              "cats": [video_basic_col, ...]}
_SPEC = {
    "features:target_enc": {
        "te": True, "ratios": [], "cats": [],
    },
    "features:longview_prior": {
        "te": False,
        "ratios": [
            ("r_ltplay_show", "long_time_play_cnt", "show_cnt"),
            ("r_complete_play", "complete_play_cnt", "play_cnt"),
            ("r_validplay_show", "valid_play_cnt", "show_cnt"),
            ("play_progress", "play_progress", None),
        ],
        "cats": [],
    },
    "features:item_cat": {
        "te": False, "ratios": [],
        "cats": ["music_id", "tag", "video_type", "upload_type"],
    },
    "features:play_quality": {
        "te": False,
        "ratios": [
            ("r_playdur_playcnt", "play_duration", "play_cnt"),
            ("r_like_show", "like_cnt", "show_cnt"),
            ("r_follow_show", "follow_cnt", "show_cnt"),
            ("r_share_show", "share_cnt", "show_cnt"),
        ],
        "cats": [],
    },
}

CANDIDATES = [
    ("features:target_enc",
     "train-log-only per-video & per-author long_view rate, Bayesian-smoothed (leakage-free label prior)"),
    ("features:longview_prior",
     "video-stat ratios that proxy long_view: long_time_play/show, complete_play/play, valid_play/show, play_progress"),
    ("features:item_cat",
     "the item categoricals the baseline ignores: music_id, tag, video_type, upload_type"),
    ("features:play_quality",
     "engagement-quality ratios: play_duration/play_cnt, like/show, follow/show, share/show"),
    ("features:combined",
     "union of whichever of the above actually improved the within-user ranking score"),
]


def _combined_spec(history: list) -> dict:
    """Union of the specs whose feature_engineer history entry has improved=True.
    Falls back to target_enc + longview_prior if nothing has improved yet.

    experiment_history entries carry no label field, so recover it from the
    hypothesis text (feature_engineer writes 'Feature engineer [<label>]: ...')."""
    won_labels = [
        lab for lab, _ in CANDIDATES[:-1]
        if any(h.get("specialist") == "feature_engineer" and h.get("improved")
               and lab in (h.get("hypothesis") or "")
               for h in history)
    ]
    if not won_labels:
        won_labels = ["features:target_enc", "features:longview_prior"]

    merged = {"te": False, "ratios": [], "cats": []}
    seen_ratio, seen_cat = set(), set()
    for lab in won_labels:
        sp = _SPEC[lab]
        merged["te"] = merged["te"] or sp["te"]
        for r in sp["ratios"]:
            if r[0] not in seen_ratio:
                merged["ratios"].append(r); seen_ratio.add(r[0])
        for c in sp["cats"]:
            if c not in seen_cat:
                merged["cats"].append(c); seen_cat.add(c)
    return merged


def feature_engineer(state: dict, tools: dict) -> dict:
    tried = state.get("tried_approaches", [])
    history = state.get("experiment_history", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    untried = [(lab, desc) for (lab, desc) in CANDIDATES if lab not in tried]
    label, desc = (untried or CANDIDATES)[0]

    spec = _combined_spec(history) if label == "features:combined" else _SPEC.get(label)
    if spec is None:  # menu exhausted, wrapped back to combined
        label = "features:combined"
        spec = _combined_spec(history)

    current_data_py = tools["read_file"]({"file_path": "data.py"})
    new_data_py = _build_data_py(label, spec)

    n_new = (2 if spec["te"] else 0) + len(spec["ratios"]) + len(spec["cats"])
    hypothesis = (
        f"Feature engineer [{label}]: {desc}. Adds {n_new} new item-side field "
        f"domain(s) to data.py (base 5 -> {5 + n_new}), quantile-bucketed into "
        f"{N_BUCKETS} the way dur_bucket already is. Current primary {primary:.4f}. "
        "Rationale: evaluate.py ranks within each user's own impression list, so "
        "only item-side signal can move GAUC/nDCG@5 -- user-side terms are "
        "invisible (this is why ablation_features.py found user features hurt). "
        + ("Target encodings are fit on the train split only (leakage-free). "
           if spec["te"] else "")
        + ("NOTE: video-stat aggregates span the whole dataset period, so these "
           "ratios are mildly forward-looking vs the train window -- "
           "organizer-provided data, disclosed in the write-up. "
           if spec["ratios"] else "")
    )

    return {
        **state,
        "hypothesis": hypothesis,
        "code_change_instruction": (
            f"(deterministic) replace the full contents of data.py with the "
            f"feature_engineer '{label}' variant: te={spec['te']}, "
            f"ratios={[r[0] for r in spec['ratios']]}, cats={spec['cats']}."
        ),
        "reasoning": hypothesis,
        "tried_approaches": tried + [label],
        "_deterministic_edit": {
            "file": "data.py",
            "old_code": current_data_py,
            "new_code": new_data_py,
        },
    }


def _build_data_py(label: str, spec: dict) -> str:
    """Generate the full data.py source for one candidate. The load()/encode()
    bodies are FIXED and branch on the three module-level config values below,
    so there is exactly one code path to get right regardless of candidate."""
    ratios_repr = repr(spec["ratios"])
    cats_repr = repr(spec["cats"])
    te_repr = repr(bool(spec["te"]))

    return f'''"""KuaiRand-Pure data load + official split + feature encoding. numpy + stdlib only.

GENERATED by agent/specialists/feature_engineer.py -- candidate: {label}
Extends the shipped 5-field encoder with item-side signal only. evaluate.py
ranks within each user's own impressions, so user-side terms cannot move the
score; every field added here is item-side and quantile-bucketed ({N_BUCKETS}
buckets, train-split edges only) exactly the way dur_bucket already is.

Config (set by feature_engineer for this candidate):
  USE_TARGET_ENC = {te_repr}   -- per-video & per-author train-log long_view rate,
                       Bayesian-smoothed (a={TE_ALPHA}); TRAIN SPLIT ONLY, no leakage.
  RATIO_COLS     = {ratios_repr}
                    -- (field, numerator, denominator|None) over
                       video_features_statistic_pure.csv dataset-wide aggregates.
                       Mildly forward-looking vs the train window (disclosed).
  VIDEO_CAT_COLS = {cats_repr}
                    -- extra item categoricals from video_features_basic_pure.csv.

Row tuple (indices 0-6 identical to the shipped data.py so baseline.py and
submit.py are unaffected):
  (date, user_id, video_id, author_id, tab, duration_ms, label, ratio_vec, cat_vec)
"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {{'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}}

N_BUCKETS = {N_BUCKETS}
TE_ALPHA = {TE_ALPHA}
USE_TARGET_ENC = {te_repr}
RATIO_COLS = {ratios_repr}
VIDEO_CAT_COLS = {cats_repr}

_TE_FIELDS = ['te_video', 'te_author'] if USE_TARGET_ENC else []
_RATIO_FIELDS = [name for (name, _num, _den) in RATIO_COLS]
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket'] + _TE_FIELDS + _RATIO_FIELDS + list(VIDEO_CAT_COLS)

USER_FEATURE_FILE = os.path.join(os.path.dirname(__file__), 'KuaiRand-Pure/data/user_features_pure.csv')
VIDEO_FEATURE_FILE = os.path.join(os.path.dirname(__file__), 'KuaiRand-Pure/data/video_features_statistic_pure.csv')


def get_feature_info(user_feature_file=USER_FEATURE_FILE, video_feature_file=VIDEO_FEATURE_FILE):
    """Real column names in the untapped user/video feature files. Never
    includes log-file per-interaction columns -- those leak the label."""
    info = {{}}
    for name, path in [('user', user_feature_file), ('video', video_feature_file)]:
        with open(path) as f:
            info[f'{{name}}_feature_columns'] = next(csv.reader(f))
    return info


def _safe_div(a, b):
    return a / b if b else 0.0


def load(data_dir):
    vid2author, vid2cat = {{}}, {{}}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
            if VIDEO_CAT_COLS:
                vid2cat[r['video_id']] = [(r.get(c) or 'UNK') for c in VIDEO_CAT_COLS]

    vid2ratio = {{}}
    if RATIO_COLS:
        with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
            for r in csv.DictReader(fh):
                vec = []
                for (_name, num, den) in RATIO_COLS:
                    a = float(r.get(num) or 0.0)
                    vec.append(a if den is None else _safe_div(a, float(r.get(den) or 0.0)))
                vid2ratio[r['video_id']] = vec

    zeros_ratio = [0.0] * len(RATIO_COLS)
    unk_cat = ['UNK'] * len(VIDEO_CAT_COLS)

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vid = r['video_id']
                rows.append((int(r['date']), r['user_id'], vid,
                             vid2author.get(vid, 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             vid2ratio.get(vid, zeros_ratio),
                             vid2cat.get(vid, unk_cat)))

    out = {{}}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out


def _edges(values, n):
    v = np.asarray(values, dtype=np.float64)
    return np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])


def encode(splits):
    """Map categorical values -> contiguous ids (unseen -> per-domain UNK slot).
    Returns (enc, dim) where enc[split] = (X int32 (N, len(FIELDS)), y, users)."""
    tr = splits['train']

    dur_edges = _edges([x[5] for x in tr], N_BUCKETS)

    te_v = te_a = None
    te_v_edges = te_a_edges = None
    if USE_TARGET_ENC:
        gp = sum(x[6] for x in tr) / max(len(tr), 1)
        vn, vp = collections.Counter(), collections.Counter()
        an, ap = collections.Counter(), collections.Counter()
        for x in tr:
            vn[x[2]] += 1; vp[x[2]] += x[6]
            an[x[3]] += 1; ap[x[3]] += x[6]

        def te_v(v, vp=vp, vn=vn, gp=gp):
            return (vp[v] + TE_ALPHA * gp) / (vn[v] + TE_ALPHA) if vn[v] else gp

        def te_a(a, ap=ap, an=an, gp=gp):
            return (ap[a] + TE_ALPHA * gp) / (an[a] + TE_ALPHA) if an[a] else gp

        te_v_edges = _edges([te_v(x[2]) for x in tr], N_BUCKETS)
        te_a_edges = _edges([te_a(x[3]) for x in tr], N_BUCKETS)

    ratio_edges = [_edges([x[7][i] for x in tr], N_BUCKETS) for i in range(len(RATIO_COLS))]

    def raw(x):
        vals = [x[1], x[2], x[3], x[4], str(int(np.searchsorted(dur_edges, x[5])))]
        if USE_TARGET_ENC:
            vals.append(str(int(np.searchsorted(te_v_edges, te_v(x[2])))))
            vals.append(str(int(np.searchsorted(te_a_edges, te_a(x[3])))))
        for i in range(len(RATIO_COLS)):
            vals.append(str(int(np.searchsorted(ratio_edges[i], x[7][i]))))
        vals.extend(str(c) for c in x[8])
        return vals

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {{}}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
'''
