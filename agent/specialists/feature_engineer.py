# agent/specialists/feature_engineer.py
"""New specialist: proposes using the video engagement-statistics columns in
video_features_statistic_pure.csv (never read anywhere in the runtime
pipeline before this) that the organizers' own ablation_features.py script
never tests -- that script only tries user_features_pure.csv's small
categorical/demographic columns and a few video_features_basic_pure.csv
categorical columns, all of which it already showed make things *worse*
(0.5953 base -> 0.5936 / 0.5933, on the real held-out test split). The
untested lever is the ~50 *numeric* engagement stats in
video_features_statistic_pure.csv (play_cnt, like_cnt, show_cnt, etc.).

Encoding approach: baseline.py's FM is purely categorical-embedding based --
every FIELDS entry is a vocab lookup, there's no continuous/linear input
path. data.py's existing dur_bucket field already shows the established
pattern for turning a continuous value into an FM-compatible field:
quantile-bucket it (edges from the TRAIN split only, so no leakage) and
treat the bucket id as a categorical domain. Every candidate here follows
that exact pattern -- no FM architecture change.

Deterministic menu, escalating richness (same spirit as ablation_features.py's
own base/item/cwm13 progression, but for the untested numeric columns):
  1. features:playcount   -- + play_cnt bucketed (1 new domain)
  2. features:engagement3 -- + play_cnt, like_cnt, show_cnt bucketed (3 new domains)
  3. features:engagement6 -- + play_cnt, like_cnt, show_cnt, comment_cnt,
                               share_cnt, download_cnt bucketed (6 new domains)

Each candidate is a FULL replacement of data.py (the file is only 64 lines,
so this is one clean edit, not a fragile multi-location patch). Unlike
training_optimizer's single-line substring edits (which compose across
iterations because they touch independent lines), these are alternative
full-file states, not additive patches -- candidate 2 is not "candidate 1
plus more", it's built fresh from the same 5-field baseline as candidate 1,
so all three are directly comparable to the real baseline, never stacked on
each other. That's why old_code is read fresh from disk every call (via
tools["read_file"]) instead of being a hardcoded original string: whatever
data.py currently contains gets replaced with this candidate's independently
-derived target content.

Zero LLM calls for any of the three candidates -- this is a closed menu end
to end, same fast path as training_optimizer/model_swapper's higher_k.
"""

# (label, [numeric video-stat columns to bucket and add], description)
CANDIDATES = [
    ("features:playcount", ["play_cnt"],
     "add play_cnt (bucketed) as a new field domain -- raw popularity signal"),
    ("features:engagement3", ["play_cnt", "like_cnt", "show_cnt"],
     "add play_cnt, like_cnt, show_cnt (bucketed) -- reach + approval + impressions"),
    ("features:engagement6", ["play_cnt", "like_cnt", "show_cnt", "comment_cnt", "share_cnt", "download_cnt"],
     "add play_cnt, like_cnt, show_cnt, comment_cnt, share_cnt, download_cnt (bucketed) -- full engagement profile"),
]


def feature_engineer(state: dict, tools: dict) -> dict:
    tried = state.get("tried_approaches", [])
    primary = state.get("current_scores", {}).get("primary") or 0.6016

    untried = [c for c in CANDIDATES if c[0] not in tried]
    label, feat_cols, desc = (untried or CANDIDATES)[0]

    current_data_py = tools["read_file"]({"file_path": "data.py"})
    new_data_py = _build_data_py(feat_cols)

    hypothesis = (
        f"Feature engineer: {desc} ({label}). Current primary {primary:.4f}. "
        "video_features_statistic_pure.csv has never been read by the runtime "
        "pipeline before -- ablation_features.py already showed the small "
        "categorical feature additions from the other feature files make "
        "things worse on the real test split (0.5953 -> 0.5936/0.5933), so "
        "this specialist deliberately targets the untested numeric engagement "
        "columns instead, bucketed the same way dur_bucket already is."
    )

    return {
        **state,
        "hypothesis": hypothesis,
        "code_change_instruction": (
            f"(deterministic) replace the full contents of data.py with a version "
            f"that also loads video_features_statistic_pure.csv and adds "
            f"{feat_cols} as quantile-bucketed field domains."
        ),
        "reasoning": hypothesis,
        "tried_approaches": tried + [label],
        "_deterministic_edit": {
            "file": "data.py",
            "old_code": current_data_py,
            "new_code": new_data_py,
        },
    }


def _build_data_py(feat_cols: list) -> str:
    """Generates the full data.py source for a given set of extra numeric
    video-stat columns to bucket and add as FIELDS. Structurally identical to
    the original 5-field data.py, just with vid2stats added to load() and the
    corresponding bucketed domains added to encode()."""
    n = len(feat_cols)
    extra_fields = ", ".join(f"'{c}_bucket'" for c in feat_cols)
    fields_line = (
        "FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket'"
        + (", " + extra_fields if n else "") + "]"
    )
    feat_cols_repr = repr(feat_cols)

    return f'''"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。

feature_engineer specialist: extends the original 5-field version with
{n} numeric video engagement-statistics column(s) from
video_features_statistic_pure.csv ({feat_cols}), quantile-bucketed (train-split
edges only, no leakage) the same way dur_bucket already is -- FM here is
purely categorical-embedding based, so this is the FM-compatible way to add
a continuous feature without changing the model architecture.
"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {{'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}}
# 5 个特征域 + feature_engineer 新增的数值型视频互动特征(分桶)。
{fields_line}
NUMERIC_VIDEO_FEATS = {feat_cols_repr}

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {{}}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    vid2stats = {{}}
    if NUMERIC_VIDEO_FEATS:
        with open(os.path.join(data_dir, 'video_features_statistic_pure.csv')) as fh:
            for r in csv.DictReader(fh):
                vid2stats[r['video_id']] = [
                    float(r[c]) if r.get(c) not in (None, '') else 0.0
                    for c in NUMERIC_VIDEO_FEATS
                ]
    zeros = [0.0] * len(NUMERIC_VIDEO_FEATS)

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             vid2stats.get(r['video_id'], zeros)))

    out = {{}}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(values, n=10):
    return np.quantile(np.asarray(values), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    dur_edges = _bucket_edges([x[5] for x in tr])
    feat_edges = [_bucket_edges([x[7][i] for x in tr]) for i in range(len(NUMERIC_VIDEO_FEATS))]

    def raw(x):
        base = [x[1], x[2], x[3], x[4], str(int(np.searchsorted(dur_edges, x[5])))]
        extra = [str(int(np.searchsorted(feat_edges[i], x[7][i]))) for i in range(len(NUMERIC_VIDEO_FEATS))]
        return base + extra

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
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
