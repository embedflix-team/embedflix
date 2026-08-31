"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

# Additional feature files available for a specialist to discover columns from
# (used only for USER-side and VIDEO-side profile/statistic columns -- NOT
# the log file's per-interaction feedback columns, see note below).
USER_FEATURE_FILE = os.path.join(os.path.dirname(__file__), 'KuaiRand-Pure/data/user_features_pure.csv')
VIDEO_FEATURE_FILE = os.path.join(os.path.dirname(__file__), 'KuaiRand-Pure/data/video_features_statistic_pure.csv')

# NOTE: the log file's per-interaction columns (is_click, is_like, is_follow,
# is_comment, is_forward, play_time_ms, ...) are deliberately NOT exposed here
# as candidate input features. LABEL ('long_view') is derived from the same
# row's play_time_ms/duration_ms, and the other is_* columns are simultaneous
# outcomes of the same impression being predicted -- using any of them as an
# FM input field would leak the label (a model can't know at serving time
# whether a user already clicked/liked the very row it's about to rank).

def get_feature_info(user_feature_file=USER_FEATURE_FILE, video_feature_file=VIDEO_FEATURE_FILE):
    """Returns the real column names available in the untapped user/video
    feature files, for a specialist (or a human) to discover what exists
    without having to open the CSVs by hand. Does NOT include log-file
    columns -- see the leakage note above."""
    info = {}
    for name, path in [('user', user_feature_file), ('video', video_feature_file)]:
        with open(path) as f:
            info[f'{name}_feature_columns'] = next(csv.reader(f))
    return info

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
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
