"""T7 -- LightGBM stack submission alignment. No FM training, no LightGBM
training: monkeypatches the FM feature to zeros so this stays a fast offline
check of the property that actually matters -- predictions come out in
data.load() row order and survive submit.py --check.

    python3 tests/test_lgbm_alignment.py
"""
import os
import sys
import tempfile

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_KIT = os.path.join(_REPO, "starter-kit")
sys.path.insert(0, _KIT)
os.chdir(_KIT)

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    line = ("PASS" if cond else "FAIL") + " - " + name
    if not cond and detail:
        line += f"  --  {detail}"
    print(line)

import model_lgbm as L
from data import load
from submit import write_submission, read_submission

# --- monkeypatch the FM feature so we don't train a 55s FM in a unit test ---
def _zero_fm(splits, seed=0, verbose=True):
    return {k: np.zeros(len(v), dtype=np.float32) for k, v in splits.items()}
L._fm_scores = _zero_fm

splits = load("./KuaiRand-Pure/data")
feats, cat_index = L.build_features(splits, verbose=False)

for sp in ("train", "valid", "test"):
    X, y, users = feats[sp]
    rows = splits[sp]
    check(f"{sp}: feature matrix has one row per data.load() row",
          X.shape[0] == len(rows), f"{X.shape[0]} vs {len(rows)}")
    check(f"{sp}: users column is in data.load() order (no sort leaked in)",
          users == [r[1] for r in rows])
    check(f"{sp}: labels match data.load() order",
          np.array_equal(y, np.array([r[6] for r in rows], dtype=np.float32)))

check("X width == len(FEATURE_NAMES)", feats["test"][0].shape[1] == len(L.FEATURE_NAMES))
check("cat_index positions are all inside the matrix",
      all(0 <= i < len(L.FEATURE_NAMES) for i in cat_index) and len(cat_index) >= 3)

# --- _group_sort: sizes sum to N, every user contiguous in the returned order ---
_, users = feats["valid"][2], feats["valid"][2]
order, groups = L._group_sort(users)
check("_group_sort: group sizes sum to N", sum(groups) == len(users))
su = [users[i] for i in order]
runs = 1 + sum(1 for a, b in zip(su, su[1:]) if a != b)
check("_group_sort: each user is one contiguous run", runs == len(groups))

# --- _rank_in_user: order-preserving, per-group dense rank ---
u = ["a", "a", "b", "a", "b"]
s = [0.1, 0.9, 0.5, 0.3, 0.2]
r = L._rank_in_user(u, np.array(s, dtype=np.float32))
check("_rank_in_user: length preserved", len(r) == len(s))
check("_rank_in_user: within group a, 0.9 ranks above 0.3 above 0.1",
      r[1] > r[3] > r[0])
check("_rank_in_user: within group b, 0.5 ranks above 0.2", r[2] > r[4])

# --- submit.py roundtrip on the test split with dummy (but aligned) scores ---
rows = splits["test"]
dummy = np.arange(len(rows), dtype=np.float64)[::-1] * 1.0  # strictly varied
tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
tmp.close()
try:
    write_submission(tmp.name, rows, dummy)
    got = read_submission(tmp.name, rows)   # raises on any misalignment
    check("submit.py --check accepts the aligned submission", len(got) == len(rows))
    check("scores round-trip unchanged", np.allclose(got, dummy))
except Exception as e:  # noqa: BLE001
    check("submit.py --check accepts the aligned submission", False, str(e))
finally:
    os.unlink(tmp.name)

# --- a deliberately mis-ordered submission MUST be rejected ---
tmp2 = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
tmp2.close()
try:
    shuffled = list(rows)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
    write_submission(tmp2.name, shuffled, dummy)
    try:
        read_submission(tmp2.name, rows)
        check("submit.py --check REJECTS a mis-ordered submission", False, "no error raised")
    except ValueError:
        check("submit.py --check REJECTS a mis-ordered submission", True)
finally:
    os.unlink(tmp2.name)

print("\n" + "=" * 70)
n_fail = sum(1 for _, ok in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
