"""Phase 0.3 -- Task Requirement 1 ("reproduce the official baseline") as a test.

The agent's whole premise is a delta over the official FM baseline, so that
baseline must be the shipped one and must reproduce its published number.
Commit 36443bc silently replaced it with an unvalidated BPR loss; this test
exists so that can never happen unnoticed again.

Runs the real pipeline (~60s per invocation). By default does a single seed-0
run. Set FULL_BASELINE_TEST=1 for the determinism (T9) + sanity checks too
(~3 min total).

    python3 tests/test_baseline_reproduces.py
    FULL_BASELINE_TEST=1 python3 tests/test_baseline_reproduces.py
"""
import hashlib
import os
import re
import subprocess
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_KIT = os.path.join(_REPO, "starter-kit")
_PY = os.path.join(_REPO, ".venv", "bin", "python")
if not os.path.exists(_PY):
    _PY = sys.executable

# Published official FM baseline, validation split (starter-kit/baseline_scores.json)
OFFICIAL_VALID_PRIMARY = 0.6016
OFFICIAL_VALID_GAUC = 0.6674
OFFICIAL_VALID_NDCG = 0.5357
# Measured spread of the restored baseline over seeds 0-4 (2026-09-01):
# primary 0.6011-0.6020, mean 0.6016, population std ~0.0003. Band = mean +/- 0.003
# (well inside the convergence epsilon of 0.002 doubled) -- generous enough not
# to flake, tight enough to catch a real regression (item-popularity is 0.5807).
PRIMARY_LO, PRIMARY_HI = 0.5986, 0.6046
ITEM_POPULARITY_PRIMARY = 0.5807
ORACLE_VALID_PRIMARY = 0.8484

_VALID_RE = re.compile(
    r"^\s*valid\s+GAUC\s+([\d.]+)\s+\|\s+nDCG@5\s+([\d.]+)\s+\|\s+primary\s+([\d.]+)",
    re.MULTILINE,
)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    line = f"{mark} - {name}"
    if not cond and detail:
        line += f"  --  {detail}"
    print(line)


def run_fm(seed):
    """Run the official FM pipeline at `seed`, return (gauc, ndcg5, primary, raw_stdout)."""
    out = subprocess.run(
        [_PY, "baseline.py", "--model", "fm", "--seed", str(seed)],
        cwd=_KIT, capture_output=True, text=True, timeout=600,
    )
    combined = out.stdout + out.stderr
    m = _VALID_RE.search(combined)
    if not m:
        return None, None, None, combined
    return float(m.group(1)), float(m.group(2)), float(m.group(3)), combined


# ---------------------------------------------------------------------------
# 0. The file on disk must actually be the official log-loss FM, not a
#    hand-edited variant. baseline_official.py is the frozen reference.
# ---------------------------------------------------------------------------
with open(os.path.join(_KIT, "baseline.py")) as f:
    live_src = f.read()
official_path = os.path.join(_KIT, "baseline_official.py")
check("starter-kit/baseline_official.py exists (frozen reference)", os.path.exists(official_path))
if os.path.exists(official_path):
    with open(official_path) as f:
        frozen_src = f.read()
    check("baseline.py == baseline_official.py (no drift from the shipped FM)",
          hashlib.md5(live_src.encode()).hexdigest() == hashlib.md5(frozen_src.encode()).hexdigest(),
          "baseline.py has diverged from the frozen official copy")

check("FM.step uses pointwise log-loss gradient (sigmoid(z) - y)/B",
      "(sigmoid(z) - y) / B" in live_src,
      "step() gradient is not the official log-loss form")
check("FM.step is NOT the unvalidated BPR variant",
      "BPR pairwise loss" not in live_src and "bpr_loss" not in live_src,
      "baseline.py still contains the contaminated BPR loss")

# ---------------------------------------------------------------------------
# 1. Reproduction: seed 0 must land on the published validation number.
# ---------------------------------------------------------------------------
g0, n0, p0, raw = run_fm(0)
check("pipeline produced a parseable valid score", p0 is not None,
      (raw[-400:] if p0 is None else ""))
if p0 is not None:
    check(f"valid primary {p0:.4f} reproduces official {OFFICIAL_VALID_PRIMARY:.4f} "
          f"(band [{PRIMARY_LO}, {PRIMARY_HI}])",
          PRIMARY_LO <= p0 <= PRIMARY_HI)
    check(f"valid GAUC {g0:.4f} ~ official {OFFICIAL_VALID_GAUC:.4f} (+/- 0.003)",
          abs(g0 - OFFICIAL_VALID_GAUC) <= 0.003)
    check(f"valid nDCG@5 {n0:.4f} ~ official {OFFICIAL_VALID_NDCG:.4f} (+/- 0.003)",
          abs(n0 - OFFICIAL_VALID_NDCG) <= 0.003)
    check("baseline beats item-popularity reference (0.5807)", p0 > ITEM_POPULARITY_PRIMARY + 0.01)
    check("baseline is well below the oracle ceiling (0.8484)", p0 < ORACLE_VALID_PRIMARY - 0.1)

if os.environ.get("FULL_BASELINE_TEST"):
    # -----------------------------------------------------------------------
    # T9. Determinism -- same seed twice must give a bit-identical score.
    #     This is what makes score_analyst's no-op detector sound.
    # -----------------------------------------------------------------------
    _, _, p0b, _ = run_fm(0)
    check("seed 0 is deterministic (identical score on re-run)",
          p0b is not None and p0 is not None and abs(p0 - p0b) < 1e-9,
          f"{p0} vs {p0b}")

    # -----------------------------------------------------------------------
    # T1-adjacent. A different seed must move the score (the pipeline is not
    #     ignoring its inputs) but stay within the measured noise band.
    # -----------------------------------------------------------------------
    _, _, p1, _ = run_fm(1)
    check("seed 1 differs from seed 0 (pipeline responds to inputs)",
          p1 is not None and abs(p1 - p0) > 1e-9)
    check("seed 1 still within the reproduction band",
          p1 is not None and PRIMARY_LO <= p1 <= PRIMARY_HI)

# ---------------------------------------------------------------------------
failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
