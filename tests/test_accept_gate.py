"""Phase 0.5 / 0.6 -- score_analyst's two-stage accept gate + no-op detector,
tested offline with fake tools (no real pipeline, no API calls).

    python3 tests/test_accept_gate.py
"""
import os
import re
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
os.chdir(os.path.join(_REPO, "agent"))

import agent
from state import initial_state

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    line = ("PASS" if cond else "FAIL") + " - " + name
    if not cond and detail:
        line += f"  --  {detail}"
    print(line)


# --- fake tools -------------------------------------------------------------
_CANNED = {}   # seed -> primary the fake pipeline reports
_CALLS = []    # ("save"|"restore", kwargs)


def _fake_run_pipeline(kw):
    ea = kw.get("extra_args", "")
    seed = int(ea.split("--seed")[1].split()[0]) if "--seed" in ea else 0
    p = _CANNED.get(seed, 0.6015)
    return f"  valid  GAUC 0.6670 | nDCG@5 0.5360 | primary {p:.4f}\n  test  GAUC 0.66 | nDCG@5 0.52 | primary 0.59"


def _fake_parse(kw):
    m = re.search(r"valid\s+GAUC\s+([\d.]+)\s+\|\s+nDCG@5\s+([\d.]+)\s+\|\s+primary\s+([\d.]+)",
                  kw["pipeline_output"])
    if not m:
        return {"GAUC": None, "nDCG@5": None, "primary": None}
    return {"GAUC": float(m.group(1)), "nDCG@5": float(m.group(2)), "primary": float(m.group(3))}


TOOLS = {
    "run_pipeline": _fake_run_pipeline,
    "parse_scores": _fake_parse,
    "save_checkpoint": lambda kw: _CALLS.append(("save", kw)) or "ok",
    "restore_checkpoint": lambda kw: _CALLS.append(("restore", kw)) or "ok",
    "read_file": lambda kw: "# code",
}


def _state(seed0_primary, best_primary, base_seed0):
    s = initial_state()
    s["iteration"] = 5
    s["current_scores"] = {"gauc": 0.667, "ndcg5": 0.536, "primary": seed0_primary}
    s["_raw_scores"] = {"GAUC": 0.667, "nDCG@5": 0.536, "primary": seed0_primary}
    s["best_scores"] = {"gauc": 0.667, "ndcg5": 0.536, "primary": best_primary}
    s["best_iteration"] = 3
    s["_best_seed0_primary"] = base_seed0
    s["next_specialist"] = "feature_engineer"
    s["experiment_history"] = []
    return s


def _run(seed0, best, base, canned):
    global _CANNED
    _CALLS.clear()
    _CANNED = canned
    out = agent.score_analyst(_state(seed0, best, base), TOOLS)
    return out, out["experiment_history"][-1], (_CALLS[0][0] if _CALLS else None)


# A: bit-identical seed-0 score == dead edit -> flagged no_op, rejected, no confirm runs
_, hA, actA = _run(0.6015, 0.6015, 0.6015, {})
check("no-op: flagged", hA["no_op"] is True)
check("no-op: not accepted", hA["improved"] is False and actA == "restore")
check("no-op: skips confirmation seeds", hA["confirm_seeds"] == 1)

# B: gain below SEED0_SCREEN_DELTA (0.0004) -> never confirmed, never accepted
_, hB, actB = _run(0.6019, 0.6015, 0.6015, {})
check("noise gain: not confirmed", hB["confirm_seeds"] == 1)
check("noise gain: rejected", hB["improved"] is False and actB == "restore")

# C: real gain, 3-seed mean clears the margin -> accepted, best + anchor updated
sC, hC, actC = _run(0.610, 0.6015, 0.6015, {0: 0.610, 1: 0.609, 2: 0.611})
check("real gain: confirmed at 3 seeds", hC["confirm_seeds"] == 3)
check("real gain: accepted", hC["improved"] is True and actC == "save")
check("real gain: best_scores holds the 3-seed mean", abs(sC["best_scores"]["primary"] - 0.610) < 1e-6)
check("real gain: seed-0 anchor advances", sC["_best_seed0_primary"] == 0.610)

# D: seed 0 clears the screen but seeds 1,2 drag the mean back under the margin -> rejected
_, hD, actD = _run(0.6035, 0.6015, 0.6015, {0: 0.6035, 1: 0.6005, 2: 0.6008})
check("unconfirmed gain: ran confirmation seeds", hD["confirm_seeds"] == 3)
check("unconfirmed gain: mean below margin -> rejected", hD["improved"] is False and actD == "restore")

# E: a non-improving FORCED feature_engineer curriculum iteration must NOT burn
#    the convergence budget (otherwise the run stops before model_swapper).
_CALLS.clear(); _CANNED = {}
from specialists.feature_engineer import CANDIDATES as _FC
sE = _state(0.6013, 0.6015, 0.6015)
sE["next_specialist"] = "feature_engineer"
sE["tried_approaches"] = [_FC[0][0]]            # a forced feature candidate just tried
sE["iterations_without_improvement"] = 3
outE = agent.score_analyst(sE, TOOLS)
check("forced feature iter: rejected", outE["experiment_history"][-1]["improved"] is False)
check("forced feature iter: convergence counter unchanged",
      outE["iterations_without_improvement"] == 3)

# F: a non-improving iteration AFTER the forced phase DOES count.
_CALLS.clear(); _CANNED = {}
sF = _state(0.6030, 0.6045, 0.6033)
sF["next_specialist"] = "loss_function_changer"
sF["tried_approaches"] = [c[0] for c in _FC] + ["model:lgbm", "loss:bpr"]
sF["iterations_without_improvement"] = 2
outF = agent.score_analyst(sF, TOOLS)
check("post-curriculum iter: convergence counter increments",
      outF["iterations_without_improvement"] == 3)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
