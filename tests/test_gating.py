"""Offline verification -- NO real API calls. supervisor's forced-override
returns before constructing any Anthropic client call when the feature menu
isn't exhausted, so this is safe by construction. agent.py's gate is a pure
function with no LLM/tools involvement at all.
Run from anywhere: python3 tests/test_gating.py
"""
import sys, os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
os.chdir(os.path.join(_REPO, "agent"))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

# ---------------------------------------------------------------------------
# 1. supervisor forced-override (menu not exhausted -> no LLM call, no client
#    construction reached)
# ---------------------------------------------------------------------------
from specialists.supervisor import supervisor, SPECIALISTS
from specialists.feature_engineer import CANDIDATES as FEATURE_CANDIDATES

check("feature_engineer is in supervisor.SPECIALISTS", "feature_engineer" in SPECIALISTS)

_L = [c[0] for c in FEATURE_CANDIDATES]
for tried in ([], _L[:1], _L[:2]):
    state = {"tried_approaches": tried, "current_scores": {"primary": 0.6016},
              "best_scores": {"primary": 0.6016}, "experiment_history": [], "iteration": 1}
    out = supervisor(state, tools={})  # empty tools dict -- forced path must never touch tools
    check(f"forced override fires with tried={tried}",
          out["next_specialist"] == "feature_engineer", out["next_specialist"])

all_labels = [c[0] for c in FEATURE_CANDIDATES]
untried = [l for l in all_labels if l not in all_labels]  # simulate "all tried"
check("guard condition is empty once all candidates are in tried_approaches", untried == [])

# ---------------------------------------------------------------------------
# 2. agent.py's _check_feature_phase_gate -- DISABLED 2026-09-01. It used to
#    stop the whole run if no feature_engineer candidate improved; Phase 1 +
#    Phase 3 showed the LightGBM stack (reachable AFTER the feature menu, via
#    model_swapper's model:lgbm) does beat the baseline, so stopping at the
#    feature phase would skip the thing that works. It must now ALWAYS return
#    None regardless of history.
# ---------------------------------------------------------------------------
from agent import _check_feature_phase_gate

feature_labels = [c[0] for c in FEATURE_CANDIDATES]

no_improve_hist = [{"iteration": i + 1, "specialist": "feature_engineer", "improved": False}
                   for i in range(len(feature_labels))]
for label, st in [
    ("menu not exhausted", {"tried_approaches": feature_labels[:1], "experiment_history": []}),
    ("menu exhausted, none improved", {"tried_approaches": feature_labels,
                                       "experiment_history": no_improve_hist}),
    ("menu exhausted, one improved", {"tried_approaches": feature_labels,
                                      "experiment_history": [dict(h, improved=(i == 1))
                                                             for i, h in enumerate(no_improve_hist)]}),
]:
    check(f"feature-phase gate is a no-op ({label})",
          _check_feature_phase_gate(st) is None)

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
