"""Offline verification -- NO real Anthropic/Tavily calls anywhere in this
file. feature_engineer's deterministic path returns before ever constructing
a client call, so this is safe by construction (not by mocking discipline).
Run from anywhere: python3 tests/test_feature_gate.py
"""
import sys, os, ast, csv

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
sys.path.insert(0, os.path.join(_REPO, "agent", "specialists"))
os.chdir(os.path.join(_REPO, "agent"))  # so `from specialists.X import Y` resolves

DATA_DIR = os.path.join(_REPO, "starter-kit", "KuaiRand-Pure", "data")
STARTER_KIT = os.path.join(_REPO, "starter-kit")

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

# ---------------------------------------------------------------------------
# 1. feature_engineer: candidate correctness against the REAL data.py + CSVs
# ---------------------------------------------------------------------------
from specialists.feature_engineer import CANDIDATES, feature_engineer, _build_data_py

check("feature_engineer has exactly 3 candidates", len(CANDIDATES) == 3, str(CANDIDATES))

with open(os.path.join(STARTER_KIT, "data.py")) as f:
    real_data_py = f.read()

for label, feat_cols, desc in CANDIDATES:
    new_code = _build_data_py(feat_cols)
    try:
        ast.parse(new_code)
        parses = True
    except SyntaxError as e:
        parses = False
        print("  syntax error:", e)
    check(f"{label}: generated data.py parses", parses)
    check(f"{label}: FIELDS line reflects {len(feat_cols)} extra domain(s)",
          f"'{feat_cols[0]}_bucket'" in new_code if feat_cols else True)

# Confirm all NUMERIC_VIDEO_FEATS column names actually exist as headers in
# the real video_features_statistic_pure.csv (would silently KeyError at
# runtime otherwise)
with open(os.path.join(DATA_DIR, "video_features_statistic_pure.csv")) as f:
    header = next(csv.reader(f))
all_cols = sorted({c for _, cols, _ in CANDIDATES for c in cols})
missing = [c for c in all_cols if c not in header]
check("all candidate columns exist in video_features_statistic_pure.csv header", not missing, missing)

# ---------------------------------------------------------------------------
# 2. feature_engineer(state, tools) -- exercise the actual function, mock
#    tools only (no Anthropic client exists in this specialist at all)
# ---------------------------------------------------------------------------
mock_tools = {"read_file": lambda kwargs: real_data_py}
state = {"tried_approaches": [], "current_scores": {"primary": 0.6016}}
out = feature_engineer(state, mock_tools)
check("feature_engineer sets _deterministic_edit", out.get("_deterministic_edit") is not None)
check("feature_engineer picks first untried candidate (features:playcount)",
      out["tried_approaches"] == ["features:playcount"], out["tried_approaches"])
check("_deterministic_edit targets data.py", out["_deterministic_edit"]["file"] == "data.py")
check("_deterministic_edit old_code matches real data.py exactly",
      out["_deterministic_edit"]["old_code"] == real_data_py)
try:
    ast.parse(out["_deterministic_edit"]["new_code"])
    check("_deterministic_edit new_code parses", True)
except SyntaxError as e:
    check("_deterministic_edit new_code parses", False, str(e))

# second call with iteration 1 already tried -> should pick candidate 2
state2 = {"tried_approaches": ["features:playcount"], "current_scores": {"primary": 0.6016}}
out2 = feature_engineer(state2, mock_tools)
check("feature_engineer advances to features:engagement3 next",
      out2["tried_approaches"] == ["features:playcount", "features:engagement3"], out2["tried_approaches"])

# all 3 tried -> menu wraps around, doesn't crash
state3 = {"tried_approaches": ["features:playcount", "features:engagement3", "features:engagement6"],
          "current_scores": {"primary": 0.6016}}
out3 = feature_engineer(state3, mock_tools)
check("feature_engineer doesn't crash once menu exhausted (wraps to candidate 1)",
      out3["_deterministic_edit"] is not None)

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
