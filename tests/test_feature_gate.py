"""Offline verification -- NO real Anthropic/Tavily calls anywhere in this
file. feature_engineer has no LLM client at all (pure deterministic menu), so
this is safe by construction.

Covers the Phase 1 item-side feature_engineer: candidate menu, generated
data.py validity against the real CSV headers, leakage discipline, the
_deterministic_edit contract, menu advancement, and the combined-candidate
union logic.

Run from anywhere: python3 tests/test_feature_gate.py
"""
import sys, os, ast, csv

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
sys.path.insert(0, os.path.join(_REPO, "agent", "specialists"))
os.chdir(os.path.join(_REPO, "agent"))

DATA_DIR = os.path.join(_REPO, "starter-kit", "KuaiRand-Pure", "data")
STARTER_KIT = os.path.join(_REPO, "starter-kit")

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    line = ("PASS" if cond else "FAIL") + " - " + name
    if not cond and detail:
        line += f"  --  {detail}"
    print(line)

from specialists.feature_engineer import (
    CANDIDATES, _SPEC, feature_engineer, _build_data_py, _combined_spec, N_BUCKETS,
)

LABELS = [c[0] for c in CANDIDATES]
check("5 candidates, combined last", LABELS == [
    "features:target_enc", "features:longview_prior", "features:item_cat",
    "features:play_quality", "features:combined"], str(LABELS))

with open(os.path.join(STARTER_KIT, "data.py")) as f:
    real_data_py = f.read()

# real CSV headers -- every column a candidate names must exist or load() KeyErrors
with open(os.path.join(DATA_DIR, "video_features_statistic_pure.csv")) as f:
    stat_header = set(next(csv.reader(f)))
with open(os.path.join(DATA_DIR, "video_features_basic_pure.csv")) as f:
    basic_header = set(next(csv.reader(f)))

LEAK_COLS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward",
             "is_hate", "play_time_ms"]

for lab, _desc in CANDIDATES:
    spec = _combined_spec([]) if lab == "features:combined" else _SPEC[lab]
    src = _build_data_py(lab, spec)

    try:
        ast.parse(src); parses = True
    except SyntaxError as e:
        parses = False; print("  syntax error:", e)
    check(f"{lab}: generated data.py parses", parses)

    # every ratio numerator/denominator column is a real stat-CSV header
    stat_cols = {c for r in spec["ratios"] for c in (r[1], r[2]) if c}
    check(f"{lab}: ratio columns exist in video_features_statistic_pure.csv",
          stat_cols <= stat_header, sorted(stat_cols - stat_header))
    # every categorical column is a real basic-CSV header
    check(f"{lab}: cat columns exist in video_features_basic_pure.csv",
          set(spec["cats"]) <= basic_header, sorted(set(spec["cats"]) - basic_header))
    # leakage: generated source must never touch a per-interaction log column
    check(f"{lab}: generated data.py references no label-leaking log column",
          not any(c in src for c in LEAK_COLS),
          [c for c in LEAK_COLS if c in src])
    # bucket count wired through
    check(f"{lab}: N_BUCKETS={N_BUCKETS} in generated source", f"N_BUCKETS = {N_BUCKETS}" in src)
    # row indices 0-6 preserved for baseline.py / submit.py
    check(f"{lab}: row tuple keeps duration_ms at index 5 / label at index 6",
          "float(r['duration_ms']), 1 if r[LABEL] != '0' else 0," in src)

# leakage-free flagship really is train-split-only
te_src = _build_data_py("features:target_enc", _SPEC["features:target_enc"])
check("target_enc: encodes off splits['train'] only (no valid/test in the TE build)",
      "tr = splits['train']" in te_src and "splits['valid']" not in te_src.split("def raw")[0])

# ---------------------------------------------------------------------------
# feature_engineer(state, tools) -- real function, mock read_file only
# ---------------------------------------------------------------------------
mock_tools = {"read_file": lambda kwargs: real_data_py}

out = feature_engineer({"tried_approaches": [], "current_scores": {"primary": 0.6016},
                        "experiment_history": []}, mock_tools)
check("sets _deterministic_edit targeting data.py",
      out.get("_deterministic_edit", {}).get("file") == "data.py")
check("old_code is the verbatim current data.py",
      out["_deterministic_edit"]["old_code"] == real_data_py)
check("picks first untried candidate (features:target_enc)",
      out["tried_approaches"] == ["features:target_enc"], out["tried_approaches"])
try:
    ast.parse(out["_deterministic_edit"]["new_code"]); ok = True
except SyntaxError as e:
    ok = False; print("  ", e)
check("new_code parses", ok)

out2 = feature_engineer({"tried_approaches": ["features:target_enc"],
                         "current_scores": {"primary": 0.6016}, "experiment_history": []}, mock_tools)
check("advances to features:longview_prior next",
      out2["tried_approaches"][-1] == "features:longview_prior", out2["tried_approaches"])

# menu exhausted -> wraps to combined, never crashes
out3 = feature_engineer({"tried_approaches": LABELS,
                         "current_scores": {"primary": 0.6016}, "experiment_history": []}, mock_tools)
check("menu exhausted -> still returns a _deterministic_edit (wraps to combined)",
      out3["_deterministic_edit"] is not None)

# combined union: two winners in history -> merged spec
hist = [{"specialist": "feature_engineer", "improved": True,
         "hypothesis": "Feature engineer [features:target_enc]: ..."},
        {"specialist": "feature_engineer", "improved": True,
         "hypothesis": "Feature engineer [features:longview_prior]: ..."},
        {"specialist": "feature_engineer", "improved": False,
         "hypothesis": "Feature engineer [features:item_cat]: ..."}]
cs = _combined_spec(hist)
check("combined union: takes te from a winner", cs["te"] is True)
check("combined union: takes ratios from a winner",
      [r[0] for r in cs["ratios"]] == [r[0] for r in _SPEC["features:longview_prior"]["ratios"]])
check("combined union: excludes the non-winner's cats", cs["cats"] == [])
check("combined fallback (no winners) -> target_enc + longview_prior",
      _combined_spec([])["te"] is True and len(_combined_spec([])["ratios"]) == 4)

print("\n" + "=" * 70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
