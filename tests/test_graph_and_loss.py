"""Offline verification -- NO real API calls. loss_function_changer is
checked as static text only (never invoked -- calling it would construct a
real Anthropic client and make a real request).
Run from anywhere: python3 tests/test_graph_and_loss.py
"""
import sys, os, ast

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "agent"))
os.chdir(os.path.join(_REPO, "agent"))

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

# ---------------------------------------------------------------------------
# graph.py wiring (import only -- build_graph() itself needs real tools/mcp,
# out of scope for an offline test; importability + dict membership is what
# matters here)
# ---------------------------------------------------------------------------
import graph
check("feature_engineer is in graph.SPECIALIST_FNS", "feature_engineer" in graph.SPECIALIST_FNS)
check("feature_engineer is in graph.SPECIALISTS", "feature_engineer" in graph.SPECIALISTS)
check("graph.SPECIALIST_FNS['feature_engineer'] is the real function",
      graph.SPECIALIST_FNS["feature_engineer"].__module__.endswith("feature_engineer"))

out = graph._extract_specialist_output({"_deterministic_edit": {"file": "data.py"}, "junk": "dropped"})
check("_extract_specialist_output keeps _deterministic_edit", "_deterministic_edit" in out)
check("_extract_specialist_output drops unlisted keys", "junk" not in out)

# ---------------------------------------------------------------------------
# loss_function_changer -- STATIC TEXT CHECKS ONLY. Never call the function.
# ---------------------------------------------------------------------------
with open("specialists/loss_function_changer.py") as f:
    src = f.read()

try:
    ast.parse(src)
    check("loss_function_changer.py parses", True)
except SyntaxError as e:
    check("loss_function_changer.py parses", False, str(e))

check("stale 'log-loss' baseline premise is gone",
      "trained with log-loss" not in src)
check("corrected premise mentions BPR is already the baseline",
      "already trained with BPR" in src)
check("single-edit-location constraint text present (mirrors model_swapper's fix)",
      "CRITICAL EXECUTION CONSTRAINT" in src and "ONE find-and-replace edit" in src)
check("instruction to edit inside FM.step() in place present",
      "INSIDE FM.step()" in src)
check("'bpr' removed from the LOSS_CHOICE menu (baseline already IS bpr)",
      "LOSS_CHOICE: [bpr | softmax | focal | warp]" not in src)
check("_decide_technique candidates no longer include a redundant 'bpr' option",
      '("bpr", "Bayesian Personalised Ranking BPR loss")' not in src)

# ---------------------------------------------------------------------------
# supervisor.py -- static check that the stale hardcoded ban is gone
# ---------------------------------------------------------------------------
with open("specialists/supervisor.py") as f:
    sup_src = f.read()
check("stale 'loss_function_changer has already been tried' ban removed",
      "loss_function_changer has already been tried" not in sup_src)
check("feature_engineer appears in ROUTE_TO enum text",
      "feature_engineer" in sup_src and "ROUTE_TO: [" in sup_src)

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
