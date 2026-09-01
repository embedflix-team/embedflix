"""Render run_log.jsonl (+ logs/run_summary.txt) into the Deliverable-3 markdown.

    python3 deliverables/make_iteration_table.py
writes deliverables/_iteration_table.md (a fragment pasted into 03_run_and_iteration_log.md).
"""
import json
import os
import re
import textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "run_log.jsonl")
SUMMARY = os.path.join(REPO, "logs", "run_summary.txt")
OUT = os.path.join(REPO, "deliverables", "_iteration_table.md")

BASELINE = 0.6016  # official validation primary


def one_line(s, n=90):
    s = " ".join((s or "").split())
    return (s[: n - 1] + "…") if len(s) > n else s


def diff_summary(code_diff):
    if not code_diff:
        return "—"
    m = re.search(r"^--- (\S+)", code_diff)
    f = m.group(1) if m else "?"
    minus = len(re.findall(r"^- ", code_diff, re.M))
    plus = len(re.findall(r"^\+ ", code_diff, re.M))
    return f"`{f}` (−{minus}/+{plus} lines)"


def verdict(e):
    rec = (e.get("recovery") or "")
    if "[NO-OP]" in rec:
        return "no-op"
    p = e.get("score", {}).get("primary")
    if p is None:
        return "error"
    # heuristic: 'improved' isn't in the log; infer from whether primary rose vs baseline meaningfully
    return "accept?" if p and p > BASELINE + 0.001 else "reject"


rows = [json.loads(l) for l in open(LOG) if l.strip()]
lines = [
    "| iter | specialist | hypothesis | code diff | GAUC | nDCG@5 | primary | Δ vs baseline | outcome | error / recovery |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for e in rows:
    it = e.get("iteration")
    hyp = e.get("hypothesis", "") or ""
    spec = "baseline" if it == 0 else (hyp.split("[")[1].split("]")[0] if "[" in hyp else hyp.split(":")[0][:22])
    sc = e.get("score", {}) or {}
    g, n, p = sc.get("GAUC"), sc.get("nDCG@5"), sc.get("primary")
    delta = f"{p - BASELINE:+.4f}" if isinstance(p, (int, float)) else "—"
    err = one_line((e.get("error") or "") + " " + (e.get("recovery") or ""), 70) or "—"
    lines.append(
        f"| {it} | {spec} | {one_line(hyp.split('REASONING:')[0], 80)} | {diff_summary(e.get('code_diff'))} "
        f"| {g:.4f} | {n:.4f} | {p:.4f} | {delta} | {verdict(e)} | {err} |"
        if isinstance(p, (int, float))
        else f"| {it} | {spec} | {one_line(hyp, 80)} | {diff_summary(e.get('code_diff'))} | — | — | — | — | error | {err} |"
    )

summary = ""
if os.path.exists(SUMMARY):
    summary = "```\n" + open(SUMMARY).read().strip() + "\n```"

with open(OUT, "w") as f:
    f.write("### Per-iteration table\n\n" + "\n".join(lines) + "\n\n### Run summary\n\n" + summary + "\n")
print("wrote", OUT, f"({len(rows)} iterations)")
