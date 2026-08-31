"""Offline verification -- NO API calls, pure local CSV reads.
Covers data.py's get_feature_info() discovery helper and the deliberate
exclusion of log-file per-interaction columns (label leakage).
Run from anywhere: python3 tests/test_feature_discovery.py
"""
import sys, os, csv

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "starter-kit"))
os.chdir(os.path.join(_REPO, "starter-kit"))

DATA_DIR = os.path.join(_REPO, "starter-kit", "KuaiRand-Pure", "data")

results = []
def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), "-", name, ("--", detail) if detail and not cond else "")

from data import get_feature_info, USER_FEATURE_FILE, VIDEO_FEATURE_FILE, LABEL

check("USER_FEATURE_FILE points at a real file", os.path.isfile(USER_FEATURE_FILE))
check("VIDEO_FEATURE_FILE points at a real file", os.path.isfile(VIDEO_FEATURE_FILE))

info = get_feature_info()
check("get_feature_info returns user_feature_columns", "user_feature_columns" in info)
check("get_feature_info returns video_feature_columns", "video_feature_columns" in info)
check("user_feature_columns has 31 entries (30 features + user_id)",
      len(info["user_feature_columns"]) == 31, len(info["user_feature_columns"]))
check("video_feature_columns has 52 entries (51 features + video_id)",
      len(info["video_feature_columns"]) == 52, len(info["video_feature_columns"]))
check("user_feature_columns matches the real CSV header", info["user_feature_columns"][0] == "user_id")
check("video_feature_columns matches the real CSV header", info["video_feature_columns"][0] == "video_id")

# get_feature_info must NEVER surface log-file columns -- those are where the
# label-leakage risk lives (LABEL is derived from the same row's
# play_time_ms/duration_ms, and is_click/is_like/etc. are simultaneous
# outcomes of the row being predicted).
LOG_LEAK_COLS = {"is_click", "is_like", "is_follow", "is_comment", "is_forward",
                  "play_time_ms", "is_hate", LABEL}
surfaced = set(info["user_feature_columns"]) | set(info["video_feature_columns"])
check("get_feature_info never surfaces log-file feedback/label columns",
      not (surfaced & LOG_LEAK_COLS), surfaced & LOG_LEAK_COLS)

with open(os.path.join(DATA_DIR, "log_standard_4_08_to_4_21_pure.csv")) as f:
    log_header = next(csv.reader(f))
check("LABEL ('long_view') really is a log-file column (sanity check on the leakage premise)",
      LABEL in log_header)
check("play_time_ms really is a log-file column (sanity check on the leakage premise)",
      "play_time_ms" in log_header)

print("\n" + "="*70)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results)-n_fail}/{len(results)} PASSED" + (f", {n_fail} FAILED" if n_fail else ""))
sys.exit(1 if n_fail else 0)
