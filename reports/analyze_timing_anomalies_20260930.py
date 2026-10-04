"""Reproduce timing-anomaly statistics from saved OJ records; no network/GPU."""

import collections
import csv
import datetime
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
COHORT = REPORTS / "2026-09-30-timing-cohort-readonly.json"


def main():
    data = json.loads(COHORT.read_text())
    submissions = data["submissions"]
    base = {v["case"]: v for v in json.loads((REPORTS / "2026-09-30-v926-score.json").read_text())["cases"]}
    statuses = collections.Counter()
    groups = collections.defaultdict(list)
    daily = collections.defaultdict(lambda: {"accepted": 0, "severe": 0})
    first = {"accepted": 0, "severe": 0}
    repeat = {"accepted": 0, "severe": 0}
    seen, severe_rows, rows, failed_zero = set(), [], [], []
    case_hits = collections.Counter()
    for sid, row in sorted(submissions.items(), key=lambda item: int(item[0])):
        sha = row.get("source_sha256")
        first_seen = sha not in seen
        seen.add(sha)
        statuses[row["status"]] += 1
        low, high_tb, failed = [], [], []
        for k, case in row.get("cases", {}).items():
            m = case.get("metrics", {})
            if not m:
                continue
            c, tk, tb = int(k), m["tk_time_ms"], m["tb_time_ms"]
            passed = m.get("pass") is True and case.get("status") == "Accepted"
            if passed and tk < 0.5 * base[c]["tk_ms"]:
                low.append(c)
                case_hits[c] += 1
            if passed and tb >= 2 * base[c]["tb_ms"]:
                high_tb.append(c)
            if not passed and tk == 0:
                failed.append(c)
        stamp = datetime.datetime.fromisoformat(row["submitTime"].replace("Z", "+00:00"))
        local = stamp.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
        flat = {
            "sid": int(sid), "submit_time_cst": local, "status": row["status"],
            "raw": row.get("displayScore"), "sha256": sha,
            "first_sha_in_cohort": first_seen, "severe_cases": sorted(low),
            "tb_ge2x_cases": sorted(high_tb), "failed_zero_cases": sorted(failed),
        }
        for c in range(1, 13):
            m = row.get("cases", {}).get(str(c), {}).get("metrics", {})
            flat[f"c{c}_tk"] = m.get("tk_time_ms")
            flat[f"c{c}_tb"] = m.get("tb_time_ms")
        if low:
            severe_rows.append(flat)
        if failed:
            failed_zero.append({"sid": int(sid), "cases": failed})
        if row["status"] == "Accepted":
            counter = first if first_seen else repeat
            counter["accepted"] += 1
            counter["severe"] += bool(low)
            daily[local[:10]]["accepted"] += 1
            daily[local[:10]]["severe"] += bool(low)
        groups[sha].append(flat)
        rows.append(flat)
    matched = []
    for sha, group in groups.items():
        lows = [r for r in group if r["severe_cases"] and r["status"] == "Accepted"]
        normals = [r for r in group if not r["severe_cases"] and r["status"] == "Accepted"]
        for lo in lows:
            if not normals:
                continue
            normal = min(normals, key=lambda r: abs(r["sid"] - lo["sid"]))
            matched.append({"sha256": sha, "low_sid": lo["sid"], "normal_sid": normal["sid"],
                            "low_tk_c10_c11_c12": [lo[f"c{c}_tk"] for c in (10, 11, 12)],
                            "normal_tk_c10_c11_c12": [normal[f"c{c}_tk"] for c in (10, 11, 12)]})
    normal_sids = [149499,149505,149508,149510,149513,149515,149517,149519,149521,149524,149529]
    normal_rows = [submissions[str(s)] for s in normal_sids]
    median_tk = {str(c): statistics.median(r["cases"][str(c)]["metrics"]["tk_time_ms"] for r in normal_rows) for c in base}
    median_tb = {str(c): statistics.median(r["cases"][str(c)]["metrics"]["tb_time_ms"] for r in normal_rows) for c in base}
    best_models = {}
    best = submissions["149493"]
    for mode in ("observed", "replace_tk11_12", "replace_all_tb", "replace_both"):
        points = []
        for k, case in best["cases"].items():
            m = case["metrics"]
            tk, tb = m["tk_time_ms"], m["tb_time_ms"]
            if mode in ("replace_tk11_12", "replace_both") and k in ("11", "12"):
                tk = median_tk[k]
            if mode in ("replace_all_tb", "replace_both"):
                tb = median_tb[k]
            points.append(math.floor(100 * tb / (tb + tk)))
        best_models[mode] = sum(points) / 12
    zero_models = {"_".join(map(str, cases)): sum(100 if c in cases else base[c]["q"] for c in base)/12
                   for cases in ((12,), (11,12), (10,11,12), (9,10,11,12), (1,2,10,11,12))}
    result = {
        "method": "Contiguous account P1 cohort; severe = accepted testcase/pass=true and tk < 50% of v926 per-case reference",
        "scope_note": "Post-hoc observational association; first source SHA is not proof of a cold compiler/cache or new worker. No randomized experiment.",
        "cohort_path": str(COHORT.relative_to(ROOT)), "checked_at": data["checked_at"],
        "complete_to_lower_bound": data["list_complete_to_lower_bound"],
        "sid_range": [min(map(int,submissions)),max(map(int,submissions))],
        "submissions": len(rows), "statuses": statuses,
        "severe_any_submission": len(severe_rows),
        "severe_full_accepted": sum(r["status"] == "Accepted" for r in severe_rows),
        "severe_testcases_by_case": case_hits,
        "first_sha_in_cohort": first, "repeated_sha_in_cohort": repeat,
        "accepted_by_day_cst": dict(daily), "severe_rows": severe_rows,
        "matched_exact_source": matched, "failed_zero_not_counted": failed_zero,
        "best_149493_conditional_models": {"normal_source_sids": normal_sids, "raw": best_models,
                                            "normal_tk11":median_tk["11"], "normal_tk12":median_tk["12"]},
        "v926_zero_tk_conditional_raw": zero_models,
    }
    with (REPORTS / "2026-09-30-timing-anomalies.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (REPORTS / "2026-09-30-timing-anomalies.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("submissions", "statuses", "severe_full_accepted", "first_sha_in_cohort", "repeated_sha_in_cohort", "severe_testcases_by_case", "matched_exact_source", "best_149493_conditional_models", "v926_zero_tk_conditional_raw")}, indent=2))


if __name__ == "__main__":
    main()
