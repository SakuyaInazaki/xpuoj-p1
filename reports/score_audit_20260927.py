"""Rebuild the 2026-09-27 P1 score audit from saved, non-secret measurements.

No platform client, credentials, network request, or GPU is used.
Run from any directory: python3 reports/score_audit_20260927.py
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import re
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
DATA = ROOT / "experiments/2026-09-20/knobs6_data"
DB = json.loads((DATA / "tk.json").read_text())
RECENT = json.loads((OUT / "2026-09-27-latest-cases.json").read_text())
ANCHORS = "146862 146866 146868 146871 146874 146877 146888 146911 146913 146915".split()
OLD_ANCHORS = (DATA / "v841.txt").read_text().split()
for sid, v in RECENT["submissions"].items():
    DB[sid] = {i: [c["tk"], c["tb"], c["sqnr"], c["score"]] for i, c in v["cases"].items()}


def case_score(tk, tb):
    return math.floor(100 * tb / (tb + tk))


def sum_score(times, baselines):
    return sum(case_score(t, b) for t, b in zip(times, baselines))


def needed_uniform(times, baselines, required):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum_score([t * (1 - mid) for t in times], baselines) >= required:
            hi = mid
        else:
            lo = mid
    return hi


def quantile(values, p):
    a = sorted(values)
    x = (len(a) - 1) * p
    lo = int(x)
    return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (x - lo)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


kernel = (ROOT / "p1/kernel.py").read_text()
EXPECTED_SHA = "5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c"
if hashlib.sha256(kernel.encode()).hexdigest() != EXPECTED_SHA:
    raise SystemExit("Current kernel differs from audited v842; select matching measurements before regenerating this report.")
tree = ast.parse(kernel)
shapes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "_KNOWN12" for t in n.targets))
targets = [80, 79, 84, 86, 82, 86, 82, 84, 77, 78, 86, 84]
rows = []
for i, (T, H, E, I, k) in enumerate(shapes, 1):
    v = [DB[s][str(i)] for s in ANCHORS]
    tk = [x[0] for x in v]
    tb = [DB[s][str(i)][1] for s in ANCHORS + OLD_ANCHORS]
    a, b, p = st.mean(tk), st.median(tb), targets[i - 1]
    s = case_score(a, b)
    limit = b * (100 - p) / p
    own_scores = [case_score(x[0], x[1]) for x in v]
    own_gain = [max(0.0, 1 - x[1] * (100 - p) / p / x[0]) for x in v]
    rows.append(dict(case=i, T=T, H=H, E=E, I=I, topk=k, local_experts=E // 4,
                     replicated_rows_per_expert=T * k / E,
                     ep_rows_per_expert=4 * T * k / E,
                     exact_v842_samples=len(v), tk_mean_ms=a, tk_median_ms=st.median(tk),
                     tk_min_ms=min(tk), tk_max_ms=max(tk), tk_sd_ms=st.stdev(tk),
                     tb_reference_samples=len(tb), tb_median_ms=b,
                     tb_p25_ms=quantile(tb, .25), tb_p75_ms=quantile(tb, .75),
                     tb_min_ms=min(tb), tb_max_ms=max(tb),
                     standardized_case_score=s, next_integer=s + 1,
                     next_integer_tk_limit_ms=b * (99 - s) / (s + 1),
                     selected_target=p, selected_target_tk_limit_ms=limit,
                     selected_target_gap_ms=a - limit,
                     selected_target_needed_reduction_pct=100 * (1 - limit / a),
                     actual_target_crossings=sum(x >= p for x in own_scores),
                     actual_score_min=min(own_scores), actual_score_max=max(own_scores),
                     all_observed_crossings_needed_reduction_pct=100 * max(own_gain),
                     continuous_case_points_per_ms=100 * b / (b + a) ** 2,
                     continuous_problem_points_per_ms=100 * b / (b + a) ** 2 / 12,
                     min_logged_sqnr_db=min(x[2] for x in v)))
write_csv("2026-09-27-score-thresholds.csv", rows)

times = [r["tk_mean_ms"] for r in rows]
refs = {key: [r[col] for r in rows] for key, col in
        [("tb_p25", "tb_p25_ms"), ("tb_median", "tb_median_ms"), ("tb_p75", "tb_p75_ms")]}
scenarios = []
for ref, baselines in refs.items():
    for gain in [0, .05, .10, .15, .20, .25, .30, .40, .50]:
        ts = [t * (1 - gain) for t in times]
        scores = [case_score(t, b) for t, b in zip(ts, baselines)]
        scenarios.append(dict(reference=ref, scenario=f"all_minus_{int(gain * 100)}pct",
                              sum_tk_ms=sum(ts), case_integer_sum=sum(scores),
                              display=round(sum(scores) / 12, 2), assumed_penalty=10,
                              board=round(sum(scores) / 12, 2) - 10,
                              **{f"c{i}": s for i, s in enumerate(scores, 1)}))
    for tag, fractions in [
        ("c9_c10_minus30pct", {9: .30, 10: .30}),
        ("c9_c10_minus30pct_c3_c4_c11_c12_minus15pct", {9: .30, 10: .30, 3: .15, 4: .15, 11: .15, 12: .15}),
    ]:
        ts = [t * (1 - fractions.get(i, 0)) for i, t in enumerate(times, 1)]
        scores = [case_score(t, b) for t, b in zip(ts, baselines)]
        scenarios.append(dict(reference=ref, scenario=tag, sum_tk_ms=sum(ts), case_integer_sum=sum(scores),
                              display=round(sum(scores) / 12, 2), assumed_penalty=10,
                              board=round(sum(scores) / 12, 2) - 10,
                              **{f"c{i}": s for i, s in enumerate(scores, 1)}))
write_csv("2026-09-27-score-scenarios.csv", scenarios)

observed = []
for sid in ANCHORS:
    values = [DB[sid][str(i)] for i in range(1, 13)]
    total = sum(case_score(v[0], v[1]) for v in values)
    observed.append(dict(sid=sid, sum_tk_ms=sum(v[0] for v in values),
                         derived_case_integer_sum=total, derived_display=round(total / 12, 2),
                         **{f"c{i}": case_score(v[0], v[1]) for i, v in enumerate(values, 1)}))
write_csv("2026-09-27-v842-observations.csv", observed)

details = {}
for path in (ROOT / "experiments/2026-09-08/results").glob("submissions_*_raw.json"):
    details.update(json.loads(path.read_text()).get("details", {}))
formula_checks = {"distinct_submissions": len(details), "accepted_case_comparisons": 0,
                  "case_mismatches": [], "full_display_comparisons": 0, "display_mismatches": [], "th_values": []}
th_values = set()
for sid, detail in details.items():
    progress = detail.get("progress", {})
    ss = []
    for tc in progress.get("testcaseResult", {}).values():
        ue = tc.get("userError") or ""
        ue = ue.get("content", "") if isinstance(ue, dict) else ue
        match = re.search(r'\{"schema_version"[^\n]*?\}', ue)
        if not match:
            continue
        x = json.loads(match.group())
        th_values.add(x["th_time_ms"])
        if not x.get("pass") or tc.get("status") != "Accepted":
            continue
        s = case_score(x["tk_time_ms"], x["tb_time_ms"])
        ss.append(s)
        formula_checks["accepted_case_comparisons"] += 1
        if s != tc.get("displayScore"):
            formula_checks["case_mismatches"].append([sid, s, tc.get("displayScore")])
    if len(ss) == 12:
        formula_checks["full_display_comparisons"] += 1
        if round(sum(ss) / 12, 2) != progress.get("displayScore"):
            formula_checks["display_mismatches"].append([sid, round(sum(ss) / 12, 2), progress.get("displayScore")])
formula_checks["th_values"] = sorted(th_values)

uniform_targets = []
for board in [75, 77, 80]:
    target = (board + 10) * 12
    own = [needed_uniform([DB[s][str(i)][0] for i in range(1, 13)],
                          [DB[s][str(i)][1] for i in range(1, 13)], target) for s in ANCHORS]
    uniform_targets.append(dict(board=board, required_integer_sum=target,
                                **{f"{name}_reduction_pct": 100 * needed_uniform(times, b, target) for name, b in refs.items()},
                                observed_own_vector_min_pct=100 * min(own),
                                observed_own_vector_median_pct=100 * st.median(own),
                                observed_own_vector_max_pct=100 * max(own)))
summary = dict(kernel_sha256=hashlib.sha256(kernel.encode()).hexdigest(), exact_v842_sids=ANCHORS,
               tb_reference_sids=ANCHORS + OLD_ANCHORS, formula_checks=formula_checks,
               representative_tk_sum_ms=sum(times), standardized_integer_sum=sum_score(times, refs["tb_median"]),
               uniform_targets=uniform_targets)
(OUT / "2026-09-27-score-audit-data.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
