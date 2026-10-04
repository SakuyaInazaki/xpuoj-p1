"""Reproduce the 2026-09-29 source/score/workload audit, without GPU or network."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc"
RECENT = [150917, 150926, 150934, 150942, 150943]


def score(t, b):
    return math.floor(100 * b / (b + t) + 1e-12)


def write_csv(name, rows):
    with (ROOT / "reports" / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    source = (ROOT / "p1/kernel.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == EXPECTED, "Production changed; re-audit."
    tree = ast.parse(source)
    shapes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_KNOWN12" for t in n.targets))
    data = json.loads((ROOT / "reports/2026-09-29-submission-audit.json").read_text())["submissions"]
    exact = {sid: d for sid, d in data.items() if d.get("source_sha256") == EXPECTED}
    chosen = {str(sid): data[str(sid)] for sid in RECENT}
    assert all(d["source_sha256"] == EXPECTED and len(d["cases"]) == 12 for d in chosen.values())
    rows = []
    for i, (T, H, E, I, k) in enumerate(shapes, 1):
        samples = [d["cases"][str(i)] for d in chosen.values()]
        ts = [d["metrics"]["tk_time_ms"] for d in samples]
        bs = [d["metrics"]["tb_time_ms"] for d in samples]
        assert min(ts) > 0
        assert all(d["metrics"]["th_time_ms"] == 0 and d["status"] == "Accepted" for d in samples)
        t, b = st.median(ts), st.median(bs)
        q = score(t, b)
        lim = b * (100 / (q + 1) - 1)
        t90 = b / 9
        F, W = 6 * T * k * H * I, 3 * E * H * I
        rows.append(dict(case=i, T=T, H=H, E=E, I=I, topk=k, samples=len(samples),
                         tk_median_ms=t, tk_min_ms=min(ts), tk_max_ms=max(ts),
                         tb_median_ms=b, tb_min_ms=min(bs), tb_max_ms=max(bs),
                         q_fixed_tb=q, next_q=q+1, next_tk_ms=lim,
                         reduction_for_next_q_pct=100*(1-lim/t),
                         tk_for_case90_ms=t90, reduction_for_case90_pct=100*(1-t90/t),
                         min_logged_sqnr=min(x for d in samples for x in d["sqnr_db"]),
                         useful_flops_T=F/1e12, effective_TFLOPS=F/(t*1e9),
                         all_fp8_weights_GiB=W/2**30, mean_rows_per_expert=T*k/E,
                         token_fp8_MiB=T*H/2**20, act_fp8_MiB=T*k*I/2**20,
                         down_fp8_MiB=T*k*H/2**20,
                         tfps_needed_if_all_cases90=F/(t90*1e9)))
    write_csv("2026-09-29-v890-thresholds.csv", rows)
    scenarios = []
    configs = [(f"all_minus_{p}pct", {i: p/100 for i in range(1, 13)}) for p in (0, 10, 20, 30, 40, 50, 55)]
    configs += [(f"c9c10_minus_{p}pct", {9: p/100, 10: p/100}) for p in (20, 30, 50)]
    configs += [("c9c10_zero_ideal", {9: 1., 10: 1.}),
                ("c1c2c9c10_zero_ideal", {1: 1., 2: 1., 9: 1., 10: 1.}),
                ("c9c10_minus30_rest_minus10", {i: .3 if i in (9, 10) else .1 for i in range(1, 13)}),
                ("c1c2_unchanged_rest_minus50", {i: .5 for i in range(3, 13)})]
    for name, changes in configs:
        qs = [score(r["tk_median_ms"]*(1-changes.get(r["case"], 0)), r["tb_median_ms"]) for r in rows]
        scenarios.append(dict(scenario=name, integer_sum=sum(qs), raw=sum(qs)/12,
                              net_assuming_10_penalty=sum(qs)/12-10,
                              **{f"c{i}": q for i, q in enumerate(qs, 1)}))
    write_csv("2026-09-29-v890-scenarios.csv", scenarios)
    targets = []
    for raw_target in (85, 87, 90):
        lo, hi = 0., 1.
        for _ in range(60):
            a = (lo+hi)/2
            total = sum(score(r["tk_median_ms"]*(1-a), r["tb_median_ms"]) for r in rows)
            if total >= raw_target*12:
                hi = a
            else:
                lo = a
        targets.append({"raw_target": raw_target, "uniform_reduction_pct": hi*100})
    checks, mismatches = 0, []
    for sid, d in data.items():
        for i, c in d.get("cases", {}).items():
            m = c["metrics"]
            if c["status"] == "Accepted" and m["th_time_ms"] == 0 and m["tk_time_ms"] > 0:
                checks += 1
                if score(m["tk_time_ms"], m["tb_time_ms"]) != c["displayScore"]:
                    mismatches.append([sid, i])
    defs = {n.name: {"line": n.lineno, "end_line": n.end_lineno} for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    summary = dict(kernel_sha256=EXPECTED, exact_source_sids=list(exact), chosen_recent_sids=RECENT,
                   scope="Descriptive medians of five existing exact-source controls; not an independent candidate A/B.",
                   sum_case_medians_ms=sum(r["tk_median_ms"] for r in rows),
                   fixed_tb_raw=sum(r["q_fixed_tb"] for r in rows)/12,
                   fixed_tb_net=sum(r["q_fixed_tb"] for r in rows)/12-10,
                   formula_checks=checks, formula_mismatches=mismatches,
                   target_models=targets, last_bound_definitions=defs)
    (ROOT / "reports/2026-09-29-v890-audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "last_bound_definitions"}, ensure_ascii=False, indent=2))
    for r in rows:
        print(f"c{r['case']:2}: tk={r['tk_median_ms']:.3f} tb={r['tb_median_ms']:.3f} q={r['q_fixed_tb']} next=-{r['reduction_for_next_q_pct']:.2f}% case90=-{r['reduction_for_case90_pct']:.2f}%")


if __name__ == "__main__":
    main()
