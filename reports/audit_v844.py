"""Offline audit of v844: exact-source samples, score thresholds and cost models."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
EXPECTED = "d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e"


def csv_write(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score(t, b):
    return math.floor(100 * b / (b + t))


def main():
    src = (ROOT / "p1/kernel.py").read_bytes()
    assert hashlib.sha256(src).hexdigest() == EXPECTED, "Baseline changed; re-audit before using samples"
    tree = ast.parse(src)
    shapes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_KNOWN12" for t in n.targets))
    data = json.loads((OUT / "2026-09-27-v844-cases.json").read_text())["submissions"]
    exact = {s: d for s, d in data.items() if d.get("source_sha256") == EXPECTED}
    assert len(exact) >= 3
    rows = []
    for i, (T, H, E, I, k) in enumerate(shapes, 1):
        samples = [d["cases"][str(i)] for d in exact.values()]
        t = [s["metrics"]["tk_time_ms"] for s in samples]
        b = [s["metrics"]["tb_time_ms"] for s in samples]
        assert all(s["metrics"]["th_time_ms"] == 0 and s["status"] == "Accepted" for s in samples)
        assert min(t) > 0
        tm, bm = st.median(t), st.median(b)
        q = score(tm, bm)
        lim = bm * (100 / (q + 1) - 1)
        F, W = 6 * T * k * H * I, 3 * E * H * I
        rows.append(dict(case=i, T=T, H=H, E=E, I=I, topk=k,
                         samples=len(t), tk_median_ms=tm, tk_min_ms=min(t), tk_max_ms=max(t),
                         tb_median_ms=bm, tb_min_ms=min(b), tb_max_ms=max(b),
                         q_fixed_tb=q, next_q=q+1, next_tk_ms=lim,
                         reduction_for_next_q_pct=100*(1-lim/tm),
                         min_logged_sqnr=min(x for s in samples for x in s["sqnr_db"]),
                         expert_flops_T=F/1e12, effective_TFLOPS=F/(tm*1e9),
                         all_expert_fp8_GiB=W/2**30, token_fp8_MiB=T*H/2**20,
                         act_fp8_MiB=T*k*I/2**20, down_fp8_MiB=T*k*H/2**20,
                         mean_rows_per_expert=T*k/E))
    csv_write("2026-09-27-v844-thresholds.csv", rows)
    ts, bs = [r["tk_median_ms"] for r in rows], [r["tb_median_ms"] for r in rows]
    scenarios = []
    configs = [(f"all_minus_{p}pct", {i:p/100 for i in range(1,13)}) for p in [0,5,10,15,20,25,30,40,50]]
    configs += [(f"c9c10_minus_{p}pct", {9:p/100,10:p/100}) for p in [10,20,30,50]]
    configs += [("c11_minus_10pct", {11:.1}), ("c1c2_minus_15pct", {1:.15,2:.15}),
                ("c9c10_minus30_c1c2_minus15_c11c12_minus10", {9:.3,10:.3,1:.15,2:.15,11:.1,12:.1})]
    for name, changes in configs:
        q = [score(t*(1-changes.get(i,0)), b) for i,(t,b) in enumerate(zip(ts,bs),1)]
        scenarios.append(dict(scenario=name, integer_sum=sum(q), display=sum(q)/12,
                              after_assumed_10_penalty=sum(q)/12-10,
                              **{f"c{i}":v for i,v in enumerate(q,1)}))
    csv_write("2026-09-27-v844-scenarios.csv", scenarios)
    checks=[]
    for sid,d in data.items():
        for i,c in d.get("cases",{}).items():
            m=c["metrics"]
            if m["th_time_ms"]==0:
                checks.append((sid,i,score(m["tk_time_ms"],m["tb_time_ms"]),c["displayScore"]))
    mismatches=[x for x in checks if x[2]!=x[3]]
    targets=[]
    for target in [75,77,80]:
        lo,hi=0.,1.
        for _ in range(60):
            a=(lo+hi)/2
            if sum(score(t*(1-a),b) for t,b in zip(ts,bs)) >= (target+10)*12:
                hi=a
            else: lo=a
        targets.append(dict(board_target=target, required_uniform_reduction_pct=100*hi))
    defs={n.name:{"line":n.lineno,"end_line":n.end_lineno} for n in tree.body
          if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef))}
    summary=dict(kernel_sha256=EXPECTED, exact_source_sids=list(exact),
                 sample_scope="Three exact-source existing submissions, descriptive only; not a new A/B experiment",
                 sum_case_medians_ms=sum(ts), fixed_tb_integer_sum=sum(r["q_fixed_tb"] for r in rows),
                 fixed_tb_after_penalty=sum(r["q_fixed_tb"] for r in rows)/12-10,
                 formula_checks=len(checks), formula_mismatches=mismatches,
                 uniform_scenarios=targets, last_bound_definitions=defs)
    (OUT / "2026-09-27-v844-audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="last_bound_definitions"},ensure_ascii=False,indent=2))
    for r in rows:
        print(f"c{r['case']:2}: tk={r['tk_median_ms']:.3f}, tb={r['tb_median_ms']:.3f}, q={r['q_fixed_tb']}, next=-{r['reduction_for_next_q_pct']:.2f}%, F={r['expert_flops_T']:.3f}T, W={r['all_expert_fp8_GiB']:.3f}GiB")


if __name__ == "__main__":
    main()
