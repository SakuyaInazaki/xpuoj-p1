"""Reproducible, offline source/result/cost audit for the September 30 guide."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "436f0a227678eb9caa255627837727f066884bbeb4bd4a0a829d08155353c8fc"
ANCHORS = (151103, 151105, 151107, 151166, 151173, 151380)


def score(t, b):
    return math.floor(100 * b / (b + t) + 1e-12)


def csv_write(name, rows):
    with (ROOT / "reports" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = (ROOT / "p1/kernel.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == EXPECTED, "Baseline changed; re-audit."
    tree = ast.parse(source)
    shapes = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "_KNOWN12" for t in n.targets))
    data = json.loads((ROOT / "reports/2026-09-30-submission-audit.json").read_text())["submissions"]
    anchors = [data[str(sid)] for sid in ANCHORS]
    assert all(d["source_sha256"] == EXPECTED and d["status"] == "Accepted"
               and len(d["cases"]) == 12 for d in anchors)
    rows = []
    for i, (T, H, E, I, k) in enumerate(shapes, 1):
        cases = [d["cases"][str(i)] for d in anchors]
        ts = [c["metrics"]["tk_time_ms"] for c in cases]
        bs = [c["metrics"]["tb_time_ms"] for c in cases]
        t, b = st.median(ts), st.median(bs)
        q = score(t, b)
        limit = b * (100 / (q + 1) - 1)
        rows.append(dict(case=i, T=T, H=H, E=E, I=I, topk=k,
                         tk_median_ms=t, tk_min_ms=min(ts), tk_max_ms=max(ts),
                         tb_median_ms=b, q_fixed_tb=q, next_q=q+1,
                         next_tk_ms=limit, next_reduction_pct=100*(1-limit/t),
                         min_logged_sqnr=min(x for c in cases for x in c["sqnr_db"]),
                         useful_TFLOPS=6*T*k*H*I/(t*1e9),
                         fp8_weight_GiB=3*E*H*I/2**30, mean_rows=T*k/E,
                         down_scratch_MiB=T*k*H/2**20,
                         fp8_scratch_read_write_plus_output_MiB=(2*T*k*H+2*T*H)/2**20,
                         continuous_raw_sensitivity_per_ms=100*b/(b+t)**2/12))
    csv_write("2026-09-30-v890-thresholds.csv", rows)
    configs = [(f"all_minus_{p}pct", {i:p/100 for i in range(1,13)}) for p in (0,5,10,20,30,40,50,55)]
    configs += [("c11c12_minus5", {11:.05,12:.05}),
                ("c11c12_minus10", {11:.10,12:.10}),
                ("c9c10_minus30", {9:.30,10:.30}),
                ("c9c10_minus50", {9:.50,10:.50}),
                ("c11c12_minus10_others_minus5", {i:(.10 if i in (11,12) else .05) for i in range(1,13)}),
                ("c1c2c9c10_zero_limit", {1:1.,2:1.,9:1.,10:1.})]
    scenarios = []
    for name, changes in configs:
        qs = [score(r["tk_median_ms"]*(1-changes.get(r["case"],0)),r["tb_median_ms"]) for r in rows]
        scenarios.append(dict(scenario=name, integer_sum=sum(qs), raw=sum(qs)/12,
                              net_model=sum(qs)/12-10, **{f"c{i}":q for i,q in enumerate(qs,1)}))
    csv_write("2026-09-30-score-scenarios.csv", scenarios)
    evidence_rows = []
    for sid, detail in sorted(data.items(), key=lambda item:int(item[0])):
        evidence_rows.append(dict(sid=int(sid),status=detail.get("status"),
            raw=detail.get("displayScore"),sha256=detail.get("source_sha256"),
            local_files=";".join(detail.get("matching_local_files",[])),
            **{f"c{i}_tk_ms":detail.get("cases",{}).get(str(i),{}).get("metrics",{}).get("tk_time_ms")
               for i in range(1,13)}))
    csv_write("2026-09-30-candidate-ledger.csv", evidence_rows)
    checks, mismatches = 0, []
    for sid, detail in data.items():
        for i, c in detail.get("cases",{}).items():
            m = c["metrics"]
            if c["status"] == "Accepted" and m["tk_time_ms"]>0 and m["th_time_ms"]==0:
                checks += 1
                if score(m["tk_time_ms"],m["tb_time_ms"]) != c["displayScore"]:
                    mismatches.append((sid,i))
    groups = {}
    for sid, detail in data.items():
        sha = detail.get("source_sha256")
        if not sha:
            continue
        group = groups.setdefault(sha, {"files": detail.get("matching_local_files",[]),
                                       "sids": [], "accepted": []})
        group["sids"].append(sid)
        if detail["status"] == "Accepted":
            group["accepted"].append(sid)
    pool = {}
    for name, sha_prefix in (("v890","436f0a"),("v926","2633cc")):
        chosen = [d for d in data.values() if d.get("source_sha256","").startswith(sha_prefix)
                  and d.get("status") == "Accepted"]
        ts = [d["cases"]["10"]["metrics"]["tk_time_ms"] for d in chosen]
        pool[name] = {"n":len(ts), "c10_mean_ms":st.mean(ts), "c10_median_ms":st.median(ts),
                      "c10_range_ms":[min(ts),max(ts)],
                      "scope":"Historical unpaired pool; descriptive, not a causal A/B estimate."}
    lo, hi = 0., 1.
    for _ in range(60):
        a = (lo+hi)/2
        if sum(score(r["tk_median_ms"]*(1-a),r["tb_median_ms"]) for r in rows) >= 1080:
            hi = a
        else:
            lo = a
    last_defs = {n.name:{"line":n.lineno,"end_line":n.end_lineno}
                 for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    summary = {"baseline_sha256":EXPECTED,"chosen_anchor_sids":ANCHORS,
               "scope":"Medians of six exact-source historical controls, not new A/B testing.",
               "case_median_sum_ms":sum(r["tk_median_ms"] for r in rows),
               "fixed_tb_raw":scenarios[0]["raw"], "uniform_reduction_for_raw90_pct":hi*100,
               "formula_checks":checks,"formula_mismatches":mismatches,
               "existing_submission_count":len(data),"source_groups":groups,
               "c10_historical_pools":pool,"last_definitions":last_defs}
    (ROOT / "reports/2026-09-30-audit.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_groups","last_definitions")},ensure_ascii=False,indent=2))
    for r in rows:
        print(f"c{r['case']:2} tk={r['tk_median_ms']:.4f} tb={r['tb_median_ms']:.4f} "
              f"q={r['q_fixed_tb']} next=-{r['next_reduction_pct']:.2f}% SQNR={r['min_logged_sqnr']:.2f}")


if __name__ == "__main__":
    main()
