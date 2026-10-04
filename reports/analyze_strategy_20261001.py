"""Reproduce October 1 guidance evidence from local records; no network/GPU."""
import ast
import collections
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"


def fingerprint(path):
    raw = path.read_bytes()
    tree = ast.parse(raw.decode())
    functions, occurrences = {}, collections.defaultdict(list)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        occurrences[node.name].append(node.lineno)
        functions[node.name] = node
    nodes = {k: ast.dump(v, include_attributes=False) for k, v in functions.items()}
    other = [ast.dump(n, include_attributes=False) for n in tree.body if not isinstance(n, ast.FunctionDef)]
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return {
        "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
        "effective_function_ast_sha256": digest(nodes), "other_top_level_ast_sha256": digest(other),
        "duplicate_definitions": {k: v for k, v in occurrences.items() if len(v) > 1},
        "functions": {k: {
            "line": v.lineno,
            "start_line": min([v.lineno] + [d.lineno for d in v.decorator_list]),
            "ast_sha256": digest(nodes[k]),
            "jit": any("jit" in ast.unparse(d) for d in v.decorator_list),
            "annotations": {a.arg: ast.unparse(a.annotation) if a.annotation else None for a in v.args.args},
            "referenced_functions": sorted({n.id for n in ast.walk(v) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in functions}),
        } for k, v in functions.items()},
        "limits": "Last Python definitions and AST only. Not a sandbox-transformed source, runtime dispatch proof, compiler cache key, or generated machine code.",
    }


def main():
    source_data = json.loads((OUT / "2026-10-01-cohort-readonly.json").read_text())
    records = source_data["submissions"]
    baselines = json.loads((OUT / "2026-09-30-v926-score.json").read_text())["cases"]
    ref = {str(r["case"]): r["tk_ms"] for r in baselines}
    paths = [ROOT / "p1/kernel.py"]
    paths += sorted((ROOT / "experiments/2026-09-30/candidates").rglob("*.py"))
    paths += sorted((ROOT / "experiments/2026-10-01/candidates").glob("*.py"))
    paths += sorted((ROOT / "experiments/2026-10-01/guide-candidates").glob("*.py"))
    local = collections.defaultdict(list)
    audits = {}
    for path in paths:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        local[sha].append(str(path.relative_to(ROOT)))
        if "2026-10-01" in str(path) or path.name == "kernel.py":
            audits[path.name] = fingerprint(path)
    groups = {g: collections.Counter() for g in ("first_in_cohort", "repeat_in_cohort", "source_unavailable")}
    statuses, seen, ledger, details = collections.Counter(), set(), [], []
    score_errors, displayed_sum_errors, strict_log_errors = [], [], []
    for sid, row in sorted(records.items(), key=lambda item: int(item[0])):
        sha = row.get("source_sha256")
        group = "source_unavailable" if not sha else "repeat_in_cohort" if sha in seen else "first_in_cohort"
        if sha:
            seen.add(sha)
        statuses[row["status"]] += 1
        complete = row["status"] == "Accepted" and set(row["cases"]) == {str(c) for c in range(1, 13)}
        complete &= all(c["status"] == "Accepted" and c["metrics"].get("pass") is True for c in row["cases"].values())
        lows, zeros = [], []
        for c, result in row["cases"].items():
            m = result["metrics"]
            if result["status"] == "Accepted" and m.get("pass") is True:
                if m["tk_time_ms"] < 0.5 * ref[c]:
                    lows.append(int(c))
                if m["tk_time_ms"] == 0:
                    zeros.append(int(c))
        if complete:
            total = sum(math.floor(100*d["metrics"]["tb_time_ms"] / (d["metrics"]["tb_time_ms"]+d["metrics"]["tk_time_ms"])) for d in row["cases"].values())
            if abs(total / 12 - float(row["displayScore"])) > 0.00501:
                score_errors.append(int(sid))
            tk_sum = sum(d["metrics"]["tk_time_ms"] for d in row["cases"].values())
            listed = row.get("listed_timeUsed")
            if listed is not None and abs(tk_sum*1000-listed) > 1.01:
                displayed_sum_errors.append({"sid": int(sid), "listed": listed, "sum_tk_us": round(tk_sum*1000, 3)})
            for c, case in row["cases"].items():
                case_logs = [l["text"] for l in row.get("logs", []) if re.search(r"tc=" + re.escape(c) + r"\b", l["text"])]
                determinants = sum(t.count("[DETERMINISM OK]") for t in case_logs)
                if determinants < 2 or len(case.get("sqnr_db", [])) < 2 or min(case.get("sqnr_db", [0])) < 22:
                    strict_log_errors.append({"sid": int(sid), "case": int(c)})
        for k, val in (("complete_ac", complete), ("low_ac", complete and bool(lows)), ("exact_zero_ac", complete and bool(zeros)), ("tle", row["status"] == "TimeLimitExceeded")):
            groups[group][k] += bool(val)
        item = {"sid": int(sid), "status": row["status"], "raw": row["displayScore"], "source_sha256": sha,
                "source_group": group, "complete_ac": complete, "low_cases": sorted(lows), "exact_zero_cases": sorted(zeros),
                "paths": local.get(sha, []), "submitTime": row.get("submitTime")}
        ledger.append(item)
        if complete and lows:
            details.append({**item, "tk_ms": [row["cases"][str(c)]["metrics"]["tk_time_ms"] for c in range(1,13)]})
    pairs = []
    for left, right in [
        ("p1_r1_all_pad_split_v2.py", "p1_r1_all_pad_split_mid_v3.py"),
        ("p1_r1_all_pad_split_mid_v3.py", "p1_r1_all_pad_split_mid_v4.py"),
        ("p1_r1_all_pad_split_mid_v3.py", "p1_r1_all_pad_split_mid_v5.py"),
        ("p1_r1_all_pad_split_mid_v6_pretanh.py", "p1_r1_all_pad_split_mid_v8_pretanh_shift.py"),
        ("p1_r1_all_pad_split_mid_v3.py", "p1_r1_all_pad_split_mid_v17_tiledshift.py"),
        ("p1_r1_all_pad_split_mid_v17_tiledshift.py", "p1_r1_all_pad_split_mid_v18_tiledshift_pretanh.py"),
    ]:
        a,b = audits[left],audits[right]
        pairs.append({"a": left, "b": right,
                      "same_effective_function_asts": a["effective_function_ast_sha256"] == b["effective_function_ast_sha256"],
                      "same_other_top_level_asts": a["other_top_level_ast_sha256"] == b["other_top_level_ast_sha256"]})
    result = {"checked_at": source_data["checked_at"], "cohort_size": len(records),
              "statuses": statuses, "groups": groups, "full_ac_low_events": details,
              "new_since_152586": [x for x in ledger if x["sid"] > 152586],
              "score_formula_mismatches": score_errors, "timeUsed_vs_sum_tk_mismatches": displayed_sum_errors,
              "full_ac_missing_two_sqnr_or_determinism_checks": strict_log_errors,
              "function_map_comparisons": pairs,
              "limits": ["Cohort first is not lifetime first or proven cold compile.",
                         "Low threshold retained at 50% of historical v926; exact zeros counted separately.",
                         "Nonrandom, dependent observations cannot estimate an intervention success probability.",
                         "Local candidate filenames may have been overwritten; platform SHA is authoritative."]}
    (OUT / "2026-10-01-strategy-analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    (OUT / "2026-10-01-source-audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2)+"\n")
    with (OUT / "2026-10-01-ledger.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(ledger[0])); w.writeheader(); w.writerows(ledger)
    print(json.dumps({k: result[k] for k in ("cohort_size", "statuses", "groups", "score_formula_mismatches", "timeUsed_vs_sum_tk_mismatches", "full_ac_missing_two_sqnr_or_determinism_checks", "function_map_comparisons")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
