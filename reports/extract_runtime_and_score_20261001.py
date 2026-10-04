"""Offline extracts supporting the guidance; never operates the judge."""
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"


def main():
    data = json.loads((OUT / "2026-10-01-cohort-readonly.json").read_text())
    records = data["submissions"]
    warning_sids = []
    for sid, row in records.items():
        if any("Profiler clears events" in x["text"] for x in row.get("logs", [])):
            warning_sids.append(int(sid))
    examples = {}
    for sid in ("152644", "152641", "151885"):
        lines = []
        for log in records[sid].get("logs", []):
            for line in log["text"].splitlines():
                if any(x in line for x in ("profiler.py:", "Running kernel,", ".kernel_cache/kernel_", "_getitem_", "exitcode  : 255")):
                    if line not in lines:
                        lines.append(line)
        examples[sid] = lines
    facts = {
        "checked_at": data["checked_at"],
        "profiler_warning_submission_count": len(warning_sids),
        "profiler_warning_sids": sorted(warning_sids),
        "log_examples": examples,
        "limits": [
            "A profiler warning proves use on that execution path, not its use/configuration on every case.",
            "The warning alone does not prove a collection bug or explain zero tk.",
            "Transformed host tracebacks do not disclose the exact source seen by Triton JIT.",
            "The judge's CUDA, CUPTI, PyTorch versions and aggregation rules remain unverified.",
        ],
    }
    (OUT / "2026-10-01-runtime-facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n")

    ref_sids = ["152241", "152248", "152251"]
    shapes = json.loads((OUT / "2026-09-30-v926-score.json").read_text())["cases"]
    case_rows = []
    best = records["152238"]
    for shape in shapes:
        c = shape["case"]
        rs = [records[s]["cases"][str(c)] for s in ref_sids]
        tk = statistics.median(r["metrics"]["tk_time_ms"] for r in rs)
        tb = statistics.median(r["metrics"]["tb_time_ms"] for r in rs)
        q = math.floor(100 * tb / (tb + tk))
        m = best["cases"][str(c)]["metrics"]
        best_q = math.floor(100 * m["tb_time_ms"] / (m["tb_time_ms"] + m["tk_time_ms"]))
        case_rows.append({**{k: shape[k] for k in ("case", "T", "H", "E", "I", "k")},
                          "tk_ms": tk, "tb_ms": tb, "q": q,
                          "next_drop_pct": 100 * (1 - tb * (100 / (q + 1) - 1) / tk),
                          "sqnr_min_db": min(v for r in rs for v in r["sqnr_db"]),
                          "best_152238_q": best_q})
    def score(rows, factor=1):
        return sum(math.floor(100*r["tb_ms"]/(r["tb_ms"]+factor*r["tk_ms"])) for r in rows) / 12
    best_sum = sum(r["best_152238_q"] for r in case_rows)
    model = {
        "normal_anchor_sids": list(map(int, ref_sids)),
        "normal_anchor_definition": "Per-case medians; a synthetic comparison window, not a new measured submission.",
        "cases": case_rows,
        "normal_raw": score(case_rows),
        "uniform_faster_scenarios": {str(p): score(case_rows, 1-p/100) for p in (5, 10, 30, 50)},
        "best_sum_q": best_sum,
        "target_sum_q": 1080,
        "best_only_c8_to_zero": (best_sum + 100-case_rows[7]["best_152238_q"]) / 12,
        "best_c8_c9_to_zero": (best_sum + 200-sum(r["best_152238_q"] for r in case_rows[7:9])) / 12,
        "best_only_c7_to_zero": (best_sum + 100-case_rows[6]["best_152238_q"]) / 12,
        "normal_c8_to_c12_zero": (sum(r["q"] for r in case_rows[:7]) + 500) / 12,
        "normal_c7_to_c12_zero": (sum(r["q"] for r in case_rows[:6]) + 600) / 12,
        "limits": "Displayed ms are rounded. Scenarios hold tb fixed and are arithmetic, not predictions. Gains cannot be added across different submissions.",
    }
    (OUT / "2026-10-01-score-plan.json").write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"profiler_warning_count": len(warning_sids), "normal_raw": model["normal_raw"], "best_sum_q": best_sum}, indent=2))


if __name__ == "__main__":
    main()
