"""Read only: extend the existing own-submission evidence; never submit code.

Uses the existing authenticated platform client. Saves no authentication material
or submitted source, only result logs and source fingerprints. Old snapshots are
not overwritten.
"""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "reports")]
from xpuoj_web import API, _session
from collect_timing_anomalies_20260930 import parse_detail, TERMINAL
from refresh_strategy_evidence_20260930 import source_features


def main():
    session = _session()

    def read(endpoint, body):
        response = session.post(API + endpoint, json=body, timeout=25)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError("Platform read returned an API error")
        return data

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = {"checked_at": stamp, "read_only": True}
    entry = read("contest/play/getContestScoreboardMe", {"contestId": 13}).get("entry") or {}
    state["me"] = {k: entry.get(k) for k in ("rank", "totalScore", "problemScores")}
    data = read("contest/play/querySubmissions", {
        "contestId": 13, "problemOrder": 1, "takeCount": 10, "locale": "zh_CN"})
    state["recent"] = [{k: r.get(k) for k in ("id", "status", "displayScore", "submitTime", "timeUsed")}
                       for r in data.get("submissions", [])]
    for key, lang in (("distributed_custom", "triton-dist"), ("single_gpu_custom", "triton-h800")):
        data_av = read("judgeClient/checkAvailability", {"language": lang, "requiredFlags": ["custom-test"]})
        state[key] = {"available": data_av.get("available")}
    (ROOT / "reports/2026-10-01-platform-readonly.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(state, ensure_ascii=False), flush=True)

    out = ROOT / "reports/2026-10-01-cohort-readonly.json"
    prior_path = out if out.exists() else ROOT / "reports/2026-09-30-late-cohort-readonly.json"
    previous = json.loads(prior_path.read_text())
    result = dict(previous)
    result.update({"checked_at": stamp, "read_only": True, "extended_from": "2026-09-30-late-cohort-readonly.json"})
    submissions = result["submissions"]
    recent = {int(r["id"]): r for r in result["recent"]}
    lower = max(int(s) for s in submissions)
    new_rows = {}
    for sid, row in submissions.items():
        if row["status"] not in TERMINAL:
            new_rows[int(sid)] = recent.get(int(sid), {"id": int(sid), "submitTime": row.get("submitTime")})
    for _ in range(30):
        rows = data.get("submissions", [])
        if not rows:
            break
        for r in rows:
            if int(r["id"]) > lower:
                new_rows[int(r["id"])] = r
                recent[int(r["id"])] = {k: r.get(k) for k in ("id", "status", "displayScore", "submitTime", "timeUsed")}
            elif str(r["id"]) in submissions and submissions[str(r["id"])]["status"] not in TERMINAL:
                new_rows[int(r["id"])] = r
                recent[int(r["id"])] = {k: r.get(k) for k in ("id", "status", "displayScore", "submitTime", "timeUsed")}
        if min(int(r["id"]) for r in rows) <= lower or not data.get("hasSmallerId"):
            break
        data = read("contest/play/querySubmissions", {
            "contestId": 13, "problemOrder": 1, "takeCount": 10,
            "maxId": min(int(r["id"]) for r in rows) - 1, "locale": "zh_CN"})
    result["recent"] = sorted(recent.values(), key=lambda r: r["id"], reverse=True)
    for i, item in enumerate(sorted(new_rows.values(), key=lambda r: r["id"])):
        sid = str(item["id"])
        detail = read("submission/getSubmissionDetail", {"submissionId": sid, "locale": "zh_CN"})
        row = parse_detail(detail)
        row.update(source_features((detail.get("content") or {}).get("code")))
        row["submitTime"] = item.get("submitTime") or row.get("submitTime")
        row["listed_timeUsed"] = item.get("timeUsed")
        submissions[sid] = row
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"{sid}: {row['status']} raw={row['displayScore']}; cases={len(row['cases'])}; {i+1}/{len(new_rows)}", flush=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cohort_size": len(submissions), "fresh_reads": len(new_rows), "path": str(out)}), flush=True)


if __name__ == "__main__":
    main()
