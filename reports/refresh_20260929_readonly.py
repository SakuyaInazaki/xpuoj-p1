"""Read the account's existing P1 records; no submission or token consumption."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import API, _session

OUT = ROOT / "reports"
SIDS = [149674, 149676, 149680, 149684, 149695, 149707, 149710,
        149740, 149745, 149747, 149752, 149768, 149772, 149773,
        149784, 149911, 149925, 149930, 149932, 149935, 149943,
        149947, 150008, 150917, 150920, 150923, 150926, 150928,
        150934, 150936, 150941, 150942, 150943, 150945]


def main():
    session = _session()

    def read(path, body):
        try:
            r = session.post(API + path, json=body, timeout=25)
            if r.status_code not in (200, 201):
                return {"read_error": "HTTP", "http_status": r.status_code}
            return r.json()
        except Exception as exc:
            return {"read_error": type(exc).__name__}

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = {"checked_at": stamp, "read_only": True}
    d = read("contest/play/getContestScoreboardMe", {"contestId": 13})
    e = d.get("entry", {})
    state["me"] = {k: e.get(k) for k in ("rank", "totalScore", "problemScores")} if e else d
    d = read("contest/play/querySubmissions", {
        "contestId": 13, "problemOrder": 1, "takeCount": 70, "locale": "zh_CN"})
    state["recent"] = [{k: v.get(k) for k in
                        ("id", "status", "displayScore", "submitTime", "timeUsed")}
                       for v in d.get("submissions", [])]
    if "read_error" in d:
        state["recent_read_error"] = d
    state["distributed_custom"] = read("judgeClient/checkAvailability", {
        "language": "triton-dist", "requiredFlags": ["custom-test"]})
    (OUT / "2026-09-29-platform-readonly.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"me": state["me"], "latest": state["recent"][:4],
                      "distributed_custom": state["distributed_custom"]}, ensure_ascii=False), flush=True)
    if not state["recent"]:
        return
    details = {"checked_at": stamp, "read_only": True, "submissions": {}}
    wanted = list(dict.fromkeys(SIDS + [x["id"] for x in state["recent"][:12]]))
    for sid in wanted:
        d = read("submission/getSubmissionDetail", {"submissionId": str(sid), "locale": "zh_CN"})
        if "read_error" in d:
            details["submissions"][str(sid)] = d
            continue
        p = d.get("progress") or {}
        source = (d.get("content") or {}).get("code")
        row = {"status": p.get("status"), "displayScore": p.get("displayScore"),
               "source_sha256": hashlib.sha256(source.encode()).hexdigest() if isinstance(source, str) else None,
               "submitTime": (d.get("meta") or {}).get("submitTime"), "cases": {}, "logs": []}
        for tc_key, tc in (p.get("testcaseResult") or {}).items():
            raw = tc.get("userError") or ""
            if isinstance(raw, dict):
                raw = raw.get("content") or raw.get("data") or ""
            raw = str(raw)
            row["logs"].append({"testcase_key": tc_key, "status": tc.get("status"), "text": raw})
            m = re.search(r"tc=(\d+)", raw)
            metric = re.search(r'\{"schema_version"[^\n]*?\}', raw)
            if not m or not metric:
                continue
            row["cases"][m.group(1)] = {
                "status": tc.get("status"), "displayScore": tc.get("displayScore"),
                "metrics": json.loads(metric.group()),
                "sqnr_db": [float(x) for x in re.findall(r"SQNR=([0-9.]+) dB", raw)],
                "determinism_lines": [s for s in raw.splitlines() if "DETERMINISM" in s]}
        details["submissions"][str(sid)] = row
        (OUT / "2026-09-29-submission-audit.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2) + "\n")
        print(f"SID {sid}: {row['status']}; {len(row['cases'])} cases; SHA {(row['source_sha256'] or '')[:12]}", flush=True)


if __name__ == "__main__":
    main()
