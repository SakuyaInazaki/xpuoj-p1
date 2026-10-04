"""Read-only OJ evidence refresh; never submits, takes a token, or prints secrets."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import API, _session

SIDS = [149493, 149522, 149524, 149600, 149601, 149603, 149604,
        149606, 149609, 149612, 149614, 149630, 149636, 149639,
        149643, 149647, 149653, 149659, 149665]


def main():
    session = _session()
    result = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "read_only": True}

    def read(path, body):
        try:
            r = session.post(API + path, json=body, timeout=25)
            if r.status_code not in (200, 201):
                return {"read_error": "HTTP", "http_status": r.status_code}
            return r.json()
        except Exception as exc:
            return {"read_error": type(exc).__name__}

    d = read("contest/play/getContestScoreboardMe", {"contestId": 13})
    e = d.get("entry", {})
    result["me"] = ({k: e.get(k) for k in ("rank", "totalScore", "problemScores")}
                    if e else d)
    d = read("contest/play/querySubmissions", {
        "contestId": 13, "problemOrder": 1, "takeCount": 60, "locale": "zh_CN"})
    result["recent"] = [{k: v.get(k) for k in
                         ("id", "status", "displayScore", "submitTime", "timeUsed")}
                        for v in d.get("submissions", [])]
    if "read_error" in d:
        result["recent_read_error"] = d
    d = read("judgeClient/checkAvailability", {
        "language": "triton-dist", "requiredFlags": ["custom-test"]})
    result["distributed_custom"] = {k: d.get(k) for k in
                                    ("available", "reason", "read_error", "http_status")}
    out = ROOT / "reports/2026-09-27-v844-platform-readonly.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"me": result["me"], "latest": result["recent"][:3],
                      "distributed_custom": result["distributed_custom"]}, ensure_ascii=False), flush=True)
    details = {"checked_at": result["checked_at"], "submissions": {}}
    for sid in SIDS:
        d = read("submission/getSubmissionDetail", {"submissionId": str(sid), "locale": "zh_CN"})
        if "read_error" in d:
            details["submissions"][str(sid)] = d
            continue
        p = d.get("progress") or {}
        source = (d.get("content") or {}).get("code")
        row = {"status": p.get("status"), "displayScore": p.get("displayScore"),
               "source_sha256": hashlib.sha256(source.encode()).hexdigest() if isinstance(source, str) else None,
               "submitTime": (d.get("meta") or {}).get("submitTime"), "cases": {}}
        for tc in (p.get("testcaseResult") or {}).values():
            raw = tc.get("userError") or ""
            if isinstance(raw, dict):
                raw = raw.get("content") or raw.get("data") or ""
            m = re.search(r"tc=(\d+)", raw)
            metric = re.search(r'\{"schema_version"[^\n]*?\}', raw)
            if not m or not metric:
                continue
            row["cases"][m.group(1)] = {
                "status": tc.get("status"), "displayScore": tc.get("displayScore"),
                "metrics": json.loads(metric.group()),
                "sqnr_db": [float(x) for x in re.findall(r"SQNR=([0-9.]+) dB", raw)],
                "determinism_lines": [s for s in raw.splitlines() if "DETERMINISM" in s],
            }
        details["submissions"][str(sid)] = row
        print(f"SID {sid}: {row['status']}; {len(row['cases'])} case records", flush=True)
    (ROOT / "reports/2026-09-27-v844-cases.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
