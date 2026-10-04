"""Read-only reconstruction of a contiguous P1 submission cohort.

Uses existing login; never submits, consumes captcha, or prints credentials.
Only list/detail read endpoints are called. Raw candidate code is hashed then
discarded; sanitized metadata, testcase logs and metrics are preserved.
"""

import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import API, _session

OUT = ROOT / "reports/2026-09-30-timing-cohort-readonly.json"
LOWER_SID = 149480
TERMINAL = {"Accepted", "WrongAnswer", "TimeLimitExceeded", "Canceled", "CompilationError", "RuntimeError"}


def parse_detail(data):
    progress, meta = data.get("progress") or {}, data.get("meta") or {}
    code = (data.get("content") or {}).get("code")
    row = {
        "status": progress.get("status") or meta.get("status"),
        "displayScore": progress.get("displayScore", meta.get("displayScore")),
        "submitTime": meta.get("submitTime"),
        "source_sha256": hashlib.sha256(code.encode()).hexdigest() if isinstance(code, str) else None,
        "cases": {}, "logs": [],
    }
    for key, tc in (progress.get("testcaseResult") or {}).items():
        raw = tc.get("userError") or ""
        omitted = 0
        if isinstance(raw, dict):
            omitted = raw.get("omittedLength", 0)
            raw = raw.get("content") or raw.get("data") or ""
        raw = str(raw)
        case = re.search(r"tc=(\d+)", raw)
        match = re.search(r'\{"schema_version"[^\n]*?\}', raw)
        row["logs"].append({"key": key, "status": tc.get("status"), "text": raw, "omittedLength": omitted})
        if case and match:
            row["cases"][case.group(1)] = {
                "metrics": json.loads(match.group()), "status": tc.get("status"),
                "displayScore": tc.get("displayScore"),
                "sqnr_db": [float(v) for v in re.findall(r"SQNR=([0-9.]+) dB", raw)],
                "determinism_checks_passed": raw.count("[DETERMINISM OK]"),
            }
    return row


def main():
    cached = {}
    for path in sorted((ROOT / "reports").glob("*cases.json")) + sorted((ROOT / "reports").glob("*submission-audit.json")):
        for sid, row in json.loads(path.read_text()).get("submissions", {}).items():
            if row.get("source_sha256") and row.get("status") in TERMINAL:
                cached[sid] = row
    if OUT.exists():
        cached.update(json.loads(OUT.read_text()).get("submissions", {}))
    session = _session()

    def read(endpoint, body):
        response = session.post(API + endpoint, json=body, timeout=25)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError("API read failed")
        return data

    result = {"checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "read_only": True, "lower_sid_inclusive": LOWER_SID,
              "list_complete_to_lower_bound": False, "recent": [], "submissions": {}}
    seen = set()
    for page_number in range(24):
        body = {"contestId": 13, "problemOrder": 1, "takeCount": 10, "locale": "zh_CN"}
        if seen:
            body["maxId"] = min(seen) - 1
        data = read("contest/play/querySubmissions", body)
        page = [r for r in data.get("submissions", []) if int(r["id"]) not in seen]
        if not page:
            break
        seen.update(int(r["id"]) for r in page)
        result["recent"].extend({k: r.get(k) for k in ("id", "status", "displayScore", "submitTime", "timeUsed")}
                                for r in page if int(r["id"]) >= LOWER_SID)
        if min(seen) < LOWER_SID or not data.get("hasSmallerId"):
            result["list_complete_to_lower_bound"] = True
            break
        if (page_number + 1) % 5 == 0:
            print(f"Read {len(result['recent'])} listing entries", flush=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    fetched = 0
    for item in sorted(result["recent"], key=lambda r: r["id"]):
        sid = str(item["id"])
        row = cached.get(sid)
        if (not row or row.get("status") not in TERMINAL or "logs" not in row
                or row.get("status") != item.get("status")
                or row.get("displayScore") != item.get("displayScore")):
            row = parse_detail(read("submission/getSubmissionDetail", {"submissionId": sid, "locale": "zh_CN"}))
            fetched += 1
        row = dict(row)
        row["submitTime"] = item.get("submitTime") or row.get("submitTime")
        row["status"] = row.get("status") or item.get("status")
        result["submissions"][sid] = row
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if fetched and fetched % 10 == 0:
            print(f"Recovered {len(result['submissions'])}/{len(result['recent'])}; fresh details {fetched}", flush=True)
    print(json.dumps({"cohort_size": len(result["submissions"]), "fresh_detail_reads": fetched,
                      "complete": result["list_complete_to_lower_bound"]}), flush=True)


if __name__ == "__main__":
    main()
