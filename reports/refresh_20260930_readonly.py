"""Read existing P1 results. Never submits, consumes captcha, or prints credentials."""
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


def main():
    session = _session()

    def read(path, body):
        try:
            response = session.post(API + path, json=body, timeout=25)
            if response.status_code not in (200, 201):
                return {"read_error": "HTTP", "http_status": response.status_code}
            data = response.json()
            if "error" in data:
                return {"read_error": "API", "error_type": str(data["error"])[:120]}
            return data
        except Exception as exc:
            return {"read_error": type(exc).__name__}

    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = {"checked_at": stamp, "read_only": True}
    data = read("contest/play/getContestScoreboardMe", {"contestId": 13})
    entry = data.get("entry", {})
    state["me"] = {k: entry.get(k) for k in ("rank", "totalScore", "problemScores")} if entry else data
    data = read("contest/play/querySubmissions", {
        "contestId": 13, "problemOrder": 1, "takeCount": 70, "locale": "zh_CN"})
    state["recent"] = [{k: item.get(k) for k in
                        ("id", "status", "displayScore", "submitTime", "timeUsed")}
                       for item in data.get("submissions", [])]
    if "read_error" in data:
        state["recent_read_error"] = data
    # The service caps pages at ten. maxId is documented in the historical
    # client notes; a large takeCount alone does not recover older results.
    seen = {int(row["id"]) for row in state["recent"]}
    for _ in range(6):
        if not seen or min(seen) <= 150945 or not data.get("hasSmallerId"):
            break
        data = read("contest/play/querySubmissions", {
            "contestId": 13, "problemOrder": 1, "takeCount": 10,
            "maxId": min(seen) - 1, "locale": "zh_CN"})
        page = [row for row in data.get("submissions", []) if int(row["id"]) not in seen]
        if not page:
            break
        seen.update(int(row["id"]) for row in page)
        state["recent"].extend({k: item.get(k) for k in
                               ("id", "status", "displayScore", "submitTime", "timeUsed")}
                              for item in page)
    state["distributed_custom"] = read("judgeClient/checkAvailability", {
        "language": "triton-dist", "requiredFlags": ["custom-test"]})
    state["single_gpu_custom"] = read("judgeClient/checkAvailability", {
        "language": "triton-h800", "requiredFlags": ["custom-test"]})
    (OUT / "2026-09-30-platform-readonly.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in state.items() if k != "recent"}, ensure_ascii=False), flush=True)
    print(f"Recovered {len(state['recent'])} listed submissions by maxId pagination.", flush=True)
    if not state["recent"]:
        return

    # Recover terminal states from the last handoff and later batches, including
    # those whose local poller stopped before the server did. Read-only only.
    wanted = {149493, 150961, 150964, 150965, 150966, 150967, 150969,
              150973, 150976, 151051, 151056}
    for path in (ROOT / "experiments/2026-09-29").glob("*.log"):
        text = path.read_text(errors="replace")
        wanted.update(int(sid) for sid in re.findall(r"\bSID\s*[=:]\s*(\d{6})\b", text))
        wanted.update(int(sid) for sid in re.findall(r"\b(15\d{4})\s+(?:display|[AN])\b", text))
    wanted.update(int(item["id"]) for item in state["recent"])
    wanted.update((151166, 151173, 151368, 151370, 151373, 151377, 151378, 151380))
    source_files = {}
    paths = [ROOT / "p1/kernel.py"] + list((ROOT / "experiments/2026-09-29/candidates").glob("*.py"))
    for path in paths:
        source_files.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(), []).append(str(path.relative_to(ROOT)))
    cache_path = OUT / "2026-09-30-submission-audit.json"
    details = json.loads(cache_path.read_text()) if cache_path.exists() else {
        "checked_at": stamp, "read_only": True, "submissions": {}}
    details["extended_at"] = stamp
    for sid in sorted(wanted):
        existing = details["submissions"].get(str(sid), {})
        if existing.get("status") in ("Accepted", "WrongAnswer", "TimeLimitExceeded", "Canceled", "CompilationError", "RuntimeError"):
            continue
        data = read("submission/getSubmissionDetail", {"submissionId": str(sid), "locale": "zh_CN"})
        if "read_error" in data:
            details["submissions"][str(sid)] = data
            print(f"SID {sid}: read_error", flush=True)
            continue
        progress = data.get("progress") or {}
        meta = data.get("meta") or {}
        source = (data.get("content") or {}).get("code")
        sha = hashlib.sha256(source.encode()).hexdigest() if isinstance(source, str) else None
        row = {"status": progress.get("status") or meta.get("status"),
               "displayScore": progress.get("displayScore", meta.get("displayScore")),
               "source_sha256": sha, "matching_local_files": source_files.get(sha, []),
               "submitTime": meta.get("submitTime"), "cases": {}, "logs": []}
        for key, tc in (progress.get("testcaseResult") or {}).items():
            raw = tc.get("userError") or ""
            omitted = 0
            if isinstance(raw, dict):
                omitted = raw.get("omittedLength", 0)
                raw = raw.get("content") or raw.get("data") or ""
            raw = str(raw)
            row["logs"].append({"testcase_key": key, "status": tc.get("status"),
                                "text": raw, "omittedLength": omitted})
            case = re.search(r"tc=(\d+)", raw)
            metric = re.search(r'\{"schema_version"[^\n]*?\}', raw)
            if case and metric:
                row["cases"][case.group(1)] = {
                    "status": tc.get("status"), "displayScore": tc.get("displayScore"),
                    "metrics": json.loads(metric.group()),
                    "sqnr_db": [float(x) for x in re.findall(r"SQNR=([0-9.]+) dB", raw)],
                    "determinism_lines": [s for s in raw.splitlines() if "DETERMINISM" in s]}
        details["submissions"][str(sid)] = row
        (OUT / "2026-09-30-submission-audit.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2) + "\n")
        names = ",".join(Path(p).name for p in row["matching_local_files"])
        print(f"SID {sid}: {row['status']} raw={row['displayScore']}; "
              f"{len(row['cases'])} cases; SHA {(sha or '')[:12]}; {names}", flush=True)


if __name__ == "__main__":
    main()
