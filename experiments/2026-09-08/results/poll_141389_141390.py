#!/usr/bin/env python3
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import API, _session  # noqa: E402

SIDS = (141389, 141390)
OUT = Path(__file__).with_name("submissions_141389_141390_raw.json")


def post(session, path, body):
    response = session.post(API + path, json=body, timeout=60)
    response.raise_for_status()
    return response.json()


def main():
    session = _session()
    terminal = set()
    payload = {"capturedAt": None, "list": None, "details": {}}
    while terminal != set(SIDS):
        listing = post(session, "contest/play/querySubmissions", {
            "contestId": 13,
            "problemOrder": 1,
            "takeCount": 10,
            "locale": "zh_CN",
        })
        payload["capturedAt"] = datetime.now(timezone.utc).isoformat()
        payload["list"] = listing
        rows = {row["id"]: row for row in listing.get("submissions", [])}
        states = []
        for sid in SIDS:
            row = rows.get(sid, {})
            status = row.get("status", "Missing")
            states.append(f"{sid}:{status}")
            if status not in ("Pending", "Running", "Missing"):
                terminal.add(sid)
                detail = post(session, "submission/getSubmissionDetail", {
                    "submissionId": str(sid),
                    "locale": "zh_CN",
                })
                payload["details"][str(sid)] = detail
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(datetime.now().astimezone().strftime("%H:%M:%S"), *states, flush=True)
        if terminal != set(SIDS):
            time.sleep(10)


if __name__ == "__main__":
    main()
