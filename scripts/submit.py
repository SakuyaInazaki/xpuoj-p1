#!/usr/bin/env python3
"""Submit P1 run_kernel file to XPUOJ and poll until finished."""
import argparse, json, sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from xpuoj_api import Client, CONTEST_ID, PROBLEM_ORDER

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="submission .py file")
    ap.add_argument("--poll", action="store_true", help="poll until terminal state")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=2400.0)
    ap.add_argument("--compile-and-run-options", default="{}", help="JSON object")
    args = ap.parse_args()
    code = open(args.file).read()
    c = Client()
    sid = c.submit(code, compile_and_run_options=json.loads(args.compile_and_run_options))
    print(f"submitted submissionId={sid}")
    if args.poll:
        deadline = time.time() + args.timeout
        while True:
            subs = c.list_submissions(5)
            info = next((s for s in subs if s["id"] == sid), None)
            if info:
                status = info.get("status")
                print(json.dumps({k: info.get(k) for k in (
                    "id", "status", "score", "displayScore", "timeUsed", "memoryUsed")}, ensure_ascii=False), flush=True)
                if status not in ("Pending", "Running"):
                    break
            if time.time() > deadline:
                print("poll timeout", flush=True)
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
