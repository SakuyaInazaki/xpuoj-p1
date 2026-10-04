#!/usr/bin/env python3
"""XPUOJ 查当前题目 scoreboard 最佳（独立版）。默认 contest 13 / problem 1。"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import Client  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contest-id", type=int, default=13)
    ap.add_argument("--problem-order", type=int, default=1)
    args = ap.parse_args()
    c = Client()
    r = c._post("contest/play/getContestScoreboardMe", {"contestId": args.contest_id})
    r.raise_for_status()
    data = r.json()
    entry = data.get("entry") or {}
    ps = entry.get("problemScores") or [{}]
    print(json.dumps({
        "totalScore": entry.get("totalScore"),
        "problemScore": ps[0].get("score"),
        "submissionId": ps[0].get("submissionId"),
        "submissionCount": ps[0].get("submissionCount"),
        "firstAcceptedAt": ps[0].get("firstAcceptedAt"),
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
