#!/usr/bin/env python3
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from xpuoj_api import Client

def main():
    c = Client()
    r = c._post("contest/play/getContestScoreboardMe", {"contestId": 13})
    r.raise_for_status()
    data = r.json()
    entry = data.get("entry") or {}
    ps = (entry.get("problemScores") or [{}])
    print(json.dumps({
        "totalScore": entry.get("totalScore"),
        "problemScore": ps[0].get("score"),
        "submissionId": ps[0].get("submissionId"),
        "submissionCount": ps[0].get("submissionCount"),
        "firstAcceptedAt": ps[0].get("firstAcceptedAt"),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
