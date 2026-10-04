#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import API, _session  # noqa: E402


def main():
    session = _session()
    score = session.post(API + "contest/play/getContestScoreboardMe", json={"contestId": 13}, timeout=60)
    score.raise_for_status()
    subs = session.post(API + "contest/play/querySubmissions", json={
        "contestId": 13, "problemOrder": 1, "takeCount": 50, "locale": "zh_CN",
    }, timeout=60)
    subs.raise_for_status()
    entry = score.json().get("entry") or {}
    problem_scores = entry.get("problemScores") or []
    p1 = problem_scores[0] if problem_scores else None
    best = max(
        (s for s in subs.json().get("submissions", []) if s.get("status") == "Accepted"),
        key=lambda s: s.get("displayScore", -1), default=None,
    )
    print("scoreboard_p1", {
        "totalScore": entry.get("totalScore"),
        "problemScore": p1.get("score") if p1 else None,
        "submissionId": p1.get("submissionId") if p1 else None,
        "submissionCount": p1.get("submissionCount") if p1 else None,
    })
    if best:
        print("recent_best", {k: best.get(k) for k in ("id", "status", "displayScore", "submitTime")})
    inflight = [
        {"id": row.get("id"), "status": row.get("status")}
        for row in subs.json().get("submissions", [])
        if row.get("status") in ("Pending", "Running")
    ]
    print("inflight", inflight)


if __name__ == "__main__":
    main()
