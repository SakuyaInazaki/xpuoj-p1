#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_web import API, _session  # noqa: E402


def text_values(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from text_values(child)


def main():
    session = _session()
    response = session.post(
        API + "contest/getContestAnnouncements", json={"contestId": 13}, timeout=60
    )
    print("http_status", response.status_code)
    if response.status_code not in (200, 201):
        print("available", False)
        return
    data = response.json()
    texts = list(text_values(data))
    terms = ("TLE", "TimeLimit", "超时", "样例", "sample")
    matches = [text for text in texts if any(term.lower() in text.lower() for term in terms)]
    print("available", True)
    print("announcement_count", len(data) if isinstance(data, list) else None)
    print("matching_text_count", len(matches))
    for text in matches:
        print(json.dumps(text[:500], ensure_ascii=False))


if __name__ == "__main__":
    main()
