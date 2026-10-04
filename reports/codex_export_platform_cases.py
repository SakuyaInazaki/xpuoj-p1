#!/usr/bin/env python3
"""Print shape-mapped XPUOJ testcase data as CSV without exposing credentials."""

import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from xpuoj_pow import Client  # noqa: E402


SHAPES = {
    1: (16384, 4096, 8, 8192, 2),
    2: (16384, 4096, 8, 14336, 2),
    3: (16384, 2048, 32, 2048, 4),
    4: (16384, 2048, 32, 1024, 4),
    5: (8192, 3584, 64, 2560, 8),
    6: (8192, 3584, 64, 1024, 8),
    7: (16384, 4096, 96, 2048, 3),
    8: (16384, 4096, 96, 1024, 3),
    9: (4096, 4096, 256, 2048, 8),
    10: (4096, 4096, 256, 1536, 8),
    11: (65536, 1024, 32, 1024, 2),
    12: (65536, 1024, 32, 2048, 2),
}


def error_text(case):
    value = case.get("userError") or ""
    if isinstance(value, dict):
        return value.get("content") or value.get("data") or ""
    return str(value)


def source_relation(remote, current):
    if remote == current:
        return "exact-current"
    diff = difflib.ndiff(current.splitlines(), remote.splitlines())
    changed = [line[2:] for line in diff if line.startswith(("+ ", "- "))]
    if changed and all(not line.strip() or line.lstrip().startswith("#") for line in changed):
        return "comment-only-current"
    return "different"


def rows(client, sid, current):
    detail = client.get_detail(sid).json()
    meta = detail.get("meta") or {}
    progress = detail.get("progress") or {}
    remote = (detail.get("content") or {}).get("code") or ""
    remote_hash = hashlib.sha256(remote.encode()).hexdigest()
    relation = source_relation(remote, current)
    result = []
    for key, case in (progress.get("testcaseResult") or {}).items():
        text = error_text(case)
        tc_match = re.search(r"tc=(\d+)", text)
        timing_match = re.search(r'\{"schema_version".*?\}', text)
        if not tc_match or not timing_match:
            continue
        timing = json.loads(timing_match.group(0))
        if timing.get("tk_time_ms") is None:
            continue
        tc = int(tc_match.group(1))
        shape = SHAPES.get(tc)
        if shape is None:
            continue
        sqnr = [float(value) for value in re.findall(r"SQNR=([\d.]+) dB", text)]
        result.append({
            "sid": sid,
            "source_sha256": remote_hash,
            "source_relation": relation,
            "submission_status": meta.get("status") or progress.get("status"),
            "submission_display_score": meta.get("displayScore", progress.get("displayScore")),
            "tc": tc,
            "T": shape[0],
            "H": shape[1],
            "E": shape[2],
            "I": shape[3],
            "topk": shape[4],
            "testcase_status": case.get("status"),
            "correctness_score": case.get("score"),
            "case_integer_score": case.get("displayScore"),
            "sqnr_1_db": sqnr[0] if len(sqnr) > 0 else None,
            "sqnr_2_db": sqnr[1] if len(sqnr) > 1 else None,
            "tk_ms": timing.get("tk_time_ms"),
            "tb_ms": timing.get("tb_time_ms"),
            "th_ms": timing.get("th_time_ms"),
            "testcase_key": key,
        })
    return sorted(result, key=lambda row: row["tc"])


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} SID [SID ...]")
    current = (ROOT / "p1" / "kernel.py").read_text()
    all_rows = []
    client = Client()
    for value in sys.argv[1:]:
        all_rows.extend(rows(client, int(value), current))
    writer = csv.DictWriter(sys.stdout, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)


if __name__ == "__main__":
    main()
