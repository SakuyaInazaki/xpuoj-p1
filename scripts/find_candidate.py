#!/usr/bin/env python3
"""Resolve an archived candidate by old path, filename fragment or SHA prefix."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("query")
parser.add_argument("--limit", type=int, default=20)
args = parser.parse_args()
with (ROOT / "archive/WORKSPACE_MOVES_2026-09-30.csv").open() as stream:
    matches = [row for row in csv.DictReader(stream)
               if args.query in row["old_path"] or args.query in row["new_path"]
               or row["sha256"].startswith(args.query)]
for row in matches[:args.limit]:
    print(f'{row["sha256"]}  {ROOT / row["new_path"]}')
print(f"{len(matches)} match(es)")
