#!/usr/bin/env python3
"""XPUOJ 查询/解析提交详情（独立版）。

用法:
    python xpuoj_status.py 117300
    python xpuoj_status.py 117300 --cases   # 打印逐点 tk/tb/score
"""
import argparse, base64, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_submit import Client  # noqa: E402


def parse_user_output(user_output):
    m = re.search(r"OJRESULT v1 \S+ (\S+)", user_output or "")
    if not m:
        return {}
    return json.loads(base64.b64decode(m.group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission_id")
    ap.add_argument("--cases", action="store_true")
    args = ap.parse_args()

    c = Client()
    d = c.get_detail(args.submission_id).json()
    meta = d.get("meta") or {}
    prog = d.get("progress") or {}
    print(json.dumps({
        "id": meta.get("id"),
        "status": meta.get("status") or prog.get("status"),
        "score": meta.get("score"),
        "displayScore": meta.get("displayScore") if meta.get("displayScore") is not None else prog.get("displayScore"),
        "timeUsed": meta.get("timeUsed"),
        "memoryUsed": meta.get("memoryUsed"),
    }, ensure_ascii=False, indent=2))

    if not args.cases:
        return
    tr = prog.get("testcaseResult") or {}
    rows = []
    for k, v in tr.items():
        j = parse_user_output(v.get("userOutput"))
        rows.append((int(v.get("input", "0").strip() or 0), v.get("status"), v.get("displayScore"),
                     j.get("tk_time_ms"), j.get("tb_time_ms")))
    for inp, status, score, tk, tb in sorted(rows):
        print(f"case {inp:>2}: status={status} score={score} tk={tk} tb={tb}")


if __name__ == "__main__":
    main()
