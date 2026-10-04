#!/usr/bin/env python3
"""XPUOJ 通用提交器（独立版，放在 xpuoj-p1 下）。

默认凭据路径 /home/sakimi26/xpuoj-p1/.secrets/xpuoj.json。
可通过环境变量 XPUOJ_SECRET_PATH 覆盖。

用法:
    python xpuoj_submit.py FILE --language <题目要求的language>
    python xpuoj_submit.py FILE --problem-order 2 --language <题目要求的language>
    python xpuoj_submit.py FILE --language <题目要求的language> --poll --interval 10 --timeout 1800
"""
import argparse, json, os, time
import requests

API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/"
DEFAULT_SECRET = "/home/sakimi26/xpuoj-p1/.secrets/xpuoj.json"


class Client:
    def __init__(self, secret_path=None):
        self.s = requests.Session()
        creds = json.load(open(secret_path or os.environ.get("XPUOJ_SECRET_PATH", DEFAULT_SECRET)))
        r = self.s.post(API + "auth/login", json=creds, timeout=30)
        r.raise_for_status()
        data = r.json()
        if "token" not in data:
            raise RuntimeError(f"login failed: {data}")
        self.token = data["token"]
        self.s.headers.update({"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})

    def _post(self, path, data=None):
        return self.s.post(API + path, json=data or {}, timeout=60)

    def submit(self, code, language, contest_id, problem_order, compile_and_run_options=None):
        content = {
            "language": language,
            "code": code,
            "compileAndRunOptions": compile_and_run_options or {},
        }
        r = self._post("contest/play/submit", {
            "contestId": contest_id,
            "problemOrder": problem_order,
            "content": content,
        })
        if r.status_code not in (200, 201):
            raise RuntimeError(f"submit HTTP {r.status_code}: {r.text}")
        data = r.json()
        if "submissionId" not in data:
            raise RuntimeError(f"submit rejected: {data}")
        return data["submissionId"]

    def list_submissions(self, take=10, contest_id=13, problem_order=1):
        r = self._post("contest/play/querySubmissions", {
            "contestId": contest_id, "problemOrder": problem_order, "takeCount": take, "locale": "zh_CN"})
        r.raise_for_status()
        return r.json().get("submissions", [])

    def get_detail(self, submission_id):
        return self._post("submission/getSubmissionDetail", {
            "submissionId": str(submission_id), "locale": "zh_CN"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--contest-id", type=int, default=13)
    ap.add_argument("--problem-order", type=int, default=1)
    ap.add_argument("--language", required=True, help="题目要求的提交语言")
    ap.add_argument("--compile-and-run-options", default="{}", help="JSON object")
    ap.add_argument("--poll", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=2400.0)
    args = ap.parse_args()

    code = open(args.file).read()
    c = Client()
    sid = c.submit(code, language=args.language, contest_id=args.contest_id,
                   problem_order=args.problem_order,
                   compile_and_run_options=json.loads(args.compile_and_run_options))
    print(f"submitted submissionId={sid}", flush=True)
    if not args.poll:
        return
    deadline = time.time() + args.timeout
    while True:
        info = None
        for s in c.list_submissions(10, contest_id=args.contest_id, problem_order=args.problem_order):
            if s.get("id") == sid:
                info = s
                break
        if info:
            status = info.get("status")
            print(json.dumps({k: info.get(k) for k in (
                "id", "status", "score", "displayScore", "timeUsed", "memoryUsed")}, ensure_ascii=False), flush=True)
            if status not in ("Pending", "Running"):
                break
        if time.time() > deadline:
            print("poll timeout; submission may still be Pending on platform", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
