"""网页通道提交 —— 逐字段复刻浏览器真实请求(2026-08-31 抓包为准)。

用法: python3 scripts/xpuoj_web.py <文件> [turnstile令牌]
     不给令牌 → 自动从本机令牌池取（~/Desktop/xpuoj-turnstile-pool，需先 start.sh + 浏览器端脚本在线）

与 xpuoj_pow.py(API-Key 通道)的区别:
  - authorization 用 .secrets/xpuoj.json 里的 sessionToken(网页登录态)
  - 必须带 x-captcha-result(turnstile 令牌, 由人工从浏览器获取)
  - body 多一个 uploadInfo: null, 且带 origin/referer/x-request-id
  - 实测不消耗 api_token 额度池
"""
import sys
import os
import json
import uuid
import time
import hashlib
import requests

API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/"
sys.path.insert(0, os.path.expanduser("~/Desktop/xpuoj-turnstile-pool"))

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")


def _ok(digest, difficulty):
    n = difficulty // 2
    for i in range(n):
        if digest[i] != 0:
            return False
    return difficulty % 2 == 0 or (digest[n] >> 4) == 0


def solve(random_data, difficulty):
    n = 0
    rd = random_data.encode()
    while True:
        d = hashlib.sha256(rd + str(n).encode()).digest()
        if _ok(d, difficulty):
            return n, d.hex()
        n += 1


def _session():
    tok = json.load(open('.secrets/xpuoj.json'))['sessionToken']
    s = requests.Session()
    s.headers.update({
        "accept": "application/json, text/plain, */*",
        "accept-language": "zh-CN,zh;q=0.9",
        "authorization": "Bearer " + tok,
        "content-type": "application/json",
        "origin": "https://xpuoj.com",
        "referer": "https://xpuoj.com/",
        "user-agent": UA,
    })
    return s


def submit(code, captcha=None, contest_id=13, problem_order=1, language="triton-dist"):
    s = _session()
    if captcha is None:
        from pool_client import take
        captcha = take("submit_problem")
    r = s.post(API + "proofOfWork/issueChallenge",
               json={"action": "submit_problem"}, timeout=30)
    r.raise_for_status()
    ch = r.json()
    t0 = time.time()
    nonce, resp = solve(ch["randomData"], ch["difficulty"])
    headers = {
        "x-captcha-result": json.dumps({"turnstile": {"token": captcha}},
                                       separators=(',', ':')),
        "x-proof-of-work": json.dumps({"id": ch["id"], "nonce": nonce,
                                       "response": resp},
                                      separators=(',', ':')),
        "x-request-id": str(uuid.uuid4()),
    }
    body = {
        "contestId": contest_id,
        "problemOrder": problem_order,
        "content": {"code": code, "language": language,
                    "compileAndRunOptions": {}},
        "uploadInfo": None,
    }
    r = s.post(API + "contest/play/submit", json=body, headers=headers, timeout=180)
    print("PoW难度%s 求解%.1fs  HTTP %s" % (ch["difficulty"], time.time() - t0,
                                            r.status_code))
    if r.status_code not in (200, 201):
        raise RuntimeError(r.text[:400])
    d = r.json()
    if "submissionId" not in d:
        raise RuntimeError(str(d)[:400])
    return d["submissionId"]


if __name__ == "__main__":
    print("SID =", submit(open(sys.argv[1]).read(), sys.argv[2] if len(sys.argv) > 2 else None))
