"""XPUOJ 新认证链路：API Key + Proof-of-Work。

平台 2026-08-30 升级后，写操作（提交/自定义测试）需要 Proof-of-Work。
算法从前端 worker (proofOfWork.worker) 还原：
  hash = SHA256(randomData + str(nonce))
  难度 d 以"半字节"计：前 d//2 字节须为 0；d 为奇数时第 d//2 字节高 4 位须为 0
解答以 HTTP 头 X-Proof-Of-Work: {"nonce":N,"response":"<hex>"} 发送。
"""
import hashlib, json, os, time
import requests

API = "https://sd629vuj4f7uh2cscrbe0.apigateway-cn-beijing.volceapi.com/api/"
SECRET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".secrets", "xpuoj.json")
CONTEST_ID, PROBLEM_ORDER, LANG = 13, 1, "triton-dist"
ACTION_SUBMIT = "submit_problem"
ACTION_CUSTOM = "custom_test"


def _ok(digest, difficulty):
    n = difficulty // 2
    for i in range(n):
        if digest[i] != 0:
            return False
    return difficulty % 2 == 0 or (digest[n] >> 4) == 0


def solve(random_data, difficulty, start=0, step=1):
    n = start
    rd = random_data.encode()
    while True:
        d = hashlib.sha256(rd + str(n).encode()).digest()
        if _ok(d, difficulty):
            return {"nonce": n, "response": d.hex()}
        n += step


class Client:
    def __init__(self):
        self.key = json.load(open(SECRET))["apiKey"]
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.key}",
                               "Content-Type": "application/json"})

    def _pow(self, action):
        r = self.s.post(API + "proofOfWork/issueChallenge",
                        json={"action": action}, timeout=30)
        r.raise_for_status()
        ch = r.json()
        t0 = time.time()
        sol = solve(ch["randomData"], ch["difficulty"])
        sol["_solve_s"] = round(time.time() - t0, 2)
        return ch, sol

    def _post(self, path, body, action=None, captcha=None):
        h = {}
        if captcha:                      # 人工过验证码后传入的 turnstile 令牌
            h["X-Captcha-Result"] = json.dumps({"turnstile": {"token": captcha}})
        if action:
            ch, sol = self._pow(action)
            h["X-Proof-Of-Work"] = json.dumps(
                {"id": ch["id"], "nonce": sol["nonce"], "response": sol["response"]})
            self.last_pow = (ch.get("difficulty"), sol.get("_solve_s"))
        r = self.s.post(API + path, json=body, headers=h, timeout=120)
        return r

    def submit(self, code, language=LANG, contest_id=CONTEST_ID, problem_order=PROBLEM_ORDER,
               captcha=None):
        r = self._post("contest/play/submit", {
            "contestId": contest_id, "problemOrder": problem_order,
            "content": {"language": language, "code": code, "compileAndRunOptions": {}},
        }, action=ACTION_SUBMIT, captcha=captcha)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"submit HTTP {r.status_code}: {r.text[:300]}")
        d = r.json()
        if "submissionId" not in d:
            raise RuntimeError(f"submit rejected: {d}")
        return d["submissionId"]

    def credit(self):
        r = self.s.post(API + "credit/getCredit",
                        json={"types": ["api_token", "custom_test"]}, timeout=30)
        return {x["type"]: x for x in r.json()["credits"]}

    def get_detail(self, sid):
        return self.s.post(API + "submission/getSubmissionDetail",
                           json={"submissionId": str(sid), "locale": "zh_CN"}, timeout=60)
