"""自定义测试通道(网页令牌)：在评测机 H800 上跑任意 Triton 代码。

createCustomTest 需要 custom_test 额度(上限3/每10分钟+1)；走网页通道时
不吃 api_token 额度池，所以一个 turnstile 令牌 = 一次 H800 运行。

用法: python3 scripts/xpuoj_ct.py <文件> <令牌> [problemId]
"""
import sys, json, time, uuid
import requests
from xpuoj_web import API, UA, solve, _session

PROBLEM_ID = 24          # 公开题 "FP8 GeMM NT"，只借它的运行环境


def create(code, captcha, problem_id=PROBLEM_ID, language="triton-h800",
           mode="GeneratedWorkload"):
    s = _session()
    ch = s.post(API + "proofOfWork/issueChallenge",
                json={"action": "custom_test"}, timeout=30).json()
    nonce, resp = solve(ch["randomData"], ch["difficulty"])
    h = {
        "x-captcha-result": json.dumps({"turnstile": {"token": captcha}},
                                       separators=(',', ':')),
        "x-proof-of-work": json.dumps({"id": ch["id"], "nonce": nonce,
                                       "response": resp}, separators=(',', ':')),
        "x-request-id": str(uuid.uuid4()),
    }
    body = {"problemId": problem_id, "mode": mode,
            "content": {"language": language, "code": code,
                        "compileAndRunOptions": {}}}
    r = s.post(API + "customTest/createCustomTest", json=body, headers=h, timeout=120)
    d = r.json()
    if r.status_code not in (200, 201) or "id" not in (d.get("data") or {}):
        raise RuntimeError(r.text[:500])
    return d["data"]["id"], s


def detail(cid, s=None):
    s = s or _session()
    r = s.post(API + "customTest/getCustomTestDetail",
               json={"id": cid}, timeout=60)
    return r.json()


def wait(cid, s=None, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = detail(cid, s)
        st = json.dumps(d)[:0]
        p = d.get("progress") or d.get("customTest") or d
        status = json.dumps(p).count("")  # placeholder
        s_txt = (p.get("status") if isinstance(p, dict) else None)
        if s_txt and s_txt not in ("Pending", "Preparing", "Compiling", "Running", "Waiting"):
            return d
        time.sleep(5)
    return detail(cid, s)


if __name__ == "__main__":
    code = open(sys.argv[1]).read()
    tok = sys.argv[2]
    pid = int(sys.argv[3]) if len(sys.argv) > 3 else PROBLEM_ID
    cid, s = create(code, tok, pid)
    print("customTestId =", cid)
    d = wait(cid, s)
    print(json.dumps(d, ensure_ascii=False)[:4000])
