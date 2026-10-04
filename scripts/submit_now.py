"""便捷提交：python3 scripts/submit_now.py <file> [turnstile令牌]

额度充足时直接提交；额度耗尽时需附带令牌（用户在浏览器过一次验证码后取得）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xpuoj_pow import Client

f = sys.argv[1]
cap = sys.argv[2] if len(sys.argv) > 2 else None
c = Client()
q = c.credit()["api_token"]
print(f"api_token 额度: {q['available']}/{q['capacity']}"
      + (f"  下次回血 {q['nextCreditAt']}" if q['nextCreditAt'] else ""))
if q["available"] == 0 and not cap:
    print("额度为 0 且未提供 turnstile 令牌 —— 请在浏览器过一次验证码后把令牌作为第二个参数传入")
    sys.exit(2)
sid = c.submit(open(f).read(), captcha=cap)
print(f"提交成功 SID={sid}  (PoW difficulty={c.last_pow[0]}, 解时={c.last_pow[1]}s)")
