"""Block until every given SID has all 12 testcases reported (or gives up)."""
import sys, os, time
sys.path.insert(0, '/Users/sakimi/Desktop/xpuoj-p1/scripts')
from xpuoj_pow import Client
from cases import fetch

sids = [int(x) for x in sys.argv[1:]]
c = Client()
deadline = time.time() + 1800
done = set()
while time.time() < deadline:
    for s in sids:
        if s in done:
            continue
        try:
            ds, o = fetch(s, c)
        except Exception as e:
            print(f"{s} poll-error {type(e).__name__}", flush=True)
            continue
        if len(o) >= 12:
            done.add(s)
            print(f"{s} DONE display={ds}", flush=True)
    if len(done) == len(sids):
        print("ALL DONE", flush=True)
        break
    time.sleep(20)
else:
    print("TIMEOUT", flush=True)
