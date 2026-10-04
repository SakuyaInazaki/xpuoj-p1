import sys
import time
sys.path.insert(0, 'scripts')
from xpuoj_web import submit
from xpuoj_pow import Client

TARGET = 85.58
N = 8
c = Client()

for i in range(N):
    sid = None
    for attempt in range(6):
        try:
            sid = submit(open('p1/kernel.py').read())
            break
        except Exception as exc:
            print('submit_retry', i, attempt, type(exc).__name__, str(exc)[:200], flush=True)
            time.sleep(90)
    if sid is None:
        print('submit_failed', i, flush=True)
        continue
    print('SID', sid, 'iter', i, flush=True)
    t0 = time.time()
    p = {}
    status = None
    while time.time() - t0 < 900:
        try:
            p = (c.get_detail(sid).json().get('progress') or {})
            status = p.get('status')
        except Exception as exc:
            print('poll_retry', sid, type(exc).__name__, str(exc)[:120], flush=True)
        if status not in (None, '-', 'Pending', 'Waiting', 'Compiling', 'Running', 'Judging'):
            break
        time.sleep(30)
    ds = p.get('displayScore')
    rows = []
    for _h, tc in (p.get('testcaseResult') or {}).items():
        if tc.get('displayScore') is not None:
            rows.append(tc.get('displayScore'))
    print('RESULT', sid, 'status', status, 'display', ds, 'sum', sum(rows), flush=True)
    if ds is not None and ds >= TARGET:
        print('TARGET_HIT', sid, ds, flush=True)
        break
    time.sleep(30)
print('DONE', flush=True)
