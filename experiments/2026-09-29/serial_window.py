import sys, os, time, json
from pathlib import Path
ROOT = Path('/Users/sakimi/Desktop/xpuoj-p1')
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, '/Users/sakimi/Desktop/xpuoj-turnstile-pool')
from xpuoj_web import submit, API, _session
from pool_client import take, PoolDown, PoolEmpty

CODE = (ROOT/'p1/kernel.py').read_text()
OUT = ROOT/'experiments/2026-09-29/serial_window_results.json'
N = int(os.environ.get('SERIAL_N', '10'))
STOP = float(os.environ.get('SERIAL_STOP', '90.0'))
s = _session()
rows = []
for attempt in range(N):
    sid = None
    for retry in range(5):
        try:
            cap = take('submit_problem', timeout=180)
            sid = submit(CODE, captcha=cap)
            break
        except (PoolDown, PoolEmpty):
            print('pool_wait', attempt, retry, flush=True)
            time.sleep(30)
        except Exception as exc:
            print('submit_err', attempt, retry, type(exc).__name__, flush=True)
            time.sleep(30)
    if sid is None:
        print('skip', attempt, flush=True)
        continue
    status = 'Pending'
    display = None
    detail = None
    for _ in range(60):
        time.sleep(15)
        try:
            d = s.post(API+'submission/getSubmissionDetail',
                       json={'submissionId':str(sid),'locale':'zh_CN'}, timeout=30).json()
        except Exception:
            continue
        p = d.get('progress') or {}
        m = d.get('meta') or {}
        status = m.get('status')
        display = m.get('displayScore')
        if status in ('Accepted','WrongAnswer','TimeLimitExceeded','CompilationError','RuntimeError'):
            detail = d
            break
    row = {'attempt': attempt, 'sid': str(sid), 'status': status, 'display': display}
    rows.append(row)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    print('RESULT', attempt, sid, status, display, flush=True)
    if display is not None and display >= STOP:
        break
print('SERIAL_DONE', len(rows), flush=True)
