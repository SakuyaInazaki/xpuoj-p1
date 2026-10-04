"""Read-only P1 evidence refresh for the late-September-30 strategy guide.

Does not submit, consume a captcha, or print authentication material. Earlier
snapshots remain unchanged. Sources are fingerprinted; only metrics/logs and
fingerprints are saved. Run from the project root.
"""
import ast
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'reports'))
from xpuoj_web import API, _session
from collect_timing_anomalies_20260930 import parse_detail, TERMINAL

OUT = ROOT / 'reports/2026-09-30-late-cohort-readonly.json'
LOWER = 149480


def source_features(code):
    if not isinstance(code, str):
        return {}
    lines = code.splitlines(keepends=True)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {'source_parse_error': True}
    functions = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        dec = [ast.unparse(x) for x in node.decorator_list]
        if not any('jit' in x for x in dec):
            continue
        raw = ''.join(lines[node.lineno-1:node.end_lineno])
        semantic = ast.dump(node, include_attributes=False)
        # The last definition is what Python ultimately binds to this name.
        functions[node.name] = {
            'text_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'ast_sha256': hashlib.sha256(semantic.encode()).hexdigest(),
            'def_line': node.lineno,
            'start_line': min([node.lineno]+[x.lineno for x in node.decorator_list]),
            'end_line': node.end_lineno,
            'lines': node.end_lineno - node.lineno + 1,
        }
    combined = '|'.join(k + ':' + v['text_sha256'] for k, v in sorted(functions.items()))
    return {'jit_functions': functions,
            'jit_text_scope': 'AST def-to-end span, not a complete runtime JIT cache key',
            'jit_text_set_sha256': hashlib.sha256(combined.encode()).hexdigest(),
            'source_lines': len(lines), 'source_bytes': len(code.encode())}


def main():
    old = json.loads((ROOT / 'reports/2026-09-30-timing-cohort-readonly.json').read_text())
    cache = old['submissions']
    if OUT.exists():
        cache.update(json.loads(OUT.read_text()).get('submissions', {}))
    session = _session()

    def read(endpoint, body):
        r = session.post(API + endpoint, json=body, timeout=25)
        r.raise_for_status()
        data = r.json()
        if 'error' in data:
            raise RuntimeError('Read endpoint returned an API error')
        return data

    result = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'read_only': True, 'lower_sid_inclusive': LOWER,
              'list_complete_to_lower_bound': False, 'recent': [], 'submissions': {}}
    seen = set()
    for _ in range(60):
        body = {'contestId': 13, 'problemOrder': 1, 'takeCount': 10, 'locale': 'zh_CN'}
        if seen:
            body['maxId'] = min(seen) - 1
        data = read('contest/play/querySubmissions', body)
        rows = [r for r in data.get('submissions', []) if int(r['id']) not in seen]
        if not rows:
            break
        seen.update(int(r['id']) for r in rows)
        result['recent'].extend({k: r.get(k) for k in
            ('id', 'status', 'displayScore', 'submitTime', 'timeUsed')}
            for r in rows if int(r['id']) >= LOWER)
        if min(seen) < LOWER or not data.get('hasSmallerId'):
            result['list_complete_to_lower_bound'] = True
            break
    fetched = 0
    for item in sorted(result['recent'], key=lambda r: r['id']):
        sid = str(item['id'])
        row = cache.get(sid)
        if (not row or row.get('status') not in TERMINAL
                or row.get('status') != item['status']):
            data = read('submission/getSubmissionDetail', {'submissionId': sid, 'locale': 'zh_CN'})
            row = parse_detail(data)
            row.update(source_features((data.get('content') or {}).get('code')))
            fetched += 1
        row = dict(row)
        row['submitTime'] = item.get('submitTime') or row.get('submitTime')
        row['listed_timeUsed'] = item.get('timeUsed')
        result['submissions'][sid] = row
        if fetched and fetched % 10 == 0:
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print(f'Recovered {len(result["submissions"])}/{len(result["recent"])}; fresh reads {fetched}', flush=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'path': str(OUT), 'submissions': len(result['submissions']),
                      'fresh_reads': fetched, 'complete': result['list_complete_to_lower_bound']}))


if __name__ == '__main__':
    main()
