"""Reproduce late P1 strategy tables from saved evidence. No network/GPU."""
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
BASE_SHA = '08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9'


def source_index(wanted):
    result = {}
    baseline = ROOT / 'experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == BASE_SHA
    baseline_ast = ast.dump(ast.parse(baseline.read_text()), include_attributes=False)
    paths = list((ROOT / 'experiments/2026-09-30/candidates').rglob('*.py'))
    paths += list((ROOT / 'p1/references').glob('*.py'))
    for path in paths:
        code = path.read_bytes()
        sha = hashlib.sha256(code).hexdigest()
        if sha not in wanted:
            continue
        tree = ast.parse(code)
        lines = code.decode().splitlines(keepends=True)
        functions = {}
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not any(
                    'jit' in ast.unparse(d) for d in node.decorator_list):
                continue
            text = ''.join(lines[node.lineno-1:node.end_lineno])
            functions[node.name] = {
                'def_line': node.lineno,
                'start_line': min([node.lineno]+[x.lineno for x in node.decorator_list]),
                'end_line': node.end_lineno,
                'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
            }
        result[sha] = {'path': str(path.relative_to(ROOT)), 'jit_functions': functions,
                      'same_module_ast_as_v12': ast.dump(tree, include_attributes=False) == baseline_ast}
    return result


def point(tb, tk):
    return math.floor(100*tb/(tb+tk))


def main():
    data = json.loads((REPORTS / '2026-09-30-late-cohort-readonly.json').read_text())
    records = data['submissions']
    shapes = {r['case']: r for r in json.loads((REPORTS / '2026-09-30-v926-score.json').read_text())['cases']}
    src = source_index({r.get('source_sha256') for r in records.values()})
    (REPORTS / '2026-09-30-late-source-map.json').write_text(json.dumps(src, ensure_ascii=False, indent=2)+'\n')
    counts = collections.Counter()
    groups = {k: {'accepted': 0, 'severe': 0, 'tle': 0} for k in ('first', 'repeat')}
    seen, events, probes, ledger = set(), [], [], []
    for sid, row in sorted(records.items(), key=lambda item: int(item[0])):
        sha = row.get('source_sha256')
        group = 'first' if sha not in seen else 'repeat'
        seen.add(sha)
        counts[row['status']] += 1
        lows, zero, fail_zero = [], [], []
        complete = row['status'] == 'Accepted' and len(row['cases']) == 12
        for c, case in row['cases'].items():
            m = case['metrics']
            passed = case['status'] == 'Accepted' and m.get('pass') is True
            complete &= passed
            if passed and m['tk_time_ms'] < .5*shapes[int(c)]['tk_ms']:
                lows.append(int(c))
            if passed and m['tk_time_ms'] == 0:
                zero.append(int(c))
            if not passed and m['tk_time_ms'] == 0:
                fail_zero.append(int(c))
        groups[group]['accepted'] += bool(complete)
        groups[group]['severe'] += bool(complete and lows)
        groups[group]['tle'] += row['status'] == 'TimeLimitExceeded'
        item = {'sid': int(sid), 'status': row['status'], 'source_sha256': sha,
                'raw': row['displayScore'], 'group': group, 'complete_ac': complete,
                'low_cases': sorted(lows), 'zero_cases': sorted(zero),
                'failed_zero_cases': sorted(fail_zero), 'path': src.get(sha, {}).get('path'),
                'submitTime': row['submitTime']}
        ledger.append(item)
        if lows:
            events.append(item)
        if '/anom_probe/' in (item['path'] or ''):
            probes.append(item)
    equivalents = [str(row['sid']) for row in ledger if row['complete_ac'] and not row['low_cases']
                   and src.get(row['source_sha256'], {}).get('same_module_ast_as_v12')]
    table = []
    for c in range(1, 13):
        shape = shapes[c]
        ms = [records[sid]['cases'][str(c)]['metrics'] for sid in equivalents]
        tk = statistics.median(m['tk_time_ms'] for m in ms)
        tb = statistics.median(m['tb_time_ms'] for m in ms)
        q = point(tb, tk)
        mrows = shape['T']*shape['k']
        capacity = ((mrows + shape['E']*127+127)//128)*128
        table.append({**{k: shape[k] for k in ('case','T','H','E','I','k')},
                      'tk_ms': tk, 'tb_ms': tb, 'q': q,
                      'next_drop_pct': 100*(1-tb*(100/(q+1)-1)/tk),
                      'M': mrows, 'mean_M_per_expert': mrows/shape['E'],
                      'useful_TFLOPs': 6*mrows*shape['H']*shape['I']/1e12,
                      'effective_TFLOPs_per_second': 6*mrows*shape['H']*shape['I']/1e9/tk,
                      'weight_GiB': 3*shape['E']*shape['H']*shape['I']/2**30,
                      'act_MiB': mrows*shape['I']/2**20,
                      'padded_capacity': capacity,
                      'extra_padded_act_MiB_max': (capacity-mrows)*shape['I']/2**20,
                      'dscl_MiB_compact': 4*mrows*(shape['H']//256)/2**20,
                      'sqnr_min': min(min(records[sid]['cases'][str(c)]['sqnr_db']) for sid in equivalents)})
    raw = lambda t: sum(point(r['tb_ms'], t(r)) for r in table)/12
    zero_models = {'+'.join(map(str, zero)): raw(lambda r: 0 if r['case'] in zero else r['tk_ms'])
                   for zero in [(12,), (11,12), (10,11,12), (9,10,11,12), (8,9,10,11,12), (7,8,9,10,11,12)]}
    best = records['152238']['cases']
    best_q = {int(c): point(v['metrics']['tb_time_ms'], v['metrics']['tk_time_ms']) for c,v in best.items()}
    best_zero_models = {'+'.join(map(str, zero)) or 'observed':
        sum(100 if int(c) in zero else best_q[int(c)] for c in best)/12
        for zero in [(), (8,), (8,9), (7,), (6,)]}
    line_samples = {}
    for sid in ('151912','152205','152238','152248','152289','152305'):
        f = src[records[sid]['source_sha256']]
        line_samples[sid] = {'path': f['path'], 'kernel_lines': {
            k: v['start_line'] for k,v in f['jit_functions'].items()
            if k in ('_route_full_kernel','_fgs_tma1_kernel_gq_tiled','_dn_tma2_f8_pad_static_kernel','_dn_tma2_f8_tiled_static_kernel')}}
    result = {'checked_at': data['checked_at'], 'source_cohort_complete': data['list_complete_to_lower_bound'],
              'sid_range': [min(map(int, records)),max(map(int,records))], 'statuses': counts,
              'groups': groups, 'events': events,
              'neutral_probe_count': len(probes), 'neutral_probe_low_count': sum(bool(p['low_cases']) for p in probes),
              'normal_pool_sids': equivalents, 'normal_pool_note': 'Full-module AST identical, different source/JIT text possible; n=3 descriptive sample, not independent performance confirmation.',
              'cases': table, 'normal_raw': raw(lambda r:r['tk_ms']),
              'global_drop_raw': {str(d): raw(lambda r:r['tk_ms']*(1-d)) for d in (.02,.05,.1,.3,.5)},
              'normal_zero_models': zero_models, 'best_case_points': best_q,
              'best_sum': sum(best_q.values()), 'raw90_point_gap': 1080-sum(best_q.values()),
              'best_zero_models': best_zero_models, 'source_line_samples': line_samples}
    (REPORTS / '2026-09-30-late-strategy-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for name, rows in [('2026-09-30-late-score.csv', table), ('2026-09-30-late-ledger.csv', ledger)]:
        with (REPORTS/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ('statuses','groups','neutral_probe_count','neutral_probe_low_count','normal_pool_sids','normal_raw','best_sum','raw90_point_gap','normal_zero_models')}, indent=2))


if __name__ == '__main__':
    main()
