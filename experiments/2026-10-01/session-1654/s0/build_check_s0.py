from pathlib import Path
import ast,copy,difflib,hashlib,json
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1'); OUT=Path(__file__).resolve().parent
s=(ROOT/'p1/kernel.py').read_text(); assert hashlib.sha256(s.encode()).hexdigest()=='08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9'
def defs(s):return {n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
o=defs(s)
def extract(name):
 n=o[name]; lo=min([n.lineno]+[d.lineno for d in n.decorator_list]);return ''.join(s.splitlines(True)[lo-1:n.end_lineno])
k=extract('_fgs_t1i_mdq_tma_pre_kernel').replace('_fgs_t1i_mdq_tma_pre_kernel','_s0_c11_tma_pre_consti_kernel').replace('    M, I, K: tl.constexpr,','    M, I: tl.constexpr, K: tl.constexpr,')
h=extract('_fgs_tma1_intq_host_pre').replace('_fgs_tma1_intq_host_pre','_s0_c11_tma_pre_consti_host').replace('_fgs_t1i_mdq_tma_pre_kernel','_s0_c11_tma_pre_consti_kernel')
old='''                    act_q8, act_rowscl = _fgs_tma1_intq_host_pre(
                        fp8_tokens_q, _dir_scl, _dir_ah, _dir_wi, _gq16, _gs16,
                        meta_expert_ids, expert_counts, meta_tile_split,
                        meta_tile_num, meta_tile_num_cum, num_tiles_total,
                    )'''
assert s.count(old)==1
new='''                    if (T, H, E, I, k) == (65536, 1024, 32, 1024, 2):
                        act_q8, act_rowscl = _s0_c11_tma_pre_consti_host(
                            fp8_tokens_q, _dir_scl, _dir_ah, _dir_wi, _gq16, _gs16,
                            meta_expert_ids, expert_counts, meta_tile_split,
                            meta_tile_num, meta_tile_num_cum, num_tiles_total,
                        )
                    else:
'''+''.join('    '+l+'\n' for l in old.splitlines())
out=s.replace(old,new)+'\n\n# S0 c11: I-only constexpr specialization.\n'+k+'\n'+h+'\n'
f=defs(out); changed=[n for n in o if ast.dump(o[n])!=ast.dump(f[n])];assert changed==['_run_replicated']
x=copy.deepcopy(f['_s0_c11_tma_pre_consti_kernel']); x.name=o['_fgs_t1i_mdq_tma_pre_kernel'].name
ia=next(a for a in x.args.args if a.arg=='I');assert ast.unparse(ia.annotation)=='tl.constexpr';ia.annotation=None
assert ast.dump(x)==ast.dump(o['_fgs_t1i_mdq_tma_pre_kernel'])
x=copy.deepcopy(f['_s0_c11_tma_pre_consti_host']);x.name=o['_fgs_tma1_intq_host_pre'].name
for a in ast.walk(x):
 if isinstance(a,ast.Name) and a.id=='_s0_c11_tma_pre_consti_kernel':a.id='_fgs_t1i_mdq_tma_pre_kernel'
assert ast.dump(x)==ast.dump(o['_fgs_tma1_intq_host_pre'])
compile(out,str(OUT/'p1_s0_c11_consti.py'),'exec')
newnames=set(f)-set(o);assert len(newnames)==2
calls=[]
for a in ast.walk(ast.parse(out)):
 if not isinstance(a,ast.Call):continue
 target=a.func.value if isinstance(a.func,ast.Subscript) else a.func
 if not isinstance(target,ast.Name) or target.id not in newnames:continue
 params=[p.arg for p in f[target.id].args.args]; bound=params[:len(a.args)]+[q.arg for q in a.keywords if q.arg in params]
 assert len(bound)==len(set(bound)) and set(bound)==set(params)
 calls.append({'name':target.id,'line':a.lineno,'positional':len(a.args)})
assert len(calls)==2
(OUT/'p1_s0_c11_consti.py').write_text(out)
(OUT/'candidate.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile='p1/kernel.py',tofile='p1_s0_c11_consti.py')))
report={'sha256':hashlib.sha256(out.encode()).hexdigest(),'changed_existing':changed,'new_helpers':{n:f[n].lineno for n in newnames},'calls':calls,'semantic_change':'I annotation only; M/counts/num_tiles remain runtime','kernel_and_host_ast':'exact after name/I annotation normalization','compile':'passed','limits':'No GPU/OJ test or speed claim'}
(OUT/'checks.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
