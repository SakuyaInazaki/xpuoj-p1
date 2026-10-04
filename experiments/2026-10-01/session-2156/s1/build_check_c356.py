from pathlib import Path
import ast,hashlib,difflib,json
OUT=Path(__file__).resolve().parent;s=(OUT/'p1_s1_c56_unified.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='3f95092adf1225f59b5e29a5bc7ad56000ca65dfeda02dd882ab6ded17c9db9a'
a='(T, H, E, I, k) in ((8192, 3584, 64, 2560, 8), (8192, 3584, 64, 1024, 8))'
b='(T, H, E, I, k) in ((16384, 2048, 32, 2048, 4), (8192, 3584, 64, 2560, 8), (8192, 3584, 64, 1024, 8))'
assert s.count(a)==2;out=s.replace(a,b); assert ast.dump(ast.parse(out.replace(b,a)))==ast.dump(ast.parse(s))
f=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)};old,new=f(ast.parse(s)),f(ast.parse(out))
assert [n for n in old if ast.dump(old[n])!=ast.dump(new[n])]==['_run_replicated']
compile(out,str(OUT/'p1_s1_c356_unified.py'),'exec')
calls=[];helpers={n for n in new if n.startswith('_s1_c6_')}
for call in ast.walk(ast.parse(out)):
 if not isinstance(call,ast.Call):continue
 target=call.func.value if isinstance(call.func,ast.Subscript) else call.func
 if not isinstance(target,ast.Name) or target.id not in helpers:continue
 params=[v.arg for v in new[target.id].args.args];bound=params[:len(call.args)]+[v.arg for v in call.keywords if v.arg in params]
 assert len(bound)==len(set(bound)) and set(bound)==set(params);calls.append({'name':target.id,'line':call.lineno})
assert len(calls)==5
producer=ast.unparse(new['_s1_c6_gq_tok_params']);consumer=ast.unparse(new['_s1_c6_mdq_g_pre_host']);gq=ast.unparse(new['_s1_c6_gq_tok_params_kernel']);md=ast.unparse(new['_s1_c6_mdq_g_pre_kernel'])
assert 'M = T * k' in producer and 'K_BRANCH=k' in producer and 'KTOP=k' in consumer and 'tl.arange(0, K_BRANCH)' in gq and 'K_BRANCH_PAD' not in gq
assert ' // KTOP' in md and 'M = int(order.shape[0])' in consumer
assert 'num_stages=3' in consumer and 'num_stages=2' in md
for n in helpers:assert ast.dump(old[n])==ast.dump(new[n])
report={'sha256':hashlib.sha256(out.encode()).hexdigest(),'parent_sha':'3f95092adf1225f59b5e29a5bc7ad56000ca65dfeda02dd882ab6ded17c9db9a','delta':'two shape predicates add complete c3 only','inverse_ast':'passed','all_helper_ast':'unchanged','bindings':calls,'k4_contract':'M=T*k/branch stride true k/MD KTOP=k; T16384 M65536; metadata untouched','cpu_contract':'reuse expanded_checks.json actual c3 T16384/E32/k4 FP32/INV','compile':'passed','c3_new_risks':['pre_nf TMA-A -> ORDER gather/SIMT A','launch4/maxnreg232 -> launch3/no maxnreg','I runtime -> constexpr'],'c3_math':'both f16x2 tanh and h*th+h; both full-tile TMA ACT and tail pointer','limits':'No GPU/OJ/AC or speed claim; not a pure c6 isomorphic extension'}
(OUT/'p1_s1_c356_unified.py').write_text(out);(OUT/'c356_unified.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile='p1_s1_c56_unified.py',tofile='p1_s1_c356_unified.py')));(OUT/'c356_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report['sha256'])
