from pathlib import Path
import ast,hashlib,difflib,json
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1');OUT=Path(__file__).resolve().parent
p=ROOT/'experiments/2026-10-01/session-1654/s1/p1_s1_c6_v2_vector.py';s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='d11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497'
old='''        elif _dir_a:
            fp8_tokens_q, fp8_tokens_s, _dir_ah, _dir_wi, _dir_scl = _gq1p_tm_params(
                x, inv_order, flat_ids, flat_weights, _gu_bnorm(gu_q, gu_s), k
            )'''
new='''        elif _dir_a:
            if (T, H, E, I, k) == (8192, 3584, 64, 1024, 8):
                _hybrid = True
                fp8_tokens_q, fp8_tokens_s, _hybrid_ah, _hybrid_wi, _hybrid_scl = _s1_c6_gq_tok_params(
                    x, inv_order, flat_ids, flat_weights, _gu_bnorm(gu_q, gu_s), k
                )
            else:
                fp8_tokens_q, fp8_tokens_s, _dir_ah, _dir_wi, _dir_scl = _gq1p_tm_params(
                    x, inv_order, flat_ids, flat_weights, _gu_bnorm(gu_q, gu_s), k
                )'''
assert s.count(old)==1;out=s.replace(old,new)
bt,nt=ast.parse(s),ast.parse(out)
assert len(bt.body)==len(nt.body)
changed=[]
for lhs,rhs in zip(bt.body,nt.body):
 if ast.dump(lhs)!=ast.dump(rhs):
  assert isinstance(lhs,ast.FunctionDef) and lhs.name=='_run_replicated';changed.append(lhs.name)
assert changed==['_run_replicated']
f={n.name:n for n in nt.body if isinstance(n,ast.FunctionDef)}
# Undo only the new c6 producer branch and demand entire parent AST identity.
run=f['_run_replicated'];branch=next(n for n in ast.walk(run) if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='_dir_a' and n.body and isinstance(n.body[0],ast.If))
original=ast.parse(old.replace('        elif _dir_a:','if _dir_a:')).body[0]
branch.body=original.body
assert ast.dump(bt)==ast.dump(nt)
# Restore parsed candidate after reversible proof.
nt=ast.parse(out);f={n.name:n for n in nt.body if isinstance(n,ast.FunctionDef)}
compile(out,str(OUT/'p1_s1_c6_unified.py'),'exec')
helpernames={'_s1_c6_gq_tok_params','_s1_c6_gq_tok_params_kernel','_s1_c6_mdq_g_pre_host','_s1_c6_mdq_g_pre_kernel'};calls=[]
for a in ast.walk(nt):
 if not isinstance(a,ast.Call):continue
 z=a.func.value if isinstance(a.func,ast.Subscript) else a.func
 if not isinstance(z,ast.Name) or z.id not in helpernames:continue
 params=[q.arg for q in f[z.id].args.args];bound=params[:len(a.args)]+[q.arg for q in a.keywords if q.arg in params]
 assert len(bound)==len(set(bound)) and set(bound)==set(params)
 calls.append({'name':z.id,'line':a.lineno,'positional':len(a.args)})
assert len(calls)==5
# Simulate existing shape/phase predicates; this does not claim judge call ordinals.
shapes=[(16384,4096,8,8192,2),(16384,4096,8,14336,2),(16384,2048,32,2048,4),(16384,2048,32,1024,4),(8192,3584,64,2560,8),(8192,3584,64,1024,8),(16384,4096,96,2048,3),(16384,4096,96,1024,3),(4096,4096,256,2048,8),(4096,4096,256,1536,8),(65536,1024,32,1024,2),(65536,1024,32,2048,2)]
gaset=set(shapes[2:10]);a_dims={(1024,1024),(1024,2048),(2048,1024),(2048,2048),(3584,2560),(3584,1024),(4096,2048),(4096,1024)}
c6=(8192,3584,64,1024,8);rows=[]
for shape in shapes:
 for n in [1,2,3,5,6,100]:
  t,h,e,i,k=shape;ga=3<=n<=5 and shape in gaset;dira=not ga and n>=3 and (h,i) in a_dims
  before=bool(ga and shape==c6);after=before or bool(dira and shape==c6)
  assert shape==c6 or before==after
  assert not after or n>=3
  if shape==c6:
   assert after==(n>=3)
   rows.append({'n':n,'GA':ga,'dir_a':dira,'parent_hybrid':before,'unified_hybrid':after,'Q_rows':t if after else t*k,'consumer':'hybrid' if after else 'cold_original'})
report={'sha256':hashlib.sha256(out.encode()).hexdigest(),'parent_sha256':hashlib.sha256(s.encode()).hexdigest(),'changed_top_level_nodes':changed,'reversible_ast_diff':'exactly c6 producer branch within existing _dir_a','all_helpers_ast':'unchanged','bindings':calls,'c6_predicate_table':rows,'other_shapes_checks':11*6,'compile':'passed','cpu_contract':'reuse v2_vector_rechecked.json 67600 branch / 202800 FP32 bits; helpers unchanged','limits':['No GPU/OJ, no fullAC claim','Non-GA old pre_far f32 tanh changes to existing hybrid f16x2; not bit equivalent','No new CALLN/GA checks or phase boundaries']}
(OUT/'p1_s1_c6_unified.py').write_text(out)
(OUT/'unified.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=p.name,tofile='p1_s1_c6_unified.py')))
(OUT/'unified_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
