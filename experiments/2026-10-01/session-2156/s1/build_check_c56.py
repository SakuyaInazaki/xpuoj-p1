from pathlib import Path
import ast,hashlib,difflib,json
OUT=Path(__file__).resolve().parent;s=(OUT/'p1_s1_c6_unified.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='befd19dcc7a0824ae49c93c91151e051ee6d3d4c8e8da17e2b1fdf40e9a02b68'
old='(T, H, E, I, k) == (8192, 3584, 64, 1024, 8)'
new='(T, H, E, I, k) in ((8192, 3584, 64, 2560, 8), (8192, 3584, 64, 1024, 8))'
assert s.count(old)==2;out=s.replace(old,new)
bt,nt=ast.parse(s),ast.parse(out);f=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)};bf,nf=f(bt),f(nt)
assert [n for n in bf if ast.dump(bf[n])!=ast.dump(nf[n])]==['_run_replicated']
assert ast.dump(ast.parse(out.replace(new,old)))==ast.dump(bt)
compile(out,str(OUT/'p1_s1_c56_unified.py'),'exec')
helpers={n for n in nf if n.startswith('_s1_c6_')};calls=[]
for a in ast.walk(nt):
 if not isinstance(a,ast.Call):continue
 target=a.func.value if isinstance(a.func,ast.Subscript) else a.func
 if not isinstance(target,ast.Name) or target.id not in helpers:continue
 params=[p.arg for p in nf[target.id].args.args];bound=params[:len(a.args)]+[k.arg for k in a.keywords if k.arg in params]
 assert len(bound)==len(set(bound)) and set(bound)==set(params)
 calls.append({'callee':target.id,'line':a.lineno})
assert len(calls)==5
shapes=[(16384,4096,8,8192,2),(16384,4096,8,14336,2),(16384,2048,32,2048,4),(16384,2048,32,1024,4),(8192,3584,64,2560,8),(8192,3584,64,1024,8),(16384,4096,96,2048,3),(16384,4096,96,1024,3),(4096,4096,256,2048,8),(4096,4096,256,1536,8),(65536,1024,32,1024,2),(65536,1024,32,2048,2)]
selected=set(shapes[4:6]);records=[]
for shape in shapes:
 for n in [1,2,3,5,6,100]:
  newhybrid=shape in selected and n>=3;oldhybrid=shape==shapes[5] and n>=3
  assert shape in selected or newhybrid==oldhybrid
  if shape in selected:records.append({'shape':shape,'n':n,'hybrid':newhybrid})
# Consumer remains the parent helper, stage3/outer2 and true k8.
md=ast.unparse(nf['_s1_c6_mdq_g_pre_kernel']);host=ast.unparse(nf['_s1_c6_mdq_g_pre_host']);gq=ast.unparse(nf['_s1_c6_gq_tok_params_kernel'])
assert 'num_stages=2' in md and 'num_stages=3' in host
assert 'tl.arange(0, K_BRANCH)' in gq and 'K_BRANCH_PAD' not in gq
assert all(k==8 and not(e==96 and i==1024) for t,h,e,i,k in selected)
report={'sha256':hashlib.sha256(out.encode()).hexdigest(),'parent':'befd19dcc7a0824ae49c93c91151e051ee6d3d4c8e8da17e2b1fdf40e9a02b68','only_change':'two complete-shape predicates c6 to c5+c6','inverse_ast':'passed','helper_ast':'unchanged','bindings':calls,'selected_predicate_table':records,'other_shape_checks':60,'stages':'original c5 MDg and pre_far outer2/launch3 equal target','cpu_contract':'reused expanded_checks.json actual c5 T8192/E64/k8 FP32/INV mappings','compile':'passed','limits':'no GPU/OJ/fullAC; >=6 oldA numerical and store path change'}
(OUT/'p1_s1_c56_unified.py').write_text(out);(OUT/'c56_unified.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile='p1_s1_c6_unified.py',tofile='p1_s1_c56_unified.py')));(OUT/'c56_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report['sha256'])
