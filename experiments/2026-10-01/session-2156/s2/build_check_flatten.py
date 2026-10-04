from pathlib import Path
import ast,copy,hashlib,difflib,json
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1');OUT=Path(__file__).resolve().parent;p=ROOT/'experiments/2026-10-01/session-1654/s2/p1_s2_c4_layout_only.py';s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8'
bt=ast.parse(s);names={'_fgs_t1i_mdq_kernel_g_s2_pad','_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad'};lines=s.splitlines(True)
for f in bt.body:
 if not isinstance(f,ast.FunctionDef) or f.name not in names:continue
 loop=next(n for n in f.body if isinstance(n,ast.For));assert isinstance(loop.iter,ast.Call)
 assert [(k.arg,ast.literal_eval(k.value)) for k in loop.iter.keywords]==[('num_stages',2)]
 assert lines[loop.lineno-1].count('num_stages=2')==1
 lines[loop.lineno-1]=lines[loop.lineno-1].replace('num_stages=2','flatten=True')
out=''.join(lines);nt=ast.parse(out);f=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)};bf,nf=f(bt),f(nt)
changed=[k for k in bf if ast.dump(bf[k])!=ast.dump(nf[k])];assert set(changed)==names
for name in names:
 a=next(n for n in nf[name].body if isinstance(n,ast.For));assert [(k.arg,ast.literal_eval(k.value)) for k in a.iter.keywords]==[('flatten',True)]
 a.iter.keywords=copy.deepcopy(next(n for n in bf[name].body if isinstance(n,ast.For)).iter.keywords)
assert ast.dump(bt)==ast.dump(nt)
nt=ast.parse(out);nf=f(nt);compile(out,str(OUT/'p1_s2_c4_flatten.py'),'exec')
helpers={k for k in nf if k not in f(ast.parse((ROOT/'experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py').read_text()))};calls=[]
for a in ast.walk(nt):
 if not isinstance(a,ast.Call):continue
 target=a.func.value if isinstance(a.func,ast.Subscript) else a.func
 if not isinstance(target,ast.Name) or target.id not in helpers:continue
 sig=nf[target.id].args;params=[p.arg for p in sig.args];assigned=set(params[:len(a.args)])
 for kw in a.keywords:
  if kw.arg is None or kw.arg in {'num_warps','num_stages','maxnreg'}:continue
  assert kw.arg in params and kw.arg not in assigned;assigned.add(kw.arg)
 assert set(params[:len(params)-len(sig.defaults)])<=assigned;calls.append({'name':target.id,'line':a.lineno})
assert len(calls)==6
for name in names:assert ast.unparse(nf[name].decorator_list[0])=='triton_dist.jit'
report={'sha256':hashlib.sha256(out.encode()).hexdigest(),'parent_sha256':hashlib.sha256(s.encode()).hexdigest(),'changes':changed,'inverse_ast':'full file identical after only two range keyword lists restored','bindings':calls,'compile':'passed','host_stages_math_layout':'unchanged AST','cpu_contract':'reuse session-1654/s2/contract_results.json 9 histograms/4157 tiles/524288 rows','limits':'no GPU/OJ/fullAC; flatten scheduling/overlap and benefit unproven'}
(OUT/'p1_s2_c4_flatten.py').write_text(out);(OUT/'flatten.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=p.name,tofile='p1_s2_c4_flatten.py')));(OUT/'flatten_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report['sha256'])
