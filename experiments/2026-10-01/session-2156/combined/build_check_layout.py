from pathlib import Path
import ast,hashlib,difflib,json,py_compile
D=Path(__file__).resolve().parent;parent=D/'p1_c356_unified_c4_flatten.py';s=parent.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='79549c5bc0151c940d72ee776cdc571f78edf4ab9740fd0a3f6d0d1fb870bd5e'
original=Path('/Users/sakimi/Desktop/xpuoj-p1/experiments/2026-10-01/session-1654/s2/p1_s2_c4_layout_only.py').read_text();assert hashlib.sha256(original.encode()).hexdigest()=='ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8'
names={'_fgs_t1i_mdq_kernel_g_s2_pad','_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad'};bt=ast.parse(s);lines=s.splitlines(True)
for f in bt.body:
 if not isinstance(f,ast.FunctionDef) or f.name not in names:continue
 loop=next(n for n in f.body if isinstance(n,ast.For));assert [(k.arg,ast.literal_eval(k.value)) for k in loop.iter.keywords]==[('flatten',True)];assert lines[loop.lineno-1].count('flatten=True')==1;lines[loop.lineno-1]=lines[loop.lineno-1].replace('flatten=True','num_stages=2')
out=''.join(lines);nt=ast.parse(out);fn=lambda t:{n.name:n for n in t.body if isinstance(n,ast.FunctionDef)};bf,nf,of=fn(bt),fn(nt),fn(ast.parse(original));assert {n for n in bf if ast.dump(bf[n])!=ast.dump(nf[n])}==names
for n in names:assert ast.dump(nf[n])==ast.dump(of[n])
for n in bf:
 if n not in names:assert ast.dump(bf[n])==ast.dump(nf[n])
p=D/'p1_c356_unified_c4_layout.py';p.write_text(out);py_compile.compile(str(p),doraise=True)
sha=hashlib.sha256(out.encode()).hexdigest();(D/'layout_vs_flatten.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),out.splitlines(True),fromfile=parent.name,tofile=p.name)));(D/'layout_checks.json').write_text(json.dumps({'sha256':sha,'parent_sha256':hashlib.sha256(s.encode()).hexdigest(),'changed_helpers':sorted(names),'changed_helpers_ast':'exact ac338 original S2','other_functions_run':'unchanged combined AST','pycompile':'passed','CPU':'fully reuse parent contracts','online':'unverified'},indent=2)+'\n');print(sha)
