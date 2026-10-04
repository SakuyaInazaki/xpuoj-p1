from pathlib import Path
import ast,hashlib,difflib,json,copy,py_compile
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1'); OUT=Path(__file__).resolve().parent
base=ROOT/'experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py'; p1=ROOT/'experiments/2026-10-01/session-2156/s1/p1_s1_c356_unified.py';p2=ROOT/'experiments/2026-10-01/session-2156/s2/p1_s2_c4_flatten.py'
b,s1,s2=[p.read_text() for p in (base,p1,p2)]
assert hashlib.sha256(s1.encode()).hexdigest()=='75c702478afa8c7d282064fad23d6a542556f30593cae8cb41a201c38da0ea7a'
assert hashlib.sha256(s2.encode()).hexdigest()=='c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b'
def funcs(s): return {n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
bf,a1,a2=map(funcs,(b,s1,s2))
def text(s,f): return ''.join(s.splitlines(True)[f.lineno-1:f.end_lineno])
br,r1,r2=[text(s,f['_run_replicated']) for s,f in ((b,bf),(s1,a1),(s2,a2))]
bl,sl=br.splitlines(True),r2.splitlines(True); changes=[]; merged=r1
for tag,i,j,k,l in difflib.SequenceMatcher(None,bl,sl,autojunk=False).get_opcodes():
 if tag=='equal': continue
 
 if tag=='insert': i-=1;k-=1
 old,new=''.join(bl[i:j]),''.join(sl[k:l]);assert old and merged.count(old)==1,(tag,old[:100]);merged=merged.replace(old,new,1);changes.append((old,new))
out=s1.replace(r1,merged,1)
added1=set(a1)-set(bf);added2=set(a2)-set(bf);assert not added1&added2
for name,f in a2.items():
 if name not in added2:continue
 start=min([f.lineno]+[d.lineno for d in f.decorator_list]);out+='\n\n'+''.join(s2.splitlines(True)[start-1:f.end_lineno])+'\n'
nf=funcs(out)
for name in bf:
 if name!='_run_replicated':assert ast.dump(nf[name])==ast.dump(bf[name]),name
for name in added1:assert ast.dump(nf[name])==ast.dump(a1[name])
for name in added2:assert ast.dump(nf[name])==ast.dump(a2[name])
restored=text(out,nf['_run_replicated'])
for old,new in reversed(changes):assert restored.count(new)==1;restored=restored.replace(new,old,1)
assert ast.dump(ast.parse(restored))==ast.dump(ast.parse(r1))
# Every added helper call is fully bound, including launch signature and required arguments.
calls=[]
for c in ast.walk(ast.parse(out)):
 if not isinstance(c,ast.Call):continue
 target=c.func.value if isinstance(c.func,ast.Subscript) else c.func
 if not isinstance(target,ast.Name) or target.id not in added1|added2:continue
 sig=nf[target.id].args;params=[x.arg for x in sig.args];assert len(c.args)<=len(params);assigned=set(params[:len(c.args)])
 for kw in c.keywords:
  if kw.arg is None:
   assert isinstance(kw.value,ast.IfExp) and isinstance(kw.value.body,ast.Dict) and [ast.literal_eval(x) for x in kw.value.body.keys]==['maxnreg'];continue
  if kw.arg in {'num_warps','num_stages','maxnreg'}:continue
  assert kw.arg in params and kw.arg not in assigned,(target.id,kw.arg,params);assigned.add(kw.arg)
 assert set(params[:len(params)-len(sig.defaults)])<=assigned;calls.append([target.id,c.lineno])
for name in added1|added2:
 if name.endswith('_kernel') or '_kernel_' in name:assert ast.unparse(nf[name].decorator_list[0])=='triton_dist.jit',name
assert '_hybrid = False' in merged and 'act_is_padded = False' in merged
shapes=[(16384,2048,32,2048,4),(16384,2048,32,1024,4),(8192,3584,64,2560,8),(8192,3584,64,1024,8)]
hybrid=set((shapes[0],shapes[2],shapes[3]));pad=shapes[1]
assert not (hybrid&{pad})
# The two hybrid predicates preserve parent AST exactly; S2 gate likewise.
assert merged.count('(16384, 2048, 32, 2048, 4), (8192, 3584, 64, 2560, 8), (8192, 3584, 64, 1024, 8)')==2
assert '(T, H, E, I, k) == (16384, 2048, 32, 1024, 4)' in merged
path=OUT/'p1_c356_unified_c4_flatten.py';path.write_text(out);py_compile.compile(str(path),doraise=True)
report={'sha256':hashlib.sha256(out.encode()).hexdigest(),'parents':[hashlib.sha256(x.encode()).hexdigest() for x in (s1,s2)],'base_sha256':hashlib.sha256(b.encode()).hexdigest(),'merge_hunks':len(changes),'helper_sets':[sorted(added1),sorted(added2)],'all_original_functions_except_run':'AST identical to 08dd base','all_added_helpers':'AST identical to respective frozen parent','run_inverse':'restoring only S2 wiring hunks yields exact C356U run AST','bindings':calls,'pycompile':'passed','dispatch':[{'shape':list(x),'GA3_5':'hybrid compact ACT' if x in hybrid else 'S2 padded ACT','dirA6plus':'hybrid compact ACT' if x in hybrid else 'S2 padded ACT','cold1_2':'original compact ACT fallback','DN':'original compact-input DN' if x in hybrid else 'S2 padded-input DN only when actual producer flag true'} for x in shapes],'CPU':'reuse parent c3/c5/c6 branch FP32 contracts and S2 layout contracts; no new batch','unverified':'combined online compilation/fullAC/performance; c3 stage/resource switch and hybrid numerical changes remain parent risks'}
(OUT/'combined_checks.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'combined_vs_c356.diff').write_text(''.join(difflib.unified_diff(s1.splitlines(True),out.splitlines(True),fromfile=p1.name,tofile=path.name)));print(json.dumps(report,indent=2))
