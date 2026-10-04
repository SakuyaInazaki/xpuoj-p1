import ast, hashlib, json, random, struct
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1')
source=(OUT/'p1_s1_c3567_vector.py').read_text(); tree=ast.parse(source)
base=ast.parse((ROOT/'experiments/2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py').read_text())
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
original={n.name:n for n in base.body if isinstance(n,ast.FunctionDef)}
for name,node in original.items():
 if name!='_run_replicated': assert ast.dump(node,include_attributes=False)==ast.dump(funcs[name],include_attributes=False),name
newnames=set(funcs)-set(original); assert len(newnames)==4
for name in newnames: assert sum(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body)==1
bindings=[]
for node in ast.walk(tree):
 if not isinstance(node,ast.Call): continue
 target=node.func
 if isinstance(target,ast.Subscript): target=target.value
 if isinstance(target,ast.Name) and target.id in newnames:
  fn=funcs[target.id]; params=[a.arg for a in fn.args.args]
  assert not any(isinstance(a,ast.Starred) for a in node.args)
  given=params[:len(node.args)]+[k.arg for k in node.keywords if k.arg in params]
  assert len(given)==len(set(given)),target.id
  assert set(given)==set(params),(target.id,set(params)-set(given))
  bindings.append({'callee':target.id,'line':node.lineno,'positional':len(node.args)})
md=ast.unparse(funcs['_s1_c6_mdq_g_pre_kernel']); host=ast.unparse(funcs['_s1_c6_mdq_g_pre_host'])
assert 'A_SCALE' not in md and 'BNORM' not in md and 'SCL' not in md
assert 'weights' not in host and 'return (act, act_scale)' in host
assert ' // KTOP' in md and 'AH + offs_m' in md and 'WI + offs_m' in md
producer=ast.unparse(funcs['_s1_c6_gq_tok_params_kernel'])
assert 'tl.arange(0, K_BRANCH_PAD)' in producer
assert not any(isinstance(n,ast.For) for n in ast.walk(funcs['_s1_c6_gq_tok_params_kernel']))
v1=ast.parse((ROOT/'experiments/2026-10-01/session-1654/s1/p1_s1_c6_v2_vector.py').read_text())
assert len(v1.body)==len(tree.body)
for lhs,rhs in zip(v1.body,tree.body):
 if isinstance(lhs,ast.FunctionDef) and lhs.name in {'_s1_c6_gq_tok_params_kernel','_s1_c6_gq_tok_params','_run_replicated'}:continue
 assert ast.dump(lhs)==ast.dump(rhs)
assert producer.count('tl.store(Q +')==1 and producer.count('tl.store(SCALE +')==1
assert 'K_BRANCH' in producer and 'INV + t * K_BRANCH + j' in producer
# Exact operation sequence retains baseline token quantization and baseline branch parameter loop.
def f32(v): return struct.unpack('<f',struct.pack('<f',v))[0]
def bits(v): return struct.unpack('<I',struct.pack('<f',v))[0]
def val(v): return struct.unpack('<f',struct.pack('<I',v))[0]
rng=random.Random(1654); scenarios=[]; bitchecks=0
for T,E,k in [(16384,32,4),(8192,64,8),(8192,64,8),(16384,96,3)]:
 ids=[e for t in range(T) for e in rng.sample(range(E),k)]
 weights=[f32(0 if b%73==0 else rng.random()) for b in range(T*k)]
 scales=[f32(2**rng.uniform(-28,-3)) for t in range(T)]
 norms=[f32(2**rng.uniform(-3,5)) for e in range(E)]
 order=sorted(range(T*k),key=lambda b:ids[b]); inv=[0]*(T*k)
 for row,b in enumerate(order): inv[b]=row
 produced=[None]*(T*k); writes=[0]*(T*k)
 for t in range(T):
  for j in range(k):
   b=t*k+j; row=inv[b]; a=scales[t]; w=weights[b]; bn=norms[ids[b]]
   bound=f32(f32(f32(f32(a*a)*bn)*bn)*abs(w)); bexp=(bits(max(bound,f32(1e-30)))>>23)&255
   produced[row]=(f32(a*.5),f32(f32(w*a)*val((244-bexp)<<23)),val((bexp+10)<<23)); writes[row]+=1
 assert all(w==1 for w in writes)
 for row,b in enumerate(order):
  t=b//k; a=scales[t]; w=weights[b]; bn=norms[ids[b]]
  bound=f32(a*a); bound=f32(bound*bn); bound=f32(bound*bn); bound=f32(bound*abs(w))
  bexp=(bits(max(bound,f32(1e-30)))>>23)&255
  expected=(f32(a*.5),f32(f32(w*a)*val((244-bexp)<<23)),val((bexp+10)<<23))
  assert tuple(map(bits,expected))==tuple(map(bits,produced[row])); assert order[inv[b]]//k==t
  bitchecks+=3
 # Batch branch gather/scatter oracle: eight lane values materialized together.
 vector_rows=[None]*(T*k)
 for t in range(T):
  padded=1<<(k-1).bit_length()
  branches=[t*k+j for j in range(padded) if j<k]
  assert len(branches)==k
  invalid=[j for j in range(padded) if j>=k]
  assert all(j>=k for j in invalid)
  destinations=[inv[b] for b in branches]
  lanes=[]
  for b in branches:
   a=scales[t]; w=weights[b]; bn=norms[ids[b]]
   bound=f32(f32(f32(f32(a*a)*bn)*bn)*abs(w)); exp=(bits(max(bound,f32(1e-30)))>>23)&255
   lanes.append((f32(a*.5),f32(f32(w*a)*val((244-exp)<<23)),val((exp+10)<<23)))
  assert len(set(destinations))==k
  for r,value in zip(destinations,lanes): vector_rows[r]=value
 assert all(tuple(map(bits,a))==tuple(map(bits,b)) for a,b in zip(vector_rows,produced))
 scenarios.append({'T':T,'M':T*k,'unique_compact_writes':T*k,'token_scale_length':T,'act_scale_length':T*k})
# Exact enabled shape set and masks, plus original stage contracts.
expected={(16384,2048,32,2048,4),(8192,3584,64,2560,8),(8192,3584,64,1024,8),(16384,4096,96,2048,3)}
run=funcs['_run_replicated']
assignment=next(n for n in ast.walk(run) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_hybrid' for t in n.targets) and isinstance(n.value,ast.Call))
assert ast.unparse(assignment.value).startswith('bool(_GA[0] and ')
comparison=next(n for n in ast.walk(assignment.value) if isinstance(n,ast.Compare))
assert set(ast.literal_eval(comparison.comparators[0]))==expected
for call in ast.walk(funcs['_s1_c6_gq_tok_params_kernel']):
 if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Attribute) or call.func.attr not in ('load','store'):continue
 addr=ast.unparse(call.args[0])
 if any(addr.startswith(x+' +') for x in ('INV','FLAT_IDS','FLAT_W','BNORM','AH','WI','ACT_SCALE')):
  assert any(k.arg=='mask' and ast.unparse(k.value)=='branch_mask' for k in call.keywords),addr
for name in ('_s1_c6_gq_tok_params_kernel','_s1_c6_mdq_g_pre_kernel'):
 assert ast.unparse(funcs[name].decorator_list[0])=='triton_dist.jit'
# Parent gather consumer/host AST already identical; each selected original _gg predicate false.
assert all(not (e==96 and i==1024) for t,h,e,i,k in expected)
assert 'num_stages=3' in host and 'num_warps=8' in host
assert 'num_stages=2' in md
report={'candidate_sha256':hashlib.sha256(source.encode()).hexdigest(),'unchanged_original_functions':len(original)-1,'new_helper_bindings':bindings,'cpu_scenarios':scenarios,'fp32_bit_comparisons':bitchecks,'new_functions':{n:funcs[n].lineno for n in sorted(newnames)},'limits':['No submitted kernel import or GPU execution','No OJ result, performance or end-to-end SQNR verification','FP32 CPU rounding contract does not prove GPU FMA contraction or extreme static scale semantics']}
(OUT/'expanded_checks.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
compile(source,str(OUT/'p1_s1_c3567_vector.py'),'exec')
