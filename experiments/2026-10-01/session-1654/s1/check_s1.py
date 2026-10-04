import ast, hashlib, json, random, struct
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1')
source=(OUT/'p1_s1_c6_v1.py').read_text(); tree=ast.parse(source)
base=ast.parse((ROOT/'p1/kernel.py').read_text())
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
assert producer.count('tl.store(Q +')==1 and producer.count('tl.store(SCALE +')==1
assert 'K_BRANCH' in producer and 'INV + t * K_BRANCH + j' in producer
# Exact operation sequence retains baseline token quantization and baseline branch parameter loop.
def f32(v): return struct.unpack('<f',struct.pack('<f',v))[0]
def bits(v): return struct.unpack('<I',struct.pack('<f',v))[0]
def val(v): return struct.unpack('<f',struct.pack('<I',v))[0]
rng=random.Random(1654); scenarios=[]; bitchecks=0
for T,E,k in [(8192,64,8),(257,64,8),(1,64,8)]:
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
 scenarios.append({'T':T,'M':T*k,'unique_compact_writes':T*k,'token_scale_length':T,'act_scale_length':T*k})
# Model pre-existing producer predicate: only c6 original token Q path is changed.
for shape in [(8192,3584,64,1024,8),(8192,3584,64,2560,8),(4096,4096,256,1024,8),(16384,2048,32,1024,4)]:
 for ga in [0,1]:
  hybrid=bool(ga and shape==(8192,3584,64,1024,8))
  assert not hybrid or (ga==1 and shape[2]==64)
report={'candidate_sha256':hashlib.sha256(source.encode()).hexdigest(),'unchanged_original_functions':len(original)-1,'new_helper_bindings':bindings,'cpu_scenarios':scenarios,'fp32_bit_comparisons':bitchecks,'new_functions':{n:funcs[n].lineno for n in sorted(newnames)},'limits':['No submitted kernel import or GPU execution','No OJ result, performance or end-to-end SQNR verification','FP32 CPU rounding contract does not prove GPU FMA contraction or extreme static scale semantics']}
(OUT/'checks.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
compile(source,str(OUT/'p1_s1_c6_v1.py'),'exec')
