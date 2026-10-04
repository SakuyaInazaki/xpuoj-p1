from pathlib import Path
import ast,copy,hashlib,json,random,py_compile
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
base=(ROOT/'p1/kernel.py').read_text()
p=OUT/'p1_s2_c4_layout_only.py'
s=p.read_text()
py_compile.compile(str(p),doraise=True)
old=ast.parse(base); new=ast.parse(s)
def defs(t): return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
o=defs(old); n=defs(new)
def dump(a): return [dump(v) for v in a] if isinstance(a,list) else ast.dump(a,include_attributes=False)
changed=[name for name in o if dump(o[name])!=dump(n[name])]
assert changed==['_run_replicated'],changed
helpers=[name for name in n if name not in o]
assert len(helpers)==6 and all('s2_' in name for name in helpers)
assert len(new.body)==len(old.body)+6
for name in helpers:
 assert len([x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name])==1
# Both math bodies are identical through q; compact scales and DROP remain identical.
for a,b in [('_fgs_t1i_mdq_kernel_g','_fgs_t1i_mdq_kernel_g_s2_pad'),('_fgs_t1i_mdq_tma_pre_nf_kernel','_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad')]:
 x=copy.deepcopy(o[a]); y=copy.deepcopy(n[b]); y.name=x.name
 assert dump(x.args)==dump(y.args) and dump(x.decorator_list)==dump(y.decorator_list)
 lx=next(z for z in x.body if isinstance(z,ast.For)); ly=next(z for z in y.body if isinstance(z,ast.For))
 assert dump(lx.iter)==dump(ly.iter)
 ix=next(i for i,z in enumerate(lx.body) if isinstance(z,ast.If))
 iy=next(i for i,z in enumerate(ly.body) if isinstance(z,ast.Assign) and isinstance(z.targets[0],ast.Name) and z.targets[0].id=='padded_row')
 assert dump(lx.body[:ix])==dump(ly.body[:iy])
 assert dump(lx.body[ix+1:])==dump(ly.body[iy+2:])
 assert 'ACT_DESC.store([padded_row, pid_n * BLOCK_N], q)' in ast.get_source_segment(s,n[b])
# DN differs only in ACT read row expression; scales, output and math are unchanged.
x=copy.deepcopy(o['_dn_tma2_f8_pad_static_kernel']); y=copy.deepcopy(n['_dn_tma2_f8_pad_static_kernel_s2_input_pad']); y.name=x.name
for node in ast.walk(y):
 if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='a_row':
  node.value=ast.parse('row_begin + local_m * BLOCK_M',mode='eval').body
assert dump(x)==dump(y)
# Every helper call meets positional/keyword argument contract.
checks=[]
for node in ast.walk(new):
 if not isinstance(node,ast.Call): continue
 f=node.func
 if isinstance(f,ast.Subscript): f=f.value
 if not isinstance(f,ast.Name) or f.id not in helpers: continue
 sig=n[f.id].args; args=[a.arg for a in sig.args]
 assert len(node.args)<=len(args)
 assigned=set(args[:len(node.args)])
 for kw in node.keywords:
  if kw.arg is None or kw.arg in ('num_warps','num_stages','maxnreg'): continue
  assert kw.arg in args and kw.arg not in assigned,(f.id,kw.arg)
  assigned.add(kw.arg)
 required=set(args[:len(args)-len(sig.defaults)])
 assert required<=assigned,(f.id,required-assigned)
 checks.append({'callee':f.id,'line':node.lineno,'positional':len(node.args)})
assert len(checks)==6,checks
# Ensure bool cannot be set by shape alone or an old producer.
run=n['_run_replicated']; flagwrites=[]
for z in ast.walk(run):
 if isinstance(z,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='act_is_padded' for t in z.targets for v in ast.walk(t)):
  flagwrites.append(ast.get_source_segment(s,z))
assert len(flagwrites)==3 and flagwrites[0]=='act_is_padded = False'
assert all('s2_pad' in v for v in flagwrites[1:])
# Launch keywords and arguments are copied without numerical or pipeline changes.
for oldhost,newhost,oldkernel,newkernel in [
 ('_fgs_tma1_intq_host','_fgs_tma1_intq_host_g_s2_pad','_fgs_t1i_mdq_kernel_g','_fgs_t1i_mdq_kernel_g_s2_pad'),
 ('_fgs_tma1_intq_host_pre_nf','_fgs_tma1_intq_host_pre_nf_s2_pad','_fgs_t1i_mdq_tma_pre_nf_kernel','_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad'),
 ('_dn_tma2_f8_pad_static_host','_dn_tma2_f8_pad_static_host_s2_input_pad','_dn_tma2_f8_pad_static_kernel','_dn_tma2_f8_pad_static_kernel_s2_input_pad')]:
 def kernelcall(fn,name):
  return next(z for z in ast.walk(fn) if isinstance(z,ast.Call) and isinstance(z.func,ast.Subscript) and isinstance(z.func.value,ast.Name) and z.func.value.id==name)
 ca=copy.deepcopy(kernelcall(o[oldhost],oldkernel)); cb=copy.deepcopy(kernelcall(n[newhost],newkernel)); cb.func.value.id=ca.func.value.id
 assert dump(ca)==dump(cb)
def swizzle(i,j,size_i,size_j,group_i=32):
 ij=i*size_j+j; group_size=group_i*size_j; group=ij//group_size
 first_i=group*group_i; current_i=min(size_i-first_i,group_i)
 return first_i+(ij%group_size)%current_i,(ij%group_size)//current_i
# Nine histogram contracts: all empty, both concentration endpoints, exact tile,
# 127/128/129 boundaries, many tails, empty middle experts, uneven dense and skew.
rng=random.Random(20261001)
sets=[('empty',[0]*32),('all_first',[65536]+[0]*31),('all_last',[0]*31+[65536]),('even',[2048]*32),
 ('boundary',[127,128,129]+[0]*28+[65536-384]),('all_tails',[1]*31+[65536-31]),
 ('sparse',[32767]+[0]*15+[129]+[0]*14+[32640])]
weights=[rng.randrange(0,1000) for _ in range(32)]
a=[65536*w//sum(weights) for w in weights]; a[-1]+=65536-sum(a);sets.append(('random_dense',a))
a=[127,128,129]*10+[0,0];a[-1]=65536-sum(a);sets.append(('many_boundaries',a))
valid_total=0;tile_total=0; stats=[]
for label,counts in sets:
 M=sum(counts); E=len(counts); P=128*((M+127*E+127)//128)
 nt=[(c+127)//128 for c in counts]; tile_total+=sum(nt)
 compact_to_pad={}; pads=set(); touched=set(); compact=0; tile_start=0
 for e,c in enumerate(counts):
  if c==0: assert nt[e]==0
  # Swizzle remains a bijection over all row tiles and all 128-column ACT tiles;
  # the persistent 132-program striding enumerates each unswizzled tile once.
  raw=[v for pid in range(132) for v in range(pid,nt[e]*8,132)]
  assert sorted(raw)==list(range(nt[e]*8))
  swizzled={swizzle(v//8,v%8,nt[e],8) for v in raw}
  assert swizzled=={(i,j) for i in range(nt[e]) for j in range(8)}
  for tile in range(nt[e]):
   compact_row=compact+tile*128; padded_row=(tile_start+tile)*128
   full=set(range(padded_row,padded_row+128))
   assert not full&touched; touched|=full; assert max(full)<P
   # DN consumes exactly these physical ACT rows but its A_SCALE remains compact.
   for lane in range(min(128,c-tile*128)):
    cr=compact_row+lane; pr=padded_row+lane
    assert cr not in compact_to_pad and pr not in pads
    compact_to_pad[cr]=pr; pads.add(pr)
    assert cr<compact+c and pr==tile_start*128+(cr-compact)
    # symbolic scale[e,local_row] pairs to the same original compact row.
    assert divmod(pr-tile_start*128,128)==divmod(cr-compact,128)
  compact+=c;tile_start+=nt[e]
 assert set(compact_to_pad)==set(range(M)) and len(pads)==M
 assert len(touched)==128*sum(nt) and len(touched)<=P
 # INV_PAD maps each original branch using compact inverse + expert delta.
 order=list(range(M)); rng.shuffle(order)
 inv=[0]*M
 for cr,branch in enumerate(order):inv[branch]=cr
 inv_pad=[compact_to_pad[cr] for cr in inv]
 assert len(set(inv_pad))==M
 for cr,branch in enumerate(order):assert inv_pad[branch]==compact_to_pad[cr]
 valid_total+=M;stats.append({'label':label,'M':M,'E':E,'capacity_P':P,'tiles':sum(nt),'effective_rows':M})
assert stats[1]['capacity_P']==69632
result={'source_sha256':hashlib.sha256(base.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(s.encode()).hexdigest(),'changed_existing_definitions':changed,'new_helpers':{h:n[h].lineno for h in helpers},'calls':checks,'cpu_histograms':stats,'full_tiles_nonoverlapping':tile_total,'valid_rows_checked':valid_total,'python_compile':'passed','kernel_math_and_existing_helpers':'AST passed','limitations':'No GPU/TMA/SQNR/performance verification; OJ required.'}
(OUT/'contract_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
