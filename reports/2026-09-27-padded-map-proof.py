#!/usr/bin/env python3
"""CPU property checks for compact-A / padded-DN / direct INV_PAD.

No torch, Triton, credentials, GPU, or submitted kernel is imported.
This validates only integer layout/scheduling and branch association/order;
it does NOT validate TMA visibility, device bounds, FP8 math, or performance.
The swizzle formula follows Triton v3.4.0 language/standard.py:72-106.
"""
from collections import Counter
import json
import random
import struct

BM = 128
KNOWN = [
    (16384,4096,8,8192,2), (16384,4096,8,14336,2),
    (16384,2048,32,2048,4), (16384,2048,32,1024,4),
    (8192,3584,64,2560,8), (8192,3584,64,1024,8),
    (16384,4096,96,2048,3), (16384,4096,96,1024,3),
    (4096,4096,256,2048,8), (4096,4096,256,1536,8),
    (65536,1024,32,1024,2), (65536,1024,32,2048,2),
]


def swizzle(i,j,ni,nj,g):
    z=i*nj+j
    base=(z//(g*nj))*g
    gg=min(ni-base,g)
    z%=g*nj
    return base+z%gg,z//gg


def f32(x):
    return struct.unpack('<f',struct.pack('<f',x))[0]


def verify(ids,E,k,nchunks=8,gm=32):
    M=len(ids)
    assert M%k==0 and all(0<=e<E for e in ids)
    buckets=[[] for _ in range(E)]
    for src,e in enumerate(ids): buckets[e].append(src)
    counts=list(map(len,buckets))
    order=[src for b in buckets for src in b]
    inv=[0]*M
    for row,src in enumerate(order):inv[src]=row
    rb=[];tb=[];r=t=0
    for n in counts:
        rb.append(r);tb.append(t);r+=n;t+=(n+127)//128
    actual_capacity=t*BM
    allocated_capacity=((M+BM-1)//BM+E)*BM
    assert actual_capacity<=allocated_capacity
    expected=[0]*M
    for e,b in enumerate(buckets):
        for local,src in enumerate(b):expected[src]=tb[e]*BM+local
    assert len(set(expected))==M
    assert all(0<=v<actual_capacity for v in expected)
    # PAD_DELTA[e] in fused offsets + sort scatter formulation.
    from_sort=[inv[src]+tb[ids[src]]*BM-rb[ids[src]] for src in range(M)]
    assert from_sort==expected
    # MD writes only after swizzle, and only the unique N tile numbered 0.
    from_md=[None]*M;writes=Counter()
    for e,n in enumerate(counts):
        nt=(n+BM-1)//BM
        seen=set()
        for im in range(nt):
            for jn in range(nchunks):
                lm,pn=swizzle(im,jn,nt,nchunks,gm)
                assert 0<=lm<nt and 0<=pn<nchunks
                assert (lm,pn) not in seen
                seen.add((lm,pn))
                if pn!=0:continue
                for lane in range(BM):
                    local=lm*BM+lane
                    if local>=n:continue
                    row=rb[e]+local
                    src=order[row]
                    pad=(tb[e]+lm)*BM+lane
                    from_md[src]=pad;writes[src]+=1
        assert len(seen)==nt*nchunks
    assert from_md==expected
    assert len(writes)==M and all(v==1 for v in writes.values())
    # Padded holes contain poison; compact / padded arrays hold unique IDs.
    padded=[None]*actual_capacity
    for src,pad in enumerate(expected):padded[pad]=src
    assert sum(v is None for v in padded)==actual_capacity-M
    # Read the same branch values/scales in j=0..k-1 order. Include values
    # whose FP32 sum is sensitive to order, and test every token.
    vals=(1e10,1.0,-1e10,0.125,-0.125,3.0,-3.0,1e-10)
    for token in range(M//k):
        old_acc=new_acc=0.0
        old_seq=[];new_seq=[]
        for j in range(k):
            src=token*k+j
            old_src=order[inv[src]]
            new_src=padded[from_md[src]]
            assert new_src is not None
            old_seq.append(old_src);new_seq.append(new_src)
            # Separate row/chunk scales would be stored at the same pad row.
            for chunk in (0,3):
                assert (old_src,chunk)==(new_src,chunk)
            old_acc=f32(old_acc+f32(vals[old_src%len(vals)]))
            new_acc=f32(new_acc+f32(vals[new_src%len(vals)]))
        assert old_seq==new_seq==list(range(token*k,(token+1)*k))
        assert struct.pack('<f',old_acc)==struct.pack('<f',new_acc)
    return dict(rows=M,experts=E,k=k,tiles=t,padded_rows=actual_capacity,
                pad_holes=actual_capacity-M,allocated_capacity=allocated_capacity,
                original_tokens=M//k,wrong_host_inferred_tokens=allocated_capacity//k)


def check_bad_preswizzle_guard():
    # Wrongly guarding original N coordinate instead of swizzled N loses
    # some rows and duplicates other rows, even in this tiny configuration.
    ni,nj,g=4,8,32
    selected=[swizzle(i,0,ni,nj,g)[0] for i in range(ni)]
    assert Counter(selected)!=Counter(range(ni))
    return selected


def main():
    rng=random.Random(20260927)
    results=[]
    # Each listed boundary appears with an empty expert; adversarial skew.
    boundary_sets=[[],[0,0,0],[0,1,127,128,129,0],[0,16384,0,0],
                   [1,127,128,129,255,256,257,0]]
    for ix,counts in enumerate(boundary_sets):
        E=max(1,len(counts));ids=[e for e,n in enumerate(counts) for _ in range(n)]
        rng.shuffle(ids)
        results.append({'case':'boundary-'+str(ix),**verify(ids,E,1)})
    for case,(T,H,E,I,k) in enumerate(KNOWN,1):
        # Mapping is tested on an arbitrary legal expert-ID range; duplicate
        # experts/token are included as a superset of the router contract.
        ids=[rng.randrange(E) for _ in range(T*k)]
        results.append({'case':'shape-c'+str(case),**verify(ids,E,k,I//128,8 if H<=1024 else 32)})
    for ix in range(80):
        E=rng.choice([1,2,8,32,64,96,256]);k=rng.choice([1,2,3,4,8])
        M=k*rng.randrange(0,1000)
        hot=rng.randrange(E)
        ids=[hot if rng.random()<0.92 else rng.randrange(E) for _ in range(M)]
        results.append({'case':'skew-'+str(ix),**verify(ids,E,k,rng.choice([1,8,16,20]),rng.choice([4,8,16,32]))})
    bad=check_bad_preswizzle_guard()
    print(json.dumps({'status':'PASS','checks':len(results),'total_rows':sum(x['rows'] for x in results),
                      'wrong_preswizzle_guard_detected':True,'wrong_guard_rows':bad,
                      'padded_shape_must_not_define_T':True,
                      'host_T_counterexamples':sum(x['wrong_host_inferred_tokens']!=x['original_tokens'] for x in results),
                      'scope':'CPU mapping, unique writers, capacity, identical branch/scale addressing and FP32 addition order only',
                      'cases':results},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
