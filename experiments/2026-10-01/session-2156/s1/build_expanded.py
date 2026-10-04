from pathlib import Path
import ast,hashlib,difflib
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1'); OUT=Path(__file__).resolve().parent
s=(ROOT/'experiments/2026-10-01/session-1654/s1/p1_s1_c6_v2_vector.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='d11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497'
def rep(a,b):
 global s
 if s.count(a)!=1:
  a='\n'+a+'\n'; b='\n'+b+'\n'
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
rep('_hybrid = bool(_GA[0] and (T, H, E, I, k) == (8192, 3584, 64, 1024, 8))','''_hybrid = bool(_GA[0] and (T, H, E, I, k) in (
                (16384, 2048, 32, 2048, 4),
                (8192, 3584, 64, 2560, 8),
                (8192, 3584, 64, 1024, 8),
                (16384, 4096, 96, 2048, 3),
            ))''')
rep('    BLOCK_H: tl.constexpr, K_BRANCH: tl.constexpr,','    BLOCK_H: tl.constexpr, K_BRANCH: tl.constexpr, K_BRANCH_PAD: tl.constexpr,')
rep('    j = tl.arange(0, K_BRANCH)','    j = tl.arange(0, K_BRANCH_PAD)\n    branch_mask = j < K_BRANCH')
rep('    dest = tl.load(INV + t * K_BRANCH + j)','    dest = tl.load(INV + t * K_BRANCH + j, mask=branch_mask, other=0)')
rep('    e = tl.load(FLAT_IDS + t * K_BRANCH + j)','    e = tl.load(FLAT_IDS + t * K_BRANCH + j, mask=branch_mask, other=0)')
rep('    w = tl.load(FLAT_W + t * K_BRANCH + j)','    w = tl.load(FLAT_W + t * K_BRANCH + j, mask=branch_mask, other=0.0)')
rep('    bn = tl.load(BNORM + e)','    bn = tl.load(BNORM + e, mask=branch_mask, other=1.0)')
rep('    tl.store(AH + dest, a * 0.5)','    tl.store(AH + dest, a * 0.5, mask=branch_mask)')
rep('    tl.store(WI + dest, (w * a) * inv)','    tl.store(WI + dest, (w * a) * inv, mask=branch_mask)')
rep('    tl.store(ACT_SCALE + dest, s)','    tl.store(ACT_SCALE + dest, s, mask=branch_mask)')
rep('        BLOCK_H=h_pad, K_BRANCH=k,','        BLOCK_H=h_pad, K_BRANCH=k, K_BRANCH_PAD=triton.next_power_of_2(k),')
# Names remain inherited to minimize unrelated textual changes; c6 helper now serves four explicit shapes.
ast.parse(s);compile(s,str(OUT/'p1_s1_c3567_vector.py'),'exec')
parent=(ROOT/'experiments/2026-10-01/session-1654/s1/p1_s1_c6_v2_vector.py').read_text()
(OUT/'p1_s1_c3567_vector.py').write_text(s)
(OUT/'expanded.diff').write_text(''.join(difflib.unified_diff(parent.splitlines(True),s.splitlines(True),fromfile='p1_s1_c6_v2_vector.py',tofile='p1_s1_c3567_vector.py')))
print(hashlib.sha256(s.encode()).hexdigest())
