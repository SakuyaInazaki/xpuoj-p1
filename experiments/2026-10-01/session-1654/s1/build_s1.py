import ast, hashlib, difflib
from pathlib import Path
ROOT=Path('/Users/sakimi/Desktop/xpuoj-p1')
OUT=Path(__file__).resolve().parent
src=(ROOT/'p1/kernel.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9'
fs={n.name:n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef)}
def extract(name):
 n=fs[name]; return '\n'.join(src.splitlines()[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno])+'\n'
def replace_once(s,a,b):
 assert s.count(a)==1,(a,s.count(a)); return s.replace(a,b)
# New GQ keeps token quantization expression and launch configuration unchanged.
gq=extract('_gq1p_tok_kernel').replace('_gq1p_tok_kernel','_s1_c6_gq_tok_params_kernel')
gq=replace_once(gq,'    X, Q, SCALE,','    X, Q, SCALE, INV, FLAT_IDS, FLAT_W, BNORM, AH, WI, ACT_SCALE,')
gq=replace_once(gq,'    BLOCK_H: tl.constexpr,','    BLOCK_H: tl.constexpr, K_BRANCH: tl.constexpr,')
params=extract('_gq1p_tm_params_kernel')
body=params[params.index('    for j in tl.static_range(K_BRANCH):'):]
body=body.replace('        tl.store(Q + dest * stride_qm + offs_h * stride_qh, q, mask=h_mask)\n','').replace('        tl.store(SCALE + dest, scale)\n','')
gq+=body
host=extract('_gq1p_tok').replace('_gq1p_tok','_s1_c6_gq_tok_params')
host=replace_once(host,'def _s1_c6_gq_tok_params(x):','def _s1_c6_gq_tok_params(x, inv_order, flat_ids, flat_weights, bnorm, k):')
host=replace_once(host,'    h_pad = triton.next_power_of_2(H)','    M = T * k\n    ah = torch.empty(M, dtype=torch.float32, device=x.device)\n    wi = torch.empty(M, dtype=torch.float32, device=x.device)\n    act_scale = torch.empty(M, dtype=torch.float32, device=x.device)\n    h_pad = triton.next_power_of_2(H)')
host=host.replace('        x, q, scale, T, H,','        x, q, scale, inv_order, flat_ids, flat_weights, bnorm, ah, wi, act_scale, T, H,').replace('BLOCK_H=h_pad,','BLOCK_H=h_pad, K_BRANCH=k,').replace('    return q, scale','    return q, scale, ah, wi, act_scale')
md=extract('_fgs_t1i_mdq_kernel_g').replace('_fgs_t1i_mdq_kernel_g','_s1_c6_mdq_g_pre_kernel')
md=replace_once(md,'    A, A_SCALE, BNORM, B_DESC, B_SCALE, W, ORDER, ACT, ACT_DESC, SCL,','    A, AH, WI, B_DESC, B_SCALE, ORDER, ACT, ACT_DESC,')
md=md.replace('        a_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)\n','').replace('        w = tl.load(W + offs_m, mask=row_mask, other=0.0).to(tl.float32)\n','')
md=replace_once(md,'        ah = a_scale * 0.5','        ah = tl.load(AH + offs_m, mask=row_mask, other=0.0)')
start=md.index('        bn = tl.load(BNORM + expert)'); end=md.index('        q = (silu * u * wi[:, None])',start)
md=md[:start]+'        wi = tl.load(WI + offs_m, mask=row_mask, other=0.0)\n'+md[end:]
md=md.replace('        tl.store(SCL + offs_m, s, mask=row_mask)\n','')
# Remove legacy explanatory comments rather than retain inaccurate descriptions.
md='\n'.join(line for line in md.splitlines() if not line.lstrip().startswith('#'))+'\n'
mdhost='''def _s1_c6_mdq_g_pre_host(a_q, act_scale, ah, wi, b_q_int, b_s_int, order, expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles, k):
    M = int(order.shape[0])
    K = a_q.shape[1]
    G, N2, K2 = b_q_int.shape
    I = N2 // 2
    act = torch.empty((M, I), dtype=torch.float8_e4m3fn, device=a_q.device)
    b_flat = b_q_int.view(G * N2, K2)
    b_desc = TensorDescriptor(b_flat, b_flat.shape, b_flat.stride(), [256, 128])
    act_desc = TensorDescriptor(act, act.shape, act.stride(), [128, 128])
    _s1_c6_mdq_g_pre_kernel[(132,)](
        a_q, ah, wi, b_desc, b_s_int, order, act, act_desc,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M, I, K,
        a_q.stride(0), a_q.stride(1), act.stride(0), act.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8 if K <= 1024 else 32, KTOP=k,
        num_warps=8, num_stages=3,
    )
    return act, act_scale
'''
new=replace_once(src,'    _dir_ah = None','    _hybrid = False\n    _hybrid_ah = None\n    _hybrid_wi = None\n    _hybrid_scl = None\n    _dir_ah = None')
new=replace_once(new,'        elif _GA[0] or (E == 256 and _FL[0]):\n            fp8_tokens_q, fp8_tokens_s = _gq1p_tok(x)','''        elif _GA[0] or (E == 256 and _FL[0]):
            _hybrid = bool(_GA[0] and (T, H, E, I, k) == (8192, 3584, 64, 1024, 8))
            if _hybrid:
                fp8_tokens_q, fp8_tokens_s, _hybrid_ah, _hybrid_wi, _hybrid_scl = _s1_c6_gq_tok_params(
                    x, inv_order, flat_ids, flat_weights, _gu_bnorm(gu_q, gu_s), k
                )
            else:
                fp8_tokens_q, fp8_tokens_s = _gq1p_tok(x)''')
new=replace_once(new,'            _gq16, _gs16 = _get_int_gu(gu_q, gu_s, 1)\n            if _dir_a:','''            _gq16, _gs16 = _get_int_gu(gu_q, gu_s, 1)
            if _hybrid:
                act_q8, act_rowscl = _s1_c6_mdq_g_pre_host(
                    fp8_tokens_q, _hybrid_scl, _hybrid_ah, _hybrid_wi, _gq16, _gs16, order,
                    meta_expert_ids, expert_counts, meta_tile_split,
                    meta_tile_num, meta_tile_num_cum, num_tiles_total, k,
                )
            elif _dir_a:''')
new+='\n\n# S1 c6: token Q and compact branch parameter producer/consumer.\n'+gq+'\n'+host+'\n'+md+'\n'+mdhost
ast.parse(new)
(OUT/'p1_s1_c6_v1.py').write_text(new)
(OUT/'candidate.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile='p1/kernel.py',tofile='p1_s1_c6_v1.py')))
print(hashlib.sha256(new.encode()).hexdigest())
