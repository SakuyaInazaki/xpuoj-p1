from pathlib import Path
import ast,hashlib,difflib,json
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
src=(ROOT/'p1/kernel.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9'
lines=src.splitlines(keepends=True)
def extract(name):
    n=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    start=min([n.lineno]+[d.lineno for d in n.decorator_list])
    return ''.join(lines[start-1:n.end_lineno])+'\n'
def once(s,old,new):
    assert s.count(old)==1,(old,s.count(old))
    return s.replace(old,new,1)
md=extract('_fgs_t1i_mdq_kernel_g')
md=md.replace('_fgs_t1i_mdq_kernel_g','_fgs_t1i_mdq_kernel_g_s2_pad')
start=md.index('        # TMA Store 卸载收尾')
end=md.index('        tl.store(SCL + offs_m',start)
md=md[:start]+'''        padded_row = (t_cum - t_num + local_m) * BLOCK_M
        ACT_DESC.store([padded_row, pid_n * BLOCK_N], q)
'''+md[end:]
pre=extract('_fgs_t1i_mdq_tma_pre_nf_kernel').replace('_fgs_t1i_mdq_tma_pre_nf_kernel','_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad')
start=pre.index('        if a_row + BLOCK_M')
end=pre.index('        tl.store(DROP + 0',start)
pre=pre[:start]+'''        padded_row = (t_cum - t_num + local_m) * BLOCK_M
        ACT_DESC.store([padded_row, pid_n * BLOCK_N], q)
'''+pre[end:]
# Copy the existing pre_nf host, retaining launch controls and DROP allocation.
preh=extract('_fgs_tma1_intq_host_pre_nf').replace('_fgs_tma1_intq_host_pre_nf','_fgs_tma1_intq_host_pre_nf_s2_pad').replace('_fgs_t1i_mdq_tma_pre_nf_kernel','_fgs_t1i_mdq_tma_pre_nf_kernel_s2_pad')
preh=once(preh,'tile_cum, num_tiles):','tile_cum, num_tiles, logical_m):')
preh=once(preh,'    M, K = a_q.shape','    M = int(logical_m)\n    K = a_q.shape[1]')
preh=once(preh,'    act = torch.empty((M, I),','    P = 128 * ((M + 127 * G + 127) // 128)\n    act = torch.empty((P, I),')
preh=once(preh,'    return act, act_s','    return act, act_s, True')
mdh='''def _fgs_tma1_intq_host_g_s2_pad(a_q, a_s, bnorm, b_q_int, b_s_int, weights, order, expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles, logical_m):
    M = int(logical_m)
    K = a_q.shape[1]
    G, N2, K2 = b_q_int.shape
    I = N2 // 2
    P = 128 * ((M + 127 * G + 127) // 128)
    act = torch.empty((P, I), dtype=torch.float8_e4m3fn, device=a_q.device)
    _scl = torch.empty(M, dtype=torch.float32, device=a_q.device)
    b_flat = b_q_int.view(G * N2, K2)
    b_desc = TensorDescriptor(b_flat, b_flat.shape, b_flat.stride(), [256, 128])
    w_sorted = weights[order]
    act_desc_g = TensorDescriptor(act, act.shape, act.stride(), [128, 128])
    _gg = (G == 96 and I == 1024)
    _fgs_t1i_mdq_kernel_g_s2_pad[(132,)](
        a_q, a_s, bnorm, b_desc, b_s_int, w_sorted, order, act, act_desc_g, _scl,
        expert_ids, counts, split_cum, tile_num, tile_cum, num_tiles,
        M, I, K,
        a_q.stride(0), a_q.stride(1),
        act.stride(0), act.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, GROUP_M=8 if K <= 1024 else 32, KTOP=_GA[1],
        num_warps=8, num_stages=4 if _gg else 3,
        **({'maxnreg': 232} if _gg else {}),
    )
    return act, _scl, True
'''
dn=extract('_dn_tma2_f8_pad_static_kernel').replace('_dn_tma2_f8_pad_static_kernel','_dn_tma2_f8_pad_static_kernel_s2_input_pad')
dn=once(dn,'        a_row = row_begin + local_m * BLOCK_M','        a_row = (t_cum - t_num + local_m) * BLOCK_M')
dnh=extract('_dn_tma2_f8_pad_static_host').replace('_dn_tma2_f8_pad_static_host','_dn_tma2_f8_pad_static_host_s2_input_pad').replace('_dn_tma2_f8_pad_static_kernel','_dn_tma2_f8_pad_static_kernel_s2_input_pad')
dnh=once(dnh,'    padded_rows, group_m=8, num_stages=4,','    padded_rows, logical_m, group_m=8, num_stages=4,')
dnh=once(dnh,'    M, K = a_q.shape','    M = int(logical_m)\n    K = a_q.shape[1]')
run=extract('_run_replicated')
run=once(run,'    act_rowscl = None','    act_rowscl = None\n    act_is_padded = False\n    _s2_c4 = ((T, H, E, I, k) == (16384, 2048, 32, 1024, 4)\n              and inv_pad is not None)')
old='''                elif H == 2048:
                    act_q8, act_rowscl = _fgs_tma1_intq_host_pre_nf(
                        fp8_tokens_q, _dir_scl, _dir_ah, _dir_wi, _gq16, _gs16,
                        meta_expert_ids, expert_counts, meta_tile_split,
                        meta_tile_num, meta_tile_num_cum, num_tiles_total,
                    )'''
new='''                elif H == 2048:
                    if _s2_c4:
                        act_q8, act_rowscl, act_is_padded = _fgs_tma1_intq_host_pre_nf_s2_pad(
                            fp8_tokens_q, _dir_scl, _dir_ah, _dir_wi, _gq16, _gs16,
                            meta_expert_ids, expert_counts, meta_tile_split,
                            meta_tile_num, meta_tile_num_cum, num_tiles_total, T * k,
                        )
                    else:
                        act_q8, act_rowscl = _fgs_tma1_intq_host_pre_nf(
                            fp8_tokens_q, _dir_scl, _dir_ah, _dir_wi, _gq16, _gs16,
                            meta_expert_ids, expert_counts, meta_tile_split,
                            meta_tile_num, meta_tile_num_cum, num_tiles_total,
                        )'''
run=once(run,old,new)
old='''            else:
                act_q8, act_rowscl = _fgs_tma1_intq_host(
                    fp8_tokens_q, fp8_tokens_s, _gu_bnorm(gu_q, gu_s), _gq16, _gs16, flat_weights, order,
                    meta_expert_ids, expert_counts, meta_tile_split,
                    meta_tile_num, meta_tile_num_cum, num_tiles_total,
                )'''
new='''            else:
                if _s2_c4 and _GA[0]:
                    act_q8, act_rowscl, act_is_padded = _fgs_tma1_intq_host_g_s2_pad(
                        fp8_tokens_q, fp8_tokens_s, _gu_bnorm(gu_q, gu_s), _gq16, _gs16, flat_weights, order,
                        meta_expert_ids, expert_counts, meta_tile_split,
                        meta_tile_num, meta_tile_num_cum, num_tiles_total, T * k,
                    )
                else:
                    act_q8, act_rowscl = _fgs_tma1_intq_host(
                        fp8_tokens_q, fp8_tokens_s, _gu_bnorm(gu_q, gu_s), _gq16, _gs16, flat_weights, order,
                        meta_expert_ids, expert_counts, meta_tile_split,
                        meta_tile_num, meta_tile_num_cum, num_tiles_total,
                    )'''
run=once(run,old,new)
old='''                        down, _dscl = _dn_tma2_f8_pad_static_host(
                            act_q, act_s, dn_q, _dn_s_folded, _dn_c,
                            meta_expert_ids, expert_counts, meta_tile_split,
                            meta_tile_num, meta_tile_num_cum, num_tiles_total,
                            _padded_rows,
                            group_m=8 if ((I == 1024 and H == 3584) or H == 1024) else 32,
                            num_stages=3 if (E == 32 and I == 1024 and H == 2048) else 4,
                        )'''
new='''                        if act_is_padded:
                            down, _dscl = _dn_tma2_f8_pad_static_host_s2_input_pad(
                                act_q, act_s, dn_q, _dn_s_folded, _dn_c,
                                meta_expert_ids, expert_counts, meta_tile_split,
                                meta_tile_num, meta_tile_num_cum, num_tiles_total,
                                _padded_rows, T * k,
                                group_m=8 if ((I == 1024 and H == 3584) or H == 1024) else 32,
                                num_stages=3 if (E == 32 and I == 1024 and H == 2048) else 4,
                            )
                        else:
                            down, _dscl = _dn_tma2_f8_pad_static_host(
                                act_q, act_s, dn_q, _dn_s_folded, _dn_c,
                                meta_expert_ids, expert_counts, meta_tile_split,
                                meta_tile_num, meta_tile_num_cum, num_tiles_total,
                                _padded_rows,
                                group_m=8 if ((I == 1024 and H == 3584) or H == 1024) else 32,
                                num_stages=3 if (E == 32 and I == 1024 and H == 2048) else 4,
                            )'''
run=once(run,old,new)
final=once(src,extract('_run_replicated'),run)
final+='\n\n# S2 c4: padded ACT producers and paired static Down input reader.\n\n'+'\n\n'.join([md,pre,mdh,preh,dn,dnh])
path=OUT/'p1_s2_c4_layout_only.py'
path.write_text(final)
(OUT/'candidate.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),final.splitlines(True),fromfile='v12/p1/kernel.py',tofile=path.name)))
print(path,hashlib.sha256(final.encode()).hexdigest())
