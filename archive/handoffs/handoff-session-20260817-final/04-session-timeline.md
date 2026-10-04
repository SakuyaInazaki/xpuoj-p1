# 04 本会话完整时间线（115143-115800）

本会话从 canonical 114970/69.33 接手，共 76 次提交。状态：Accepted 若干、WA 诊断/失败、TLE 3 次。

## 阶段概览

1. **115143-115222：case9/10 全量 replicated + 低精度 GEMM**
   - 修复 int64 expert 偏移；case10 FP8 both 通过。
   - case9 lowmem full FP8 replicated，raw 69.67。
   - 关键产出：115209。
2. **115593-115687：API/JIT 源码探测与失败 kernel 变体**
   - 读取官方 group_gemm 源码、签名、wrapper kwargs。
   - 排除 official FP8 BF16 输出、B 转置、persistent、BM64/w4/s2 等。
3. **115691-115714：swizzle 优化**
   - FP8/INT8 custom GEMM 加 `tl.swizzle2d`。
   - raw 从 69.67 -> 70.00 -> 71.08。
4. **115722-115800：FP8 gateup epilogue 融合与失败/TLE 探索**
   - 115738 fused FP8 gateup+SwiGLU+amax，实际最快。
   - INT8 融合 gateup TLE；case2 FP8 fused TLE。
   - 115705（GM2）靠 case6 tb 异常成为平台最佳。

## 全部提交表

| id | 状态 | display | timeUsed | 日志 | 说明 |
|---:|---|---:|---:|---:|---|
| 115143 | Accepted | 68.42 | 58545 | submit_case9_static_shape_cache.log | 修复 duplicate _get_static_cache 的 id-key；实际无单点收益 |
| 115147 | WrongAnswer | 65.58 | 1900715 | submit_diag_case9_static_114970.log | shape-key 命中诊断；case9 phase |
| 115156 | WrongAnswer | 5.75 | 20338605 | submit_case10_fp8both_lowmem.log | case10 lowmem FP8 第一版；worker 崩溃，数据不可信 |
| 115159 | WrongAnswer | None | 50558854 | submit_case10_fp8both_lowmem_r2.log | 同上重试；全局 TLE/崩溃 |
| 115161 | WrongAnswer | None | 78258442 | submit_case10_fp8both_lowmem_r3.log | 同上重试；全局 TLE/崩溃 |
| 115165 | WrongAnswer | 66.67 | 1901646 | submit_case10_fp8both_chunkquant.log | _quant_weight_fp8 chunked + case10 full FP8 both；case10 非有限值（int32 expert 偏移溢出） |
| 115175 | Accepted | 68.67 | 57500 | submit_case10_fp8down_chunkquant.log | chunk quant + case10 FP8 down only；case10 3.808 |
| 115180 | Accepted | 69.33 | 58488 | submit_case10_fp8down_chunkquant_staticfix.log | + static shape cache；无收益 |
| 115185 | Accepted | 69.33 | 57204 | submit_case10_fp8both_int64ptr.log | 修复 _fp8_group_gemm_kernel int64 expert 基址；case10 full FP8 both 通过，case10 3.123 |
| 115189 | Accepted | 68.42 | 59132 | submit_case10_fp8both_int64ptr_staticfix.log | + static shape cache；无收益 |
| 115190 | Accepted | 69 | 56903 | submit_case10_fp8both_int64ptr_r2.log | 115185 复测；case10 3.127，case9 4.904 |
| 115196 | Accepted | 67.67 | 60809 | submit_case10_fp8both_int64ptr_case9fp8.log | + case9 local FP8；本轮 case9 8.0，未采用 |
| 115202 | Accepted | 68.33 | 59445 | submit_case10_fp8both_int64ptr_r3.log | 115185 复测；case9 6.64 噪声 |
| 115207 | WrongAnswer | 63.92 | 1866719 | submit_case9_repl_fp8_regular.log | case9 full replicated 普通 FP8 权重量化路径：OOM |
| 115209 | Accepted | 69.67 | 56185 | submit_case9_repl_fp8_lowmem.log | case9 full replicated FP8 lowmem；晋升当时 canonical，raw 69.67 |
| 115214 | Accepted | 69.42 | 56641 | submit_case9_repl_fp8_lowmem_r2.log | 115209 复测 |
| 115217 | WrongAnswer | None | 46201672 | submit_diag_115209_repl.log | 诊断脚本缺 _diag_mark（脚本错误） |
| 115219 | WrongAnswer | None | 119350320 | submit_diag_115209_repl_r2.log | 115209 phase 诊断 |
| 115222 | Accepted | 69.33 | 56570 | submit_route_e256_bm64.log | E256 route BM64 试验；未超过 |
| 115593 | WrongAnswer | 23.5 | 94340516 | submit_fp8_bm256.log | FP8 BM256 + metadata block_m=256；FP8 case 全错 |
| 115595 | Accepted | 44.08 | 198619 | submit_fp8_w4.log | FP8 num_warps=4；明显变慢 |
| 115597 | WrongAnswer | None | 81856377 | submit_fp8_bt.log | FP8 B 转置布局；case1 illegal memory，随后全局失败 |
| 115599 | Accepted | 69 | 56624 | submit_fp8_small.log | case4/6/11 开 FP8；case4/11 变慢，case6 打平 |
| 115602 | WrongAnswer | None | 36217702 | submit_probe_group_gemm_api.log | API probe（含 try/except）；Language validation 崩 |
| 115603 | WrongAnswer | None | 36526216 | submit_probe_group_gemm_api_r2.log | API probe（含 inspect/json import）；Import validation 崩 |
| 115604 | WrongAnswer | None | 42060348 | submit_probe_group_gemm_api_r3.log | API probe（访问 __doc__）；dunder attribute 被禁 |
| 115607 | WrongAnswer | None | 41909191 | submit_probe_group_gemm_api_r4.log | 成功读取 group_gemm 模块对象列表 |
| 115609 | WrongAnswer | None | 41523305 | submit_probe_transposed_sig.log | transposed_moe_grouped_gemm 签名 probe |
| 115610 | WrongAnswer | None | 42054778 | submit_probe_2weights_sig.log | moe_grouped_gemm_2weights 签名 probe |
| 115612 | WrongAnswer | None | 41389595 | submit_probe_kernel_sig.log | JIT kernel 直接调用 probe；不可在 kernel scope 外调用 |
| 115616 | WrongAnswer | None | 41535166 | submit_probe_outdtype.log | out_dtype probe；torch.tensor 被禁 |
| 115618 | WrongAnswer | None | 41986843 | submit_probe_outdtype_r2.log | out_dtype probe；wrapper 不支持 out_dtype |
| 115622 | WrongAnswer | None | 42252530 | submit_probe_mixed_dtype.log | official FP8/混合 dtype probe；dot_k_const 不支持 fp8 |
| 115626 | Accepted | 65.17 | 69748 | submit_fp8_bm64half.log | FP8 BM64 half-tile；变慢 |
| 115630 | Accepted | 67 | 63822 | submit_fp8_s2.log | FP8 num_stages=2；变慢 |
| 115633 | WrongAnswer | None | 42584603 | submit_probe_kw.log | 确认 wrapper 支持 num_warps |
| 115638 | WrongAnswer | None | 41097001 | submit_probe_jit_sig.log | JIT kernel no-arg 签名 probe |
| 115641 | WrongAnswer | None | 41097949 | submit_probe_jit_dir.log | JITFunction dir 读取可用属性 |
| 115642 | WrongAnswer | None | 41752775 | submit_probe_jit_src.log | 读取 official kernel arg_names/signature/src |
| 115643 | WrongAnswer | None | 41157953 | submit_probe_jit_src2.log | 读取 official kernel 完整源码 |
| 115647 | WrongAnswer | None | 42428626 | submit_probe_fp8_official.log | official FP8 both probe；dot 不支持 fp8e4nv |
| 115653 | WrongAnswer | None | 41896636 | submit_probe_gg_dir.log | group_gemm 模块 dir |
| 115655 | WrongAnswer | None | 41556312 | submit_probe_dot_src.log | 读取 dot_k_const 源码 |
| 115658 | Accepted | 53.67 | 121287 | submit_fp8_bf16dot.log | FP8 dot 前显式转 BF16；严重变慢 |
| 115660 | WrongAnswer | 17.75 | 113977358 | submit_fp8_persistent.log | FP8 persistent 132 programs；FP8 case 全错 |
| 115666 | Accepted | 68.92 | 56760 | submit_case6_fp8.log | 仅 case6 FP8；打平，未采用 |
| 115668 | WrongAnswer | None | 41314198 | submit_probe_moe_sig.log | moe_grouped_gemm wrapper 签名确认 |
| 115672 | WrongAnswer | None | 41535842 | submit_probe_kw2.log | 确认 wrapper 支持 num_stages/GROUP_SIZE_M，不支持 PERSISTENT |
| 115678 | Accepted | 69.67 | 56230 | submit_route_bk128.log | route BK128；与当时 canonical 打平 |
| 115682 | WrongAnswer | 17.67 | 44285786 | submit_fp8_persistent_tritonjit.log | FP8 persistent 改 @triton.jit；仍错 |
| 115687 | WrongAnswer | None | 42564988 | submit_probe_build_src.log | 读取 build_block_row_idx_info_kernel 源码/签名 |
| 115691 | Accepted | 70 | 54398 | submit_fp8_swizzle.log | FP8 GEMM 加 tl.swizzle2d（GROUP_M=8）；raw 70.0 |
| 115696 | Accepted | 71.08 | 51419 | submit_int8_swizzle.log | INT8 GEMM 也加 swizzle；case2 12.10，raw 71.08 |
| 115700 | Accepted | 70.08 | 51752 | submit_gm16.log | GROUP_M=16；tk 略差 |
| 115703 | Accepted | 70.08 | 52477 | submit_gm4.log | GROUP_M=4；tk 略差 |
| 115705 | Accepted | 71.67 | 53703 | submit_gm2.log | GROUP_M=2；case6 tb=45.559 异常，raw 71.67 成为平台最佳 |
| 115711 | Accepted | 70.17 | 51906 | submit_fp8gm16.log | FP8 GM16 + INT8 GM8；未超过 |
| 115713 | Accepted | 69.83 | 52260 | submit_int8gm16.log | INT8 GM16 + FP8 GM8；未超过 |
| 115714 | Accepted | 71 | 52316 | submit_int8_bn128_sw.log | INT8 BN128 + swizzle；未超过 |
| 115722 | WrongAnswer | 64.92 | 1902480 | submit_case2_int8down_sw.log | case2 down 改 INT8；case2 12.95，且 lowmem case9 UnboundLocal bug |
| 115727 | Accepted | 70.17 | 52046 | submit_bf16_gm8.log | official BF16 显式 GROUP_SIZE_M=8/num_stages=3；无收益 |
| 115729 | WrongAnswer | 24.92 | 97592108 | submit_fused_gateup_swiglu.log | fused FP8 gateup+SwiGLU v1；kernel 变量名 bug |
| 115732 | WrongAnswer | 17.67 | 44248519 | submit_fused_gateup_swiglu_r2.log | fused FP8 v2；A 未量化导致 dot 混合 dtype 错误 |
| 115737 | WrongAnswer | None | 46159835 | submit_fused_gateup_swiglu_r3.log | fused FP8 v3；脚本误替换 _linear_bf16 的 a_q |
| 115738 | Accepted | 71.08 | 51044 | submit_fused_gateup_swiglu_r4.log | fused FP8 v4 正确；raw 71.08 / timeUsed 51044，实际最快 |
| 115743 | TimeLimitExceeded | None | 473544353 | submit_fused_int8_gateup.log | INT8 gateup 融合 SwiGLU；TLE |
| 115757 | Accepted | 70.08 | 51435 | submit_fused_gateup_swiglu_r5.log | 115738 复测；raw 70.08 |
| 115760 | Accepted | 70.5 | 51332 | submit_fused_fp8gm16.log | fused gateup GROUP_M=16；未超过 |
| 115767 | Accepted | 69.83 | 52189 | submit_fused_bn64.log | fused gateup BN64；变慢 |
| 115769 | TimeLimitExceeded | None | 473010222 | submit_case2_fp8_fused.log | case2 改 FP8 fused gateup；TLE |
| 115778 | Accepted | 69.83 | 53175 | submit_fused_bk64.log | fused gateup BK64；变慢 |
| 115781 | Accepted | 70.58 | 51403 | submit_down_gm16.log | down GROUP_M=16；未超过 |
| 115784 | WrongAnswer | 11.92 | 18553958 | submit_route_bk256.log | route BK256；H<4096 shared memory OOR |
| 115789 | Accepted | 70.17 | 51970 | submit_fused_gm2.log | fused gateup GROUP_M=2；未超过 |
| 115792 | TimeLimitExceeded | None | 473578801 | submit_fused_int8_bn64.log | fused INT8 gateup BN64/w4/s2；仍 TLE |
| 115800 | Accepted | 70.42 | 51182 | submit_fused_gateup_swiglu_r6.log | 115738 复测；raw 70.42 |
