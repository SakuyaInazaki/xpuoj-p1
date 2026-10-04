# 04 本会话完整时间线与 49 次提交（submission 116130-116568）

## 阶段概览

1. **116130-116195：接手后第一轮参数/constexpr 扫描**
   - 发现 FP8 GEMM `K: tl.constexpr`（fused+down 同时改）成为 scoreboard best 116142。
   - per-tile scale、逆置换、routing topk 等均为负。
2. **116215-116324：TMA 与 per-row FP8**
   - 探测出 host-side TensorDescriptor 可用、device-side TMA 不可用。
   - v80：down B host-TMA，timeUsed 42592。
   - v95/v98：per-row scale 逐步推广到 down act 与 token gather。
   - v98 成为当前实际性能 base。
3. **116542-116568：追加排除实验**
   - num_ctas、routing TMA、case2 s4、fused persistent/M-only row-amax 均失败或变慢。

## 全部提交表

| id | 状态 | raw | timeUsed | 日志 | 说明 |
|---:|---|---:|---:|---|---|
| 116130 | Accepted | 66.67 | 63403 | submit_v62_fp8act_tiled.log | per-tile FP8 activation scale + tiled down GEMM |
| 116135 | Accepted | 73.33 | 44121 | submit_v64_invscatter.log | 用 index assignment 求逆置换替代 argsort |
| 116142 | Accepted | 74.75 | 43040 | submit_v65_kconst.log | FP8 fused gateup + persistent down 的 K 改 tl.constexpr（scoreboard best） |
| 116148 | Accepted | 74.25 | 43126 | submit_v66_nkconst.log | v65 基础上 fused I / down N 也 constexpr |
| 116152 | Accepted | 74.17 | 43206 | submit_v67_routeconst.log | v65 基础上 route GEMM N,K constexpr |
| 116157 | Accepted | 73.08 | 44542 | submit_v68_fused_iconst.log | v65 基础上仅 fused I constexpr |
| 116160 | Accepted | 73.17 | 44492 | submit_v69_down_nconst.log | v65 基础上仅 down N constexpr |
| 116164 | Accepted | 74.08 | 43554 | submit_v70_routekconst.log | v65 基础上仅 route K constexpr |
| 116169 | Accepted | 73.92 | 43149 | submit_v71_strideconst.log | v65 基础上 fused/down stride constexpr |
| 116175 | Accepted | 73.67 | 44035 | submit_v72_fused_kconst.log | 仅 fused gateup K constexpr |
| 116178 | Accepted | 73.58 | 44577 | submit_v73_down_kconst.log | 仅 persistent down K constexpr |
| 116183 | Accepted | 73.67 | 44004 | submit_v74_v65_repeat.log | v65 原文件复测 |
| 116193 | Accepted | 73.5 | 44766 | submit_v75_route_topk_logits.log | routing 改为 topk(logits) 后归一化 |
| 116195 | Accepted | 73 | 44681 | submit_v76_topk_bf16.log | routing 直接 topk(BF16 logits) |
| 116215 | WrongAnswer | 0 | 41129644 | submit_probe_tma.log | TMA API 探测（含 __version__，被 sandbox 拦截） |
| 116216 | Accepted | 72.92 | 46973 | submit_v77_down_bk256.log | down GEMM BK256/s2 |
| 116220 | WrongAnswer | 0 | 41105346 | submit_probe_tma2.log | TMA API 探测（tl.make_tensor_descriptor 等） |
| 116227 | WrongAnswer | 0 | 101204328 | submit_v78_down_tma_bdesc.log | device-side TMA descriptor down |
| 116229 | WrongAnswer | 0 | 47511501 | submit_v79_down_tma_tritonjit.log | device-side TMA + @triton.jit |
| 116233 | WrongAnswer | 0 | 40727995 | submit_probe_tma_host.log | 探测 triton.tools.tensor_descriptor.TensorDescriptor |
| 116236 | Accepted | 74.33 | 42592 | submit_v80_down_tma_hostdesc.log | host TensorDescriptor B TMA down（v80，单次 timeUsed 最快） |
| 116244 | Accepted | 74 | 43056 | submit_v81_all_tma_plain.log | case2 gateup 也走 B TMA |
| 116249 | Accepted | 73.75 | 43766 | submit_v82_fused_tma.log | fused gateup B TMA |
| 116252 | Accepted | 73.92 | 43716 | submit_v83_tma_s4.log | TMA down num_stages=4 |
| 116255 | Accepted | 73.58 | 43783 | submit_v84_tma_cstore.log | down 增加 C TMA store |
| 116256 | Accepted | 72.33 | 48304 | submit_v85_tma_bn128.log | TMA down BN128 |
| 116260 | Accepted | 73.92 | 43705 | submit_v86_swiglu_act_cache.log | case2 SwiGLU 先写 act BF16 再量化 |
| 116262 | Accepted | 73.83 | 48100 | submit_v87_tma_ab.log | down A+B 双 TMA |
| 116264 | Accepted | 71.08 | 49783 | submit_v88_tma_contig.log | TMA down contiguous-M 调度 |
| 116266 | Accepted | 74.25 | 43730 | submit_v89_tma_case2gather.log | TMA base + case2 gather BM256 |
| 116270 | Accepted | 71.92 | 48198 | submit_v90_tma_s2.log | TMA down num_stages=2 |
| 116275 | Accepted | 73.5 | 44443 | submit_v91_v80_repeat.log | v80 复测 |
| 116281 | Accepted | 74.5 | 43345 | submit_v92_tma_flatten.log | TMA outer loop flatten |
| 116287 | Accepted | 73.5 | 43883 | submit_v93_tma_warpspec.log | TMA inner loop warp_specialize |
| 116290 | Accepted | 73.75 | 43882 | submit_v94_v92_repeat.log | v92 复测 |
| 116295 | Accepted | 74.08 | 43427 | submit_v95_rowscale_fp8.log | fused down activation per-row scale |
| 116302 | Accepted | 74 | 44136 | submit_v96_rowscale_flatten.log | v95 + flatten |
| 116305 | Accepted | 74.33 | 43389 | submit_v97_case2_row.log | case2 SwiGLU per-row FP8 |
| 116310 | Accepted | 74.67 | 42881 | submit_v98_token_row.log | token per-row + down TMA + act per-row（当前实际 base） |
| 116315 | Accepted | 74 | 43642 | submit_v99_case2_token_row.log | case2 token 也 per-row |
| 116318 | Accepted | 73.83 | 43524 | submit_v100_v98_repeat.log | v98 复测 |
| 116324 | Accepted | 73.42 | 44123 | submit_v101_row_gather_bm256.log | token row-gather BM256 |
| 116542 | WrongAnswer | 5.92 | 46825324 | submit_v102_numctas2.log | down+fused 加 num_ctas=2 |
| 116545 | Accepted | 72.58 | 47323 | submit_v104_numctas_down.log | 仅 down TMA num_ctas=2 |
| 116549 | Accepted | 73.42 | 44229 | submit_v103_route_tma.log | routing GEMM 改 host TensorDescriptor TMA |
| 116553 | Accepted | 73.5 | 44178 | submit_v105_case2_s4.log | case2 plain gateup num_stages=4 |
| 116559 | WrongAnswer | 6.17 | 138579321 | submit_v106_fused_persist_row.log | fused gateup persistent + batched row amax（-inf init） |
| 116563 | WrongAnswer | 6.08 | 137214239 | submit_v107_fused_persist_row_zero.log | fused gateup persistent + batched row amax（零 init） |
| 116568 | WrongAnswer | 6.08 | 128549928 | submit_v108_fused_monly.log | fused gateup M-only 网格 + batched row amax |
