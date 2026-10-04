# XPUOJ P1 接手继续迭代（submission 116663-116745）

## 10 秒结论

- **实际性能 base 再次晋升：116735 / v136，当前 `p1/kernel.py` 已是该文件。**
  - 116735：Accepted，raw 74.25，timeUsed **43212**；另一次 116730=43300。
  - 同窗口旧 base v130 对照 116736=43924，v136 快约 0.7ms。
- scoreboard best：116716 raw 75.83，total **65.83**（case1/case5 tb 异常，不代表实际性能）。
- v136 优化栈 = v115 + 三个新有效项：
  1. 所有 activation amax atomic 改 `sem="relaxed"`（v129）；
  2. fused gateup persistent `num_stages=4`（v130）；
  3. replicated 路径 topk_ids 不再 int64->int32->int64 来回转换（v136）。

## 关键提交表

| id | 文件 | 说明 | raw | timeUsed |
|---:|---|---|---:|---:|
| 116663 | kernel_diag_v115_phases.py | v115 逐 phase CUDA-event 诊断（WA） | 0 | - |
| 116666 | kernel_v120_fused_persist_major_fixed.py | 修正 v106 M-major persistent | 69.25 | 53467 |
| 116667 | kernel_v121_cache_tma_desc.py | TMA descriptor 缓存 | 74.00 | 43583 |
| 116668 | kernel.py（v115） | 对照 | 73.58 | 43906 |
| 116672 | kernel_v121...r2 | descriptor 缓存复测 | 73.75 | 43969 |
| 116675 | kernel_v122_meta_num_sms132.py | metadata num_sms=132 | 75.25 | 44304 |
| 116676 | kernel_v123_fused_grid160.py | fused grid=160 | 69.75 | 52407 |
| 116679 | kernel_v125_buf_cache.py | 中间 tensor buffer cache | WA/TLE | - |
| 116683 | kernel_v126_packed_sort.py | tensor.sort 打包排序 | WA guard | - |
| 116685 | kernel_v127_binned_amax_case2.py | case2 amax 分 bin | 73.92 | 43256 |
| 116688 | kernel.py（v115） | 对照 | 73.50 | 44361 |
| 116690 | kernel_v127...r2 | binned amax 复测 | 73.00 | 44393 |
| 116694 | kernel_v128_packed_sort_functional.py | functional torch.sort | WA guard | - |
| 116695 | kernel_v129_relaxed_atomics.py | **amax atomic sem=relaxed** | 74.33 | **42664** |
| 116696 | kernel_v129...r2 | relaxed 复测 | 74.00 | 43537 |
| 116701 | kernel_v130_fused_s4.py | **+ fused num_stages=4** | 73.83 | 42987 |
| 116702 | kernel.py（v129） | 对照 | 73.83 | 43539 |
| 116707 | kernel_v130...r2 | v130 复测 | 74.25 | **42985** |
| 116708 | kernel.py（v129） | 对照 | 73.67 | 43587 |
| 116711 | kernel_v131_fused_s5.py | fused num_stages=5 | WA OutOfResources | - |
| 116716 | kernel_v132_fused_grid128_s4.py | fused grid=128 | 75.83 | 43541 |
| 116718 | kernel_v133_case2_row_relaxed_s4.py | case2 row SwiGLU/down | 73.67 | 43809 |
| 116724 | kernel_v134_tma_desc_cache_s4.py | v130 + TMA descriptor cache | 74.08 | 43005 |
| 116727 | kernel_v135_fused_w16_s4.py | fused num_warps=16 | 73.50 | 44852 |
| 116730 | kernel_v136_no_id_conversions.py | **topk_ids 保持 int64** | 74.08 | 43300 |
| 116735 | kernel_v136...r2 | **当前 base** | 74.25 | **43212** |
| 116736 | kernel.py（v130） | 对照 | 73.50 | 43924 |
| 116745 | kernel_v137_inplace_weight_norm.py | topk_weights /= | 73.83 | 43280 |

## 当前 base 检查

```bash
cd /home/sakimi26/xpuoj-p1
sha256sum p1/kernel.py p1/kernel_116735_backup.py
python -m py_compile p1/kernel.py
python scripts/best_score.py
```

期望：

```text
3b9a9c4f32fcd62083fbeadcfb649694dbbeb200fec6aa5f0066f5c8ab08e429  p1/kernel.py
3b9a9c4f32fcd62083fbeadcfb649694dbbeb200fec6aa5f0066f5c8ab08e429  p1/kernel_116735_backup.py
{"totalScore": 65.83, "problemScore": 65.83, "submissionId": 116716, ...}
```

## 有效改动实现位置

- `_gather_tokens_row_amax_order_kernel` / `_fused_gateup_swiglu_kernel_rowA_persistent_tiles` /
  `_act_amax_fp8_kernel` / `_swiglu_amax_kernel`：`tl.atomic_max(..., sem="relaxed")`。
- `_fused_gateup_swiglu_rowA_persistent_tiles` host：grid=132，num_stages=4。
- `_run_replicated`：`flat_ids = topk_ids.reshape(-1)`，不再 `.to(int32)` 再 `.to(int64)`。

## 负结果（勿重试）

- fused num_stages=5：OutOfResources。
- fused num_warps=16、grid=128/160：慢。
- case2 full-fused/row-SwiGLU-row-down：仍 TLE/慢。
- metadata num_sms=132、token gather M-only、weights 融合 gather：慢。
- packed sort（tensor.sort / torch.sort functional）：sandbox 禁止。
- 中间张量全局 buffer cache：WA/TLE。
- TMA descriptor cache、case2 binned amax、in-place topk_weights：无稳定收益。

## 下一步

1. 继续在 v136 上同窗口 A/B；当前 case2 仍是最大单点。
2. 可研究 fused persistent 的 GROUP_M 扫描（GM4/GM16 尚未在 s4 上重测）。
3. 可对 v136 再做一次 phase 诊断，确认 relaxed atomic 后 gather/gateup 占比。

---

# 追加迭代（submission 116747-116788）

## 当前实际 base 再晋升

- **116773 / v143，当前 `p1/kernel.py` 已是该文件。**
  - timeUsed 42869 / 42945；同窗口 v142 对照 43778 / 43811。
- v143 = v142 + fused gateup `GROUP_M=16`。
- v142 = v140 + case2 SwiGLU amax/quant 也改 orderW。
- v140 核心：fused gateup 直接 `w = W[ORDER[offs_m]]`，不再做 `flat_weights[order]`。

## 关键提交表（追加）

| id | 文件 | 说明 | raw | timeUsed |
|---:|---|---|---:|---:|
| 116747 | kernel_v138_fused_gm16_s4.py | v136 + fused GM16（无 orderW） | 74.42 | 42830 |
| 116749 | kernel.py（v136） | 对照 | 74.17 | 43234 |
| 116751 | kernel_v140_fused_orderW.py | **fused gateup orderW** | 74.50 | 42343 |
| 116753 | kernel_v141_orderW_not_i8192.py | orderW 排除 I=8192 | 74.33 | 43826 |
| 116754 | kernel_v140...r2 | **orderW 复测，曾晋升** | 74.92 | **41808** |
| 116758 | kernel_v142_case2_swiglu_orderW.py | **case2 SwiGLU orderW** | 74.67 | 42762 |
| 116760 | kernel.py（v140） | 对照 | 73.50 | 43809 |
| 116767 | kernel_v142...r2 | v142 复测 | 74.33 | 43160 |
| 116768 | kernel.py（v140） | 对照 | 73.58 | 43823 |
| 116771 | kernel_v143_orderW_gm16.py | **+ fused GM16** | 74.08 | 42945 |
| 116772 | kernel.py（v142） | 对照 | 74.33 | 43299 |
| 116773 | kernel_v143...r2 | **当前 base** | 74.17 | **42869** |
| 116774 | kernel.py（v142） | 对照 | 73.67 | 43804 |
| 116783 | kernel_v144_orderW_gm4.py | GM4 扫描 | 74.17 | 42972 |
| 116784 | kernel.py（v143 GM16） | 对照 | 74.08 | 43778 |
| 116787 | kernel_v144...r2 | GM4 复测 | 73.92 | 43388 |
| 116788 | kernel.py（v143 GM16） | 对照 | 73.50 | 43811 |

## 结论

- orderW 是近期最大有效改动：fused gateup 与 case2 SwiGLU 都直接在 kernel 内
  用 `order` 索引原始 route weights，省掉整段 `flat_weights[order]`。
- fused orderW 的 GROUP_M：GM16 与 GM8 基本打平，GM16 略好；GM4 较慢。
- 同窗口对照仍然不可省；本轮 v140 r2 的 41808 是低噪声窗口，不代表可持续。

---

# 再追加迭代（submission 116792-116839）

## 当前实际 base 再晋升

- **116808 / v147，当前 `p1/kernel.py` 已是该文件。**
  - timeUsed 43232 / 42659；同窗口 v143 对照 43763 / 43812。
- v147 = v143 + case2 token gather 改为 custom order-derived gather：
  - `_gather_tokens_from_order_kernel`：`src = ORDER[offs_m]; row = src // k`；
  - 省掉 case2 的 `token_idx = token_idx[order]` 和 `x[token_idx].contiguous()`。

## 关键提交表（追加）

| id | 文件 | 说明 | raw | timeUsed |
|---:|---|---|---:|---:|
| 116792 | kernel_v145_order32.py | 全部 order 转 int32 | 76.00 | 43123 |
| 116794 | kernel.py（v143） | 对照 | 74.00 | 43230 |
| 116795 | kernel_v146_order32_not_case12.py | order32 排除 case1/2 | 73.75 | 43790 |
| 116798 | kernel_v147_case2_gather_from_order.py | **case2 custom order gather** | 74.42 | 42659 |
| 116799 | kernel.py（v143） | 对照 | 74.42 | 43812 |
| 116808 | kernel_v147...r2 | **当前 base** | 73.92 | **43232** |
| 116810 | kernel.py（v143） | 对照 | 73.75 | 43763 |
| 116812 | kernel_v148_case2_gather_bm256.py | case2 gather BM256 | 73.92 | 43163 |
| 116813 | kernel.py（v147） | 对照 | 74.08 | 43219 |
| 116816 | kernel_v149_case2_gather_amax.py | case2 gather+amax atomic | 73.83 | 43122 |
| 116817 | kernel.py（v147） | 对照 | 74.08 | 43204 |
| 116823 | kernel_diag_v147_phases.py | v147 phase 诊断（WA） | 0 | - |
| 116828 | kernel_v150_case2_gather_partial_amax.py | case2 gather+partial amax | 74.08 | 43117 |
| 116830 | kernel.py（v147） | 对照 | 74.08 | 43190 |
| 116833 | kernel_v151_orderW_gm24.py | fused GM24 | 74.17 | 42756 |
| 116834 | kernel.py（v147） | 对照 | 73.83 | 43731 |
| 116837 | kernel_v151...r2 | GM24 复测 | 74.00 | 43286 |
| 116839 | kernel.py（v147） | 对照 | 76.00 | 43222 |

## 结论

- case2 custom order gather 是有效改动，保留。
- case2 gather+amax（atomic 或 partial）无稳定收益，保留单独 `_quant_act_fp8`。
- order32 整体无收益。
- fused GM24 两次与对照互有胜负，不采用；GM16 维持。

---

# 第三次追加（submission 116841-116869）

## 当前实际 base 再晋升

- **116858 / v155，当前 `p1/kernel.py` 已是该文件。**
  - timeUsed 43315 / 43174；同窗口 v147 对照 43747 / 43746。
- v155 = v147 + **E<=16 的 routing GEMM 用 BM64/BN16/w4**。
  - case1/2 的 route GEMM 从 128 个 block 提升到 256 个 block，改善 SM 占用。

## 关键提交表

| id | 文件 | 说明 | raw | timeUsed |
|---:|---|---|---:|---:|
| 116841 | kernel_v153_inv_sort_side_stream.py | inv argsort 放 side stream | WA determinism | - |
| 116842 | kernel.py（v147） | 对照 | 74.08 | 43274 |
| 116844 | kernel_v154_case2_quant_bm256.py | case2 token quant BM256 | 73.92 | 43253 |
| 116845 | kernel.py（v147） | 对照 | 74.08 | 43203 |
| 116847 | kernel_v155_route_e8_bm64.py | **route E8 BM64** | 74.17 | 43174 |
| 116848 | kernel.py（v147） | 对照 | 73.50 | 43747 |
| 116858 | kernel_v155...r2 | **当前 base** | 74.00 | **43315** |
| 116859 | kernel.py（v147） | 对照 | 73.67 | 43746 |
| 116863 | kernel_v156_route_e8_bm64_w8.py | route E8 BM64 w8 | 73.83 | 43763 |
| 116864 | kernel.py（v155） | 对照 | 74.17 | 43218 |
| 116868 | kernel_v158_route_n32_bm64.py | route N<=32 BM64 | 74.00 | 43184 |
| 116869 | kernel.py（v155） | 对照 | 73.67 | 43665 |

## 结论

- route E8 BM64/w4 有效，保留。
- route E8 BM64/w8 慢；route N<=32 BM64 未显示稳定收益，不采用。
- side-stream 做 inv argsort 会 determinism fail，关闭。
- case2 token quant BM256 无收益。

---

# 第四次追加（submission 116879-116897）

## 当前实际 base 再晋升

- **116882 / v159，当前 `p1/kernel.py` 已是该文件。**
  - timeUsed 43131 / 43198；同窗口 v155 对照 43777 / 43714。
- v159 = v155 + **route E8 的 BLOCK_K 从 64 提到 128**（BM64/BN16/w4/s3）。
  - case1/2 路由再快约 0.08-0.23ms。

## 关键提交表

| id | 文件 | 说明 | raw | timeUsed |
|---:|---|---|---:|---:|
| 116879 | kernel_v159_route_e8_bk128.py | **route E8 BK128** | 74.17 | 43198 |
| 116880 | kernel.py（v155） | 对照 | 73.75 | 43714 |
| 116882 | kernel_v159...r2 | **当前 base** | 74.08 | **43131** |
| 116883 | kernel.py（v155） | 对照 | 73.58 | 43777 |
| 116888 | kernel_v160_route_n64_bm64.py | route N<=64 BM64 | 75.75 | 43210 |
| 116889 | kernel.py（v159） | 对照 | 73.92 | 43742 |
| 116892 | kernel_v160...r2 | N64 BM64 复测 | 73.75 | 43767 |
| 116893 | kernel.py（v159） | 对照 | 74.00 | 43358 |
| 116896 | kernel_v161_route_e8_bk256.py | route E8 BK256 | 73.83 | 43804 |
| 116897 | kernel.py（v159） | 对照 | 73.92 | 43285 |

## 结论

- route E8 BK128 有效，保留。
- route E8 BK256 慢。
- route N<=64 BM64 两次互有胜负，不采用。
