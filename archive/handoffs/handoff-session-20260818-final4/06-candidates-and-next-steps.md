# 06 候选文件、当前差距、下一步

## 推荐 base

- **`p1/kernel_116882_backup.py` = 116882 = v159：当前实际性能 base。**
- scoreboard 文件：`p1/kernel_v145_order32.py`（116792，raw 靠异常，不推荐作为性能 base）。
- 上一稳定：`p1/kernel_116858_backup.py`、`p1/kernel_116808_backup.py`。

## 关键候选文件

| 文件 | submission | raw | timeUsed | 建议 |
|---|---:|---:|---:|---|
| `p1/kernel.py` / `p1/kernel_116882_backup.py` | 116882 | 74.08 | **43131** | **当前 base** |
| `p1/kernel_116858_backup.py` | 116858 | 74.00 | 43315 | route E8 BM64/BK64 |
| `p1/kernel_116808_backup.py` | 116808 | 73.92 | 43232 | case2 custom gather |
| `p1/kernel_116773_backup.py` | 116773 | 74.17 | 42869 | orderW + GM16 中间版 |
| `p1/kernel_116754_backup.py` | 116754 | 74.92 | 41808 | fused orderW 单次最快 |
| `p1/kernel_116735_backup.py` | 116735 | 74.25 | 43212 | no id conversions |
| `p1/kernel_116707_backup.py` | 116707 | 74.25 | 42985 | relaxed + s4 |
| `p1/kernel_116627_backup.py` | 116627 | 75.00 | 42753 | persistent fused + order token |

## 当前 raw 下一档差距

以 v159 两次 tk 中位数和典型 tb 估算：

| case | tk 中位 | 典型 tb | 当前分 | 下一档需 tk <= | 缺口 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.841 | 18.11 | 75 | 5.720 | 0.121 |
| 2 | 10.115 | 28.96 | 74 | 9.653 | 0.462 |
| 3 | 2.284 | 7.04 | 75 | 2.224 | 0.060 |
| 4 | 1.593 | 5.07 | 76 | 1.513 | 0.080 |
| 5 | 4.124 | 12.70 | 75 | 4.009 | 0.115 |
| 6 | 2.388 | 7.91 | 76 | 2.363 | 0.025 |
| 7 | 3.404 | 10.38 | 75 | 3.279 | 0.125 |
| 8 | 2.308 | 7.22 | 75 | 2.279 | 0.029 |
| 9 | 3.333 | 8.45 | 71 | 3.285 | 0.048 |
| 10 | 2.735 | 6.71 | 71 | 2.611 | 0.124 |
| 11 | 2.045 | 5.66 | 73 | 1.988 | 0.057 |
| 12 | 2.995 | 8.17 | 73 | 2.871 | 0.124 |

case2 仍是绝对最大单点；多个 case 只差 0.03-0.13ms，继续微调仍有现实收益。

## 下一步优先级

1. **case2 plain gateup GEMM 结构性优化**。
   - 当前 10.1-10.4ms，是最大单点。
   - 已排除：full-fused（多次 TLE）、split gate/up、TMA B、BN128、BK64/256、
     s2/s4、GM4/GM16、nonpersistent、BM64+metadata、orderW 无法用于 gateup 本身。
   - 可研究：更细的 N 分块与 SwiGLU 量化流水、gateup C 布局调整（例如 [M,2,I] 或
     分块交错），但任何改动要先证明不比当前 plain BF16 C 写出更慢。
2. **case2 token quant**。
   - v147 phase 诊断（116823）显示 case2 `token_quant_done-gateup_done` 约 2.59ms（事件口径）。
   - 已排除：gather+amax 融合（scalar/partial）、quant BM256、binned amax、case2 token per-row。
   - 可研究：BK=256 在 v159 上是否仍慢（历史 115843 慢，但当前上下文不同）；或
     amax 与 quant 的 hierarchical reduction。
3. **route 继续微调**。
   - E8 已到 BM64/BN16/BK128/w4/s3；BK256、w8 已排除。
   - E32/E64 的 BM64 两次互有胜负，若再做必须用同窗口三组对照，勿单次晋升。
4. **fused gateup orderW 的 GROUP_M 收尾**。
   - GM16 当前；GM8 基本打平，GM4 慢，GM24 不稳定。
   - 若继续，建议 GM24/GM32 各做两次同窗口，不要用单次。
5. **不要做**：
   - case2 full-fused；
   - side-stream inv argsort（determinism fail）；
   - 全量中间 buffer cache；
   - packed sort（sandbox 禁止 tensor.sort/torch.sort functional）；
   - device-side `tl.make_tensor_descriptor`；
   - num_ctas、A+B TMA、C TMA store、route TMA；
   - fused s5、w16、grid128/160；
   - M-major fused persistent；
   - 重复提交同一文件碰 tb。
