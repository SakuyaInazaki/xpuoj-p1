# 2026-09-27 P1 direct INV_PAD 工作摘要

## 当前榜面

- 只读 scoreboard：P1 score **75.17**，submissionId **149493**，submissionCount 3285。
- 该 75.17 来自 `diag_v2_disable` 的一次基线等价运行；它的 c11 tk=0.325 ms、c12 tk=0.000 ms 属于平台计时异常/热窗口，不是真实结构收益。
- 生产文件已从 v842 晋升为 v843。

## 生产文件

- 旧生产 v842：`5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c`
- 新生产 v843：`a6442ca8d4a717f2e52b4769b697c623c125ab0fd62ce1bf7d859c25d6dcca07`
- 文件：`p1/kernel.py` = `p1/kernel_v843_direct_invpad.py` = `experiments/2026-09-27/candidates/direct_invpad_all_fp8_tiled_c910.py`

## 机制

在十个 FP8-down 正式用例上做同一件事：

1. 稳定 counting sort 的 scatter kernel 额外写 `INV_PAD[original_branch] = compact_pos + delta[expert]`；`delta` 由新增的 `_pad_delta_kernel` 从 `TOT` 计算。
2. DN 输出按 128-row padded expert block 存储：`_dn_tma2_f8_pad_kernel` 或 `_dn_tma2_f8_tiled_pad_kernel` 用 TensorDescriptor TMA store；CSCL 按 padded row 写入。
3. final gather 直接把 `INV_PAD` 当现有 gather kernel 的 source-row map 传入；保持原 branch 累加顺序。
4. 只对启用 case 生效：c3,c4,c5,c6,c7,c8,c9,c10,c11,c12；c1/c2 不进入该路径。

## 关键 SID

- 最早 s3+flatten 版：SID 149486 / 149489，sample/case1 TLE；关闭。
- 没有 flatten 的 s3 版：SID 149496，Accepted，但 c4 比同窗 anchor 慢约 4%；关闭。
- 确认 flatten s4 可用：SID 149502 / 149503 / 149506，均 Accepted；推进为 v9。
- 扩展到 c3/c5/c6/c7/c8/c9/c10/c11/c12：SID 149509 / 149511 / 149514 / 149520 / 149522，均 Accepted。
- 对应 anchor：SID 149510 / 149513 / 149515 / 149521 / 149524，均 Accepted，SQNR 逐案相同。
- c2 BF16 padded 试做：SID 149526，Accepted，但 c2 tk 相对 anchor 无稳定收益，未纳入 v843。

## 真实配对结论

v9 正常窗口（SID 149522 vs 149524，c10/c11/c12 未被平台异常加速）显示：

- c3 −1.3%，c4 约 0%，c5 −0.5%，c6 −1.2%，c7 −1.4%，c8 −1.0%，c9 −0.5%，c10 −1.0%，c11 −1.6%，c12 −1.0%。
- 方向一致为正收益，但幅度约 1%，不足以把多数 case 推过下一档积分门槛；c9 最容易受益，c4/c7/c12 可能偶发过档。
- 平台 c10/c11/c12 偶发 tk 近零/极低，导致单发 display 可达 84–85；这是环境噪声，不能作为结构收益承诺。
- c1/c2 仍是大幅上分的主要缺口，后续需要不同的结构路线。

## 下一步建议

1. 保持 v843 为回退基线，任何新候选先用同窗 anchor 配对。
2. 优先审计 c1/c2 的 MD/DN 大 GEMM 与 final path；当前 direct INV_PAD 对 c1/c2 没有稳定收益。
3. 对 c9/c10 单独确认 v843 是否稳定把 c9 推过 77；若稳定，只保留该 case 可降低风险。
4. 暂停对 c10/c11/c12 异常低 tk 的追逐；需要新机制或更长确认批。

## 继续迭代补充（packed-key 局部排序与 c1）

- 新增 packed-key scatter 原型：`_sort_scatter_packed_kernel`，用 `tl.sort` + `tl.gather` + `tl.associative_scan` 复现稳定局部排名；CPU 整数 oracle 对 E=32/64/96/256、block=64/128/256 均逐元素通过。
- 只替换 c9/c10：SID 149530（首次出现平台异常低 tk，不适合判效）、149534（正常窗口）。
  - 相对 v843 anchor SID 149532：c9 2.490 vs 2.493、c10 1.895 vs 1.910；约 0.1–0.8%，但整案 Σtk 未改善。
- 扩展到 c7/c8/c9/c10：SID 149536 Accepted。
  - 相对 149532：c7 2.175 vs 2.184、c8 1.230 vs 1.239、c9 2.473 vs 2.493、c10 1.896 vs 1.910；touched 约 0.4–0.8%，但 c1 等同窗控制回退，Σtk 28.863 vs 28.800，不晋升。
- c1 padded TMA：SID 149528 Accepted，但 c1 4.640 vs 同窗 v842 anchor 4.601，慢约 0.8%，关闭。
- 上述 packed/c1 候选都不进入 v843 生产；生产仍为 v843。

## v844：c9/c10 compact-packed 路径

- SID 149600 关闭 c9/c10 padded INV_PAD（相对 v843 anchor 149601）：c9 2.472=2.472，c10 1.883 vs 1.895；c9 中性，c10 约 −0.6%。
- SID 149603 / 149604 在关闭 c9/c10 padding 的同时，用 packed-key sort 替换 c9/c10 scatter：
  - 149603 vs 149601：c9 2.465 vs 2.472（−0.3%），c10 1.872 vs 1.895（−1.2%）。
  - 149604 vs 149606：c9 2.454 vs 2.459（−0.2%），c10 1.866 vs 1.897（−1.6%）。
  - 其余 case 未改；两份均 Accepted、SQNR/determinism 相同。
- 该路径减少了 c9/c10 的 padded Down/CSCL 容量与 final 读取，同时用 packed sort 去掉 dense one-hot 局部排名。
- 生产晋升为 v844：`p1/kernel.py` = `p1/kernel_v844_packed_compact_c910.py`，SHA
  `d1f1e0697c3e99a86f3d34bea6509755be6f70470a2373ec0bd355819c671f9e`。
- 预期真实积分收益：c9 基本持平，c10 约 +1 个整数 q；不足以单独冲 77。

## 主计算链负结果补充

- c1/c2 MD ACT TMA store：
  - SID 149609 / 149612，Accepted。
  - c1 约 4.99–5.02 ms，c2 约 8.51–8.52 ms；相对 v844 正常水平 4.60/7.84 ms 明显变慢。
  - 结论：c1/c2 MD pointer store 已优于 TMA descriptor store，关闭。

- c11/c12 token-only `_g` MD：
  - SID 149614，Accepted。
  - c11 0.917 ms vs 正常 0.887 ms；c12 1.567 ms vs 正常 1.48–1.49 ms。
  - 结论：c11/c12 的 sorted-A + TMA A 路径更快，token-only pointer gather 关闭。

- c1/c2 单 `[BM,256]` accumulator + 宽 B load：
  - SID 149630，Accepted。
  - c1 4.706 ms vs 正常 4.60 ms；c2 7.817 ms vs 正常 7.82–7.85 ms。
  - 结论：宽 B load 的 layout permute/split 成本抵消了少一次 B load 的收益，c1 明确变慢；关闭。

## c1/c2 gran=1 interleaved GU 负结果

- 尝试把 c1/c2 静态 GU 从 gran=128 块间 gate/up 布局改为 gran=1 输出通道交错，复用 `_fgs_t1i_mdq_tma_kernel` 单 `[BM,256]` accumulator 路径。
- 首次 SID 149635 因漏传 A strides 在 determinism check 中 TypeError，未进入正式数值/性能。
- 修正后 SID 149636 Accepted，SQNR 逐案相同，但：
  - c1 4.722 ms vs 正常 4.60 ms
  - c2 8.014 ms vs 正常 7.82–7.85 ms
- 结论：gran=1 交错布局 + `_tma` epilogue 对 c1/c2 仍慢，关闭；当前 c1/c2 的 gran=128 双 TMA 设计更优。

## compact TMA DN 与 packed warp 补充

- c4 compact DN TMA store，flatten=True：SID 149647，首测试点 TLE，关闭。
- c4 compact DN TMA store，FLAT=False：SID 149653，Accepted，但 c4 0.849 ms，正常 v844 约 0.79 ms，变慢，关闭。
- c7/c8 packed sort warps 4、c9/c10 warps 2：SID 149643，Accepted，但 c7/c8/c9/c10 相对 149639 变慢，关闭；packed warp 8 保持。
- 结论：v844 的 c4–c8/c11/c12 padded direct INV_PAD + c9/c10 compact packed 仍是当前最优已测结构。

## final gather output TMA store 补充

- 对 `_gather_branch_sum_f8_kernel` 和 BF16 `_gather_branch_sum_kernel_tiled` 增加 `Out_DESC.store`，已知 12 案 T/H 都是 block 整数倍，走无 fallback TMA store。
- SID 149659 Accepted，SQNR/determinism 通过；与同窗 v844 anchor SID 149665 对比：
  - c1 +1.28%（变慢）
  - c2 −0.4%
  - c3 −1.25%
  - c4 −0.4%
  - c5 ~0，c6 +0.4%，c7 −0.3%，c8 +0.1%，c9 +0.3%，c10 +0.4%
  - c11 −11.6%、c12 −6.35% 属于已知平台低 tk 异常，不能作为真实收益。
- 去除异常后整体约中性偏慢；且 c3/c4/c7 即使有 <1.5% 收益也不足以跨积分档位，因此不晋升。
- 最终生产保持 v844。

