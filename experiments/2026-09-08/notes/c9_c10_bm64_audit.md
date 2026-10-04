# c9/c10 BM64 short audit

现役 c9/c10 的 BM 粒度都是 128。call 3–5 的 MD 为 `_fgs_tma1_kernel_gq`：BM128/BN128/BK128、persistent grid 132、GM32、w8/s4；DN 为 `_dn_tma2_f8_kernel`：BM128/BN256/BK128、persistent grid 132、GM32、w8/s4、`FLAT=true`。其他 call 会切换 kernel 变体，但仍使用 BM128 metadata；call 1–2 的 DN 是 GM8/s3，不能与上述稳态配置混称。

`T*topk/E = 4096*8/256 = 128` 是每专家平均行数的算术事实。仓内没有找到 c9/c10 的真实 `expert_counts` 或 `num_tiles_total` 日志；`num_tiles_total` 由 GPU metadata kernel 生成后直接被 persistent kernel 读取。因此“BM128 平均 1.5 tile、padding 额外 FLOPs 约 50%”依赖 counts 围绕 128 分布的模型，不是实测事实。

直接历史证据：

- V284，SID 120865：对 `E>=128` 同步使用 BM64 metadata，并把 c9/c10 的 MD/DN 改为 grid 132 的 BM64。c9 为 3.113 ms，对照约 2.86 ms，慢约 9%；c10 为 2.433 ms，对照约 2.26 ms，慢约 7.5%。记录归因为 BM64 增加 B 权重重读。
- V571：在 BM128 主循环后用 BM64 回收尾块，第三次提交 Accepted 后 c9/c10 分别慢约 19.5%/19.6%；现存短记录没有保留该次 SID。此前两次 TLE 不能替代这次有效结果。

grid 264 是另一条占用率实验轴，不能替代以上 grid 132 证据。V386/V387 的 BM64/BN256 只在 `E<=96` 分支调用，不覆盖 c9/c10。现有直接负证据足以关闭 c9/c10 BM64 与单独尾块回收路线，但不足以把 50% padding 写成已测事实。
