# 下一步（按性价比）

1. **case2 仍是最大单点（10.18ms）**。full-fused FP8 会 TLE，两段式 fused 也慢。
   可尝试：
   - case2 gateup 用 FP8 plain，但为它单独扫描 GROUP_M=4/16、down BN128/256、persistent grid 数；
   - 把 case2 SwiGLU amax/quant 的 BLOCK_M 从 128 调到 64（BM256/BN512 不要试，已 TLE/慢）。
2. **case9/10 的 FP8 persistent down 与 gateup**：
   - 当前 fused gateup 非 persistent，v6 的 persistent fused 无收益；
   - 可试 case9/10 单独 GROUP_M=16 或 BN64（全局扫描历史上都慢，但当前 amax 已变）。
3. **B 转置布局 [G,K,N]**：
   - 旧实验 115597 非法访存且 SQNR 错，疑似 stride/量化拷贝有 bug；
   - 当前 custom kernel 已大改（int64、no-mask、persistent），可重新做最小修复。
4. **量化 tile 微调**：
   - FP8 activation amax/quant 当前 BM128/BK128，BK256 已证慢；
   - 可试 BM64 或 BM256（保持元素数 <=32K）。
5. **不要做**：case2 full-fused、swiglu BN512、route 去 mask、官方 BF16 num_sms=132、fused s2、tl.sigmoid 替换。
