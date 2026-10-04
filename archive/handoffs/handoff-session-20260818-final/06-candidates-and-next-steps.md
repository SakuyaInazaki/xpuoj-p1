# 06 候选文件、当前差距、下一步

## 推荐 base

- **`p1/kernel_116310_backup.py` = 116310**：当前实际性能 base。
- scoreboard 文件：`p1/kernel_116142_backup.py`（不推荐作为性能 base）。
- 旧稳定文件：`p1/kernel_115907_backup.py`。

## 关键候选文件

| 文件 | submission | raw | timeUsed | 建议 |
|---|---:|---:|---:|---|
| `p1/kernel.py` / `p1/kernel_116310_backup.py` | 116310 | 74.67 | **42881** | **当前 base** |
| `p1/kernel_v98_token_row.py` | 116310 | 74.67 | 42881 | 当前 base 命名文件 |
| `p1/kernel_116142_backup.py` | 116142 | 74.75 | 43040 | scoreboard best，tb 异常 |
| `p1/kernel_v80_down_tma_hostdesc.py` | 116236 | 74.33 | 42592 | 单次最快中间版 |
| `p1/kernel_v95_rowscale_fp8.py` | 116295 | 74.08 | 43427 | down act per-row 中间版 |
| `p1/kernel_115907_backup.py` | 115907 | 74.00 | 43538 | 旧稳定 base |
| `p1/kernel_115950_backup.py` | 115950 | 74.08 | 44017 | 前 scoreboard best |

## 当前 raw 80 差距

以 v98 两次 tk 中位数和典型 tb 估算，当前多数 case 距下一档只差 0.02-0.17ms：

| case | tk 中位数 | 典型 tb | 当前分 | 下一档需 tk <= | 缺口 |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.077 | 18.00 | 74 | 6.000 | 0.077 |
| 2 | 10.249 | 28.90 | 73 | 10.154 | 0.095 |
| 3 | 2.321 | 7.00 | 75 | 2.211 | 0.110 |
| 4 | 1.599 | 5.00 | 75 | 1.579 | 0.020 |
| 5 | 4.180 | 12.70 | 75 | 4.011 | 0.169 |
| 6 | 2.389 | 7.50 | 75 | 2.368 | 0.021 |
| 7 | 3.330 | 10.20 | 75 | 3.221 | 0.109 |
| 8 | 2.226 | 7.10 | 76 | 2.121 | 0.105 |
| 9 | 3.355 | 8.35 | 71 | 3.247 | 0.107 |
| 10 | 2.747 | 6.65 | 70 | 2.716 | 0.030 |
| 11 | 1.879 | 5.65 | 75 | 1.784 | 0.095 |
| 12 | 2.853 | 8.15 | 74 | 2.717 | 0.136 |

raw 80 需要 12 点平均 80，仍需结构性加速；但 raw 75-76 只需多个 case 各快 0.05-0.15ms，
是现实目标。**任何新优化先看 timeUsed 和 tk 中位数，不要看单次 tb 异常。**

## 下一步优先级

1. **本地/远程 4×H800 NCU/SASS profiling**。
   - 当前 custom FP8 dot 的真实 tensor-core 利用率未知；盲扫已到边际。
   - 重点：case2 plain gateup persistent kernel 的 register/spill、smem、L2；
     fused gateup 的 dual-dot 调度；down TMA kernel 的 TMA 利用率。
2. **case2 gateup 结构性优化**。
   - 当前 10.15-10.35ms，是最大单点。
   - 已排除：full-fused、split gate/up、TMA B、BN128、BK64/256、s2/s4、GM4/16、
     nonpersistent、BM64+metadata。
   - 可研究：更细 split-N、TMA 新版本特性、gate/up 输出复用、与 SwiGLU quant 的流水线。
3. **fused gateup schedule**。
   - 当前非 persistent + per-row AMAX 是有效组合。
   - persistent/M-only 重排已因 SQNR 错误放弃；但可做更细 phase 诊断，
     判断 token quant、dual dot、epilogue 各自占比。
4. **routing/prep 微调**。
   - 当前 gather+per-row amax + per-row quant 已是最佳。
   - route TMA、topk(logits)、逆置换均未胜；不要重试。
   - 可试 BM64/BH64/BN 变体并用 phase 诊断确认，但预期收益小。
5. **不要做**：
   - device-side `tl.make_tensor_descriptor`；
   - `num_ctas=2`；
   - per-K-block activation scale（tiled down）；
   - fused gateup persistent/M-only batched row-amax（SQNR 错）；
   - case2 full-fused、split gate/up、B 转置、fused gather+quant、down slot；
   - TMA route、TMA case2 gateup、TMA fused gateup、A+B TMA、C TMA store、contiguous-M TMA；
   - down BK256/s2、BN128、s4、num_ctas；
   - case2 token per-row、SwiGLU act 缓存、row-gather BM256；
   - 重复提交同一文件碰 tb。

## 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_116310_backup.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```
