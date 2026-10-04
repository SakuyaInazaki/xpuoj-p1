# 06 候选文件、当前差距、下一步

## 推荐 base

- **`p1/kernel_115907_backup.py` = 115907**，raw 74.00 / timeUsed **43538**。
- 当前 `p1/kernel.py` = 115950 只是 scoreboard 最佳（case2 tb 异常），不应作为性能 base。

## 关键候选文件

| 文件 | submission | raw | timeUsed | 建议 |
|---|---:|---:|---:|---|
| `p1/kernel_115907_backup.py` | 115907 | 74.00 | **43538** | **推荐 base** |
| `p1/kernel.py` / `p1/kernel_115950_backup.py` | 115950 | 74.08 | 44017 | scoreboard 最佳；tb 异常 |
| `p1/kernel_v50_case2_gather_bm256.py` | 115943 | 73.83 | 43467 | 单次最快；复测波动大 |
| `p1/kernel_115854_backup.py` | 115854 | 72.92 | 45372 | 上阶段 base |
| `p1/kernel_v16_swiglu_bn256.py` | 115854 | 72.92 | 45372 | 同 115854 |
| `p1/kernel_115738_backup.py` | 115738 | 71.08 | 51044 | 早期实际最快 |
| `p1/kernel_115705_backup.py` | 115705 | 71.67 | 53703 | 接手时平台最佳 |

## 当前 raw 80 差距

以稳定 115907 的 tk 估算，raw 80 要求 12 点平均 80 分；按近期 tb，总 tk 需从
约 43.5ms 降到约 28ms，即约 1.55 倍综合加速。主要缺口：

| case | 115907 tk | 80 分粗估需 tk | 缺口 |
|---:|---:|---:|---:|
| 1 | 6.00 | ~4.5 | -1.5 |
| 2 | 10.17 | ~7.2 | -3.0 |
| 3 | 2.32 | ~1.74 | -0.6 |
| 4 | 1.60 | ~1.25 | -0.35 |
| 5 | 4.16 | ~3.16 | -1.0 |
| 6 | 2.39 | ~1.93 | -0.46 |
| 7 | 3.39 | ~2.56 | -0.83 |
| 8 | 2.31 | ~1.79 | -0.52 |
| 9 | 3.41 | ~2.11 | -1.30 |
| 10 | 2.80 | ~1.68 | -1.12 |
| 11 | 2.02 | ~1.41 | -0.61 |
| 12 | 2.98 | ~2.04 | -0.94 |

结论：剩余优化需要结构性改动，不是参数微调。

## phase 诊断结论（115892/115898，v16 时代）

case2 各阶段参考值：
- activation quant：1.709ms
- gateup FP8 GEMM：6.590ms
- SwiGLU amax+FP8 quant：1.445ms
- down FP8 GEMM：2.552ms
- routing/prep：约 0.5/3.0ms（event 数据有噪声）

case2 gateup GEMM 仍是最大单点。其他 case 的 fused gateup 也是主要时间。

## 下一步优先级

1. **本地/远程 4×H800 profiling（NCU/SASS）**。
   - 当前 custom FP8 dot 的真实 tensor-core 利用率未知；远程盲扫已到边际。
   - 重点看 case2 plain gateup persistent kernel 的 register/spill、smem、L2 命中。
2. **case2 gateup GEMM 结构性优化**：
   - full-fused/BN64/BM64/split 均已失败；
   - 可研究方向：更细的 split-N、TMA 或 triton_dist 新版特性、
     gate/up 输出复用、与 SwiGLU quant 的流水线。
3. **fused gateup 的 schedule 优化**：
   - persistent 已试过 132 无收益；可试 grid=264/66 并用 phase 诊断定位。
4. **routing/prep 优化**：
   - 当前 route E256 的 BN64/BK128/BM64 变体都慢；
   - 可试 BM32/BN128 或 split-K，并用 phase 诊断确认 routing 是否真是瓶颈。
5. **不要做**：
   - case2 full-fused FP8（BN128/BN64）、INT8 fused、case2 split gate/up；
   - B 转置布局（修复后也极慢）；
   - fused gather+quant（TLE）；
   - down 写 slot；
   - SwiGLU BN512/BM64/w4/w16、quant BM256/BK256；
   - case2 gateup/down BN128、BK64、s2、GM4/GM16、nonpersistent、BM64+metadata；
   - route BN64/BK128/BM64、down grid264；
   - 重复提交同一文件碰 tb。

## 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_115907_backup.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```
