# 06 候选文件、当前差距、下一步

## 推荐 base

- **`p1/kernel_fused_gateup_swiglu.py` = 115738**，raw 71.08 / timeUsed 51044。
- 这是当前实际最快、最值得继续迭代的 base。115705 只是平台显示最佳，不应作为性能 base。

## 关键候选文件

| 文件 | submission | raw | timeUsed | 建议 |
|---|---:|---:|---:|---|
| `p1/kernel.py` / `p1/kernel_gm2.py` | 115705 | 71.67 | 53703 | 平台最佳；GROUP_M=2，case6 tb 异常虚高 |
| `p1/kernel_fused_gateup_swiglu.py` | 115738 | 71.08 | **51044** | **推荐 base** |
| `p1/kernel_115696_backup.py` | 115696 | 71.08 | 51419 | INT8+FP8 swizzle，无 FP8 epilogue 融合 |
| `p1/kernel_115691_backup.py` | 115691 | 70.00 | 54398 | 仅 FP8 swizzle |
| `p1/kernel_115209_backup.py` | 115209 | 69.67 | 56185 | 本会话早期 canonical |
| `p1/kernel_case10_fp8both_int64ptr.py` | 115185/115190/115202 | 69.33/69.00/68.33 | 57204/56903/59445 | case10 int64 修复里程碑 |
| `p1/kernel_case9_repl_fp8_lowmem.py` | 115209 | 69.67 | 56185 | case9 lowmem 里程碑 |

## 当前 raw 80 差距

以最快 115738 的 tk 估算，raw 80 要求 12 点平均 80 分。按近期 tb，总 tk 需从约 51ms 降到约 28ms。主要缺口：

| case | 115738 tk | 80 分粗估需 tk | 缺口 |
|---:|---:|---:|---:|
| 1 | 6.61 | ~4.5 | -2.1 |
| 2 | 12.23 | ~7.2（tb29）/ ~10.9（tb43） | -5.0 或 -1.3 |
| 3 | 2.75 | ~1.74 | -1.0 |
| 4 | 1.97 | ~1.25 | -0.7 |
| 5 | 4.90 | ~3.16 | -1.7 |
| 6 | 3.12 | ~1.93 | -1.2 |
| 7 | 3.98 | ~2.56 | -1.4 |
| 8 | 2.85 | ~1.79 | -1.1 |
| 9 | 3.76 | ~2.11 | -1.6 |
| 10 | 3.14 | ~1.68 | -1.5 |
| 11 | 2.36 | ~1.41 | -0.95 |
| 12 | 3.38 | ~2.04 | -1.3 |

结论：需要约 1.8 倍综合加速，不是参数微调能达到。

## 下一步优先级

1. **修复 INT8 gateup 融合 SwiGLU 的 TLE/寄存器问题**。
   - case2 是最大单点。当前 INT8 gateup 仍先写 `[M,2I]` BF16 gateup，再 `_swiglu_quant_fp8` 两遍读。
   - 方向：BLOCK_M=64 + BLOCK_N=64/32、单 accumulator 复用、两阶段共享内存、或只把 INT8 gateup epilogue 改成写 gate/up FP8/BF16 分片。
2. **修复 custom FP8 B 转置布局 [G,K,N]**。
   - 可改善 B tile 访存；当前 transposed 版本 case1 illegal memory，需要本地或最小 probe 定位。
   - 参考 official kernel 用 `offs_bn % N`、int64 row offsets、逐步开放部分 case。
3. **FP8 persistent kernel**。
   - 当前 132-programs 多 tile loop 全错；应先用 grid=total_tiles*nblocks 的单 tile loop 验证，再逐步减少 grid。
4. **down GEMM epilogue 与 final gather 融合**。
   - 当前 down 输出按 expert 排序的 `[T*k,H]` BF16，再 `_gather_branch_sum`。可尝试 down 直接写 per-token 的 k 个 BF16 slot 或 FP32 partial。
5. **本地 4×H800 NCU/SASS**。
   - 当前 custom fp8 dot 的真实 tensor-core 利用率未知；远程盲扫已到边际。
6. **不要再做**：
   - 重复提交同一文件碰 tb；
   - BM256/metadata block_m=256、BM64 half-tile、w4/s2/GM16/GM4/GM2、route BK256、official FP8 BF16 输出、B 转置未修复版、persistent 未修复版、INT8 fused TLE 原样重试。

## 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_fused_gateup_swiglu.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```
