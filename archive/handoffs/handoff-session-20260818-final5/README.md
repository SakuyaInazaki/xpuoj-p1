# XPUOJ P1 MegaMoE 续接（当前 base 117300/v233）

> final4 的 v159 之后继续迭代。当前实际 base 已是多次 tiling/微调后的组合。

## 10 秒结论

- **当前实际性能 base：`p1/kernel.py` = 117300（v233）。**
  - 117300：Accepted，raw 73.92，timeUsed 43266。
  - 后续复测 117304=43325，case2 相比对照稳定快约 0.16-0.26ms。
- 当前 base 组合：
  1. final gather tiling：case1/2 BT8/BH1024/w8；其余 BT32/BH1024/w32。
  2. case2 token gather：BM64/BH256/w8。
  3. case2 token FP8 quant：BM128/BK64。
  4. case2 SwiGLU：BN64（amax s1 / quant s2）。
  5. route：E8 BM64/BN16/BK128/w4；E32 BM64/BN32/BK128/w4；
     E64 BM128/BN64/BK128/w8；E96 BM64/BN128/BK128/w4。
  6. 非 case2 fused gateup：BM128/BN128/BK128/GM32/w8/s4/grid132。
- scoreboard best：116961，raw 76.67，扣罚后 66.67（case5 tb 异常）。
- submissionCount 已 904；扣罚上限 10。

## 检查

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_117300_backup.py
python -m py_compile p1/kernel.py p1/kernel_117300_backup.py
```

期望 SHA：

```text
14329d914f56144d2ab181eee096e96b61f6862ced5af23af8c7e9e967f1a8c5  p1/kernel.py
14329d914f56144d2ab181eee096e96b61f6862ced5af23af8c7e9e967f1a8c5  p1/kernel_117300_backup.py
```

## 有效晋升（累计）

| id | 改动 | raw | timeUsed |
|---:|---|---:|---:|
| 117009 | final gather BT32/w32 | 74.50 | 42779 |
| 117050 | route N96 BM64/BK128 | 73.67 | 43403 |
| 117077/117135 | case2 SwiGLU BN64 | 74.75/73.92 | 42795/43137 |
| 117151/117155 | route N96 w4 | 75.75/74.17 | 43002/43179 |
| 117157/117160 | fused GM32 | 74.25/74.17 | 42966/43215 |
| 117207/117211 | route E32 BM64 | 74.17/74.00 | 42980/43235 |
| 117218/117221 | case2 gather BM64 | 74.17/73.83 | 42903/43206 |
| 117234/117237 | case2 gather BM64/BH256 | 74.17/74.42 | 42961/43233 |
| 117300/117304 | case2 token quant BK64（4 窗中 3 窗 case2 更快） | 73.92/73.75 | 43266/43325 |

## 已排除（避免重试）

- case2 dual gateup/interleave/amax/normal：慢或 OOR。
- case2 SwiGLU BN32、gather BM32/BH512。
- non-case2 gather BM64/BH256：3 窗中 2 窗更慢，不采用。
- token quant BM64/BK32：不稳定。
- down w4/w16；fused grid264/GM64/GM32+s3。
- route E8 s4/BM32、E64 BM64/w4、E32 BM32/BN16。
- case2 gateup grid264/BK256/s2。
- final H1024 BT64/w16、H3584 BH512、case9/10 BT16：无稳定收益。

## 下一步

1. case2 plain gateup 仍是最大单点；参数已扫尽，建议 profiling 或结构性改动。
2. 平台偶发长时间 Pending；提交后异步处理，不要同步死等。
3. 同窗口 A/B 仍投当前 `p1/kernel.py` 作对照。
