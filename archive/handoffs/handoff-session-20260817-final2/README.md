# 第二段会话交接（submission 115818-115879）

本段从 `handoff-session-20260817-final/`（115705 平台最佳 / 115738 实际最快）接手，
最终晋升 **115854**：

- raw displayScore：**72.92**
- scoreboard：**62.92**（扣罚 10）
- timeUsed：**45372**（12 点 tk 总和 45.372ms，此前最快 115738 为 51044）
- `p1/kernel.py` / `p1/kernel_115854_backup.py` 与 115854 逐字节一致。

## 本段有效改动（按采纳顺序）

1. 修正 FP8 down persistent kernel 的 launch bug：`kernel[((132,),)]` -> `kernel[(132,)]`。
   旧实验 115660/115682 其实从未跑起来（TypeError: tuple cannot be interpreted as integer）。
2. 删除 custom FP8/INT8/fused gateup GEMM 内层 K-remainder mask。
   当前所有 K 均为 BLOCK_K=128 整数倍；115819 timeUsed 49629。
3. 删除 custom GEMM 的 N-col mask（N 均为 BLOCK_N 整数倍），并删除 FP8/SwiGLU quant kernel 的 M/K mask。
   115825 raw 71.42 / timeUsed 48912。
4. 自写 FP8 activation amax kernel，替代 `a.abs().max()` 物化：
   单遍 load -> abs -> block max -> atomic_max。**本段最大单项收益**。
   115834 raw 71.83 / timeUsed 47532。
5. case2 gateup 从 INT8 改为 plain FP8 grouped GEMM：
   115839 单点 case2 10.433（原 11.6-12.2）。合并进 115841 raw 72.17 / timeUsed 46168。
6. case4/6/11 从官方 BF16 grouped GEMM 改为 FP8 fused gateup + FP8 down：
   115850 raw 72.58 / timeUsed 45801。
7. case2 的 SwiGLU amax/quant kernel BLOCK_N 128 -> 256：
   115854 raw 72.92 / timeUsed **45372**。

## 115854 逐点

| case | tk | tb | 单点分 |
|---:|---:|---:|---:|
| 1 | 6.136 | 18.049 | 74 |
| 2 | 10.184 | 28.788 | 73 |
| 3 | 2.458 | 7.730 | 75 |
| 4 | 1.730 | 5.073 | 74 |
| 5 | 4.403 | 12.670 | 74 |
| 6 | 2.635 | 7.610 | 74 |
| 7 | 3.606 | 10.277 | 74 |
| 8 | 2.511 | 7.154 | 74 |
| 9 | 3.540 | 8.395 | 70 |
| 10 | 2.927 | 6.711 | 69 |
| 11 | 2.131 | 5.647 | 72 |
| 12 | 3.111 | 8.151 | 72 |

注：case3 tb=7.73 偏高，但 timeUsed 45.37ms 是实际最快。

## 失败/无效实验（不要重试）

| id | 实验 | 结果 |
|---:|---|---|
| 115830 | route GEMM 去 mask + SwiGLU BF16 去 mask | 实际变慢，timeUsed 49689 |
| 115833 | fused gateup persistent grid=132 | 与 baseline 打平，无收益 |
| 115836 | 官方 BF16 GEMM num_sms=132 | 实际变慢；raw 高是 case1 tb=40.5 异常 |
| 115843 | FP8 activation quant BLOCK_K=256 | timeUsed 47225，慢于 128 |
| 115848 | case2 使用 full-fused FP8 gateup（同其他 case） | TLE，确认非偶然 |
| 115856 | case2 两段式 fused gateup（先写 gate BF16，再算 up+act） | case2 13.014，明显慢 |
| 115859 | case2 SwiGLU quant BLOCK_N=512 | TLE |
| 115860 | case2 gateup GROUP_M=16 | timeUsed 45777，慢 |
| 115861 | fused gateup num_stages=2 | timeUsed 52889，明显慢 |
| 115866 | `g/(1+exp(-g))` 改 `tl.sigmoid` | timeUsed 46065，慢 |
| 115876 | case2 SwiGLU quant BLOCK_M=64 | timeUsed 45779，慢 |
| 115878 | case2 down FP8 BLOCK_N=128 | timeUsed 46599，慢 |
| 115879 | case2 gateup FP8 BLOCK_N=128 | timeUsed 48664，慢 |

## 接手命令

```bash
cd /home/sakimi26/xpuoj-p1
python scripts/best_score.py
sha256sum p1/kernel.py p1/kernel_115854_backup.py
python -m py_compile p1/kernel.py
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```

关键 SHA-256：

```text
84fb09987acf1761397f1bda2c87d05780f02eb826df3abeefaaf07065424426  p1/kernel.py (=115854)
84fb09987acf1761397f1bda2c87d05780f02eb826df3abeefaaf07065424426  p1/kernel_115854_backup.py
```
