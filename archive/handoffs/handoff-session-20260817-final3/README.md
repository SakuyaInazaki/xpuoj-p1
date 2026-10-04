# 第三段会话交接（submission 115892-115961）

从 115854（raw 72.92 / scoreboard 62.92 / timeUsed 45372）继续迭代：

- scoreboard 最佳：**115950**，raw 74.08 / 扣罚后 64.08（case2 tb=46.98 异常）
- 稳定实际最快：**115907**，raw 74.00 / timeUsed **43538**
- 当前 `p1/kernel.py` = 115950；性能 base 为 `p1/kernel_115907_backup.py`

## 有效改动

1. 新增 `_gather_tokens_amax_kernel`：
   - 在按 expert 排序 gather token 的同时计算全局 activation amax（单遍读 + atomic_max）。
   - 替代 `x[token_idx].contiguous()` + 单独 amax pass。
   - 对除 case2 外的所有 case 有效（115905：raw 73.33 / timeUsed 43978）。
2. case2 保持 torch gather + 原 `_quant_act_fp8`（两遍 sorted tokens），
   因为 case2 用 fused gather+amax 反而慢约 0.12ms。
   该 hybrid 为 115907：raw 74.00 / timeUsed **43538**。

## 115907 逐点

| case | tk | tb | 单点分 |
|---:|---:|---:|---:|
| 1 | 6.003 | 18.108 | 75 |
| 2 | 10.172 | 28.943 | 73 |
| 3 | 2.315 | 7.111 | 75 |
| 4 | 1.598 | 5.121 | 76 |
| 5 | 4.158 | 12.770 | 75 |
| 6 | 2.390 | 7.642 | 76 |
| 7 | 3.390 | 10.283 | 75 |
| 8 | 2.308 | 7.280 | 75 |
| 9 | 3.411 | 8.381 | 71 |
| 10 | 2.795 | 7.167 | 71 |
| 11 | 2.015 | 5.666 | 73 |
| 12 | 2.983 | 8.190 | 73 |

## 本段 phase 诊断结论（115898，v16 时代）

case2 各阶段参考值：
- activation quant：1.709ms
- gateup FP8 GEMM：6.590ms
- SwiGLU amax+FP8 quant：1.445ms
- down FP8 GEMM：2.552ms
- routing/prep：约 0.5/3.0ms（event 数据有噪声）

所以 case2 下一阶段的重点是 gateup GEMM（6.59ms）和 routing/prep。

## 失败/无效实验（不要重试）

| id | 实验 | 结果 |
|---:|---|---|
| 115895 | case2 gateup nonpersistent | timeUsed 45501，慢 |
| 115896 | route E256 BM64（初版有 UnboundLocal bug） | WA；修复后仍无收益 |
| 115898/115892 | phase 诊断提交 | 用于取 phase 数据 |
| 115899 | case2 gateup num_stages=2 | 47787，慢 |
| 115900 | case2 gateup BLOCK_K=64 | 48169，慢 |
| 115903 | FP8 quant BLOCK_M=256 | 45864，慢 |
| 115905 | 全 case fused gather+amax（v31） | 有效，但 case2 略慢 |
| 115908 | route E256 BM128/BK128/s2 | 44556，慢 |
| 115911 | gather BLOCK_H=256 | 44748，慢 |
| 115915 | B 转置 [G,K,N] 修复 stride 版 | 正确但 299912，严重慢 |
| 115918 | gather num_warps=4 | 44100，慢 |
| 115919 | case2 gateup BM64（metadata 未同步） | SQNR 3dB，WA |
| 115922 | case2 SwiGLU quant num_warps=16 | 44101，慢 |
| 115923 | route E256 BM64/BK128 | 44131，慢 |
| 115924 | down H=1024 BLOCK_N=128 | 44936，慢 |
| 115927 | case2 full-fused FP8 BN64 | TLE（再次确认） |
| 115929 | case2 SwiGLU quant num_warps=4 | 46617，慢 |
| 115930 | case2 用 x 的 amax（避免 sorted 重复读） | 44083，慢 |
| 115933 | gather 不带 amax + 单独 x amax | 43938，慢 |
| 115935 | case2 down nonpersistent | 44122，慢 |
| 115937 | case2 gateup GROUP_M=4 | 43916，慢 |

## 追加实验（115939-115961）

| id | 实验 | 结果 |
|---:|---|---|
| 115939/115942 | fused gateup BM64 + metadata64 | 首次 tuple bug；修复后 TLE |
| 115943 | case2 gather BM256/H128/w8（v50） | timeUsed 43467，case2 10.084，实际略快但 raw 受 tb 波动 |
| 115944/115946 | fused gather+FP8 quant（省 sorted BF16） | 首版 pointer bug，修复后 TLE |
| 115950 | case2 gather BM256/H128/w4 | raw **74.08**，case2 tb=46.98 异常，timeUsed 44017 |
| 115951 | 全 case gather BM256/H128 | 44023，慢 |
| 115954 | case2 gather BM256/H64 | 44542，慢 |
| 115956 | case2 gather BM512/H64 | 44571，慢 |
| 115957 | v50 复测 | 44517，波动，无稳定收益 |
| 115959 | case2 plain GEMM BM64 + metadata64 | TLE |
| 115961 | down persistent grid=264 | 44406，慢 |

## 追加实验（115964-115966）

| id | 实验 | 结果 |
|---:|---|---|
| 115964 | route E256 BLOCK_N=64/BM128/BK64 | 44092，慢 |
| 115965 | down 直接写 [T,k,H] slot + final 顺序归约 | 45099，down 写 slot 的代价大于 final 收益 |
| 115966 | case2 gate/up 拆成两个独立 FP8 GEMM | TLE |

## 下一步

1. case2 gateup GEMM 仍是绝对瓶颈。BN128/BK64/BK128/s2/GM4/GM16/nonpersistent 都已试过，BM64 需同步 metadata BLOCK_SIZE_M=64 后才可重试。
2. routing/prep 中 argsort+gather 疑似仍较大，可对 case2 单独做更细 phase 诊断。
3. B 转置已证明在当前 Triton 版本严重变慢，不要继续。
4. 当前最优所有专家 GEMM 全 FP8；量化 tile BM128/BK128 是实测最佳。
