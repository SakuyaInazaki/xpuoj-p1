# 方向 A / c2 分配清理晋升记录（2026-09-30）

基线：`p1/references/kernel_v926_measured.py`
SHA-256 `2633cc995eb2563e22c1a212df1256141bdc1b520fe841d4dd4c15c0d382e7bf`。

新生产候选：`p1/kernel.py`
SHA-256 `938ad5c850fb11c828a90d2e9c5e457c141b5981bc4611dc3c684d1cf43ca9bb`
对应候选文件 `experiments/2026-09-30/candidates/p1_dirA_c11_v9_c2clean_meas.py`。

## 改动

1. 新增 `_gq1p_tm_params_kernel` / `_gq1p_tm_params`：在 c11/c12 的 GQ 中按 sorted row 单写 `AH/WI/ACT_SCALE`。
2. 新增 `_fgs_t1i_mdq_tma_pre_kernel` / `_fgs_tma1_intq_host_pre`：MD 直接消费上述参数，删除每个 N tile 重复的 BNORM 加载、bound/bit-op、scale 写。
3. `_run_replicated` 只在 `_dir_a`（E=32, H=1024, I in 1024/2048, call>=3, _GA=0）时切换预计算支路；其他支路逐语句保持 measured 基线。
4. 移除 c2 direct-GQ 分支未读取的 `tokens_sorted` 与 `_gateup_shadow` 分配请求（约 2 GiB）。

静态核对：相对 measured，除上述函数外全部函数 AST 相等，非函数顶层语句顺序不变。

## 关键实现约束

Triton 3.4 系下，把 LIN persistent TMA-MD 的尾部 `tl.store(SCL...)` 删除后，TTGIR pass 会在 `_fgs_t1i_mdq_tma_pre_kernel` 上报 `PassManager::run failed`。试过 `flatten=False`、去 `maxnreg`、AH/WI 乘 1.0 均不能保留 LIN；最终在 pre kernel 内加入一个只写 1 元素的 dummy `tl.store(DROP + 0, 0.0)`，即可让 flatten 支路编译通过且不改变输出。该 dummy 支出结果记录中保留。

## 性能对照（同窗 tik/tb 均为 ms）

控制 A：SID 151866 / 151905（v926 measured）
候选 B：SID 151901/151903（v8，无 c2 清理）与 151912/151918（v9，含 c2 清理）。
四发顺序近似 A-B-B-A 加 B 重复；下表中 B 取 v8/v9 两组。

| 案 | A tk | B tk | 变化 |
|---|---:|---:|---:|
| c11 | 0.874 / 0.874 | 0.853 / 0.859 / 0.857 / 0.856 | −1.7% 到 −2.4% |
| c12 | 1.471 / 1.473 | 1.446 / 1.449 / 1.446 / 1.449 | −1.6% 到 −1.7% |

未触达案无系统回退；c2 清理单独收益不能稳定分辨，但未观察到负向趋势。两次 v9 均完整 AC，SQNR 与其他案保持原水平（c11 min 23.14 dB，c12 min 22.90 dB）。

## 平台观察

本轮新增证据：连续 187 发队列中 7 次 TLE 全部是首次出现的 SHA，重复 SHA 无 TLE；v926 清理版首测 SID 151862 TLE，而 measured 同码重测正常。这与“首次源码/冷 JIT 生命周期影响采集/执行”的假设相容，但仍不是因果证明。对新 SHA 至少预留一次重复提交或一次冷启动 TLE 的时间预算。

## 结论

方向 A 达到指南门槛（c11 两对同向且 ≥1.5% 降时），晋升到 `p1/kernel.py`。c2 分配清理作为低风险 host 清理保留。下一步结构主线仍是 c9 方向 B，按单案两对、2% 门槛验证。
