# 当前状态

更新时间：2026-09-09（平台与本地文件已复核；实验文件仍归入 2026-09-08 批次）

## 可信边界

- 当前开发入口是 `p1/kernel.py`，SHA-256 `dd46bdebb7be2eed2f1ebe1106be258789bde35e4ee6c9421d5756b163f426b9`。截至 2026-09-09 未换底盘；不要用历史 handoff 中的 `kernel.py` 映射覆盖它。
- 最后实时复核的榜面最佳为 **72.75**，SID **141408**；最近明确核验的累计提交数为 **2585 次（SID 141465 时点）**，目标 75 尚未达到。该次 displayScore 82.75 含 c3 `tb=42.634 ms` 异常，不能解释为稳定提速。
- P1 最后明确已知的评测环境是 distributed Triton 3.4.0；截至 2026-09-08 没有升级证据。
- 基线验收仍有一项边界：不同 call 数走到的量化路径尚未证明任意跨 call 逐位一致；这是一项待补证据，不能仅据此断定规则违规。
- 根目录旧 `README.md` 曾停留在榜面 66.00，旧 `HANDOFF.md` 更早，均已归档，不能作为当前结论。
- 本地没有 P1 的 4×H800 等价运行环境；本地静态检查不能替代平台正确性和性能结果。

## 当前目标

- 保持 `p1/kernel.py` 为稳定对照，所有候选独立保存到 `experiments/2026-09-08/candidates/` 或明确的 `p1/` 历史文件。
- 榜面达到 75 需要某次提交 displayScore 至少 85.00，即 12 案整数分之和至少 1020。
- 性能判断使用同窗口候选/锚对和逐案数据；不能把单次 `tb` 异常或单一 `Σtk` 阈值当成充分证据。

## 已确认结论

- SID 141390 是本轮同窗口稳定基线，Accepted、displayScore 81.00、12 案 `Σtk=30.436 ms`。
- SID 141389 在 samples 配置 1 阶段 TLE；TLE 前两次 SQNR 检查通过，但没有产生 schema 计时。该次 TLE 原因未定位，不能断定为编译失败或双 CTA 候选本身失败。
- SID 141395 是双 CTA 短 K 候选的唯一顺序同码诊断：全 12 案 Accepted、displayScore 81.08、`Σtk=30.133 ms`，逐案 SQNR 与 SID 141390 一致。目标 c4/c6/c8/c11 的 raw 分别慢约 6.0%/8.4%/7.6%/5.9%；未触达 8 案 median ratio 为 0.97835，校正后目标案分别慢约 8.3%/10.8%/9.9%/8.3%。root 已淘汰该路线，不再扩范围或扫描参数。
- SID 141397 是固定 shape dispatch 候选的全量提交：外层 `compile.success=true`，但在 samples 配置 1 首调约 500.99 秒后 TLE，零 SQNR、确定性或 schema 计时，且没有 `ptxas` 栈；现有证据无法区分 JIT、运行或同步问题。root 已停止该候选且不重发。
- SID 141408 是仅 c5/c7 固定原 call 3 数学路径的 partial 候选：全 12 案 Accepted、displayScore 82.75、`Σtk=29.750 ms`，确定性通过；c5 SQNR `23.13/23.13`，c7 `23.11/23.12`。相对 SID 141390，c5/c7 raw 分别快 2.81%/2.66%，未触达 10 案 median ratio 为 0.983942、sum ratio 为 0.978476，校正后的真实小差约 0.5–1.2%，不足以认定为可靠加速，因此不晋升 `p1/kernel.py`。c3 的 `tb=42.634 ms` 异常抬高了已核实的榜面分。
- SID 141410 是 partial 底盘的 MD 重放诊断：全 12 案 Accepted，所有 SQNR 与 SID 141408 一致；同一 MD kernel 的新增增量在 c5 约 1.748–1.750 ms、c7 约 1.310–1.312 ms，未触达 10 案 median ratio 为 1.001742、sum ratio 为 1.001256。该结果证明固定 c5/c7 路径上的探针增量有效，并取代旧 call-gated 差分解释；它不是优化候选。
- SID 141411 是同一 partial 底盘的 DN 重放诊断：全 12 案 Accepted；同一 DN kernel 的新增增量在 c5 约 0.861–0.864 ms、c7 约 0.651–0.653 ms。它是固定路径增量测量，不是优化候选。
- SID 141413 的 MD BN64 双 CTA 候选在 samples 阶段完成两次 SQNR 后 TLE，没有目标案数据；SID 141415 是唯一同码顺序复现，全 12 案 Accepted，但按未触达案校正后 c5 慢约 22.5–23.3%、c7 慢约 13.9–14.6%。root 已淘汰该路线，不再扫参数；首次 TLE 本身仍未定位。
- SID 141422 的 graphsafe bulk 草案约 500.978 秒 TLE，samples 阶段零 SQNR、零计时且没有目标案数据；代码仍缺 `put→fence` 的 CTA issuance 同步，因此该版本已作废，也不能用于估算收益。
- SID 141431 是补入两次 CTA sync 后的 bulk 同路线候选：外层编译成功，但仍在 samples 配置 1 首调约 500 秒后 TLE，零 SQNR、零 schema 计时且没有 c9 数据。当前无法取得 EP 收益预算，bulk 路线冻结。
- SID 141442 是保持原 JIT 行号布局的全固定路径候选：外层 `compile.success=true`，但在 samples 配置 1 首调约 500.916 秒后 TLE，零 SQNR、零确定性、零 schema 计时。该结果没有精度或速度数据，不能归因为数学路径失败；全量 stable 及其 SHA `448277e8…` 的 FP6 子候选均冻结且不晋升。
- SID 141446 的 c9/c10 FP6 roundtrip 探针被语言校验器拒绝以下划线开头的新函数绑定，零目标 SQNR；合规改名的 SID 141448 非目标 10 案 Accepted，但 c9/c10 在 FP6 执行前因沙箱禁止 `Tensor.data_ptr()` 而失败，仍无目标 SQNR。这两次结果都不是精度或编译失败证据。
- SID 141451 去除 `data_ptr()` 后，非目标 10 案 Accepted，c9/c10 首调 SQNR 分别为 22.63/22.64 dB；确定性复调进入 full-FP8 权重重建后 OOM，因此第二次 SQNR 和跨调用确定性未知。记录中没有对象 ID，不能判断缓存 miss 来自 guard 还是 clone。
- SID 141454 的静态 shape-cache FP6 精度底盘全 12 案 Accepted，且每案两次确定性检查均通过；c9 SQNR 为 22.63/22.64 dB，c10 为 22.63/22.65 dB。它只验证同一测试点内固定权重的 shape 复用，不覆盖连续两个同 shape、不同权重测试点，不能作为完整缓存合同或压缩提速证据。
- SID 141457 的 packed-MD release 为目标 partial WA：非目标 10 案 Accepted；c10 SQNR 22.63/22.65 dB、确定性通过，但 `tk=12.955 ms`，约为 SID 141454 的 6.50 倍；c9 两次 SQNR 22.63/22.64 dB 后在确定性复调 OOM，没有有效计时。平台未给出寄存器、shared、spill 或栈信息，OOM 根因及 identity cache 未命中来源均未定位，不能确定归因为包装器或 clone。
- SID 141460 的静态派生 cache 对照保持 packed GPU 代码不变：非目标 10 案 Accepted；c9 仍在确定性复调 OOM、无有效计时，c10 `tk=13.141 ms`，校正后约为 SID 141454 的 6.59–6.60 倍。该 host-cache 改动没有消除观测到的现象，但不能据此定位原因。
- SID 141462 的公开 stdout/资源诊断同样为目标 partial WA：c9 复调 OOM、无计时，c10 `tk=13.030 ms`；平台运行字段没有捕获任何 `FP6_*` 标记，也未返回寄存器、spill 或 shared 数字，因此没有新增运行期诊断证据。
- SID 141465 的 c9 w32 bulk 通信预算全 12 案 Accepted；相对 SID 141454，未触达 11 案校正后的 c9 总增量为 **0.80673/0.81102 ms**（median/sum）。该数包含 quant、combine、CTA sync/fence/signal 与最终 reuse barrier，不是纯链路带宽，也不足以支持大改 EP。
- SID 141469 的主动 RuntimeError 资源诊断在 c9/c10 均取得 `regs=255`、`spills=180`、`shared=131088`；Triton 3.4 driver 将 `n_spills` 定义为 `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4`，所以 180 表示 720 bytes/thread local memory，不是 180 条 spill 指令或已证实的 180 个寄存器，也不能单独解释 OOM。错误发生在 cold launch 返回后、SQNR/确定性/计时前，仅用于暴露资源数据，根因仍未完整定位。
- SID 141470 的 thread0 fence 对照全 12 案 Accepted，c9 `tk=3.451 ms`；相对 SID 141454 的校正总增量为 **0.81839–0.82100 ms**，相对 SID 141465 校正后又慢 **0.00738–0.01828 ms**，没有观察到 fence0 收益。root 已关闭当前 bulk 优化和完整 EP 投入，两版均不重试。
- SID 141473 的 FP6 packed BN64 partial 诊断在 c9 取得 `regs=255`、`spills=10`（40 bytes/thread local memory）、`shared=90128`；c10 Accepted、`tk=8.926 ms`，相对 SID 141454 校正后仍慢约 4.475–4.480 倍。资源压力虽下降，但没有形成提速。
- SID 141474 的 FP6 packed SWAR4 partial 诊断在 c9 取得 `regs=255`、`spills=4`（16 bytes/thread local memory）、`shared=131088`；c10 SQNR 22.63/22.64 dB、确定性通过、`tk=6.393 ms`，相对 SID 141454 校正后仍慢 3.2119 倍。root 已关闭当前两平面 FP6→FP8 MD 路线，不组合 tile、不修整合或缓存。
- 缓存审计确认晋升前缺口：题面只保证同一测试点内权重与 `topk` 不变，切换测试点后可能变化；当前活跃的部分权重派生缓存只按 shape 命中，同 shape 换权重时不会更新，不能假设“同 shape 永远同权重”。本轮暂不修改缓存。
- SID 139314 是 2026-09-05 时的历史榜面最佳，但含有利 `tb` 采样；它不是稳定速度基线。
- 2026-09-05 审计确认 `p1/kernel.py` 与保存的 #76/V692 参照在三次复核中性能等价，差异低于可靠检测尺度。
- v811d（SID 139917）通过正式正确性，displayScore 81.67；case 9 最薄 SQNR 余量只有约 +0.22 dB，不能把精度探针直接当成可用加速。
- 历史融合 c11 候选多次在 `ptxas sm_90a` 阶段失败。SID 140261（s2）与 140263（s3）均在 **2026-09-05 提交，2026-09-08 读取终态**，不是本轮新提交：前者 Accepted 80.75，c11 1.280 ms，机器校正后比锚 140253 的 1.006 ms 慢约 27%；后者 Accepted 81.25，c11 1.037 ms，按未触达 11 案校正后慢约 4.5–4.9%。本方向已关闭，不再复测。
- int6 c9 原型曾 500 秒 TLE，未形成可复用性能收益。
- 完整的 2026-09-05 证据仍在 `reports/`；它是历史基线，不自动代表 2026-09-08 实时状态。

## 本轮工作区

- 双 CTA 候选：`experiments/2026-09-08/candidates/dn_two_cta_shortk.py`
- 固定 shape dispatch 候选：`experiments/2026-09-08/candidates/stable_shape_dispatch.py`，SHA-256 `67340914ae3759c84dd541ec5cece0f8305fdeb8a517286bc8105fc05128f9ea`；静态检查与独立审查通过，但 SID 141397 首调 TLE，已停止。
- 基于该候选的 MD/DN c5/c7 重放诊断已冻结且未提交；入口为 `experiments/2026-09-08/notes/stable_profile_c5_c7.md`。
- partial c5/c7 候选：`experiments/2026-09-08/candidates/stable_c5_c7_dispatch.py`，SHA-256 `64ad7cf0f5b9a1c39747c326310d0084e259bd96681b4cc546eede85f73c1b50`；SID 141408 全 12 案 Accepted，可作为诊断底盘，但不晋升主基线。
- 基于 partial 的 MD/DN 重放探针为 `experiments/2026-09-08/candidates/c57_profile_md.py` 与 `experiments/2026-09-08/candidates/c57_profile_dn.py`：SID 141410 与 141411 均已全量 Accepted。旧全量 stable 底盘的两份探针继续冻结。
- MD BN64 双 CTA 候选 `experiments/2026-09-08/candidates/md_bn64_two_cta_c57.py` 已由 SID 141415 证明明显变慢，正式淘汰。
- `experiments/2026-09-08/candidates/bulk_c9_comm_graphsafe.py`（SHA-256 `a9dc6b66853183be4a9f0fee44729533292044fc49f3e9e63c24dcd614e01bf8`）对应已作废的 SID 141422；旧 `p1/codex_bulk_c9_comm_probe.py`（SHA-256 `36e094b5f17426de2329dede5148564aa2d8caf6c5877985587c495d02922b40`）同样禁止自动重发。
- 最终 bulk 候选 `experiments/2026-09-08/candidates/bulk_c9_comm_ctasync.py`，SHA-256 `b1233b36bcf09298057868fba63128e63bd08bdfb589592ffe01c11cd4754fd2`；它在 graphsafe 版基础上增加两次 CTA sync，已通过 root 的 diff 与官方示例审查，但 SID 141431 已在样例首调 TLE。该方向现已冻结，不自动重发。
- 行号对齐的全量 stable 候选 `experiments/2026-09-08/candidates/stable_shape_dispatch_aligned.py`，SHA-256 `1a832e87435d1ce266ce5e8f6d384fdd95a88161c8a1bc2fedf4d815692492af`；SID 141442 无有效正确性或计时数据，已冻结。基于它的 `fp6_c9_c10_roundtrip_stable.py`（SHA-256 `448277e8b6f79dd54778020ad21106e078becd598d33c7da2faa7b27b40285de`）同样冻结、未晋升。
- c9/c10 FP6 roundtrip 静态 shape-cache 诊断候选为 `experiments/2026-09-08/candidates/fp6_c9_c10_roundtrip_static.py`，SHA-256 `6b6384ecc21d77f963da6897001842a272538cfa98560c2f6d8a96ff3e919897`；对应 SID 141454 已全量 Accepted，但仍不解决跨测试点同 shape 换权重。
- packed-MD release `experiments/2026-09-08/candidates/fp6_c9_c10_packed_md_release.py` 对应 SID 141457，已因 c10 明显变慢且 c9 复调 OOM 停止；静态派生 cache 对照 `experiments/2026-09-08/candidates/fp6_c9_c10_packed_md_static_cache.py`，SHA-256 `23d1fd793336a1dc1ebe13cbdd55edbcbd0be7bdc0b046b13204eb57095101e2`，对应 SID 141460，未消除相同现象；stdout 诊断 SID 141462 也没有捕获运行期标记。三者均不晋升、不重发。
- 固定数学底盘的 c9 通信预算：w1 备用 `experiments/2026-09-08/candidates/bulk_c9_comm_stable_append.py`，SHA-256 `1d9ab6b4713699a100b5104303371b4a59db3bf68692dc686fb22a5e06dfbea4`，未提交；w32 版 `experiments/2026-09-08/candidates/bulk_c9_comm_stable_w32.py`，SHA-256 `e05ed1f8e263cb72d5a2d037ba9f67cd6dadadf5eb0a568cdf398b9372d30f33`，对应 SID 141465；thread0 fence 版 `experiments/2026-09-08/candidates/bulk_c9_comm_stable_w32_fence0.py`，SHA-256 `3d113eca28aac092593155e30d753500fa281368364c031dc40893616b7da936`，对应 SID 141470。两次均全量 Accepted，但后者没有收益；当前 bulk 优化已关闭。旧 `36e094…`、`a9dc6b…`、`b1233b…` 仍冻结，不得重发。
- packed-MD 资源诊断候选 `experiments/2026-09-08/candidates/fp6_c9_c10_packed_md_resource_error.py` 对应 SID 141469；BN64 版 `experiments/2026-09-08/candidates/fp6_packed_bn64_c9_resource_c10_perf.py`，SHA-256 `9d90cc4a570ed649238b389acc512601fccb64969c6fc6eee1cde01c68c06dde`，对应 SID 141473；SWAR4 版 `experiments/2026-09-08/candidates/fp6_packed_swar4_c9_resource_c10_perf.py`，SHA-256 `ca1f98b730764867626323f3ded8830d2811b8fd6726acb8173210b67a887da1`，对应 SID 141474。该两平面 FP6→FP8 MD 路线已关闭，均不得自动重发。
- 平台与实验结果：`experiments/2026-09-08/results/`
- SID 141397 结构化结果：`experiments/2026-09-08/results/stable_shape_dispatch_141397_vs_141390.json`
- SID 141397 原始结果：`experiments/2026-09-08/results/submissions_141397_raw.json`；脱敏错误尾部：`experiments/2026-09-08/results/sid141397_tle_sanitized.txt`
- SID 141442 结构化结果：`experiments/2026-09-08/results/stable_shape_dispatch_aligned_141442.json`；原始结果：`experiments/2026-09-08/results/submissions_141442_raw.json`；脱敏错误尾部：`experiments/2026-09-08/results/sid141442_tle_sanitized.txt`
- 本轮终态结果索引：`experiments/2026-09-08/notes/results_index.md`
- 双 CTA 短笔记：`experiments/2026-09-08/notes/dn_two_cta_shortk.md`
- 双 CTA 静态验证入口：`experiments/2026-09-08/notes/verify_dn_two_cta.py`

## 下一步

1. 当前平台 **0 项在途**。SID 141470 已终态且没有 fence0 收益；当前 bulk 优化与完整 EP 投入均已关闭，旧 SHA 不得自动重发。
2. c9/c10 FP6 roundtrip 的 SID 141454 只是一份全量 Accepted 的精度控制底盘；packed-MD 的 SID 141457/141460/141462/141473/141474 均只有 partial 结果，且当前两平面 FP6→FP8 路线已关闭。跨测试点同 shape 换权重的缓存合同缺口，以及早期 c9 OOM/cache miss 来源未定位的边界继续保留为历史限制。
3. 通过验收的后续提交仍由唯一平台操作者按 [提交说明](SUBMISSION.md) 顺序执行并保存逐案结果。
4. 只把平台终态和可复算结果提升为本页结论；晋升前还必须修正 shape-only 权重缓存缺口，并保留不同 call 数路径的一致性证据缺口。

## 旧资料标注

- `archive/handoffs/`：2026-08 的累计交接快照，内容保留但状态已过期。
- `archive/transcripts/`：长会话原文，可能含旧认证材料；只在确需追溯时定向搜索，不复制认证内容。
- `archive/legacy/README.pre-2026-09-08.md`：旧根入口，仅供审计。
- `reports/`：2026-09-05 专项报告与可执行导出器，路径保持原位；归档索引见 `archive/reports/README.md`。
