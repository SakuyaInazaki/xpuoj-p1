# cross-domain-translator

- model: `deepseek/deepseek-v4-flash`
- track: `grounded`
- contract_ok: `true`
- cost: `$0.002104`

## IDEA A-1
### CLAIM
把专家权重在预热期沿 H 维做结构化重排（列置换），使 FP8 分块缩放的刻度落在更均匀的列簇上，从而让同一 FP8 核心在相同 tile 下减少缩放/转置开销而不改数学。
### EVIDENCE
[EVIDENCE] 权重静态仅限单测试点、可预热预处理：`1-full.md:225`。[EVIDENCE] 缓存需区分新测试点：`docs/STATE.md:69`。按比例改动/剪枝已失败：`docs/HISTORICAL_NO_REPEAT.md:13`。
### SPECULATIVE_LEAP
[INFERENCE] 均匀化缩放可改善 FP8 反量化摊销；材料未给缩放开销占比，属推断。
### NOVELTY_GAP
[HYPOTHESIS] 非查新结论；[INFERENCE] 与已关闭 tile/accumulation 族不同，改动的是权重布局前提。
### MINIMUM_TEST
本地 c2 单 GPU，对比成熟 FP8 核心有无列重排的 MD/DN 阶段耗时，同输入同精度。
### KILL_CRITERION
阶段耗时无改善或 SQNR<22 dB 即淘汰。
### FEASIBILITY
依赖 FP8 核心与预热权重预处理，P1 Triton 版本未知（`docs/STATE.md:8,21`）。
### RISK
确定性与布局正确性；重排引入额外拷贝。
### CONFIDENCE
中低。

## IDEA B-1
### CLAIM
把 gather/sum 与 down 投影的边界重画：不物化 [M,H] 再求和，而是让 down 的 K 分块沿路由分支做归约累加，减少一次全量中间张量写读。
### EVIDENCE
[EVIDENCE] 数据流含 gather/sum：`mechanism-facts.md:3,11`。[EVIDENCE] 阶段诊断不含最后 gather：`docs/STATE.md:40`。[EVIDENCE] 直连未排序 token gather 已关闭：`docs/HISTORICAL_NO_REPEAT.md:6-18`。
### SPECULATIVE_LEAP
[INFERENCE] 融合可省一次 M*H 边界流量；材料未给该边界占比，属推断。
### NOVELTY_GAP
[HYPOTHESIS] 非查新结论；失败前提是“直连 gather”而非“归约边界重画”。
### MINIMUM_TEST
本地 c2 对比成熟 FP8 核心与融合边界的端到端阶段计时，验证逐字节一致输出。
### KILL_CRITERION
端到端无增益或非 byte-identical 即淘汰。
### FEASIBILITY
需 Triton 归约/原子累加支持，P1 能力未知。
### RISK
归约顺序影响确定性；并行度下降。
### CONFIDENCE
中低。
