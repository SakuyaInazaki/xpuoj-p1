# mechanism-builder

- model: `deepseek/deepseek-v4-pro`
- track: `grounded`
- contract_ok: `true`
- cost: `$0.006438`

## IDEA A-1

### CLAIM
把 MD 的逐 token 归一化/scale 计算与 gather/sum 合并成单一 fuse，消除一次中间 [M,I] 的整张物化。

### EVIDENCE
MD 产出 FP8 [M,I]+row scales，DN 前需逐行 scale 修正后 down project，中间 [M,I] 整张落 HBM。诊断仅覆盖 stage 内，未含路由/量化/通信/gather 各中间张量。[EVIDENCE] experiments/2026-09-12/c1_c2_rank_compute_comm_audit.md:3-18；docs/STATE.md:40

### SPECULATIVE_LEAP
若 MD 的缩放可延迟到 DN 的 dot-K 内按行 scale 融合（以分组 scale 而非单独 [M,I] 物化），可省去一次全量 HBM 往返。

### NOVELTY_GAP
未找到材料指出现有实现已做 scale-fused DN。

### MINIMUM_TEST
本地定长 shape 的 kernel 时间对拍：fused 版 vs 成熟 FP8 控制，比较 HBM 写量与 e2e 时序。

### KILL_CRITERION
若 fused 版无写量下降，或 SQNR<22 dB，弃。

### FEASIBILITY
中；需 Triton 支持跨层 scale 延迟。

### RISK
精度漂移，需回退。

### CONFIDENCE
中低。

## IDEA A-2

### CLAIM
将 DN 的分组 scale 量化从逐 token 改为按 rank 局部常量缓存复用，消除每 token 重复 scale 计算。

### EVIDENCE
G2：现有 shape-only 权重缓存跨测试点会失效，说明静态量可缓存但需正确失效键。[EVIDENCE] docs/STATE.md:69

### SPECULATIVE_LEAP
分组 scale 在同一 shape 内对多数 token 恒定，可缓存结果并以形状+内容摘要为失效键。

### NOVELTY_GAP
未找到现行 scale 缓存的材料。

### MINIMUM_TEST
本地同 shape 不同 weight 两测试点对拍，验证缓存命中与正确失效。

### KILL_CRITERION
若缓存命中不省时，或跨点复用出错，弃。

### FEASIBILITY
高。

### RISK
正确性误判。

### CONFIDENCE
中。
