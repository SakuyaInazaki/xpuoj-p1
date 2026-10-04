# constraint-opportunist

- model: `deepseek/deepseek-v4-flash`
- track: `grounded`
- contract_ok: `true`
- cost: `$0.002196`

## IDEA L-1
### CLAIM
把 c1/c2 每 rank 现在隐含的“统一 M 行计数”执行图，改成按本地 top-k 分桶后插入一个仅本地、不跨 rank 的 token 重排序，使 MD/DN 的 grouped GEMM 用真正同质、可整除 tile 的分组，减少 padding 与尾块浪费；不改精度、不改跨 rank 集合通信。

### EVIDENCE
[EVIDENCE] 当前稳态路径每 rank 自持 T=16384、M=T*k=32768 行，无激活集合通信 (`c1_c2_rank_compute_comm_audit.md:3-10`)；权重在测试点内静态、可用于预热 (`1-full.md:225`)。[INFERENCE] 上述 M 行列序被假定沿原始 token 顺序落入 tile，尾块/padding 与不均匀专家长度会浪费 FP8 core 周期。[HYPOTHESIS] 本地分桶能把这部分浪费变成真实吞吐。

### SPECULATIVE_LEAP
“direct unsorted token gather”已关闭 (`HISTORICAL_NO_REPEAT.md:6-18`)。此处改变其失败前提：不是把 token 直接乱序喂给核，而是先做廉价本地分桶+计数，使每个专家的行区间对齐 tile 尺寸，再去 gather，因而消除的正是当年未处理的尾块/padding 成本。[INFERENCE]

### NOVELTY_GAP
未找到材料支持 P1 侧对该分桶重排做过 A/B；不声称学术新颖性。

### MINIMUM_TEST
单卡/单 rank、冻结权重、固定输入，比较“现行行序”与“本地分桶行序”下 MD/DN 的 grouped GEMM tile 利用率与端到端 rank 时间；以现行成熟 FP8 路径为同期控制。

### KILL_CRITERION
若 MD 或 DN 任一 stage 时间改善 <3%，或插入分桶/重排后净收益为负（重排开销 ≥ 节省），即杀死。

### FEASIBILITY
纯本地 torch 分桶 + 现有 FP8 核，无需新集合通信；一天内可得到单 rank 信号。依赖 U2（P1 编译器行为未知）。

### RISK
分桶本身引入索引/重排开销；可能被现有实现已对齐的部分抵消。

### CONFIDENCE
中低。

## IDEA L-2
### CLAIM
把 MD 的 FP8→FP32 放大采用“通道级行缩放”，改为在预热期对静态权重做一次离线 outlier 重分布（本地、权重端），使激活只需更粗的 per-tile 缩放，减少内核内 FP32 scale 广播与反量化指令，从而提高 FP8 核效率；数学上仍是 FP32 累加。

### EVIDENCE
[EVIDENCE] 全密度路径为 FP8/BF16 混合精度，成熟控制 (`1-full.md:227`；`STATE.md:40`)；权重静态可预热 (`1-full.md:225`)。[EVIDENCE] “单遍激活量化把 scale 乘法插入 dot K 块”已关闭 (`HISTORICAL_NO_REPEAT.md:6-18`)。[INFERENCE] 该关闭针对激活端在线量化；权重端离线 pre-transform 是不同对象。

### SPECULATIVE_LEAP
改变失败前提：不在 kernel 内对激活逐 K 块插 scale，而是在预热期把权重改成更利量化的分布，让激活量化 scale 取值更平滑，从而减少核内浮点指令。[HYPOTHESIS]

### NOVELTY_GAP
未找到材料支持 P1 上权重端量化友好重分布 A/B；不声称学术新颖。

### MINIMUM_TEST
固定输入，在成熟 FP8 核上比较“原权重”与“重分布后权重”的 MD/DN 时间与 SQNR；同期控制为现行 FP8 路径。

### KILL_CRITERION
若 SQNR <22 dB，或重分布权重未使 MD/DN 时间改善 ≥3%，或输出不能字节一致（确定性失败），即杀死。

### FEASIBILITY
权重端离线变换 + 现有 FP8 核；一天内可得单卡 MD/DN 信号。依赖 F11 权重缓存跨测试点更新正确（`1-full.md:225`）。

### RISK
重分布可能改变数值分布触发精度回退；收益可能小于测量噪声。

### CONFIDENCE
低。
