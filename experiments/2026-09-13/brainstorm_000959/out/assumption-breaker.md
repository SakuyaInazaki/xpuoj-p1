# assumption-breaker

- model: `deepseek/deepseek-v4-pro`
- track: `grounded`
- contract_ok: `true`
- cost: `$0.006781`

## IDEA AT-1

### CLAIM
当前全密度 FP8 核心隐含假设「MD/DN 必须对全部 M 列做等深宽、一次性 FP32 累加」。结构上改为**按 topk 分支分组的块量化输出**，把 M=T·k 路由行的 down 阶段分解为 k 个独立块，中间不直接加回统一 [T,H]，而是在 FP32 末端做精确定点归约，从而消去一次跨块隐式中间读写。这是数据流重排，不是 tile 重调。

### EVIDENCE
上游 MD/DN 有用工作为 7.697/3.848 TF，stage 诊断 6.14/2.75 ms 且明确不含路由、分配、量化、cache、通信与 final gather（`mechanism-facts.md:15-17`；`docs/STATE.md:40`）。F5 明确 k=2 分支归属同一 rank，无 activation collective（`evidence-map.md` F5）。因此「gather/sum k branches」目前是一条热路径但未单独计量。

### SPECULATIVE_LEAP
把 final gather 从「先调 epilogue 再跨 branch 归约」改成「先按 branch 独立 block 输出、再定点归约」，能省掉一次跨 K 块的中间 buffer 往返。

### NOVELTY_GAP
材料未对 gather/sum 阶段单独切出计时，无法定位其占比（`evidence-map.md` U1、G1）。

### MINIMUM_TEST
对 c2 配置，在现有实现里对「down→sum/gather」单独打标，跑一次 12 案，看该段相对 6.14/2.75 ms 的占比；若 <5% 则直接证伪。

### KILL_CRITERION
分块归约版本 end-to-end 相对 FP8 控制无 3 案以上各降 ≥1 ms，或 SQNR 跌破 22 dB，即终止。

### FEASIBILITY
中；需重写 down epilogue，但可控。

### RISK
改动影响「identical inputs → byte-identical outputs」的定点归约顺序，可能破坏确定性（F10）。

### CONFIDENCE
低。

---

## IDEA AT-2

### CLAIM
当前实现假设「权重在 test point 内 static，故可在 untimed warmup 预处理后，用 shape-only 命中缓存」（`mechanism-facts.md:19`）。拆掉「shape 相等即命中」这条假设，改为**内容签名缓存**：以权重内容（或其强散列）为 key 区分新 test point，避免 shape 复用导致 stale cache（`evidence-map.md` G2），其机制改变的是缓存失效判据而非 tile。

### EVIDENCE
F11 明确 weights/topk 仅 test point 内 static、切换后可变；G2 明示 shape-only 派生缓存会在 shape 重复时 stale（`evidence-map.md` F11、G2）。`mechanism-facts.md:19` 要求「correct cache must distinguish a new test point even when tensor shapes repeat」。

### SPECULATIVE_LEAP
内容签名的开销可在 untimed warmup 中摊掉，timed 路径不承担额外 hash，仅承担一次 key 比较。

### NOVELTY_GAP
材料未给出 P1 端 warmup 是否计入/如何计时的确切事实（`evidence-map.md` F12、G3）。

### MINIMUM_TEST
构造两个同 shape 不同 weight 的连续 test point，验证当前 shape-only 缓存是否产出错误结果；再测内容签名版在 timed 段的净增开销。

### KILL_CRITERION
若 timed 段 hash/比较开销 >0.1 ms/案，或 warmup 实际计入得分，则签名的正确性收益被抵消，终止。

### FEASIBILITY
中高；仅改缓存 key 逻辑。

### RISK
若 P1 warmup 不计时，本改动无得分贡献；且散列本身有碰撞风险。

### CONFIDENCE
低。

---

两个 idea 分别拆掉「gather/sum 非瓶颈」与「shape-only 缓存足够」两条隐藏假设，均为数据流/缓存正确性层面的结构边界，非参数扫。
