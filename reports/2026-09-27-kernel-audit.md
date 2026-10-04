# 当前 kernel 机制审计与可交给实现 agent 的方向

审计日期：2026-09-27。此报告只核查源码、已有实验与 CPU 整数映射；没有新 GPU 性能结果，没有提交平台，没有修改生产文件。

## 1. 基线与证据等级

实际入口 `p1/kernel.py` 是 **v842**，与 `p1/kernel_v842_sort_nw4_c1112.py` 完全相同；SHA-256：

`5cfd6a80d066d8a729ff1b6741541c1c70df7df75eeffe40c67440d4fe962a3c`

文件 6724 行、275458 bytes。AST 有 170 个函数定义、160 个不同名字、10 组重名；总 83 个带 `triton_dist.jit` 的定义，其中有效最后定义 80 个。`docs/CODE_MAP.md` 仍对应 dd46… 旧版；`docs/STATE.md` 页首 v836 也过时，页末 09-20 v842 才匹配当前源码。以下行号均对应上面 SHA。

本报告用 **事实** 指源码或保存的测量能直接支持的结论；**推断** 指有机制依据但没有本轮 GPU 测量；**假设** 指需要实验判定的收益。旧平台 Accepted 只能证明该提交在那次执行的检查通过，不能证明任意输入、任意调用顺序、跨测试点缓存更新都正确。

用户补充 `/Users/sakimi/Desktop/addinfo/addinfo.md` 中的赛事方答复已确认：P1 为 Triton-dist/Triton 3.4；四卡 H800 80G SXM、默认 700 W，GPU 间拓扑均 NV8；500 秒是一次 torchrun 中多个测试点的总预算。答复写“十个测试点”，项目日志为 12 案，数量需澄清，但已经足以推翻“单个 kernel 享有 500 秒编译预算”的解释。通信 kernel 须用 `triton_dist.jit`；本报告没有再建议版本探针或重新 init/finalize。

## 2. 当前真实调用图

最后生效的关键定义：

| 函数 | 行号 | 事实 |
|---|---:|---|
| `run_kernel` | 6472–6724 | 全局 `_CALLN` 每调用 +1；没有按 shape 重置 |
| `_run_replicated` | 6024–6470 | 12 个已知 case 全部进入复制专家路径 |
| `_prepare_moe_metadata` | 4118–4162 | 覆盖 278 行旧定义，BM128 元数据 |
| `_get_static_cache` | 4320–4334 | 只在 generic fallback；不是 12 案稳态缓存 |
| `_run_kernel_a2a` | 4338–4455 | 覆盖 2117 行旧定义；12 案不走 |
| `_get_full_weights` | 4491–4516 | 覆盖 4461 行 identity 版；实际为 shape key |
| `_fgs_tma2_int_kernel` | 5178 起 | 覆盖 4938 行旧定义；不能只读首次命中 |

所有 12 案在静态权重缓存命中后 **均无跨 GPU token 通信**。冷路径将全专家权重复制到各 rank，本地 token 在本卡完成全部选中专家计算；不能再把 c9/c10 的当前瓶颈解释成原始 EP 通信问题。

### 2.1 呼叫 3–5 的路径表

`M=T*k`；MD 表示融合 gate/up GEMM + SwiGLU + route weight + FP8 activation；DN 表示 down GEMM + FP8 output。所有 MD、DN 的 `tl.dot` 均 FP8×FP8→FP32 累加；函数名字里的 `int` 指 interleave，不代表 INT8 MMA。

| case | T/H/E/I/k | M | GQ 与 MD | DN 参数（BM128/BN256/BK128，grid132，w8） |
|---|---|---:|---|---|
| c1 | 16384/4096/8/8192/2 | 32768 | token 一次量化、散射复制到 sorted A；双 TMA MD `_fgs_tma2_int_pm_q8_kernel` | GM4、s4、FLAT=false |
| c2 | 16384/4096/8/14336/2 | 32768 | 同 c1；还保留两个无用分配，旧删除实验没有测得端到端收益 | GM32、s3、FLAT=false |
| c3 | 16384/2048/32/2048/4 | 65536 | token-only FP8 A；`_fgs_t1i_mdq_kernel_g` pointer gather A/TMA B，TMA full-tile ACT store | GM32、s4、FLAT=true |
| c4 | 16384/2048/32/1024/4 | 65536 | 同 c3 | GM32、s3、FLAT=true |
| c5 | 8192/3584/64/2560/8 | 65536 | 同 c3 | GM32、s4、FLAT=true |
| c6 | 8192/3584/64/1024/8 | 65536 | 同 c3 | GM8、s3、FLAT=true |
| c7 | 16384/4096/96/2048/3 | 49152 | 同 c3 | GM32、s4、FLAT=true |
| c8 | 16384/4096/96/1024/3 | 49152 | 同 c3，但 MD s4/maxnreg232，其他 c3–7 为 s3、不设 cap | GM32、s4、FLAT=true |
| c9 | 4096/4096/256/2048/8 | 32768 | token-only FP8 A；`_fgs_tma1_kernel_gq_tiled`，两路 tiled TMA B | tiled GM32、s4、FLAT=true |
| c10 | 4096/4096/256/1536/8 | 32768 | 同 c9 | tiled GM32、s4、FLAT=true |
| c11 | 65536/1024/32/1024/2 | 131072 | token 一次量化、散射复制到 sorted A；`_fgs_t1i_mdq_tma_kernel` 双 TMA，LIN=true | GM8、s4、FLAT=true |
| c12 | 65536/1024/32/2048/2 | 131072 | 同 c11 | GM8、s4、FLAT=true |

MD：c1/c2 为 BM128/BN128（两块 accumulator）/BK128、GM16、w8/s4/maxnreg232；c9/c10 为 BM128/BN128（两块）/BK128、GM32、w8/s4/maxnreg232；c3–8 为 BM128/实际合并输出 256/BK128、GM32；c11/c12 为 BM128/实际合并输出 256/BK128、GM8、w8/s4/maxnreg232。`_EPP=2` 使 c1/c2/c9/c10 用 f32 tanh；c3–8/c11/c12 是代码中固定的 f16x2 tanh。

共同后段：DN 写每行每 256 列一个 FP32 scale 的 FP8 buffer；`_gather_branch_sum_f8`（4715）按原 top-k slot j=0…k-1 读取 `inv_order`，逐次 FP32 累加再写 BF16 output。它不是 FP 原子归并。

### 2.2 “稳态”必须包含调用序号条件

`_GA=1` 仅在全局 call 3…5、已知 c3–10 shapes 时成立（6485–6488），不是所有 call>=3 都成立。

- c3–8 在 call>=6 改为 `_gq1p_tm` 生成 sorted、复制的 A。c3/c4 (`H<=2048`) 改走 `_fgs_t1i_mdq_tma_kernel`；c5–8 改走 `_fgs_t1i_mdq_kernel`。c8 的 s4/r232 优化只落在 `_g` 分支，不覆盖这个晚期分支。
- c9/c10 是新 tiled 分支：`E==256 and _FL` 使其始终 token-only GQ，MD/DN tiled host 也不再受旧 `_GA` 3…5 限制。不能沿用旧 CODE_MAP 对其 call6 的推断。
- call1/2 router 仍走 torch softmax/topk 或 fused GEMM-softmax + torch topk；call>=3 才 `_route_full`。c1/c2 的 call1 还用 torch stable argsort。
- c1–8/c11/c12 的 call1/2 存在 BF16 activation、中间落盘、不同量化/下投影分支。c4/c11/c12 的 BF16-act DN 条件在 call>=3 被 `act_q8 is not None` 的更早分支覆盖；不能据变量名断言它们稳态是 BF16×FP8。

因此需要“相同输入跨 call1/2/3/5/6/7 比较”独立于单次 SQNR。只在 `_CALLN>=3` 插入探针而拿平台 min(tk) 判性能，也可能量到未武装 call；现有历史已经记录这个陷阱。

## 3. 量化、布局与缓存

**事实：** `_quant_weight_fp8`（2886）按权重每输出行 amax/448 做 E4M3FN 量化，scale FP32，静态派生。`_gq1p_tm`（871）每 token 量化一次后按 inv_order 复制 k 份；`_gq1p_tok`（856）只写一份 token，MD 根据 ORDER//k 随机取行。当前已经实现这些常见消重，不能作为新增优化。

MD 不再单独生成 BF16 SwiGLU 后做整行 amax；融合 epilogue 用 `a_scale² * expert_bnorm² * abs(route_weight)` 的指数构造幂次行尺度，所有 N tiles 写相同 row scale，然后直接写 FP8 ACT。这是当前性能的关键。该固定缩放启发式没有任意 BF16 输入下的误差/溢出证明；旧样本 SQNR≈23 dB 只比 22 dB 门槛留约 1 dB 余量，不能视作任意继续砍精度的授权。

c1/c2 静态 GU 以 128 列块 gate/up 交错；c3–8/c11/c12 用 gran=1 的逐输出通道 gate/up 交错。`_get_int_gu`（6005）负责这两种 layout。c9/c10 `_get_full_fp8_weights_lowmem_tiled`（3790）按 128×128 GU / 256×128 DN 的物理 tile 连续布局缓存，MD/DN 保留 TMA descriptor；这已经是 v836 的已晋升结构。

当前活跃缓存包括 `_FULL_WEIGHT_CACHE`、`_FULL_FP8_CACHE`、`_FULL_FP8_LOWMEM_TILED_CACHE`、`_BNORM_CACHE`、`_INT_GU_CACHE`，键都是 shape（后者加 gran）。清空只留下最近一项并不能修复“连续两个相同 shape、不同专家权重”错误。题面 225 行明确只承诺**同一测试点**静态，切换测试点后必须更新。09-14 identity 守卫导致高代价/OOM，是实现性能负例；没有充分日志时也不能把“每个调用一定 clone 权重”当作唯一已证解释。正确缓存边界需要赛事方提供允许且可靠的 generation/storage/version 合同，或实现能检测变化的合规机制；抽样指纹有碰撞漏检，不能声称完全修复。

## 4. 首个性能主线：c4 DN 的 padded TMA 输出 + 直接 INV_PAD

**已有事实：** `experiments/2026-09-13/results/public24_f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595_parsed.json` 六组确认里，padded 输出 TMA 的 DN 本体均快；旧版单测 DN 快约 2.8–3.2%。但是独立 `PAD_ROW` 构造和 final 的额外索引吃掉收益，整体 ratio of means 1.000875，三快三慢，未晋升。旧组合不可原样重跑。现 c4 DN 已变 s3，应在 v842 的 s3 对照上重新测机制，不能直接沿用旧 s4 百分比。

**新机制：** 不需要将 MD activation padded。保留当前紧凑 ACT 和 DN 输入 TMA；仅 DN output/CSCL padded，直接构造“原 branch→padded row”的逆表。fin 一次查该逆表，后续 FP8 dequant + j 顺序累加完全不变，从结构上消掉旧 PAD_ROW launch 与第二次 map lookup。

设专家 e 的有效行数为 n[e]，紧凑起点 `R[e]=sum_{j<e}n[j]`，tile 起点 `U[e]=sum_{j<e}ceil(n[j]/128)`。紧凑行 `r=R[e]+local` 对应 padded 行：

```text
p(r) = 128*U[e] + local
INV_PAD[ORDER[r]] = p(r)
```

已有 MD/DN metadata 中 `U[e]=t_cum-t_num`，因此不用 CPU `.item()` 求偏移。

### A. 最小实现：MD epilogue 写 INV_PAD

目标文件只复制 v842 为候选，先只门控 c4。

1. 分配 `INV_PAD[M]`，初测 int64，与现 inverse dtype 一致。不要覆盖旧 inv_order，因为晚期 call 的 GQ 仍用它生成 compact sorted A。
2. 在 `_fgs_t1i_mdq_kernel_g`（5745）的 epilogue，**对 swizzle 后的 `pid_n==0`**，重新 load `src=ORDER[offs_m]`，masked 写：

```text
pad=(t_cum-t_num+local_m)*128+lane
INV_PAD[src]=pad
```

   每个有效行恰好写一次。不要将 src 向量提升到 K 循环前，避免扩大寄存器活跃范围。现 MD TMA store 的全 tile/尾 tile 分支仍保持。
3. 对 c4 call>=6 的 `_fgs_t1i_mdq_tma_kernel` 同样支持；或采用下面 B 方案一次覆盖所有调用路径。不得只给计时窗口造表，遗漏另一路调用。
4. DN 从 ACT 仍用 `a_row=R[e]+local_m*128`；将输出改成 descriptor store 到 `c_row=(t_cum-t_num+local_m)*128`。**CSCL 也必须写 padded row**，否则 fin 仍需第二张 map 或会用错尺度。
5. 容量预分配 `P=128*(ceil(M/128)+E)`，c4 P=69632。Down `[P,H]` FP8、CSCL `[P,H/256]` FP32，安全容量无需读取 GPU num_tiles。c4 相对原 Down 多最多 8 MiB，CSCL 多 128 KiB，INV_PAD 额外 512 KiB。实际填充行数与真实 counts 有关。
6. `_gather_branch_sum_f8`（4715）目前从 `down.shape[0]//k` 推 T；**必须改成 `output.shape[0]` 或传入原始 T**。padded capacity/k 不等于 token 数，否则会越界写 output。fin kernel 本体可保留，传 `INV_PAD`，它对 Down 和 CSCL 使用同一 padded 行。
7. 无效 padded 行永远不被 final 读取；数学验证只比较所有有效行和最终 output，并额外验证 holes 不可能被索引。若为调试给 holes 填 poison，只能在验证阶段，不能漏算正式填充成本。

**风险/必测：** 新 md 分支/重载 ORDER 会增加 epilogue 时间；TMA output 异步可见性必须用真实 GPU 重复输出检查；所有 swizzle、空专家、极斜 counts 需覆盖；跨 case 缓存和跨 call 路径仍独立检查。禁止 FP32 atomic_add 到 output，因为它会改变求和顺序和确定性。

### B. 后续更稳的整链实现：sort scatter 直接写 INV_PAD

`_csort_offsets_kernel`（731）已经拿到 `TOT[E_PAD]`。同时求 `tile_counts=ceil(TOT/128)` 与 prefix，生成：

```text
PAD_DELTA[e] = 128*prefix(tile_counts)[e] - prefix(TOT)[e]
```

在 `_sort_scatter_kernel`（692）算出当前稳定位置 `pos` 后，增加 `INV_PAD[offs]=pos+PAD_DELTA[ids]`。继续保留原 `INV[offs]=pos`。

- 优点：每个 branch 明确单写；不碰主 GEMM 的寄存器/流水；在所有 sort 路径都可统一接口；fin 无新 lookup；后续能与第 6 节 metadata 融合共享 prefix。
- 代价：scatter 每项多读一个 delta、多写一个 int64。小型 cache-resident delta（c4 为 32 项）预计成本很小，但这只是推断。
- 不要把 PAD_DELTA 查询移到 fin：那会重建旧的二次索引税。
- 起步不改排序稳定性、不改 int32/64、不改精度，不与排序算法改造合并。

### 实验顺序、上界与停止条件

1. 单卡 custom：v842 c4 真正的 MD→DN→fin，先用同一解析 ORDER/metadata。A/B 保持同数学。可用预生成 INV_PAD 做一次 **DN+fin 的理想上界诊断**，但正式候选时间必须包含构造该 map 的 MD/scatter 差额。
2. 首次最小候选在 MD epilogue 出表；同时测 MD-only、DN-only、MD+DN+fin，明确不能只报 DN-only。若 MD epilogue抵消了收益，再允许一次 B 版对照，不扫几十个配置。
3. 数学门：有效 Down FP8 bits、CSCL bits、最终 BF16 bits 全量比较，输出 finite；同输入重复 20 次；不同路由/全空尾/1,127,128,129边界及 skew。
4. 采用旧 DN 数据的**量级预算**仅约 7–8 μs/c4，按当时约 0.85 ms 整案估计小于 1%。这不是承诺：新 s3 可能改变收益。预计整题积分增量小，适合作为结构流程的第一条可证伪候选，不足独力追榜。
5. kill：理想 DN+fin 无稳定 ≥1% 收益就不实装；实装后主链无稳定 ≥0.5% 收益或 ≥5 μs 绝对改善（取更严格者）则关闭；出现任何 byte mismatch/非法地址先修最多一个明确 bug，不能用放宽精度解释 map bug。

### 本轮已完成的 CPU 映射验证

`reports/2026-09-27-padded-map-proof.py` 不导入 torch/kernel，涵盖全部 12 个 shape、空专家、1/127/128/129/255/256/257、80 组强 skew，以及 swizzle 不同 GM。结果：**97 组、939997 个 row，全部通过**。

检查映射单射、容量上界、ORDER/inv/inv_pad 一致、MD swizzle 后 N=0 单写、两个方案出表一致、padded holes 不被取到、每个 token 的 branch/scale 关联与 FP32 加法顺序不变。还捕获了“swizzle 前 N=0 guard”的反例。结果 JSON：`reports/2026-09-27-padded-map-proof-results.json`。这不是 GPU/TMA/编译/性能验收。

## 5. 次级结构方向：用 packed-key 局部排序替代 one-hot prefix scatter

**事实：** 现 `_sort_scatter_kernel` 为每个 chunk 构造 `[BLOCK,E_PAD]` one-hot，再沿 BLOCK 做 cumsum，再全 E 归约。c9/c10 64×256；c7/c8 128×128。以每个 branch 的局部名次为目的，却产生 O(M*E_PAD) 中间比较/扫描工作。最近 warp 数 sweep（`knobs11.md`）没有更换这个算法。

**建议先选 c9/c10 的 BLOCK64 scatter-only 原型**，保留 histogram/colscan/offsets/metadata：

1. 构造 32-bit key `(expert_id << 6) | local_index`，尾项用最大 sentinel；局部 `tl.sort` ascending。
2. 由排序 key 分离 expert 与原始 index；expert 相邻变更产生 segment-start 位置，对该位置做 max associative scan，得到局部 rank=`sorted_position-last_start`。
3. load `BASE[expert*C+chunk]`；global pos=base+rank；写 `ORDER[pos]=chunk*64+orig_index`、`INV[orig_branch]=pos`。
4. 因 key 的低位 original_index，等专家仍按原先输入顺序，输出与稳定 counting sort **精确相同**。仅整数操作，无错误预算消耗。
5. 可另立一步把 `_sort_hist_kernel` one-hot reduction 换 `tl.histogram`；先做版本/编译最小验证，不把官网 main 文档当 3.4 可用性证明。不要第一版上手写 match.any，因为 Triton lane layout 需要额外证明。

源锚点：624、692、750。Triton v3.4 的语言 standard.py 包含 sort 和 associative_scan 的实现，可用官方版本源码核对：[v3.4 standard.py](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/language/standard.py)。本建议没有断言 bitonic sort 一定更快；寄存器 layout conversion 和 shuffle 可能抵消指令减少。

**最小实验：** 全量 ORDER、INV、COUNTS byte equality（E8/32/64/96/256，随机/全同专家/交错/末块不整除）；单卡 custom 测完整 sort，而不仅替换 kernel；再串上 baseline MD/DN 检查结果精确不变。收益上界为旧 sort 阶段全部耗时，当前没有 v842 逐阶段时间，不提供虚构毫秒。

**kill：** c9/c10 sort 子链不能稳定快 ≥15% 或节省 ≥10 μs，就不集成；若只有在改变 expert内次序下才能赢，第一轮直接停止，保持精确稳定-order合同。先评估一次 layout，禁止退化成 warp/stage 盲扫。旧 BM64、EP、量化/route 融合负结果不直接覆盖这个整数算法，但必须单独证明端到端收益。

## 6. 可与 INV_PAD 共享的低上限方向：offsets + metadata 合并

当前 `_csort_offsets_kernel` 对所有 E 的 TOT 做每 expert prefix；随后 `_prepare_moe_metadata` 又 launch imported `build_block_row_idx_info_kernel`，132 CTA 各自读 counts、cumsum、ceildiv，再为 row tile 找 expert。其源码位于 `work/repo/python/triton_dist/kernels/nvidia/group_gemm.py:40`，本地只证明实现逻辑，不能证明远端 patch 完全相同。

可以保留 histogram、colscan，扩展 offsets kernel：每 expert CTA 已知自己的 count/row offset，再用同一 TOT 得到 tile prefix，向该 expert 的连续 tile 区间直接写四张 metadata（expert、row_begin、tile_num、tile_cum），由 pid0 写 num_tiles；同时产生 PAD_DELTA。这样减少独立 metadata launch与部分重复计算，避开 GPU→CPU `.item()`。

每 expert CTA 的 tile 数循环可以覆盖任意 skew；不能按均匀路由固定容量/固定上限。BM128 capacity 仍 `ceil(M/128)+E`。必须逐项比对原六元组中实际被消费的值，不比较未初始化容量尾部。现 helper 中两个纯中间数组可不返回，但不能把删分配本身当收益。

最小实验只替换 offsets+metadata，保持 sort scatter/GEMM不变，先全量 metadata equality，再测 hist→scatter→metadata 子链与全链。其收益上界仅旧 metadata 阶段与被删 launch 费用，预计微秒级；若子链节省不足 3 μs 或新 metadata 算术导致 ≥0.5% 主链回退则停止，不花多轮正式提交。适合作为第 4 节 B 版的配套实现，而非独立“冲榜大方向”。

## 7. 必做的实验基础：减少实际编译工作集，不能将删死代码误写为提速

AST 保守依赖图（从 run_kernel、跨所有分支追踪本地函数名字）找到 123 个可达函数、37 个不可达函数；不可达部分 1016 行/20 个 JIT 定义，加上被覆盖旧定义 420 行。它只是源码清理机会，不等于省 20 次编译：本地 `triton_dist/jit.py:222–269` 表明 decorator 构造 wrapper，调用 run/warmup 才进入 JIT。

另外官方 Triton3.4 `JITFunction.cache_key` 包含 starting_line_number。直接删前面的死代码可能改变所有后续内核 cache key，使冷编译反而增加。首先采用保留空白行的清理（保留每个有效函数源码与起始行），同时记录 kernel源、dtype/layout、constexpr/launch options 的特化键。远端 patch 是否相同需保留不确定性。

真正可能降低 500 秒总预算的是减少**实际执行的不同数学路径与特化**：现 call1/2/3/6 会触发多个旧 MD/DN/router/gather 族；统一为一次预热就可复用的、与调用编号无关的数值路径，才能同时减少特化并解决任意调用确定性。但旧“全 shape 强制 call3”曾总预算 TLE，禁止不计实际编译工作量直接重发。

建议分两步：

1. 只做离线 entrypoint/case/call-state 的静态 manifest、保行清理、sha/AST 证明；单卡 custom逐个编译实际会用的 family并记录资源/compile错误，不能用现代本地 GPU 镜像外推远端性能。
2. 各选一个小范围 case family，将首次调用与后续调用的**数学结果**统一；再做跨 call和shape A→B→A验证。没有权重失效合同前，缓存修复不能被“shape不同”这个历史样本特征代替。冷/暖整套执行需报告 wall time；收益是编译余量与实验可实施性，不默认计分tk下降。

## 8. 本轮明确不应重开

- c9/c10 全 BM64，以及 BM128 主循环+BM64 尾回收均有直接负例；后者 V571 约 +19.5/+19.6%，见 `experiments/2026-09-08/notes/c9_c10_bm64_audit.md`。不能把平均行数128自动当作已有实测50%padding。
- EP 已有 09-20完整数值正确实装负例；NV8 新硬件说明不使这个已测协议的物理时间消失。
- FP16 accumulator/BM256、INT4/INT6、packed FP6、ordinary sorted-A、launch-parameter sweep 等已广泛实测，除非改变了关键前提，否则不重扫。
- 新 TMA 方案只因消掉旧独立 map和二次lookup成本而重新立项；若把它写成“再次试 padded output”而忽略这些区别，会重复旧失败。
- 正式运行没有测得可靠阶段成本时，不把某个负归因写成硬件不支持/绝对不可能；先用数学门、可重算配对结果与限定上下界收口。
