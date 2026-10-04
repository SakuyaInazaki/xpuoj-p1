# Candidate review checklist

- Before freezing a GeneratedWorkload candidate, statically inspect every
  `@triton.jit` function for free variables. Numeric module globals must be
  passed as explicit `tl.constexpr` parameters at every launch; only `tl`,
  `triton`, builtins, and explicit JIT helpers may remain as free names.

Run the source-only preflight from the repository root:

```sh
python3 scripts/check_jit_globals.py experiments/2026-09-13/candidates/legacy_mma_int4_throughput_probe.py
```

The command exits 1 and reports the source line, JIT function, and variable for
each direct read of a plain module-level number, string, or tuple constant.
Pass such values as explicit `tl.constexpr` kernel parameters (including at the
launch site), or declare an intentionally JIT-visible module constant with a
`tl.constexpr` annotation. The checker parses AST only and never imports or
executes the target. It is deliberately limited to this known global-capture
failure mode; a clean result is not a complete Triton compilation or validity
check.

## c4 output-store diagnostic disposition

- [`fp8_c4_dn_tma_store_custom_bench.py`](fp8_c4_dn_tma_store_custom_bench.py), CID `de45e5a9-16ed-475f-ab49-b733daa621d0`: candidate output TMA plus `flatten=True` failed JIT/MLIR verification before `P1MD`. This is compiler-form evidence only and does not show that output TMA itself has no benefit.
- `fp8_c4_dn_tma_store_custom_bench_zerofix.py` changed only the zero-row check from FP8 bit encoding to FP32 numeric comparison. It was **not submitted**, because it did not address the compiler failure.
- [`fp8_c4_dn_tma_store_custom_bench_noflat.py`](fp8_c4_dn_tma_store_custom_bench_noflat.py), CID `aa4ff079-1264-4b84-bff5-73398e324b8b`: baseline retained `flatten=True`; candidate combined output TMA with `flatten=False`. All exact gates passed, but three DN+gather groups were 14.35–14.60% slower. Both DN kernels reported 229408 bytes shared memory, while registers rose 170→231; the register increase is correlated with this result but is not proven to be its only cause. This exact combination is closed without parameter sweeps or extension to other cases.

Full terminal evidence is in [`../results/INDEX.md`](../results/INDEX.md).

## c4 padded output-TMA disposition

- [`fp8_c4_dn_tma_padded_custom_bench.py`](fp8_c4_dn_tma_padded_custom_bench.py), CID `edea0c78-e34a-4bdb-b31d-82636ed70f0c`: all exact mapping/math gates passed. DN-only was 2.80–3.18% faster; the main rowmap→DN→mapped-gather path was 0.15–2.14% faster in the initial three groups.
- [`fp8_c4_dn_tma_padded_custom_bench_confirm.py`](fp8_c4_dn_tma_padded_custom_bench_confirm.py), CID `f5c14b6e-30cf-4e52-8c94-cf7bdb1a5595`: six longer balanced groups reproduced DN-only gains, but the taxed main path split three faster and three slower with ratio of means `1.000875×`.
- The padded + independent PAD_ROW structure is not promoted and must not receive another same-code timing run or store-parameter sweep. The DN-body result remains evidence limited to this c4 geometry, so it does not close output TMA generally. A later attempt requires a distinct structure with evidence that it removes the added preparation/indexing cost.

## aux / md-epilogue 直接测量探针（已冻结，未提交）

两份探针都以 `p1/kernel.py`（SHA-256 `dd46bdeb…`）为基线，diff 纯追加、无删除、无新
`@triton_dist.jit` 源（两份都仍是 79 + 2 个 jit 源）。目的是把标定模型里从未被直接测量的
两项残差（aux 2.30 ms、per-tile 固定开销 3.01 ms）换成实测值，机制是「保持输出逐比特不变、
把某个组件跑两遍」，Δtk 即该组件成本。两份都**没有**提交。

| 文件 | SHA-256 | 闸门 | diff |
|---|---|---|---|
| [`probe_aux2x.py`](probe_aux2x.py) | `bdc5d99748e8b3c8260aaa979f8fdc7900d15a67846024358f96d60fcfe706f1` | `_AX2[0]` | +66 行 |
| [`probe_epi2x.py`](probe_epi2x.py) | `026a70f5e5061d52cd90c9954b925abb8e3eaa6f1d5341d8d429ced85ddf0ddd` | `_EPI2X[0]` | +144 行 |

### 闸门折叠证明

两份都能机械折叠回基线：删掉基线里不存在的顶层名（`_AX2` / `_EPI2X`、`_EPZ_CACHE`、
`_epi2x_args`）、删掉 test 里出现闸门名的 `If`（均无 `orelse`）、删掉 value 调用
`_epi2x_args` 的 `Assign`、删掉名为 `ZP` / `EPI2X` 的形参与 launch 关键字。折叠后
`ast.dump` 与 `p1/kernel.py` 完全相等（probe_aux2x 17 个 If；probe_epi2x 5 个 If、
4 个 Assign、8 个形参、8 个关键字）。

### probe_aux2x 覆盖范围

17 个插入点都形如 `if _AX2[0] and _CALLN[0] >= 2: <阶段>; del <结果>`，第二次调用与基线
逐条同参数。按静态分支推演，第 2~6 次调用中 12 案经过的每一个 aux 阶段都被翻倍，无漏项：
`_route_full` / `_route_gemm_softmax`+`torch.topk`+`_topk_renorm_flat`、
`_counting_sort_order`（两处）、`_gq1p_tok` / `_gq1p_tm`、`_prepare_moe_metadata`、
`flat_weights[order]` 与 `_fgs_tma1_intq_host` 里的 `weights[order]`、
`_quant_act_fp8_row_from_amax`（三处）、`_gather_branch_sum_f8` / `_gather_branch_sum`。
第 1 次调用（首判、500 s 编译预算最紧）逐条等于基线。`_q8_blk2row` 与 `_strip_amax` 在
`_FL[0]==1` 的 12 案下不可达，未触碰。

### probe_epi2x 覆盖范围与刻意取舍

触达的 4 个 md 内核恰是第 3~5 次调用时 12 案的 md 内核：`_fgs_tma2_int_pm_q8_kernel`
(c1/c2)、`_fgs_t1i_mdq_kernel_g` (c3–c8)、`_fgs_tma1_kernel_gq` (c9/c10)、
`_fgs_t1i_mdq_tma_kernel` (c11/c12)。第 6 次调用时 c5–c10 会落到未打点的
`_fgs_t1i_mdq_kernel` / `_fgs_tma1_kernel`，第 2 次调用时 12 案全部落到 `_fgs_tma1_kernel`。

第二遍**不从 `acc` 重算**：`acc`（[128,256] fp32 = 128 reg/thread）若跨第一遍保活，ptxas
峰值会到约 300 reg 并在 255 处溢出，spill 流量会污染 Δtk 甚至拖慢 K 主循环。现在 mdq 家族
从 `gr`/`u` 接、gq/pm 家族从 `acc_g`/`u` 接，峰值约 240 reg。代价是不覆盖
`acc * b_sc`（mdq，[128,256] 一次乘）或 u 半边的两次乘（gq/pm），估计覆盖收尾
指令数的 75%（mdq）/ 84~88%（gq/pm）。`tl.where` 放在 fp8 转换之后，q/q2 各只占
16 reg/thread。

## aux_bundle：route+量化融合 + 排序链 4→3（已冻结，未提交）

| 项 | 值 |
|---|---|
| 文件 | [`aux_bundle.py`](aux_bundle.py) |
| SHA-256 | `0b76eb76bf20c3068a27872d47d768c038fde7e22e402f08e40c9816d9ae0da1` |
| 基线 | `p1/kernel.py`，SHA `dd46bdeb…` |
| 闸门 | `_AXB[0]`；折叠证明 [`aux_bundle_astproof.py`](aux_bundle_astproof.py) |
| diff | +202 / −2 行（两处删除只是把基线的 `if` 改成 `elif`） |
| jit 源 | 79+2 → **80**+2（新增 `_route_fq_kernel`；`_sort_scatter_off_kernel` 本已存在但是死码） |

两项改动，作为一个整体测：

1. `_route_fq` = `_route_full` + `_gq1p_tok` 融合，仅 c3~c8（`_AXGSET`）。结构是
   **倒序 amax 预扫 + 原样 route K 循环里顺带出 FP8**。倒序是为吃 L2 的 LRU；
   amax 放预扫（而不是像 V722 放在 dot 循环、量化另起一趟）是为了让预扫的流水
   buffer 成为 route 循环的真子集，smem 峰值不超过 `_route_full`。闸门 `_CALLN>=3`
   覆盖到第 6 调用，使 `_route_full_kernel` 对这 6 案**一次都不用编译**；`_gq1p_tok_kernel`
   也彻底消失 ⇒ c3~c8 的 JIT 数净 **−1**。
2. `_counting_sort_order` 在 `_FL[0]` 且 `_CALLN>=2` 且 `e_pad!=16` 时改走
   colscan → `_sort_scatter_off_kernel` 三内核链，`counts` 直接取 `tot[:E]`。
   即历史 V627/V628/V675a/V711a 的「改法 1」，V711a 复发已 Accepted（不是编译 bug），
   实测只值约 0.05 ms。

计时路径 launch：c1/c2 不变，c3~c8 每案 −2，c9~c12 每案 −1，合计 **−16**。
Σtk 预期 0.13~0.21 ms（raw +0.06~0.10），其中 DRAM 项 50~132 µs 取决于 L2 命中率
（LRU 模型给 f≈0.36~0.43，上限 f=1 时 132 µs）。**这低于 0.3 ms 的可叠加门槛，
提交前须先接受这个量级。**

V722 归因与本候选的规避见会话报告；本候选相对 V722 移除了四处新奇点：
`tl.range(num_stages=2)`、`tl.store(eviction_policy=…)`、第二条独立多级 smem 流水、
以及 call 6 仍要编译 `_route_full_kernel` 的额外 JIT。

## epi_port：md epilogue 移植到 c9/c10 与 c1/c2（已冻结，未提交）

| 项 | 值 |
|---|---|
| 文件 | [`epi_port.py`](epi_port.py) |
| SHA-256 | `70c347cc7f216403b09b09baa44cef19fdb95115177aa71388e3ccba3792826c` |
| 基线 | `p1/kernel.py`，SHA `dd46bdeb…`（= `p1/kernel_v760a_tanh.py`） |
| 闸门 | `_EPP[0]`：0 = 基线收尾，1 = 新收尾 + `tanh.approx.f16x2`（待测档），2 = 新收尾但 tanh 留 f32 pack=1（对冲档） |
| 折叠证明 | [`epi_port_astproof.py`](epi_port_astproof.py)（丢 `_EPP`、2 个 `EPP` 形参、2 个 `EPP` launch 关键字、4 个 `If`→`orelse` ⇒ `ast.dump` 与基线相等） |
| diff | +153 / −44 行；−44 全部是把基线收尾整段缩进进 `else:`，无语义删除 |
| jit 源 | 79+2 → **79+2**（零新增；`EPP` 是 `tl.constexpr` 形参，一次运行只用一个值 ⇒ 编译特化数也不变） |
| 数值 | [`epi_port_sqnr_screen.py`](epi_port_sqnr_screen.py)：四案孤立 epilogue ΔSQNR 为 **−0.0001 dB / +0.0000 dB**，fp8 载荷 99.85~99.88% 逐位相同，差异元素恰好移动一个 e4m3 台阶；`s`/`inv` 表达式逐字未动 ⇒ 行尺度逐比特不变 |

两处改动，纯原地算术；不加内核、不动循环调度、不动 tile 配置、不动装载位置与
eviction 提示。

1. `_fgs_tma1_kernel_gq`（c9/c10）：全套移植 `_fgs_t1i_mdq_kernel_g` 的收尾 ——
   V716 代数折叠（`a_scale` 折进 g 半边的逐行 `ah`，u 半边的 `a_scale` 并进
   `wi = (w·a_scale)·inv`）+ `tanh.approx.f16x2` + `silu = h·th + h`。
   本内核此前**既没拿到过 V716 折叠、也没拿到过 tanh**：V716 当时只改了三个
   「md 内核」（`_fgs_t1i_mdq_kernel_g` / `_fgs_t1i_mdq_tma_kernel` /
   `_fgs_tma2_int_pm_q8_kernel`），c9/c10 反而是那几发的漂移锚。
2. `_fgs_tma2_int_pm_q8_kernel`（c1/c2）：**只做** `tanh.approx.f32 pack=1` →
   `tanh.approx.f16x2 pack=2`，并把 `(g·0.5)·(1.0+th)` 收成 `h = g·0.5` 加一条
   FFMA。**刻意不做 V716 折叠** —— 该折叠在本内核上已有判题机配对实测负结果：
   V716 全量版 c1 **+0.89%** / c2 **+1.31%**（同发其余八案 −0.49~−1.67%），
   紧接着的 V716b 就是把本内核单独退回未折叠式，净 raw 从 +0.040 升到 +0.055。
   相比历史 [`p1/kernel_v761_tanhc12.py`](../../../p1/kernel_v761_tanhc12.py)
   （净 raw −0.07，未晋升），本档多出 FFMA 收敛、并去掉了 `(g*0.5)` 写两遍
   依赖 CSE 的写法。

### 未解决的判别：f16x2 与 f32 两种 tanh 宽度

`tanh.approx.f16x2` 把每瓦片 MUFU 发射从 16384 降到 8192（−512 clk），代价是新增
每瓦片 8192 条 `cvt.rn.f16x2.f32` 加 16384 条 `cvt.f32.f16`。CUDA 吞吐表把
「all other type conversions」列为每 SM 16/clk，若确实如此这两趟 cvt 值 1536 clk，
会把 512 clk 的 SFU 收益吃穿。两边都有判题机证据且互相矛盾：

- **D17**（V472，md 八案中未被 q8 改动的五案）：f16x2 **5/5 全负，+0.24~0.86%**，
  当时的死因推断正是「fp32→fp16 转换是净增的 ALU」。
- **V760a**（v751 → v760a，判题机配对）：同一替换读到净 raw **+0.065**，
  但同一发里**代码完全未变的 c9/c10 也移动了 −0.54~−1.50%** ⇒ 该发的共模漂移
  至少约 1%，这个 +0.065 不能算干净读数。

因此 `_EPP[0] = 2` 留作一行对冲：它保留 c9/c10 的全部确定收益（原式是
MUFU.EX2 + MUFU.RCP 两趟 = 2048 clk，换成一趟 f32 tanh = 1024 clk），且完全不产生
cvt 往返。在最悲观的 cvt 模型下 2 档严格优于 1 档；在最乐观模型下 1 档多赚约
320~512 clk/瓦片。

**未提交。** 若一次配对后 c9/c10 与 c1/c2 的符号不一致，按内核拆开重测，不要按案拆。

## md 跨瓦片流水（flatten）三探针：c11/c12 专用（已冻结，未提交）

基线 = `p1/kernel.py` = `p1/kernel_v820_epi_m2.py`，SHA-256
`1ae6a4d24b1ce5f6d1be80a2d7afad8a2c85e4f753772781fa18df43491e6d38`。
三份都只改 `_fgs_t1i_mdq_tma_kernel`（c11/c12 的 md 内核），闸门 `_MDF[0]` +
`tl.constexpr` 形参 `FLT`，launch 处 `FLT=(1 if (_MDF[0] and _FL[0] and K == 1024) else 0)`：
在 `_fgs_tma1_intq_host` 里 `K==1024` 只能是 c11/c12（c3/c4 走同一内核但 K=2048 ⇒ 取 0；
c5–c8 走 `_fgs_t1i_mdq_kernel`，c9/c10 走 `_fgs_tma1_kernel_gq`，c1/c2 走
`_fgs_tma2_int_pm_q8_kernel`）。两组 K 本来就是不同编译特化 ⇒ **JIT 源码数（79+2）与编译
特化数都不变**；c11/c12 的第 1、2 次调用不经过本内核，500 s 首调预算不受影响。

| 发序 | 文件 | SHA-256 | FLT=1 相对基线的改动 | diff |
|---|---|---|---|---|
| ① F1 | [`mdflat_keepstore_c1112.py`](mdflat_keepstore_c1112.py) | `ba927224eba3b929df26a854789f1d06da4bc2a659f6c6156c4dbead9ac1e736` | **只有**外层 `tl.range(…, num_stages=2)` → `tl.range(…, flatten=True)`；epilogue、条件描述符 store 全部逐字不动 | +161 / −74 |
| ② F2 | [`mdflat_c1112.py`](mdflat_c1112.py) | `011d33e02b067c2383042584f34de73f817fc2d23a397ec9703a9341e5d36d65` | flatten + 收尾条件描述符 store 换成无条件掩码指针 store（避开 `scf.if` arity 失败，与 dn 已晋升的 FLAT 形态完全同构） | +160 / −71 |
| ③ F3 | [`mdptr_c1112.py`](mdptr_c1112.py) | `b25fd32817cd51d951a1e5b2ce08bf5150c3f594283408f0084a080d2d731801` | 只做 store 替换，外层循环保持 `num_stages=2`（F2 的归因对照：F2−F3 = 纯 flatten 效应） | +155 / −71 |

`−71/−74` 全部是把基线循环整段缩进进 `else:`，无语义删除：`else:` 分支与基线那 76 行循环
**逐字节相同**。`tl.dot` 顺序、K 循环、累加顺序、整段 epilogue 算术在三份里都逐字未动；
F2/F3 的 store 替换对满瓦片写的是同一个 `q`、同一套 `offs_m/offs_n/row_mask`，
所以输出逐比特相同（`SCL` 那一条 store 也未动）。

折叠证明 [`mdflat_astproof.py`](mdflat_astproof.py)（SHA-256
`025b02942f234e7144b7509d530100fa10d4a2c3db35924eeba43123946ff643`）：丢 `_MDF`、1 个 `FLT`
形参、1 个 `FLT` launch 关键字、1 个 `If`→`orelse` ⇒ 三份的 `ast.dump` 与基线完全相等。
`python3 -m py_compile` 与 `scripts/check_jit_globals.py` 三份全过。

### 为什么需要 F2（以及为什么 F1 仍值得先发）

同一平台上有两条互相矛盾的编译事实，且差别只在「条件描述符 store 是不是循环体最后一句」：

- 09-05 沙箱 `md_kernel`（`ONS=None, OFLAT=True`）里 `flatten=True` **加**「满瓦片
  `ACT_DESC.store` / 尾瓦片掩码指针 store」条件分支**编译通过并跑出了数**（mdflat
  c4 −694%、c8 −747%）。该内核的条件分支后面还跟着 `tl.store(SCL + offs_m, …)`。
- 09-13 c4 dn 探针 CID `de45e5a9…`（源码 SHA `5d69abb6…`）里同一构造**在 MLIR
  verification 失败**（`source has 1 operands, but target successor needs 5`）。该内核的
  条件分支**就是循环体最后一句**。

我们的 `_fgs_t1i_mdq_tma_kernel` 属于前者（`if` 之后还有 `tl.store(SCL …)`），所以 F1
有真实的编译机会；F2 则彻底不含这个构造，作为 arity 失败时的备份路线。

### 历史负结果的适用边界（为什么这两案从未被测过）

| 历史读数 | 实际测的是什么 | 与本探针的差别 |
|---|---|---|
| 真实内核 flatten「+31%~+129%」 | flatten **叠**显式 outer `num_stages`（当时 `ONS=1`，作者自述「flatten 配 num_stages 在真实内核上炸掉」） | 本探针把 `num_stages=2` 整个去掉，与已晋升的 `_dn_tma2_f8_kernel` 同形 |
| 沙箱 mdflat −694% / −747% | `SHAPES_TO_RUN` 只含索引 3 与 7 = **c4/c8**，而 `ga_set()=(3,4,5,6,7,8,9)` ⇒ 两案都是 `AMODE=2`（按 `ORDER` 指针散射取 A） | c11/c12 是 `AMODE=0`（A 走 `TensorDescriptor`），跨瓦片只需携带 2 个标量坐标，而 `AMODE=2` 要携带整块 `[BLOCK_M,BLOCK_K]` 地址状态 |
| 手工预取 mdpf −27.6/−23.9%、mdpfa −12.5/−7.6% | 寄存器里携带下一瓦片的 B（mdpf ≈224 reg/thread）或 A（mdpfa ≈192） | 与 flatten 无关：wgmma 的 B 必须从 smem 取，寄存器携带强制回写；flatten 让编译器在 smem 里预取，不经寄存器 |

### 判读

其余 10 案代码逐字未变，作同发共模漂移锚（历史共模约 ±1%）。

| F1 读数 | 结论 |
|---|---|
| c11/c12 都快 ≥2% | flatten 在 md 的 TMA-A 路径上成立；历史负结果是「`AMODE=2` 指针取 A」+「flatten 叠显式 outer num_stages」的产物，不是 md 的性质。晋升，并按下表推广 |
| 与基线差在 ±1% 内 | 无收益也无害；不晋升，改发 F2 看去掉 TMA store 后 flatten 是否能动起来（TMA store 会在 epilogue 里插一次 reg→smem 回写，可能正是流水交叠被堵的地方） |
| 慢 5% 以上 | md 侧 flatten 与 A 的取数方式无关地失败 ⇒ 归因收敛到 epilogue 在瓦片边界的活跃度/累加器保活；路线关闭，不扫参数 |
| MLIR/JIT 失败（arity） | 判定为编译构造问题而非机制否证，改发 F2；若 F2 也失败则 flatten 在 md 路线关闭 |

| F2 / F3 读数 | 结论 |
|---|---|
| F3 ≈ 基线，F2 明显快 | flatten 成立且 TMA store 在 c11/c12 上不值钱 ⇒ 直接晋升 F2 |
| F3 明显慢（= TMA store 有价值），F2 ≈ 基线 | flatten 的收益恰被放弃 TMA store 抵掉；要两者兼得须先解决 arity（F1 或 padded 结构），不在本轮 |
| F2 与 F3 同幅度变化 | 全部差异来自 store 形态，flatten 本身零效应 ⇒ 关闭 |

### 收益上限（标定模型，E_fix=3480 cy、fill/drain+前言≈2000 cy、132 SM、1.755 GHz）

瓦片数按 `M = T·topk`、`BLOCK_M=128`、`num_block_n = I/128` 计；该口径下
c11+c12 = 24576 瓦片 ⇒ E_fix 合计 0.369 ms，与既有标定的 0.37 ms 一致（全表 112640 瓦片
= 1.69 ms，比标定的 1.97 ms 少的约 16% 是尾瓦片浪费）。

| 范围 | 瓦片 | E_fix | flatten 可攻的 fill/drain 上限 |
|---|---|---|---|
| c11 | 8192 | 0.123 ms | **−0.071 ms** |
| c12 | 16384 | 0.246 ms | **−0.142 ms** |
| c11+c12 | 24576 | 0.369 ms | **−0.213 ms**（≈ raw +0.10） |
| 再加短 K 的 c3/c4（H=2048，符合 dn 实测的 K≤2560 规律） | +12288 | +0.185 ms | 合计 **−0.319 ms**（≈ raw +0.15） |
| 全 12 案 md（含 c1/c2/c5–c10 的长 K） | 112640 | 1.69 ms（标定 1.97） | **−0.97~1.13 ms**（≈ raw +0.46~0.54） |

推广的已知阻力：dn 实测 flatten 的收益随 K 单调衰减并在 K>2560 变号（I=1024 −3~5%、
I=2048 −0.3~1.3%、I=8192/14336 反而 +2.8/+3.3%），而 md 侧的 K 就是 H：只有 c11/c12
（H=1024）与 c3/c4（H=2048）落在正号区。c1/c2 的 md 内核用 `num_stages=4` + 196640 B
smem + `maxnreg=168`，距 227 KB 上限只剩 ~35 KB（< 一级 48 KB），且它的外层是**裸
`range`**（无 `num_stages` 属性）；c3–c8 的 `_fgs_t1i_mdq_kernel_g` 正是沙箱测出
−694/−747% 的 `AMODE=2` 路径，未拿到 TMA-A 结论前不得推广。

## md (m,n) 线性化假设：**证伪**（基线里本来就是线性化的两层嵌套）

委派任务的前提是「md 内核是三层嵌套 `for m_tile: for jn: for k:`，而 dn 是两层，所以
flatten 在 md 上摧毁了 K 循环流水」。**这个前提在冻结基线 `p1/kernel.py`
（SHA `1ae6a4d2…`）上不成立。**

| 内核 | 外层循环 | 内层 | 行号 |
|---|---|---|---|
| `_dn_tma2_f8_kernel`（已晋升 `FLAT=(K<=2560)`） | `for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, flatten=FLAT)` → `pid_m = tile_id // num_block_n; pid_n = tile_id % num_block_n` | 唯一 `for k in range(0, tl.cdiv(K, BLOCK_K))` | 4305 / 4324 |
| `_fgs_t1i_mdq_tma_kernel`（c11/c12） | `for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid, num_stages=2)` → **同一套 `// num_block_n` / `% num_block_n` 解码** | 唯一 `for k in range(0, tl.cdiv(K, BLOCK_K))` | 5270 / 5292 |
| `_fgs_t1i_mdq_kernel_g`（c3–c8, AMODE=2） | 同上 | 同上 | 5369 / 5393 |
| `_fgs_t1i_mdq_kernel`（长 K） | 同上 | 同上 | 5195 / 5218 |

`grep` 全文 68 个循环：**没有任何内核是三层嵌套**，每个 GEMM 内核都是「线性 (m,n) 瓦片
循环 + 唯一 k 循环」，m 主 n 次，`tl.swizzle2d(local_m, pid_n, …)` 在解码之后按专家分组
重排 n 顺序，md 与 dn 逐字同构。

⇒ 委派要求的 `mdlin_c1112.py`（线性化 + flatten）与已实测的
[`mdflat_keepstore_c1112.py`](mdflat_keepstore_c1112.py)（SID 143324，c11 +142% / c12 +104%）
**是同一个变换**；`mdlin_noflat_c1112.py`（线性化 + `num_stages=2`）与基线**逐字节相同**。
按原文交付会白烧 2 次提交、0 bit 新信息，因此没有按原文交付。

### 改交付：前言下沉（md 前言集合 → dn 前言集合）

把 flatten 的 md/dn 差异重新收敛到源码里**唯一还没被控制**的结构差：md 在 k 循环**之前**
load 三个只被收尾消费的向量，dn 在 k 循环**之后**才 load 它的 scale。

```
md 5288-5290 (k 循环之前)            dn 4328-4329 (k 循环之后)
  a_scale = tl.load(A_SCALE + offs_m …)   a_scale = tl.load(A_SCALE + offs_m …)
  b_sc    = tl.load(B_SCALE + …[2*BN])    b_scale = tl.load(B_SCALE + … + offs_n)
  w       = tl.load(W + offs_m …)         (无 w)
```

依据 Triton 3.4 `lib/Dialect/TritonGPU/Transforms/FuseNestedLoops.cpp`：`flatten=True` 只
是给外层打 `tt.flatten`（L24），融合把 `prologue(i)` 整段塞进 `scf.if T == 0`，并把 prologue
里「后续仍被使用的中间 SSA 值」全部提成融合循环的 loop-carried iter_arg，else 分支原值
透传（L465-467 自述 *"This is why fusion will increase liveranges"*）。所以 md 比 dn 多背
`a_scale[128]f32 + b_sc[256]f32 + w[128]f32` 三个向量的 iter_arg + 三次被谓词化的全局 load。

| 文件 | SHA-256 | LIN=1 相对基线 | diff |
|---|---|---|---|
| [`mdlin_c1112.py`](mdlin_c1112.py) | `736411a42da642756df2a42d5e59375aefe54f95437dca2d8cbd2db773da1dd4` | 三条 load 移到 `acc = tl.dot(...)` 之后 + 外层 `num_stages=2` → `flatten=True` | +168 / −76 |
| [`mdlin_noflat_c1112.py`](mdlin_noflat_c1112.py) | `6068bdf290963f02b5cfb952d291a094f3ee7f88ef7463844bfa8588cb50cf55` | **只**移三条 load，外层保持基线 `num_stages=2`（单因子对照） | +169 / −76 |

两份互相只差外层那一行（已 `diff` 确认）。闸门 `_MDL[0]` + `tl.constexpr` 形参 `LIN`，
launch 处 `LIN=(1 if (_MDL[0] and _FL[0] and K == 1024) else 0)`。K 循环、`tl.dot` 顺序、
整段 epilogue 算术、条件描述符 store 全部逐字未动。

**输出逐比特相同**：`A_SCALE`/`B_SCALE`/`W` 都是只读入参（`W = weights[order]` 新张量），
k 循环里只有两次 TMA load + `tl.dot`，**没有任何 store**；内核唯一的两次 store 写
`ACT`/`SCL`（都是 `torch.empty` 新缓冲，与三个入参不别名）⇒ 纯代码移动，收尾的消费顺序与
算术表达式完全未变。

折叠证明 [`mdlin_astproof.py`](mdlin_astproof.py)（SHA `07d0d46811a5…`）：丢 `_MDL`、1 个
`LIN` 形参、1 个 `LIN` launch 关键字、1 个 `If`→`orelse` ⇒ 两份 `ast.dump` 与基线完全相等。
`py_compile` 与 `scripts/check_jit_globals.py` 两份全过，jit 源码数 79+2 不变。

### 收益上限 vs 下沉的代价（诚实的低先验）

- **线性化前言 ×nbn 的代价 = 0**：循环本来就是线性的，下沉不增加任何每瓦片工作量，
  只改三条 load 的位置。委派里估的「前言 ×nbn（c11 nbn=8、c12 nbn=16）」不存在。
- **下沉省下的量级**：三个向量共 512 个 f32 lane / 256 线程 = **2 reg/thread** 的 iter_arg
  select。融合后 c11 的融合循环迭代数 = 8192 瓦片 × inner_len 8 = 65536（/132 SM ≈ 496/CTA），
  每迭代省约 2 条 select ⇒ **≈16 cy/瓦片**。
- **flatten 能攻的 fill/drain ≈ 2000 cy/瓦片**（c11 −0.071 ms、c12 −0.142 ms、合计
  −0.213 ms ≈ raw +0.10，口径同上文标定表）。
- ⇒ 16 cy/瓦片的 liverange 账**远不足以解释 +140%**，也不足以自己赚回 2000 cy。本探针只在
  「真正的堵点是那三次被谓词化的全局 load 本身（阻塞 pipeliner 给 TMA load 排 stage），
  而不是 liverange」时才会赢。**先验低，但它是源码里最后一个未控制的结构差。**

### 判读

其余 10 案代码逐字未变，作同发共模漂移锚（历史共模约 ±1%）。

| 读数 | 结论 |
|---|---|
| `mdlin` 相对锚快 ≥2% | 假设成立：md 的 flatten 死于「prologue 里的三次谓词化 load」，不是 md 的性质。晋升，并按 K≤2560 规律考虑推广到 c3/c4 |
| `mdlin` ≈ 基线（±1%），`mdlin_noflat` ≈ 基线 | 下沉无害但 flatten 的 +140% 另有其因；md 侧 flatten 关闭，归因转向未控残差（见下） |
| `mdlin` 仍 +50% 以上 | 与前言集合无关 ⇒ flatten 在 md 路线**彻底关闭**，不再扫任何 flatten 变体 |
| `mdlin_noflat` 与基线的差 | = **下沉本身**的代价/收益（不含 flatten）。若它单独快 ≥2%，那是一个与 flatten 无关的独立收益，可单独晋升 |

### md 与 dn 之间剩下的未控残差（本探针不动）

1. launch `num_stages=3`（md）vs `4`（dn，K=1024）。`FuseNestedLoops.cpp:876-887` 的
   `numStages = max(外层 tt.num_stages, 内层 tt.num_stages)`，两者都没有显式属性 ⇒ 取 1 ⇒
   不打属性 ⇒ 都回落到 kernel 级默认（`getNumStagesOrDefault`），所以 md 的融合循环按 3 段、
   dn 按 4 段流水。
2. `GROUP_M=8`（md, K≤1024）vs `32`（dn, K=1024）——只影响 prologue 的标量 swizzle。
3. 收尾重量：md 的 reshape/split + `tanh.approx.f16x2` + 3.5 U 逐元素；dn 的
   `tl.max(tl.abs(acc), axis=1)` 跨 lane 归约。
4. **已排除**：收尾里的嵌套 `scf.if`（条件描述符 store）——F2 `mdflat_c1112.py` 换成无条件
   掩码指针 store 后仍 +143%/+106%，与 F1 同幅度。
5. **已排除**：k 循环本体形状——md 与 dn 的 `tl.dot` 都是 `[128,128]fp8 × [128,256]fp8`、
   accumulator 都是 `[128,256]f32`（md 的 `2*BLOCK_N=256`，dn 的 `BLOCK_N=256`），逐字同构。

**未提交。**

## 把 v830 的「前言下沉 + 跨瓦片流水」推到其余三个 md 内核（已冻结，未提交）

基线 = `p1/kernel.py` = **v830** = `mdlin_c1112.py`，SHA-256
`736411a42da642756df2a42d5e59375aefe54f95437dca2d8cbd2db773da1dd4`（已含 `_MDL=[1]` /
`LIN`，c11/c12 实测 c11 −3.42% / c12 −3.12%，3 组配对）。下面三份**各只动一个内核**，
其余案的代码逐字不变（含 c11/c12 仍是 v830 形态）作同发共模锚。

### 计时调用（call 3~5）的案 → md 内核映射（静态推导；仓库里没有 `sim.py`）

入口 `_CALLN[0]+=1; _FL[0]= key in _KNOWN12; _GA[0]= _FL and 3<=_CALLN<=5 and key in _GASET`，
`_GASET` = c3~c10（8 项）。12 案全部 `use_fp8=True / use_int8=False`，c2 从第 2 调起
`c2_fused=True ⇒ use_c2_plain=False`，所以 12 案都进 `elif use_fp8 and not use_int8 and
not use_c2_plain`。

| case | (T,H,E,I,topk) | 分支 | host | md 内核 | A 取数 | k 趟数=H/128 | launch |
|---|---|---|---|---|---|---|---|
| c1 | 16384,4096,8,8192,2 | `_CALLN>=3 and E==8` + `_FL[0]` | 就地 launch | `_fgs_tma2_int_pm_q8_kernel` | TMA | 32 | ns=4/w8/**maxnreg=168**/smem 196640 |
| c2 | 16384,4096,8,14336,2 | 同上 | 就地 launch | 同上 | TMA | 32 | 同上 |
| c3 | 16384,2048,32,2048,4 | `E<=96 and _q8_act` | `_fgs_tma1_intq_host` → `_GA[0]` | `_fgs_t1i_mdq_kernel_g` | `ORDER` 指针散射 | 16 | ns=3/w8 |
| c4 | 16384,2048,32,1024,4 | 同上 | 同上 | 同上 | 指针散射 | 16 | ns=3/w8 |
| c5 | 8192,3584,64,2560,8 | 同上 | 同上 | 同上 | 指针散射 | 28 | ns=3/w8 |
| c6 | 8192,3584,64,1024,8 | 同上 | 同上 | 同上 | 指针散射 | 28 | ns=3/w8 |
| c7 | 16384,4096,96,2048,3 | 同上 | 同上 | 同上 | 指针散射 | 32 | ns=3/w8 |
| c8 | 16384,4096,96,1024,3 | 同上 | 同上 | 同上 | 指针散射 | 32 | ns=3/w8 |
| c9 | 4096,4096,256,2048,8 | `_CALLN>=2` + `E==256` + `_GA[0]==1` | `_fgs_tma1_host(q8=True)` | `_fgs_tma1_kernel_gq` | 指针散射 | 32 | ns=4/w8 |
| c10 | 4096,4096,256,1536,8 | 同上 | 同上 | 同上 | 指针散射 | 32 | ns=4/w8 |
| c11 | 65536,1024,32,1024,2 | `E<=96 and _q8_act` | `_fgs_tma1_intq_host` → `K<=2048` | `_fgs_t1i_mdq_tma_kernel`（v830 `LIN=1`） | TMA | 8 | ns=3/w8 |
| c12 | 65536,1024,32,2048,2 | 同上 | 同上 | 同上 | TMA | 8 | ns=3/w8 |

- 早先本 README 写「c3/c4 走 `_fgs_t1i_mdq_tma_kernel`、c5~c8 走 `_fgs_t1i_mdq_kernel`」
  只在 `_GA[0]==0` 时成立；计时窗内 c3~c8 全部落在 `_fgs_t1i_mdq_kernel_g`。
- **`_fgs_t1i_mdq_kernel` (长 K, 5189) 在 12 案的 call 3~5 里一次都不跑。**
- `_fgs_t1i_mdq_tma_kernel` 的 **K=2048 特化只在第 6 次及以后调用**才可能被 c3/c4 取到
  （`_GA[0]` 的 `3<=_CALLN<=5` 窗口一关，c3/c4 的 `K=2048<=2048` 才走进去），那时
  `LIN` 门上的 `K == 1024` 取 0 ⇒ 基线形态。**本轮不改它的 K 门。**

### 三份候选

统一规则（三份完全同一条）：把 k 循环**之前**、结果**只被收尾消费**的语句整句搬到
k 循环之后（相对顺序不变），外层瓦片循环的 `num_stages` 换成 `flatten=True`。
K 循环体、`tl.dot` 顺序、整段 epilogue、两条 store 逐字不动 ⇒ **输出逐比特相同**
（被搬的 load 全是只读入参，k 循环内零 store，两条 store 的目标都是 `torch.empty`
新缓冲，与入参不别名）。

| 发序 | 文件 | SHA-256 | 闸门 / constexpr | 内核（案） | 下沉的语句 | diff |
|---|---|---|---|---|---|---|
| ① | [`mdlin_g_c38.py`](mdlin_g_c38.py) | `25674bd8bf9e3ef0eea7f6f6dffc9ebf8cfdf0b51f9818071ed788cc4bb613ca` | `_MDLG[0]` / `LNG` | `_fgs_t1i_mdq_kernel_g`（c3~c8） | `offs_n`, `a_scale`, `b_sc`, `w` | +161 / −73 |
| ② | [`mdlin_gq_c910.py`](mdlin_gq_c910.py) | `9f36834ff873ba86b8996f95e9a3b401425eec9c49f0bea6c7ee50fed1644d3b` | `_MDLQ[0]` / `LNQ` | `_fgs_tma1_kernel_gq`（c9/c10） | `offs_n`（三向量基线里**本来就在循环后**） | +220 / −104 |
| ③ | [`mdlin_q8_c12.py`](mdlin_q8_c12.py) | `6c1baec32753f6209a5fc59cf5a8de628e94eed8c5585e000499eee4a3924449` | `_MDLP[0]` / `LNP` | `_fgs_tma2_int_pm_q8_kernel`（c1/c2） | `n_rows`, `offs_m`, `offs_n`, `row_mask`；并把 `maxnreg` 168→232 | +235 / −108 |

无法下沉的跨迭代状态：

- **①/②（AMODE=2）**：`rows = tl.load(ORDER + offs_m, …) // KTOP`（`a_ptrs` 在循环前就要用，
  且 ① 里下沉 `a_scale` 后它还得活过循环）、`a_ptrs`（`[128,128]` 指针块，循环内 `+=`）、
  `row_mask`（循环里当 A 的 load 掩码）、`offs_m`（喂 `row_mask`）、`b_row*`。
  ⇒ 这两个内核的融合前言里**必然还留一次谓词化全局 load（`ORDER`）**，达不到 c11/c12
  「前言零向量 load」的状态；`a_ptrs` 正是 09-05 沙箱 −694/−747%（`SHAPES_TO_RUN`=c4/c8，
  都是 AMODE=2）真正在测的东西。
- **③（A/B 全 TMA）**：k 循环只吃 `a_row` / `b_row_g` / `b_row_u` 三个标量，下沉后融合前言
  **只剩统一标量**（`expert`/`row_begin`/`local_m`/`t_num`/`t_cum`/`a_row`/`b_row_*`）+ 两个
  `tl.zeros`，是四个 md 内核里最干净的前言。

### c1/c2 的寄存器 / smem 评估

- **smem 不变（196640 B）**：196640 = 3 个 TMA 操作数 × 4 stage × 16384 B + 32 B 屏障。
  `FuseNestedLoops.cpp:876-887` 取 `max(外层 tt.num_stages, 内层 tt.num_stages)`，两层都没有
  显式属性（`flatten=True` 只打 `tt.flatten`）⇒ 回落到 kernel 级 `num_stages=4`，多缓冲深度、
  操作数个数、block shape 全不变；本内核收尾里没有任何 smem 算子（两个独立 `[128,128]`
  累加器，无 reshape/split/跨 lane 归约）。多缓一块也才 213024 B，仍在 227 KB 内。
- **寄存器是唯一真风险**：`acc_g+acc_u = 2×[128,128] f32 = 32768/256 线程 = 128 reg/thread`，
  已占 168 cap 的 76%；c2 单卡 profile 实测正是 `168 regs / 0 spill / 196640 shared`，即**贴顶零余量**。
  融合新增的 loop-carried 只有上列统一标量（≈8）+ 融合循环自己的 `i`/`T`（2）+ 各自一条
  `T==0` 的 select；**向量 iter_arg 为 0**（这正是把 `offs_m`/`offs_n`/`row_mask`/`n_rows`
  一起下沉的原因，留着的话还要多约 1.5 reg/thread 的 i32[128]/i1[128] 及其 select）。
  ⇒ 估计 +10 统一寄存器，在 168 下大概率转成 spill。
- **把 cap 抬到 232 不花占用率**：smem 196640 ⇒ 每 SM 只放 1 个 CTA（2×196640 > 232448），
  1 CTA = 8 warp = 256 线程，寄存器文件 65536/SM ⇒ 可用 **256 reg/thread**；232×256 = 59392
  < 65536。占用率早已被 smem 钉死，所以 232 只可能消 spill，不可能掉占用率——与历史
  「232 对 168 单独测是噪声级」一致。代价是它成了本文件里的第二个因子：③ 若读数 ≈0，
  先拆 cap 再判 flatten。

### 预期 Δ%（标定 + dn 先例）

口径：E_fix=3480 cy/瓦片，其中 flatten 可攻的 fill/drain+前言 ≈2000 cy；132 SM @1.755 GHz
⇒ 每瓦片上限 8.634e-6 ms。瓦片 = ⌈T·topk/128⌉ × (I/128)，合计 112640（与旧表一致）。
分母用 SID 143335（= 本基线）的逐案 tk。
**实现率标定**：c11 上限 −7.19% 实测 −3.42%（0.48×），c12 上限 −8.84% 实测 −3.12%（0.35×）。
dn 先例换成 k 趟数（dn 的 K=I、BLOCK_K=128）：8 趟 −3~5%、16 趟 −0.3~1.3%、64 趟 +2.8%、
112 趟 +3.3% ⇒ **变号点约在 20~24 趟**；md 的 k 趟数 = H/128。

| case | 内核 | 瓦片 | 上限 ms | 上限 %tk | k 趟 | 预期 Δ%（下界=0.35~0.48×上限，上界=dn 变号区） |
|---|---|---|---|---|---|---|
| c3 | ① g | 8192 | −0.0707 | −4.97% | 16 | **−2.4 ~ −0.3%**（dn 16 趟带 −0.3~1.3%） |
| c4 | ① g | 4096 | −0.0354 | −4.19% | 16 | **−2.0 ~ −0.3%** |
| c5 | ① g | 10240 | −0.0884 | −3.07% | 28 | **−1.5% ~ +2%**（已过变号点，不预测符号） |
| c6 | ① g | 4096 | −0.0354 | −2.62% | 28 | **−1.3% ~ +2%** |
| c7 | ① g | 6144 | −0.0530 | −2.28% | 32 | **−1.1% ~ +3%**，更可能为正（变慢） |
| c8 | ① g | 3072 | −0.0265 | −1.94% | 32 | **−0.9% ~ +3%** |
| c9 | ② gq | 4096 | −0.0354 | −1.37% | 32 | **−0.7% ~ +3%** |
| c10 | ② gq | 3072 | −0.0265 | −1.34% | 32 | **−0.6% ~ +3%** |
| c1 | ③ q8 | 16384 | −0.1414 | −2.99% | 32 | **−1.4% ~ +3%** |
| c2 | ③ q8 | 28672 | −0.2475 | −3.03% | 32 | **−1.5% ~ +3%** |

十案上限合计 **−0.760 ms**；按 0.40 实现率是 −0.304 ms ≈ raw +0.14。但按 k 趟数
先例，其中 0.62 ms（28/32 趟的 8 个案）落在变号区，所以**诚实先验是三份加起来
只有 c3/c4 有把握，合计约 −0.10 ms ≈ raw +0.05**。

### 判读（预注册）

| 读数 | 结论 |
|---|---|
| ① c3/c4 快 ≥1.5% 且 c5~c8 不慢 | k 趟数律在 md 上比在 dn 上更宽；晋升 ①，并把 `LNG` 收成 `K<=2048` 或整体开 |
| ① c3/c4 快、c5~c8 慢 | 变号点确认在 16~28 趟之间；`LNG` 改成 `K<=2048` 单独晋升（下一发） |
| ①/② 任一 ≥+50% | flatten 在 AMODE=2 上被 `a_ptrs[128,128]` iter_arg 打死 = 09-05 沙箱 −694/−747% 的真因；两条路径一起关闭，不再扫 AMODE=2 的 flatten 变体 |
| ② ≈ 基线而 ① 明显慢 | 差别只剩「①多下沉了三条谓词化 load」⇒ 反向再次确认 c11/c12 的机制归因 |
| ③ ≈ 0 | 先拆 `maxnreg`（232 vs 168）再判 flatten；不要按案拆 |
| ③ 慢且 `n_spills`>0 | cap 不是瓶颈、融合的 liverange 是；q8 路线关闭 |

折叠证明 [`mdlin3_astproof.py`](mdlin3_astproof.py)（SHA-256
`222e2f107aa51725b8f05c026608a4ba9a57c0f795616ea1566fb4cde595e9ef`）：三份各丢 1 个模块闸门
名、1 个 constexpr 形参、1~2 个 launch 关键字、1~2 个 `If`→`orelse`（③ 的第二个是复制出来
的 host launch，折回后含 `maxnreg=168`）⇒ `ast.dump` 与基线完全相等；jit 源码数三份都是
79+2 不变。`python3 -m py_compile` 与 `scripts/check_jit_globals.py` 三份全过。
另外机器校验了两件事：每份「闸门分支」与「基线分支」的循环体**语句多重集完全相同**
（纯重排，无新增/删除语句），且重排后 def-before-use 成立。

**未提交。**
