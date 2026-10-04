# `codex_mddn_fuse_c11.py` 审计记录

## 结论

这组文件只覆盖 c11 `(T=65536,H=1024,E=32,I=1024,k=2)`。首次提交的 `codex_mddn_fuse_c11.py` 保留了 v800 的第二次调用启用逻辑，仅用于信息探针；正式候选 `codex_mddn_fuse_c11_allcalls.py` 改为只按输入 shape 守门，所有 c11 调用执行同一条完整融合计算。`codex_mddn_fuse_c11_s2.py` 与 all-calls stage-3 版相比只把 md/dn 流水级数从 3 改为 2，用于共享内存归因；`codex_mddn_fuse_c11_s2_w16.py` 再单独把融合 launch 的 8 warps 改为 16 warps，用于检验跨相寄存器并集。当前优先候选 `codex_mddn_fuse_c11_noinline.py` 保留原 8 warps、stage 3、md BN128、dn BN256 和每 256 列 down 尺度，把两相隔离到两个真正的 noinline device function。它们没有修改 `kernel.py`、`solution.py`、`kernel_v800_fuse.py` 或公共脚本。

旧记录的 1.4--1.5 ms 是 c3--c12 全体 act 流量和全案固定成本的合计估算，不能作为本候选的收益预期。c11 的基线 act 写读是 268.44 MB，132 个 CTA 的私有槽是 17.30 MB；按 2.75 TB/s 和已测 60%--79% 裸露比例，单纯流量项约 0.055--0.072 ms。算上可摊薄的元数据前言和一次 launch/wave 尾，乐观总收益约 0.15--0.27 ms tk。不能用全局固定的 raw/ms 系数换算：以锚 140221 的 c11 `tk=1.008 ms, tb=5.595 ms` 代入 `floor(100*tb/(tb+tk))`，case 分数当前为 84；省 0.15 ms 时为 86，省 0.27 ms 时为 88，对 12 案总分分别约增加 0.167 和 0.333。仅流量下界会落到 case 85，即约 +0.083 总分。单次性能读数只作方向性证据，应结合锚和后续扩案结果判断。

## 正确性审计

- `_FUSESET` 只含 c11。正式 all-calls/s2 版本由 `_FL[0]` 和输入 shape 守门，不依赖调用次数；样例槽和其余 11 案回落现役路径。
- `_FU[0]` 机械折叠为 0 后，AST 与 `kernel_v760a_tanh.py` 完全相同。
- md 与 dn 的 `BLOCK_K=128`、`tl.dot` K 顺序、缩放、FP8 转换和 dn 输出尺度公式均取自现役内核。中间结果仍先舍入到 e4m3，再由 dn 读取，因此没有删掉量化边界。
- grid 固定为 132。CTA `pid` 只访问 `SCR[pid*128:(pid+1)*128,:]`；同一 CTA 的后续 m tile 才复用该槽。不同 CTA 不读写同一 scratch 行，不存在跨 CTA 的生产者/消费者关系。
- md 的 scratch 写是普通 `st.global.cg`，dn 的 scratch 读是普通 `ld.global.cg`。两者之间有统一控制流上的 `tl.debug_barrier()`。Triton 3.4.0 NVIDIA 后端将 block barrier 降为 `bar.sync`；CUDA 的 block 同步语义对参与线程先前的 global/shared 访问提供顺序。第二个 barrier 位于 dn 消费后、下一轮 md 覆盖前。
- TMA descriptor 在融合核中只用于读 A、gate/up B 和 down B。scratch 和输出使用普通 pointer store/load，所以本设计没有 TMA store 与另一 CTA 的可见性问题。descriptor load 的值在 `tl.dot` 中被消费，相关 async wait 由 Triton lowering 管理。
- 对 200 组随机的 32 专家计数做了 CPU 索引覆盖检查：每个有效输出行恰好写一次，132 个私有 scratch 槽无别名。

## 资源预测

md 相和 dn 相每个 stage 都需要 A 16 KiB 加 B 32 KiB。stage 3 时单相为 144 KiB。源码控制流里两个 128x256 FP32 accumulator 顺序生存，但现有 ptxas 255 结果不能证明 Triton/PTX 后端确实将它们的寄存器区间复用；如果后端按整 kernel 合并分配，CTA 总寄存器需求仍可能接近两者之和。

如果 Triton 3.4.0 对两个顺序流水环做共享内存活跃期复用，metadata.shared 应接近 144 KiB，H800 上为 1 CTA/SM，132 个 CTA 正好覆盖 132 SM。如果没有复用，两环合计正好是 294,912 B（288 KiB），超过日志所示 232,448 B 硬件上限。仓库历史提交 114967 曾明确报告 `Required: 294912, Hardware limit: 232448`；这个数值与本核的两相 stage-3 静态分配完全吻合。140224 把两相都降至 stage 2 后仍以同样的 ptxas 255 失败，所以 288 KiB shared 不是单一根因；它可能与寄存器分配问题同时存在。

每个 128x256 FP32 accumulator 约占 32,768 个 FP32 标量，即 8 warps 时平均约 128 registers/thread，与现役单相内核相同；融合还增加了行尺度、索引和循环状态。若两组 accumulator 被合并分配，CTA 仅 accumulator 就约占 65,536 个寄存器。把 launch 改成 16 warps 只会把平均线程份额从约 256 降到约 128 registers/thread，并不会降低 CTA 的 accumulator 总数；因此 140241 的 w16 仍失败，不能排除 CTA 总寄存器容量或跨相 liveness。反过来，ptxas 通常会结合 launch bounds 做限寄存器与 spill，所以 exit 255 本身也不能确诊寄存器超限。这里仍是根因假设，需要 ptxas 资源报告或能编译的分相对照验证。

c11 scratch 为 17.30 MB。132 个并发 CTA 大约覆盖 4--5 个专家，活跃 gate/up 与 down 权重工作集估计约 12--15 MB，合计低于 H800 的约 50 MB L2。`.cg` 和 `evict_last` 是缓存提示，命中率影响性能而不影响结果。

### noinline 资源隔离候选

Triton v3.4.0 的 `ModuleAllocation` 对整个 call graph 做 shared 分配：每个 call op 被视为大小等于 callee shared footprint 的 virtual scratch，而普通 call scratch 的活跃区间只覆盖该 call operation；模块所需 shared 从 root function 的分配取最大值。`codex_mddn_fuse_c11_noinline.py` 因而把每个 m tile 的完整 md 和 dn 相分别放入 `@triton.jit(noinline=True)` 函数，caller 只持有标量状态并在顺序调用间放统一 barrier。理论 metadata.shared 应接近 `max(144,144) KiB` 加少量 barrier 开销，而不是两相求和。函数 ABI 也把两个 accumulator 的寄存器分配隔离开。

noinline 函数不能接收非标量 block tensor，因此 md 将原先跨相保存在寄存器里的 128 行 FP32 尺度写入 `SSCR[pid*128:(pid+1)*128]`，dn 再从相同私有槽读取。SSCR 总容量只有 `132*128*4 = 67,584 B`；每个 m tile 额外写读 1 KiB，c11 全程不超过约 1.08 MB，不改变每 256 列 down 量化尺度。两个 helper 只接收标量 pointer、TensorDescriptor、标量索引/stride 和 constexpr，不返回任何 block tensor。官方 v3.4.0 单元测试包含 noinline 函数内 descriptor 访问、descriptor 参数以及 shared-memory dot，说明这些上游能力本身存在；实际 triton_dist fork 与 noinline WGMMA call graph 的组合仍必须由 P1 真机 JIT 验证。

ABI 性能代价约为每个 m tile 两次 device call；c11 的有效行至少需要 1024 个 tile，加上 32 个专家各自向 128 行取整后总数最多约 1055，平均每 CTA 约 7.8--8.0 个 tile，即每 CTA 约 15.5--16 次调用。若每次为几十到数百周期，预期远小于 0.1 ms，但 noinline 会阻断跨函数优化，并可能引入参数搬运或 local stack，所以必须以 tk 实测为准。

### `.cg` 与 `evict_last` 的 PTX 组合

140220、140224、140241 和 140247 的四个 c11 失败版都有一个更直接的共同点：新 SCR/SSCR 普通 global load/store 同时指定 `cache_modifier='.cg'` 与 `eviction_policy='evict_last'`。Triton v3.4.0 的 NVIDIA lowering 对 `evict_last` 同时发出 `L1::evict_last` 和动态 `L2::cache_hint`，而 `.cg` 独立发出 cache-operation qualifier，最终形态会同时包含 `.cg.L1::evict_last.L2::cache_hint`。NVIDIA PTX ISA 8.7 的 `ld`/`st` 文法把 `.cop`（包含 `.cg`）与 `.level1::eviction_priority` 放在互斥的两条产生式中；`.cg` 本身又表示只在全局层级缓存、旁路 L1。上游 Triton 3.4 单测分别覆盖 cache modifier 和 eviction policy，但没有覆盖二者组合。仓库只有未找到 AC/SID 证据的 `kernel_v660b.py` 也出现过该组合，现役基线没有可确认的组合先例。因此非法 PTX 组合目前比纯资源超限更直接地解释“所有结构版本恒定 ptxas 255”；前文 shared/register 推断降为次级假设，只有删除该组合后仍失败才继续按资源路径归因。

`codex_mddn_fuse_c11_noinline_noevict.py` 是相对 140247 的单变量修复：只从四个 SCR/SSCR load/store 删除 `eviction_policy='evict_last'`，保留 `.cg`、两个 barrier、8 warps、stage 3、tile、量化和所有数学。它仍让 scratch 绕过 L1 并进入 L2；删除的只是额外驱逐提示，不改变内存一致性或值。AST 检查确认旧版恰有四个 `.cg+evict_last` 调用，新版有四个 `.cg` 调用且没有组合；SHA256 为 `714b9bb3638645aac6a344e77f0bd2ebf61486950e91875b50c47639a1f5a1cb`。

这次归因也修正了审计方法：API 分别接受两个关键字，只能证明各参数独立受支持，不能证明 lowering 后的组合满足目标 PTX 文法。后续凡是叠加 cache、eviction、memory semantic 或 scope qualifier，都应检查目标版本的最终组合产生式或已有同组合的真机 AC，不能只查 Python API 签名。

140254 对这个单变量修复给出 Accepted：c11 两道 SQNR 为 23.27/23.31，`tk=1.172 ms, tb=5.630 ms`。配对锚 140253 的 c11 为 1.006 ms，所以 noinline 融合慢 0.166 ms（16.5%），没有晋升价值；但同一源码删除四个 eviction hint 后 ptxas 255 消失，强证实非法 qualifier 组合是此前恒定编译失败的共同原因。当前有两个严格的单 kernel 对照：`codex_mddn_fuse_c11_s3_noevict.py` 从原 all-call stage-3 版只删除 SCR store/load 两处 `eviction_policy`，SHA256 为 `81a6b19630780892390690bf22a11040dc7f79ccea5c29ede280a21070441e54`；`codex_mddn_fuse_c11_s2_noevict.py` 对 stage-2 版作同样两处删除，SHA256 为 `a4dc8032f0b693bf4aedf4ddcd5b9d547847765e35a6178f8a2aca91cb3ce4d2`。两份都保留 `.cg`、8 warps、两个 barrier、tile 和数学。stage-3 的 288 KiB、stage-2 的 192 KiB 都只是按源码的静态相加估计，是否复用以及是否 OOR 必须由修复非法 PTX 后的平台结果确认。

## 已完成验证与未完成项

已完成：Python `py_compile`；关闭融合后的 AST 等价；白名单静态检查；随机 metadata/行覆盖/scratch 槽检查；与现役两相数值顺序逐段人工对照。首次 stage-3 探针提交 140220 的 c11 在首次 JIT 的 ptxas `sm_90a` 阶段以 exit 255 失败，没有产生 SQNR 或 tk；同一提交其余路径通过，配对锚 140221 全部通过且 c11 两道 SQNR 最低 23.31 dB。stage-2 诊断 140224 仍在同一个 c11 ptxas 阶段 exit 255，其余 case 通过，同样没有 c11 SQNR 或 tk。因此 288 KiB shared 超限不是充分解释。stage-2/w16 诊断 140241 也以相同的 c11 ptxas 255 失败，其余 11 案通过；由于 w16 不改变 accumulator 的 CTA 总寄存器数，这个结果只排除了“单纯降低每线程平均寄存器份额即可编译”，没有排除跨相寄存器并集。当前优先假设是跨相资源区间未被复用，或 Triton 3.4 对同一 kernel 内两个 staged dot loop 的 codegen 失败；两者都尚未确诊。

noinline 版本已通过 Python `py_compile` 和结构 AST 检查：只含 c11 shape 门控、两个 helper 都是 `noinline=True`、无 block return、caller 的两个 barrier 均在统一外层循环上。另对 200 组随机的 32 专家计数重做行覆盖和 SCR/SSCR 私有槽检查，所有有效输出行恰写一次，同一 pid 只按顺序复用自己的槽。本机仍无法执行 Triton JIT 或数值 oracle；这些检查不能替代 P1 的两 oracle、determinism 和全调用验证。140247 的 noinline 原版仍在 c11 ptxas `sm_90a` 阶段 exit 255、没有 SQNR/tk，其余 11 案通过；由于该版仍带四处可疑的 `.cg+evict_last`，这个结果尚不能否定 noinline 资源隔离机制。

本机没有可运行的 Triton/CUDA 环境，因此本机检查没有 JIT、ptxas、H800 数值或性能实证。Python 可编译不代表 Triton JIT 可编译。平台只返回 `CalledProcessError` 的命令和 return code；部署版把 stderr 重定向到 `make_cubin` 内部的临时文件但没有把内容附到异常，外层合法 `try/except` 只能取得空的 `stderr/output`，无法补回根因文本。

下一步先测 `codex_mddn_fuse_c11_s3_noevict.py`，再测 s2 对照。s3 若 Accepted 可直接评估融合本体并与 noinline 的 1.172 ms 比较；若明确报告 shared OOR，s2 提供资源安全对照；若两者都编译通过，二者差值给出 pipeline 深度收益。不能用 s2 与 noinline s3 的差值单独归因 ABI，因为两者还同时改变 stage。若单 kernel 编译通过但都不快，则停止 c11 融合路线。`codex_mddn_fuse_c11_s2_w16.py` 只保留为已完成的归因样本，不再扫 stage/warps。

如果 noinline 仍 ptxas 255，下一项只做一个分相编译诊断，不再改 stage 或 warps。首选是 **md-only noinline**：让一个外层 kernel 按现有 c11 m-major 映射调用完整 `_fused_md_phase`，但把结果写回标准完整中间 act/scale 缓冲，随后复用现役独立 dn 与 final 路径生成正常输出。它保持每次调用结果完整且输入相关，同时只保留一个 noinline staged-dot 相；若仍失败，根因集中到 noinline md/TMA/tanh/WGMMA 或该 m-major 循环形态；若通过，则 combined-module/cross-phase 资源或第二个 noinline 调用更可疑。这个诊断需要把 helper 的输出行从私有 `offs_sm` 改为全局 `offs_m`，并保持现役 FP8 act 与逐行 scale 布局。它只用于可编译性归因，不据其附加 launch 性能做晋升判断。若 md-only 通过，再决定是否有必要做对称 dn-only；不同时提交两者。

另一条纸面备选是在 8 warps、stage 3 下把 md `BN_MD=64`、dn `BN_DN=128`：两相均为 `(16 KiB A + 16 KiB B) * 3 = 96 KiB`，合计 192 KiB；两个 accumulator 也都降到约 64 registers/thread。代价是 c11 的 md 与 dn N-tile 数都翻倍，两个 A 操作数各多读 536.9 MB，合计约 1.074 GB；若命中 L2 理想约 0.07--0.12 ms，若落到 HBM 则约 0.39 ms，并且短 K 的 fill/drain 次数翻倍。dn 输出量化尺度还会从每 256 列变为每 128 列，需同步修改 `CSCL` 与 final gather 的尺度索引，不能当作两行扫参。历史 V498 的 c11 md BN64 整案慢 3.1--3.9%，而 E8 的 dn BN128 慢 29--55%（形状不同，只能作风险上界）。只有 w16 能通过编译后，才值得决定是否实现这条窄 N 方案。

## 语义依据

- [Triton v3.4.0 `core.py`](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/language/core.py)
- [Triton v3.4.0 NVIDIA load/store lowering](https://github.com/triton-lang/triton/blob/v3.4.0/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp)
- [Triton v3.4.0 shared-memory allocation and liveness](https://github.com/triton-lang/triton/blob/v3.4.0/lib/Analysis/Allocation.cpp)
- [Triton v3.4.0 JIT function/noinline handling](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/compiler/code_generator.py)
- [Triton v3.4.0 noinline TensorDescriptor tests](https://github.com/triton-lang/triton/blob/v3.4.0/python/test/unit/language/test_tensor_descriptor.py)
- [Triton v3.4.0 noinline shared-dot tests](https://github.com/triton-lang/triton/blob/v3.4.0/python/test/unit/language/test_core.py)
- [NVIDIA PTX ISA 8.7 load/store qualifier grammar](https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld)
- [CUDA block synchronization and global-memory ordering](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#synchronization-functions)
- [Triton maintainer explanation of `num_stages` and shared-memory growth](https://github.com/triton-lang/triton/discussions/512)
