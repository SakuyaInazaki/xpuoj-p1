# XPU-OJ 第一届算子优化比赛 — 工作流程（可执行版）

> 配套文件：`background.md`（背景/调研）、`goal.md`（目标/里程碑）。本文件给出从环境搭建到结果复现的完整流程，每步含具体命令、脚本与产出物。
> 约定：`$PROJ` = 本地项目根目录（建议 `~/xpuoj`）；所有路径相对 `$PROJ`。
> 信息分级：[A] 题包事实；[B] 调研结论；[C] 待确认——[C] 项在执行到相关步骤时若与平台实际不符，以平台为准并回写本文档。

---

## 0. 前置确认（M0，1-3 天）

**0.1 报名与取题 [A/C]**
1. 在 https://xpuoj.com 注册正式账号，报名比赛：https://xpuoj.com/contest/13 。
2. 登录后进入三题题目详情页，**逐字记录**：run_kernel 函数签名（参数顺序/类型/张量语义）、输出张量约定、测试点数量与配置（维度：hidden/E/topk/seq/n/C/r/T 等）、baseline 说明、T_b/T_h 数值（若有）、约束校验的精确口径（SQNR 计算方式、容差）。
3. 阅读官方评测指南：https://xpuoj.com/d/2 与白名单帖：https://xpuoj.com/d/3 ，记录提交格式、沙箱限制（Triton/TileLang 白名单、1 个 jit + 1 个 autotune 限制、禁用同步调用）。
4. 加入赛事微信群；向主办方确认以下 [C] 项并记录答复：
   - 赛题一评测方式（4 卡如何启动、NVSHMEM 对称内存如何分配、评测环境中的 Triton-distributed 版本）；
   - 赛题一是否允许预热预处理（题包只在赛题二、三写明 [A]）；
   - 赛题二、三是否也要求两次运行逐字节一致（题包只在赛题一写明 [A]）；
   - 评测机 CUDA/PyTorch/Triton 精确版本。

**产出**：`$PROJ/REPRO.md`（版本清单、题目签名、[C] 答复、时间表）。

**0.2 赛题一评测方式确认（关键路径）**
- 若主办方提供样例代码或环境镜像：直接克隆/拉取，跳过 1.3 的自建环境。
- 若未提供：本地按官方示例流程搭建（见 1.3），并准备「最小正确提交」验证平台侧环境（先提交官方示例改写的 run_kernel，确认编译与 4 卡启动方式）。

**0.3 校准提交（获取真实 T_b/T_h 锚点）**
- 对 P2/P3：先提交一版「朴素但正确」的实现（直接按参考语义写，不优化），确认返回 >0 分并记录 T_k 与（排行榜/详情页如有）T_b/T_h。
- 对 P1：评测方式确认后提交官方示例移植版。
- **产出**：`$PROJ/benchmarks/calibration.md`（每次提交的分数与时间记录）。此步骤同时验证提交链路，占 1-3 次提交/题（预算 60 次内）。

---

## 1. 环境搭建

### 1.1 通用（P2/P3 共用）

```bash
# 本地 Python 环境（版本与评测机对齐 [C]；评测机 CUDA < 12.6，本地先装 cu121/cu124 兼容版）
conda create -n xpuoj python=3.10 -y && conda activate xpuoj
pip install torch --index-url https://download.pytorch.org/whl/cu124   # 具体版本号对齐评测机
pip install triton==<评测机同版本>      # [C] 版本待确认；沙箱限制下提交代码仅依赖 torch/triton
pip install tilelang                   # 若选 TileLang 路线

# 自测：确认 GPU 可用
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

验证判据：`get_device_name(0)` 返回 H800，capability `(9, 0)`。

### 1.2 剖析工具（P2/P3）

```bash
# NVIDIA 官方 profile 工具（本地）
# Nsight Compute: https://developer.nvidia.com/nsight-compute 下载 .deb/.dmg 安装
ncu --version
nsys --version
```

### 1.3 赛题一环境（Triton-distributed，[B/C] 见 background.md §5.1）

```bash
mkdir -p $PROJ/p1 && cd $PROJ/p1
git clone https://github.com/ByteDance-Seed/Triton-distributed
cd Triton-distributed

# 官方推荐：NGC PyTorch 25.04 容器（保证 NVSHMEM/驱动匹配）
# docker pull nvcr.io/nvidia/pytorch:25.04-py3   # [C] 本地有无 docker/GPU 容器能力
# 裸环境安装（需 CUDA >= 12.4 且驱动支持）：
pip uninstall -y triton              # 官方包与 Triton-distributed 内嵌 Triton 3.4 冲突，必须替换
pip install triton_dist-3.4.0-cp312-cp312-linux_x86_64.whl   # 版本 [C]：以仓库 README 的 wheel 为准（v0.0.2 为 cp312）
pip install nvshmem4py-cu12==0.1.2 nvidia-nvshmem-cu12==3.3.9 cuda.core==0.2.0
```

**验证判据**：`python -c "import triton_dist; print(triton_dist.__version__)"` 成功；`python -c "import triton; print(triton.__version__)"` 显示 3.4.x（fork 版）。

**跑通官方 EP MoE 示例（确认 4 卡链路）**：

```bash
cd python
# 官方测试内部用 multiprocessing spawn 多进程；确认机器可见 4 张卡
nvidia-smi -L   # 期望 4×H800
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest triton_dist/test/nvidia/test_ep_moe_inference.py -x -s
```

验证判据：测试全绿（分布式路由→A2A→group GEMM→combine 全链路）。

**本地 4 卡 A2A 带宽基线（可选）**：`git clone https://github.com/NVIDIA/nccl-tests && make && ./build/alltoall_perf -b 8M -e 1G -f 2 -g 4`（NCCL 路线参考值；Triton-distributed 走 NVSHMEM，此项仅作上限参考）。

**产出**：`$PROJ/p1/env.md`（安装命令、版本号、官方示例跑通记录）。

### 1.4 项目骨架

```bash
mkdir -p $PROJ/{p1,p2,p3,benchmarks,scripts,logs}
# scripts/ 下放公共脚本（§2.2/§3.1/§4.3）
```

---

## 2. 正确性验证

### 2.1 参考实现（本地，FP32/FP64 口径，与平台「独立参考」同语义）

| 赛题 | 参考实现要点（实现时必须逐字对齐，[B] 公式见 background.md §5） |
| --- | --- |
| P1 | 单卡等价 FP32 实现：gate_weight 全精度算 logits → top-k → 每个 token 对选中专家算 gate/up、SwiGLU、权重乘、down，按 token 累加；再核对 4 卡分布式路径与其一致性（同输入同输出） |
| P2 | FP32 朴素：逐 Q head、逐 slice 算局部 attention（含 mask），按 LSE 精确合并（`LSE=log(exp(LSE1)+exp(LSE2))`、`O=exp(LSE1−LSE)·O1+exp(LSE2−LSE)·O2`），最后 sink 修正（`lse'=log(exp(lse)+exp(lse_sink))`、`O'=O·exp(lse−lse')`）；全部 FP32 |
| P3 | FP32 逐 token 朴素：RMSNorm(展平 nC) → `H̃=α·(x⃗'φ)+b` → σ / 2σ / Sinkhorn-Knopp（exp + 行列交替归一，**迭代 20 次**，顺序与 eps [C] 与赛题文档核对）→ `h0=H^pre·x` → F(h0)=Up(SiLU(Down(h0))) → `x'=H^res·x + (H^post)ᵀ·F(h0)` |

产出：`$PROJ/{p1,p2,p3}/reference.py`（供 sqnr 与对拍使用）。

### 2.2 验证脚本 `scripts/verify.py`

```bash
# 用法: python scripts/verify.py <p1|p2|p3> --kernel <run_kernel 调用> --ref <参考输出> --sqnr-threshold 22
```

脚本必须实现 4 项检查（对应 goal.md §3 的 C1-C4）：

```python
import torch, numpy as np

def sqnr(ref, out):
    """SQNR(dB)。ref/out 均为 torch 张量（bf16 或 fp32）。"""
    r = ref.to(torch.float64); o = out.to(torch.float64)
    num = (r * r).sum(); den = ((r - o) ** 2).sum()
    return float(10 * torch.log10(num / den + 1e-30))

def check_finite(out):
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"

def check_input_unchanged(before, after):
    """输入不可变：按字节比较。before 为调用前深拷贝。"""
    for b, a in zip(before, after):
        assert torch.equal(b.contiguous().view(torch.uint8),
                           a.contiguous().view(torch.uint8)), "输入被修改"

def check_deterministic(run_once):
    """两次运行逐字节一致（赛题一强制 [A]；二三默认执行）。"""
    o1 = run_once(); o2 = run_once()
    assert torch.equal(o1.contiguous().view(torch.uint8),
                       o2.contiguous().view(torch.uint8)), "两次运行输出不一致"
```

验证判据（goal.md M1）：SQNR ≥ 阈值+5 dB；四项检查全绿。**P3 的 Sinkhorn 对拍**：参考与 kernel 的系数张量在 FP32 内逐位一致后再进行整体 SQNR。

### 2.3 边界用例（写进 `scripts/verify.py` 的边界用例集）

- P2：单 slice / 多重叠 slice / 无可见 K 的行（lse=−inf）/ sink 极大（真实权重被压小）/ sink 极小 / GQA 不同 ratio（含 ratio=1 退化）。
- P3：T=1 / T=64（小 batch 延迟场景）/ 极端 r / 含零残差的 token。
- P1：topk=1 与 topk=8、token 分布极端不均（某专家 0 token、某专家超长）。

**产出**：`verify.py` + `$PROJ/benchmarks/verify_log.md`（每次运行记录）。

---

## 3. 性能剖析

### 3.1 计时 harness `scripts/bench.py`（模拟评测：预热 + 取平均）

```python
import torch, time

def bench(fn, *args, warmup=100, iters=2000, **kw):
    """预热 warmup 次不计时，测 iters 次取平均。近似评测口径 [C]（评测用 cupti）。"""
    for _ in range(warmup): fn(*args, **kw)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn(*args, **kw)
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms/次

# 多次重复取均值与标准差（复现性判据 std/mean <= 2%）
```

- 计时纪律 [B/C]：**run_kernel 内禁止 cudaDeviceSynchronize/torch.cuda.synchronize**（平台会污染计时）；本地比对时全部放在外层。
- 用法：`python scripts/bench.py p2 --iters 2000`，输出 `$PROJ/benchmarks/bench_<date>.csv`（kernel 版本、T_k、std、T_k/T_b、估算分数）。

### 3.2 剖析（P2/P3 优化前必须做一次）

```bash
# 1) 总览级
nsys profile --trace=cuda,nvtx -o $PROJ/logs/p2_trace python scripts/bench.py p2 --iters 200
# 2) 内核级（看占用率/访存/瓶颈）
ncu --set full -o $PROJ/logs/p2_ncu --launch-skip 200 --launch-count 5 python scripts/bench.py p2 --iters 300
# 3) 快速查关键指标
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats python ...
```

**判据**：定位瓶颈类型——(a) 计算受限（SM/算力利用率高）→ 优化 matmul/张量核利用；(b) 访存受限（DRAM 吞吐近峰值、SM 空转）→ 优化 tiling/复用/融合；(c) 延迟受限（占用率低、小规模）→ 减 launch、persistent kernel；(d) 通信受限（仅 P1）→ 查 A2A 与重叠。

### 3.3 理论上限估算 `scripts/roofline.py`（T_h 自估，[C] 与平台口径对齐前仅作参考）

```python
# 按题意：T_h ≈ max(flops / peak_tflops, bytes / peak_bw) [B/C]
PEAK_FLOPS = 989e12   # H800 BF16 稠密 [B]
PEAK_BW    = 3.35e12  # H800 HBM3 [B/C]
def t_h(flops, bytes_):
    return max(flops / PEAK_FLOPS, bytes_ / PEAK_BW) * 1e3   # ms
```

以 P2 为例的估算：`flops = 2·T·Hq·N·Hk`（可见部分）、`bytes ≈ 读 Q/K/V + 写 O`；以 P3 为例：`bytes ≈ 2·T·n·C·2B`（读改写），`flops = 2·T·C·(r + r + n + n²)` 量级。**产出的 T_h 仅用于相对判断，分数估算以平台为准**。

---

## 4. 迭代优化（M1→M5）

> 每轮迭代固定循环：**改代码 → verify.py 四连全绿 → bench.py 计时 → 记录 →（性能达标才）提交**。任何未过约束的版本不得提交（0 分浪费次数 [A]）。

### 4.1 赛题二优化路径（按 background.md §5.2）

1. **v1（M2，FA2 级）**：从 Triton 教程 `06-fused-attention.py`（https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html）改造：Q 行分块 + K/V 块循环；**所有 m/l/acc 用 FP32**；加 GQA 索引（`kv = q_head // ratio`）。先跑通无 mask 退化路径。
2. **v2（M3，静态 solver）**：
   - 预热期（同测试点输入不变 [A]）：把 slice 配置编译为「每 (q-block, slice) 的 k-block 起止表」与 mask 类型表，kernel 内零分支索引（FFA 博客：https://SandAI-org.github.io/MagiAttention/docs/main/blog/magi_attn.html）；
   - 多 slice：slice 级并行 → 局部 (O,LSE) → **确定性**归约合并（固定顺序两两 logsumexp，禁用 atomic 依赖顺序）→ sink 后处理融合（公式见 background.md §5.2，lse_sink 预热期算好）；
   - GQA：grid 按 (q-block, kv-head) 划分，kernel 内循环复用 KV；
   - 边界：lse=−inf 行显式置 0；sink 极值用 logsumexp 稳定形式。
3. **v3（M4，冲刺）**：Triton `warp_specialize=True`（sm90）+ host-side TensorDescriptor（TMA）；或迁移 CUDA/FA3 级（flash-attn `hopper/` 参考）；slice 按面积排序做负载均衡。
4. **不做**：FlashDecoding++ unified max（破坏精确 LSE，与赛题语义冲突 [B]）、FP8（无精度余量）[B]。

### 4.2 赛题三优化路径（按 background.md §5.3）

1. **v1（M2，论文 3 核）**：
   - 核 A：系数 GEMM——`φ=[φ^pre|φ^post|φ^res]` 拼成 `[nC, n²+2n]`，一次 `tl.dot`（bf16 x，tf32 φ，fp32 累加）；epilogue 除 RMSNorm 因子、加 b（RMSNorm 重排 + 权重吸收，数学等价）；
   - 核 B：轻量系数核——σ、2σ、Sinkhorn(20 次，FP32) 逐 token 并行；
   - 核 C：F_post,res 融合——`h0=H^pre·x`（元素级加权和）→ Down GEMM → SiLU epilogue → Up GEMM → 与 `H^res·x` 相加写回（元素级外积，不用 K=4 的小 bmm）。
2. **v2（M3，persistent 单核）**：读 x 一次 → 片上算系数（φ 驻 L2）→ h0 → F → 混合 → 写 x' 一次；Sinkhorn 压入寄存器；`$PROJ/p3/kernel.py` 按 T 与 r 模板化（T 小 → 单核消 launch；r 大 → F 的 GEMM 走大块 tiling）。
3. **v3（M4，冲刺）**：按 r/T 双维 autotune；若 F 的 FLOP 占比高，Down/Up 用更大 BLOCK_N 与更高 num_stages；CUDA Graph 捕获（若沙箱允许 [C]）。
4. **不做**：mHC-lite/TBP/Newton 对偶等数学变体（对拍失败风险 [B]）；BF16 下做 exp/Sinkhorn。

### 4.3 赛题一优化路径（按 background.md §5.1）

1. **v1（M2）**：官方 `test_ep_moe_inference.py`（https://github.com/ByteDance-Seed/Triton-distributed/tree/main/python/triton_dist/test/nvidia）移植为赛题 run_kernel；**路由索引改为稳定排序**（`flatten().argsort(stable=True).argsort()` 或 Triton 内确定性实现），**禁用 `tl.atomic_add` 计数路由**；combine 固定顺序；固定 num_warps/num_stages/tiling。
2. **v2（M3，fused megakernel）**：`ep_all2all_fused.py`（官方「megakernel with token optimization」）——把 dispatch/combine 的 NVSHMEM put/signal 与专家 GEMM 流水重叠；per-expert 对齐（`ALIGNMENT_BY_EXPERT`）；gate/up 合成 `[E, 2N, K]` 一次 GEMM + SwiGLU 切片；top-k 权重 FP32。
3. **v3（M4，冲刺）**：虚拟专家（`all_to_all_vdev_2d_offset.py`，token 分布不均时）；双缓冲（`call_count % 2`）与缓冲上限按最大 batch 设置；A2A 与 GEMM 的比例调优。
4. **确定性铁律**：两次运行逐字节一致是硬约束 [A]——每次改动后跑 `check_deterministic`，不达标不提交。

### 4.4 调参与提交节奏

- 提交预算：每题 ≤ 60 次（见 goal.md §1）。策略：本地全绿 + 计时达标才提交；每 3 天集中提交一轮（每次提交前把 verify+bench 日志存 `benchmarks/`）。
- Triton autotune：本地先用完整 autotune 找最优配置，**提交版固定写死参数**（沙箱仅 1 个 autotune [C]，避免评测时反复自调浪费计时）。
- 分数估算器 `scripts/score.py`：实现题包公式 [A] `floor(100/max(0.0001, 1 + (Tk−Th)/(Tb−Th)))`，T_h 用 3.3 节估算或平台值；本地每次 bench 后输出估算分数。

---

## 5. 结果复现与终局（M5）

### 5.1 复现归档（在 `$PROJ/REPRO.md` 维护）

```markdown
# REPRO.md
## 环境
- CUDA: <版本, 评测机 [C]> / 本地 <版本>
- PyTorch: <版本> / Triton: <版本> / Triton-distributed: <commit hash> / TileLang: <版本>
- 评测机硬件: H800×4（P1）/ H800×1（P2/P3）[A]
## 代码
- 每题最终提交的 commit/文件清单（run_kernel 入口文件路径）
- 固定编译参数（num_warps/num_stages/block 尺寸/autotune 配置）
## 数据
- benchmarks/ 下每次 bench 日志（版本→T_k→std→T_k/T_b→估算分数）
- 平台提交记录表（提交 ID、时间、分数、罚分）
## 确定性
- verify.py 的 4 项检查最后一次全绿记录（含随机种子/输入哈希）
```

### 5.2 复现步骤（打包成 `scripts/repro.sh`）

```bash
# 1) 重建环境（记录于 REPRO.md）→ 2) 冻结依赖版本（pip freeze > requirements.txt）
# 3) 跑 verify（四连全绿）4) 跑 bench（3 次，std/mean ≤ 2%）
# 5) 对照 REPRO.md 中的平台提交记录，确认最终提交对应 commit 与本地一致
```

**判据**：冻结后 3 次重复测量 T_k 稳定（std/mean ≤ 2%）；verify 四连全绿；平台最终提交 = 冻结 commit。

### 5.3 最终提交清单

| 项 | 内容 |
| --- | --- |
| P1 | run_kernel（Triton-distributed，MegaKernel），4 卡启动方式按 [C] 答复 |
| P2 | run_kernel（CUDA/Triton/TileLang 三选一，与本地验证一致） |
| P3 | 同上 |
| 时间线 | 冻结版 09-25 前；09-25~10-01 仅小修补；10-01 23:59 截止 [A] |
| 收尾 | 复核「不满足约束 0 分」[A] 的检查项、输入不可变、无同步调用、无越权访问行为（参赛规范 [A]） |

---

## 6. 附录：脚本骨架索引

| 脚本 | 位置 | 用途 |
| --- | --- | --- |
| verify.py | scripts/ | SQNR/有限/输入不变/确定性 4 连 + 边界用例集 |
| bench.py | scripts/ | 预热+平均计时，输出 CSV 与估算分数 |
| roofline.py | scripts/ | T_h 自估（flops/峰值 vs bytes/带宽） |
| score.py | scripts/ | 题包评分公式实现 |
| repro.sh | scripts/ | 环境重建+验证+基准复现 |
| reference.py | {p1,p2,p3}/ | 各题 FP32 参考实现 |
| kernel.py | {p1,p2,p3}/ | 各题 run_kernel 提交入口 |

**开工顺序建议**：先做 §0.1（取题+[C] 确认）→ §1.1/1.2（P2/P3 环境）→ §2.1/2.2（P2 参考+验证，M1）→ §4.1 v1（M2）→ 并行推进 §1.3（P1 环境，依赖 [C]）→ 依里程碑推进。
