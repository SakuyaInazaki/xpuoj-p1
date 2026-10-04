# c9 G64 int6 packed gate/up：首版实现审计

日期：2026-09-05  
候选：`p1/codex_int6_c9_g64_bk64.py`  
基线：已 Accepted 的 `p1/kernel_v811d_q6_gu_g64.py`（SID 139917）

## 数值契约

首版严格沿用 v811d 的顺序：BF16 静态权重先按输出行量化为 FP8，再对 FP8 值沿 K 维按 G64 计算 FP32 `amax/31` scale，round/clamp 到 `[-31,31]`。内核从 q4/q2 位平面重建 code，执行 `(code-32)*scale`，随后显式转为 FP8 再参与 dot。它没有改成 BF16 直接量化，也没有沿用“直接量化精度应更好”的未验证假设。

CPU 逐值模型与 v811d `_q6round` 数值完全一致。唯一位级差异是 int6 的单一零码不能保存 FP8 `-0` 的符号位；两者转 FP32 后逐值相等，dot 语义相同。

## 布局与容量

c9 gate/up 的 dense FP8 为 `256*4096*4096 = 2^32` 字节，即 4 GiB：

- q4：2 GiB；
- q2：1 GiB；
- FP32 G64 scale：256 MiB；
- 合计：3.25 GiB，减少 0.75 GiB / 18.75%。

首版不把 group scale 降成 FP16，因为那会改变已 Accepted 探针的量化/反量化语义。q4 的容量恰为 `2^31` 字节，内核先将 expert/row base 提升到 int64；校验器覆盖了最后一行、最后一字节的索引等式。

## 为什么首版是 BK64

TMA 可搬运紧凑 uint8 tile，却不能融合 q4/q2 两平面和 scale 的解包。K64 时每输出行读取 q4 32 字节、q2 16 字节和一个 FP32 scale；q2 的小 inner box 以及两平面合并使普通 `tl.load` 更直接。解包在寄存器完成。

K32 不是数学限制，只是旧吞吐探针控制 live range 的保守选择。BK128 可在一个原始 K step 中处理两个 G64 scale，但 BM=BN=128 的双累加器已经占据主要寄存器；解包阶段的 int32/FP32 中间张量可能造成 spill。FP8 最终值可被打包，不能简单按“每个 FP8 scalar 一个 32-bit register”估算压力，真正风险是 widened code 和 FP32 reconstruction 的 live range。BK64 先把 gate、up 解码顺序化，在两次 dot 之间缩短中间值寿命，并把 dot 数量控制为原路径的两倍。

## 调度、缓存和回退

专用途径仅由运行时 `(E=256, I=2048, H=4096)` 触发，其他输入继续执行基线。所有 c9 调用都使用同一 packed 权重路径；只保留基线已有的 activation gather/output-q8 模式，以保证后续 down 路径的接口不变。

缓存只保存静态权重派生物，不保存输出。key 包含三个源 tensor 的对象身份、shape、stride、dtype、device、rank 和 world size；命中后再以 `is` 检查三个源对象。替换输入会先清除旧派生 buffer，再重建。缓存保留源对象引用，避免 Python id 复用形成伪命中。

后续只读复核发现一个需要单独记录的边界：上述 key 能识别“换成另一个 tensor 对象”，不能识别“同一对象被原地改写”。普通 PyTorch 的私有 `Tensor._version` 通常会随原地写递增，但现有 P1 证据不能证明这个下划线属性能通过评测代理/静态规则；历史 sandbox 明确拒绝下划线开头的属性访问，因此不能把它当成已获许可的修复。题包交接记录给出的实际约束是：每个 testcase 独立进程，同一 testcase 的静态权重只读且固定，并明确允许按 shape 缓存；在该生命周期保证下不存在同对象原地变权重的正确性问题，跨 testcase 也不会复用 Python cache。相反，记录还说明同一 testcase 的多次调用可能获得新的 tensor 对象，因此本候选的 identity key 可能导致每次重新 gather/量化/打包，属于严重性能风险，并可能与 SID 140230 的 501 秒 TLE 相容；该次日志没有形状或栈，不能据此确诊。BK64 冻结文件未改。若另做修订版，应优先采用题包明确许可的 testcase 内 shape/metadata 静态权重缓存，而不是未经平台证明的 `._version`，并重新做独立审计与平台验证。

## 收益上限与停止条件

按 2.75 TB/s，少读 0.75 GiB 的理想上限约 0.27 ms。基线 c9 `tk` 约 2.59 ms、`tb` 约 8.4 ms，若理想收益全部兑现，单案整数分大致可增加 2，折合总 raw 约 0.17；真实收益更可能小于此值。

最大风险是 BK64 令 dot/loop 次数翻倍，普通 load、位运算、FP32 乘法和寄存器压力抵消带宽收益。第一次平台运行应先看 JIT/ptxas 是否成功，再看 correctness/SQNR，最后只以 c9 配对 `tk` 判断。若明显回退，应停止扩展；下一步只值得尝试用 `inline_asm_elementwise` 缩短 4 值解码、scale 和 FP8 convert 的 live range，再评估恢复 BK128 原 K-loop，不应重做已经判负的 K32 分解。

## 本地验证边界

`python3 p1/codex_int6_verify.py` 已通过：实际候选 packer AST 执行、逐值 pack/unpack、零组/端点、容量、索引和结构检查；候选、草案和校验器都通过 `py_compile`。本机没有 GPU，尚未验证 Triton 3.4 JIT、ptxas 资源或 GPU 数值/性能。

## BK128 未提交草案

独立文件 `p1/codex_int6_c9_g64_bk128_draft.py` 恢复原路径的 K128 dot 次数。每个 K step 读取 q4 64 字节/行和 q2 32 字节/行，并分别加载两个 G64 FP32 scale。它先解 gate、dot，再解 up、dot，避免两套 reconstruction 同时存活。

草案将连续 4 个输出 lane 的两个 q4 byte 和一个 q2 byte 按 32-bit word 处理。`inline_asm_elementwise(pack=4)` 用位掩码/移位一次生成 4 个 uint8 code，替代 Triton 层完整的 `h/l/shift/mask/code` 中间链。校验器对全部 `h0/h1` byte 组合及覆盖 0/1、稀疏位、交替位、全 1 的 q2 byte 验证了 word 解码代数，并检查两个 asm 点、K128 和双 G64 scale 索引。

它仍会在 `(code-32)`、FP32 scale 乘法和 FP8 convert 阶段形成一个完整 K128 B tile；只有 GPU 编译才能确定编译器是否把最终 FP8 四值打包并及时释放 code/scale。继续判据应是：BK64 首测至少成功 JIT 且性能亏损主要可归因于 dot 次数翻倍，或编译资源显示 BK128 仍有明显余量。若 BK64 已接近/超过 0.27 ms 的理论带宽收益幅度而变慢，BK128 才值得一次 ptxas 探针；若主要瓶颈是每值 FP32 scale/cvt，而不是 dot 次数，则 BK128 不会消除该成本，应停止该路线。
