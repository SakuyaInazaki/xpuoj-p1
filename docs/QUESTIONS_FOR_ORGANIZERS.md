# P1 待赛事方确认的问题

以下问题截至 2026-09-13 **尚未获得赛事方答复**，可直接转发。第 1–4 项优先级最高。

1. **【优先】P1 线上环境的精确版本是什么？** 请提供 Triton、`triton-dist` 的版本或 commit、PyTorch、CUDA Toolkit、`ptxas`、NVIDIA driver，以及评测镜像的名称、tag 或 digest；若比赛期间升级，也请说明生效时间。

2. **【优先】P1 是否可以开放四卡 custom test？** 若已有入口，请提供调用方式、支持的 mode、语言名、输入格式、额度和日志范围；单卡 `triton-h800` custom 无法验证 `triton-dist` 的四卡通信路径。

3. **【优先】P1 是否允许导入并使用 Gluon 的 Hopper WGMMA 实现？** 请明确允许的包/模块、版本、是否可随提交携带源码，以及对内联 PTX、外部编译产物和运行时编译的限制。

4. **【优先】各阶段的超时和可见日志分别是什么？** 请区分源码/模块编译、首次 JIT、数学正确性、确定性复测和正式计时，给出每阶段超时、总超时、超时归类方式，以及参赛者可见的编译器、PTXAS、运行时和判题日志范围。

5. **相同 shape 的不同测试点之间，权重如何更新？** 评测器是否会复用同一个 tensor 对象并原位改写内容，还是创建新对象？派生权重或量化结果的 cache 生命周期应限定在单次调用、单个测试点还是整个进程？是否有可靠的测试点边界或版本标识可用于失效 cache？

6. **正式计时的准确范围是什么？** 请说明是否包含首次 JIT、权重预处理、量化、缓存构建、分配、通信和同步；四个 rank 的时间如何汇总；哪些基于静态权重的预处理允许放在计时区外或跨调用复用？

7. **P1 评测硬件的可用资源和运行策略是什么？** 请提供每卡可用显存、GPU 型号与互联拓扑/NVLink 或 NVSwitch 情况、NVSHMEM 配置与对称堆上限、动态/静态 shared-memory 上限，以及是否锁定 GPU/显存频率或功耗状态。

8. **Triton CTA planner 对 `num_ctas=2` 是否存在已知限制？** 将 P1 的 TMA 内核移至单卡 Triton 3.6 辅助 custom 诊断时，`num_ctas=2` 触发 `PlanCTA.cpp:212: CTA tiling is already determined`；P1 自身目前版本未知。两套环境是否存在相关已知限制？请确认这是否属于特定 descriptor/TMA/WGMMA 组合限制或不受支持的用法，并提供推荐规避方式或可用版本。

## 可提供的复现

- 单卡 Triton 3.6 辅助 custom CID `dd535374-7eba-488f-bb1b-c1afc5fb86d9`：`num_ctas=2` 在 CTA planning 阶段触发 `PlanCTA.cpp:212: CTA tiling is already determined`。可提供[脱敏错误与结果记录](../experiments/2026-09-12/results/public24_dd535374-7eba-488f-bb1b-c1afc5fb86d9_parsed.json)。
- 单卡 Triton 3.6 辅助 custom CID `de45e5a9-16ed-475f-ab49-b733daa621d0`：输出 TMA descriptor store 与 `tl.range(..., flatten=True)` 组合在 MLIR verification 阶段触发 `scf.if` region/result arity 不匹配。完整源码见 [`fp8_c4_dn_tma_store_custom_bench.py`](../experiments/2026-09-13/candidates/fp8_c4_dn_tma_store_custom_bench.py)，脱敏错误见[错误记录](../experiments/2026-09-13/results/public24_de45e5a9-16ed-475f-ab49-b733daa621d0_user_error.txt)。
