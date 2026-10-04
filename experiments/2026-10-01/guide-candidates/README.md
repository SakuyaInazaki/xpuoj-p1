# 本轮指引生成的未测布局候选

依据 [优化任务书第 5 节](../../../docs/OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)。**这些文件没有由本次指导工作提交到 OJ**，没有 GPU 正确性、性能或异常命中结论。

- `anchor_sid152976_88361f262ddd.py`：原字节冻结的 SID 152976（历史 AC/raw89.00）源码；SHA `88361f262dddcb58986b23b9d54d9219c1645529b98752e8b768e19fc2ace673`。
- `p1_layout_J1_unmeasured.py`：只交换 MDg 与相邻 host 定义，MDg JIT 提交起始行 6007→6056。
- `p1_layout_J2_unmeasured.py`：只交换 TMA pre 与相邻 host 定义，JIT 提交起始行 6156→6177。
- `p1_layout_J12_unmeasured.py`：组合上述两处交换。

所有函数体（含同名重复定义顺序）、其他顶层 AST、窗口外 JIT 提交行号保持；未添加 GPU launch、延时、调用阶段分支或 profiler API。该控制不保证线上改写后缓存键相同/不同。

完整 SHA 和静态验证：[清单](../../../reports/2026-10-01-controlled-layout-probes.json)。可复算生成器：[脚本](../../../reports/build_controlled_layout_probes_20261001.py)。唯一平台执行者先核在途与已提交 SHA，再按任务书最多 6 发预算执行；不要与其他会话重复投递，禁止原地覆写这些文件。
