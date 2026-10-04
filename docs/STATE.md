# P1 当前状态

最新依据是 [异常证据与结构优化任务书](OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)，截点 **2026-10-01 14:51:56（Asia/Shanghai）**。接手先核源码 SHA，报告截点以后可能还有其他执行者的提交。

- **平台**：P1 net **79.08**，最佳 **152238 / raw 89.08**，累计提交 **3522**；账号总榜第 4、总分 250.08，不是 P1 排名。raw90/net80 目标尚未达到。
- **生产锚**：[kernel.py](../p1/kernel.py)，v12 SHA `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`；正常同码对照 152241/152248/152251，正常约 raw82。生产文件本轮未改。
- **新异常锚**：**152976 完整 AC / raw89.00**，c8=0.471 ms，c9–c12 全零。已冻结 [原码](../experiments/2026-10-01/guide-candidates/anchor_sid152976_88361f262ddd.py)，SHA `88361f262dddcb58986b23b9d54d9219c1645529b98752e8b768e19fc2ace673`。原码复测优先于继续扩大布局搜索。
- **最近其他结果**：152995 全案 tk 正常、raw81.92；153006 raw83.42、c11/c12=0.302/0.460 ms，属于非零低值。
- **连续窗口**：283 发（207 AC、43 WA、21 Canceled、12 TLE）；首次 SHA 的完整 AC 130 发中有 17 次低值、9 次精确零值；重复 SHA 的 77 发 AC 均无低值。不是独立随机样本，不能推算下一发命中率。
- **计时机制**：21 个 SID 有 torch.profiler cycle 警告；线上 host 源码被改写。CUPTI 缓冲耗尽尚未证实；旧 E1 辅助 launch 方案撤销。`timeUsed` 等于显示 Σtk 的微秒值，不能当整批 wall time。
- **关闭/降级**：R1 c11/c6 正常无收益，强制 A c3/c4/c5 未赢，R3 有参数错误及缺诊断的 rank2 exit255；普通 stage/tanh 扫描无可辨收益。旧 B、slot DN、EP2 等不重复立项。
- **当前任务**：J 有限布局对照；S1 token-only Q + compact AH/WI/ACT_SCALE + gather MD；S2 c4 双 producer padded ACT + 配套 DN；S0 备选 I constexpr。任务书给出总预算 24 发、实现合同与停止条件。
- **已准备候选**：[J1/J2/J12](../experiments/2026-10-01/guide-candidates/README.md)只改变一个或两个相邻 JIT/host 定义顺序，本地语法和完整函数 AST 控制通过；**未上 OJ，未证实 GPU 提速/异常命中**。
- **环境**：4×H800，Triton/triton-dist3.4 系，整批500秒；精确 fork/计时器版本未知。distributed custom 不可用，单卡 custom 可用但不能替代 P1 四卡。
- **正确性**：题目明确切换测试点须更新静态缓存；现 `_CALLN/_GA`、缓存键和 static scale 边界仍需独立审计。两组 SQNR/确定性通过不证明任意调用协议都正确。
- **本轮交付验证**：3471 个 branch 映射、10413 次 FP32 标量 bit 比较、9 组布局边界、1341 个完整 tile 不重叠；仅 CPU 合同。没有修改生产 kernel、没有新增正式/custom 提交、没有另启 agent。

详情与证据入口：[最新任务书](OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)、[CODE_MAP](CODE_MAP.md)、[提交说明](SUBMISSION.md)。禁止使用 `deepseek-brainstorm`。用户自行分派 coding agent；正式平台保持唯一提交者。
