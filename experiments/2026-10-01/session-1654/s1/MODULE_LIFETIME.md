# S1调用路径：只读决策信息

确定事实：生产run_kernel无条件递增模块全局_CALLN，未按shape重置；_GA只在已有3–5调用窗口且shape属于_GASET时为真。S1 v1/v2仅在原token Q producer且_GA为真+c6完整shape时启用，consumer按_hybrid返回参数配对；源码本身不能保证计时调用恰在3–5。

本地未找到线上triton_dist_runner.py或triton_sandbox/executor.py完整源码。现有回溯只能提供线上路径/调用片段，不能证明模块每case重新导入。同一OJ_JUDGE_TASK_ID出现在12case日志不能证明同模块；整批500秒也不能证明同module。

直接历史进程证据：283发窗口的SID149972有逐case独立torch.distributed.run launcher PID：tc1=2832785、tc4=2834206、tc6=2834884等；对应各自per-rank目录和失败worker PID。其日志证明确有逐case另启动torchrun的历史失败执行路径，不能把_CALLN一定跨12case累计作为事实。此提交失败后各case被分别执行，不足以证明当前完整AC正常路径或S1测量调用边界。抽取见module-lifetime-evidence.json。

当前AC日志152238/152241/152976只公开tc1完整launcher片段，其他case主要SQNR/确定性汇总；没有module导入标识、每次run_kernel序号或S1实际执行签名。warmup=1/iters=2/testdata_groups=2只说明runner配置，无法还原SQNR/确定性/测量的具体调用顺序。因此S1在c6正式计时调用中是否被执行仍未知；性能相近也不是未执行证明。

可设计后续S1统一配对版（本轮不实现）：

1. 在_run_replicated现有GQ分发前，用完整c6 shape与现有q8 MD eligibility建立局部标记；predicate覆盖现有`use_fp8 and not use_int8 and not use_c2_plain`、`_CALLN>=3 and E<=96 and _q8_act`。不使用_GA来选新链，不新增调用阶段规则。
2. 标记为真时，无论旧_GA/旧_dir_a，均使用tokenQ+compact AH/WI/ACT_SCALE producer。其他producer保持原合同，包括冷call1/2 fallback。
3. 在同一个既有q8 MD分支里优先按该标记调用gather pre consumer；未启用则走旧A或旧MD host。只从实际新producer取得参数；DN/fin不变。
4. 保持topk映射、量化和数值结合、tile/stage；不增加计时/输出探针或无用launch。新增helper或复用现有S1 helper；独立文件冻结。
5. 必须验证call1/2旧compact、call>=3新的tokenQ/compactscale所有路径配对，且其他case完全不变。全AC和正常同窗对照判断收益。它覆盖更多既有producer，在非_GA调用中也替换sortedQ/A consumer；这是新的性能实验，不能当v1的小参数复测。

v1/v2未改，未提交OJ。决策：不因_CALLN全局无reset直接否定v1已执行；也不把现有日志当计时执行证明。统一配对方案可消除_GA可见性依赖，但须独立验收与实测。
