# Kineto Out-of-range 缺失机制：只读结论

结论：存在已发布历史实现支持的机制，但本项目没有直接匹配字段；不能定位为P1病因，也不足以支持新布局/结构触发实验。关闭此研究支路，继续有独立计算理由的S1/S2。没有改候选、环境、时钟或profiler，没有新增launch、OJ/custom提交。

| 新事实 | 来源/条件 | 对P1的边界 |
|---|---|---|
| 原始issue报告kernel事件缺失，坏机器Out-of-range计数3、好机器0；两者CPU GPU out-of-order计数均0 | [官方issue996](https://github.com/pytorch/kineto/issues/996)，2024-10-04，作者torch2.3.0+cu121/WSL2；是原始用户报告，无已确认根因 | 外部症状支持过滤机制存在，不证明本项目同版本/同原因 |
| 已发布PyTorch v2.3.0固定Kineto为3f30237e868ca92b46b309da17d84b37be373a6e | [官方子模块元数据](https://api.github.com/repos/pytorch/pytorch/contents/third_party/kineto?ref=v2.3.0) | 未宣称线上P1为torch2.3；main只用于说明现在拆入GenericActivityProfiler，不据此推断线上 |
| 活动起点早于采集窗口，或活动终点晚于窗口，即计数并丢弃，handleGpuActivity直接return | [固定历史源码](https://github.com/pytorch/kineto/blob/3f30237e868ca92b46b309da17d84b37be373a6e/libkineto/src/CuptiActivityProfiler.cpp#L407)，407–418、635–650；窗口由startTrace/stopTrace的system_clock取值 | 不要求GPU停算、不要求timestamp=0，不要求buffer耗尽；全kernel缺失需全部对应GPU记录越界，部分越界可造成部分缺失 |
| 历史GpuActivity把CUPTI start转换为Unix epoch微秒，duration由end-start纳秒转微秒 | [历史CuptiActivity.h](https://github.com/pytorch/kineto/blob/3f30237e868ca92b46b309da17d84b37be373a6e/libkineto/src/CuptiActivity.h#L40) | 时钟域/窗口不同步、零/错误时间戳、异步工作越窗口均能满足条件；现数据不能分辨哪个条件实际发生 |
| CPU/GPU timestamp次序检查仅warning并加计数，随后仍log活动 | 固定旧源码607–650 | wrong-order不是此版本中的直接删除条件；GPU过滤先于次序检查，因此无wrong-order也不能排除GPU已越界丢弃 |
| 283发原始结果及近期6个raw JSON无直接Kineto签名；只有21SID profiler cycle警告 | scan-results.json，近期153072/153115/153123/153131/153137/153151；scan_existing.py可复算 | 未见INFO/VLOG不等于未发生：日志级别及公开日志覆盖未知 |

关键可验证字段：Out-of-range计数、TraceActivity outside of profiling window的活动起终点/窗口起终点、Profile time range、Processed GPU records及Record counts、GPU op timestamp警告与CPU GPU out-of-order计数。283发中这些全部未命中。全项目旧结果的四处out-of-range字符串仅否定性审计字段/文字（false或not an out-of-range failure），不是Kineto日志。

可以保留：profiler窗口/时钟转换/聚合遗漏为未排除假设。不能排除：CUPTI零/错时间戳、活动丢失、cycle读取问题、跨rank聚合；也不能从此机制推出当前zero的稳定触发条件。必须先取得线上版本、原始GPU活动/窗口时间和范围丢弃计数，才可能将诊断缩窄。issue原程序的时间阶段差异也是症状，不自行归因为系统时钟跳变。

不存在本轮可据此推荐的新增合规结构实验：普通数学优化改变运行时长/编译时间，可能改变采集覆盖但因果未知，不能据此重开已失败J路线。S1/S2的实验理由仍是重复标量工作/ACT布局成本，异常独立入账。
