# 21:56接手五分钟决策

生产实算SHA f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a，现为153151布局锚，旧README/v12当前描述过时。未改生产/候选，未提交。

现存17:20后notes+近期审计支持：最好异常153151完整AC raw89.75/net79.75；c8=.466、c9=.063、c10–12=0，c1tb40.455。真正正常可靠锚仍08dd的152241/248/251约raw82。新窗口名义正常最高153250 raw82.17，但c4tb6.953vs常态约4.94，多3个case整数分来源是tb上浮，不能晋升为结构正常锚。153278/153322 raw81.92/82.08；c7分别2.181/2.185，153250的c7=2.179。因此153151的c7=2.097未在这些等价新身份样本保持。

S0已有153193完整AC raw81.83，c11=.855vs约.857无≥1.5%信号，关闭。J已关闭。晚间p1_S1_c8ext/ident/shift名称并非真正S1：仍旧sortedQ A的等价c67 helper；只扩shape/改名/移位，未实现tokenQ+compact参数/gather consumer。不能因名字含S1就认为hybrid已测。

真正hybrid冻结v1 SHA05828fef7359fd70159046bc026b414219d7fd28559146cd8793799aaaa15fa7、v2 SHAd11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497；现存本地audit无这两个SHA结果。S2 SHAac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8也未见结果。平台最新状态由唯一执行者确认。

最多两项选择：

1. 首选真正S1 vector的c6统一配对版。复用v2全部helper，以当前f9ca锚独立派生；仅在既有完整c6 q8 MD eligibility覆盖所有实际producer时tokenQ+compactAH/WI/ACT_SCALE，按实际标记优先选gather pre consumer。冷call1/2原样。必要修补是取消新链对_GA3–5的依赖（不新增调用阶段规则），生产者消费者条件同步；完整shape限定，DN/fin不变。旧v1/v2有实际测量路径未知，直接提交不能排除未测到新链。向量版可减少展开标量指令链，但无GPU收益保证；正常两对约≥2%才扩c3，c6上档约需2.4%。
2. 次选已静态验收S2 c4 layout-only；可直接使用冻结08dd独立候选进行首次GPU正确性验证，或机械rebase到当前f9ca保留c67文本，由主agent选基座。双producer/实际padded标记/配套DN合同已完备，无需混flatten或stage调整。它首轮只改ACT布局，风险和改动比S1更大，正常c4上档需约7%，≥2%不等于立即加分；因此排在S1后。

不再测S0、纯改名/注释/shift或扩旧A副本。剩余两小时应先获取真实新结构首测而非再次探测旧文本身份。最高异常距门槛3整数分是条件账，不保证普通结构增益能叠到异常样本；c1tb若恢复，原距离即变化。
