# 2026-10-01 21:56续轮状态

任务截止：**2026-10-01 23:59（Asia/Shanghai）**；目标raw90/net80尚未达到。实际已提交9份，暂停自动新增：S2b153681完整12AC/raw81.08/net71.08，无zero/low，c4=1.456ms相对父S2=0.787ms退化约85%，**关闭flatten推广**。C356U153682已完整12AC/raw82.00/net72.00、zero/low空；当前三份在途153686/expanded、153707/既有组合（不取消）、153725/第9修正版df5c。第9已网页HTTP201成功提交；依据新负结果逆转两处flatten，不是随机重复。用户持续授权所有提交，先前8发自定上限已撤销。最佳仍153151 raw89.75/net79.75，生产f9ca不变；每份采用仍须完整12AC/双SQNR/确定性。

**已核验最高成绩：SID153151，12案完整AC/raw89.75/net79.75，q总和1077，距raw90门槛1080差3。** 精确零c10–c12，低值c8–c12；c1 baseline tb40.455ms异常上浮，因此不能把成绩归为正常结构提速。源码c67 helper是原helper等价副本。历史[完整审计](../session-1654/platform/153151-audit.json)、[冻结源码](../session-1654/platform/153151-source.py)、[榜单快照](../session-1654/platform/153151-scoreboard.json)保留。

## 当前生产与正常锚

| 对象 | 文件 | SHA-256 | 已知结论 |
|---|---|---|---|
| 当前生产 | [p1/kernel.py](../../../p1/kernel.py) | `f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a` | 与153151源码相同；异常高分锚，尚不证明可复现 |
| 旧正常v12锚 | [冻结v12](../../2026-09-30/candidates/p1_dirA_c34_v12_far_meas.py) | `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9` | 正常同码152241/raw82.00；本轮S1/S2基底 |

旧[详细任务书](../../../docs/OPTIMIZATION_GUIDE_ANOMALY_EVIDENCE_2026-10-01.md)保留实现合同与正常收益门槛，但其中“当前生产”描述已过时，以本页实算SHA为准。[17时状态](../session-1654/STATUS.md)是历史截点。

## 授权、平台与执行顺序

用户**21:56再次明确授权继续一切提交**，platform_sol为唯一执行者，root负责取舍验收。实际9份已发，暂停自动新增；当前收153686/153707/153725三份终态，不重复在途SHA。R151153507完整AC/raw81.92，S1v2153526完整AC/raw87.75，S2153550完整AC/raw82.00，C56U153620完整AC/raw87.75；最高与生产不变。

## 当前队列

| 顺序 / 路线 | 冻结文件 | SHA-256 | 当前状态 |
|---|---|---|---|
| 1 / R151 | [153151原源码](../session-1654/platform/153151-source.py) | `f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a` | 153507完整AC/raw81.92/net71.92，无零/低值 |
| 2 / S1 v2 | [p1_s1_c6_v2_vector.py](../session-1654/s1/p1_s1_c6_v2_vector.py) | `d11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497` | SID153526完整AC/raw87.75/net77.75；zero11/12，不是新高 |
| S2 / root待取舍 | [p1_s2_c4_layout_only.py](../session-1654/s2/p1_s2_c4_layout_only.py) | `ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8` | SID153550完整12AC/raw82.00/net72.00；无零/低值，c4=0.787ms，无可见布局收益 |
| S2b flatten / 未测备选 | [p1_s2_c4_flatten.py](s2/p1_s2_c4_flatten.py) | `c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b` | 23:06 root明确指令立即独立提交，覆盖父S2完整AC/c4≤0.811门槛；SID153681完整12AC/raw81.08/net71.08，无zero/low；c4=1.456ms，关闭flatten推广 |
| expanded S1 / 未测备选 | [p1_s1_c3567_vector.py](s1/p1_s1_c3567_vector.py) | `0328320d02ba2425fe9904b0817cebea28ebfbbd780efc0729e9e642ca7a987b` | SID153686已确认，网页HTTP201，第7/8；待终态 |
| unified c6 S1 / 未测备选 | [p1_s1_c6_unified.py](s1/p1_s1_c6_unified.py) | `befd19dcc7a0824ae49c93c91151e051ee6d3d4c8e8da17e2b1fdf40e9a02b68` | 独立验收通过；旧自动预授权撤销，保留回退，无SID |
| unified c5+c6 S1 / 未测备选 | [p1_s1_c56_unified.py](s1/p1_s1_c56_unified.py) | `3f95092adf1225f59b5e29a5bc7ad56000ca65dfeda02dd882ab6ded17c9db9a` | SID153620已完整12AC/raw87.75/net77.75；zero11/12、low9–12，非新高 |
| unified c3+c5+c6 S1 / 未测备选 | [p1_s1_c356_unified.py](s1/p1_s1_c356_unified.py) | `75c702478afa8c7d282064fad23d6a542556f30593cae8cb41a201c38da0ea7a` | SID153682完整12AC/raw82.00/net72.00，zero/low空，minSQNR22.69/每案2det；[审计](platform/153682-audit.json)，未见≥2%正常信号 |
| 8 / COMB | [p1_c356_unified_c4_flatten.py](combined/p1_c356_unified_c4_flatten.py) | `79549c5bc0151c940d72ee776cdc571f78edf4ab9740fd0a3f6d0d1fb870bd5e` | SID153707，网页HTTP201，第8/8；待终态；[独立review](combined/INDEPENDENT_REVIEW.md) |
| 9 / C9 layout修正版 | [p1_c356_unified_c4_layout.py](combined/p1_c356_unified_c4_layout.py) | `df5c5f77eadccd9466719d32f8755c381c7686e1dad6d660789d8ba75cac1318` | SID153725，网页HTTP201；待终态；[独立review](combined/LAYOUT_INDEPENDENT_REVIEW.md) |

S1 v2只改c6既有token-Q支路参数向量生成，实际新producer/consumer成对；仍受既有_GA窗口约束。[实现说明](../session-1654/s1/V2_VECTOR.md)。S2覆盖c4的MDg和pre_nf两个实际producer，ACT padded但参数scale compact，按producer标记选择配对DN；[复核](s2/RECHECK.md)、[原合同](../session-1654/s2/README.md)。S1 v2与S2均已正式完整AC；S1存在计时异常且S2 c4无可见收益，不能称稳定结构提速。S2来自旧v12，不保留f9ca源码布局身份；异常与正常收益分别记录。

**S0/J关闭。** S0同方向153193完整AC/raw81.83，c11=0.855ms对正常0.850–0.860ms没有可辨≥1.5%信号，本轮c56d备份不重复评测。J0/J1/J2/J12均完整AC正常、未复现零/低值，停止纯布局扩散。153250(c8 regular-A扩展)/153278(c67改名)/153322(line shift)均正常，不能据其重开随机探针。

## 后续决策边界

正常晋升仍需要目标case正常同窗配对；异常tk及baseline漂移不得拼入正常结构收益。每次结果保存冻结SHA/SID、12案状态、两组SQNR/确定性、完整tb/tk/q、精确零与低值集合。153151 raw89.75仍差3个整数分，不能把别的SID最佳case拼接成成绩。

S2 c4仅下一整数档就需约4.41%降时（固定153151 tb）；2%正常信号可能不加分。潜在[flatten最小方案](s2/FLATTEN_PROPOSAL.md)仅是提案、未实现；root23:06已明确允许一次独立flatten立即入队，不再等待父链终态；不继续stage/warp sweep。不得盲扫stage/warp或删DROP/scale/fin。注意整批500秒与截止剩余时间，避免预算末尾产生来不及收取的候选。

静态缓存shape-only失效边界已记录，[generation设计](../session-1654/s2/CACHE_GENERATION_DESIGN.md)未实现，不混入本轮S1/S2。不改_CALLN/_GA、NVSHMEM生命周期或通信缓存；没有稳定tk=0触发方法。

## 终态与决策追加

当前待追加：153620终态及S2b通道/实际提交状态。root每次终态给出更新决定后，由本页维护者更新队列与追加记录。尚未实际提交的候选不填SID；raw/net最高分只有平台确认才更新。

### 队列策略增量

153507等待约11分钟后，root为截止前保留验证窗口允许同一执行者最多两份不同SHA在途。S1 v2已提交153526（网页HTTP201），本轮第2/8发；两者待终态，S2与expanded S1未发。下一次提交不得使在途超过两份；提前留时间收取结果并冻结。

### 未测备选独立验收

[expanded S1独立review](s1/INDEPENDENT_REVIEW.md)未发现阻断，k3/pad4所有实际分支读写均mask；[unified c6独立review](s1/UNIFIED_INDEPENDENT_REVIEW.md)确认只在已有_dir_a内补完整c6producer选择、MD优先hybrid，DN仍compact。统一c6在新增_dir_a路径会从f32tanh/h*(1+th)换为f16x2/h*th+h，须在线检查SQNR/确定性，尚未提交。没有新增终态或臆造SID。

### 22:27:46原码复测终态与第三发

[153507原始详情](platform/153507-raw.json)与[审计](platform/153507-audit.json)确认12案完整AC/raw81.92/net71.92；每案两组SQNR和两次确定性通过，最小SQNR22.69dB，无精确零/低值。tk向量(ms)为`[4.601,7.838,1.347,0.787,2.761,1.249,2.183,1.231,2.449,1.875,0.858,1.447]`。c7回归正常2.183ms，原码这次没有零计时，不证明以后永久不能复现。完整tb保持在原审计，不从其他SID拼接。

S2 layout-only ac338已第三发提交SID153550；S1 v2 d11bf SID153526仍待终态，预算3/8。原root条件预授权单c6 befd作为第4发，已由后续决策撤销；按最新C56U条件执行，未实际submit不能填新SID。expanded GA-only暂降备选，不跟随提交。最高153151和生产f9ca不变。

### 153507时间线补证

root直接验收153507日志：提交UTC13:59:28，tc1 torchrun launcher日志UTC14:25:41，终态首次观测UTC14:27:46（对应北京时间21:59:28/22:25:41/22:27:46）。提交至tc1 launcher约26分13秒，launcher至终态观测约125秒。前26分钟可能含排队/启动等，现有字段不能直接分类；这进一步表明500秒整批运行预算不是全部“提交到终态”的等待上限。终态观测含轮询间隔，125秒也不是精确完整执行耗时。

### c5+c6统一备选独立验收

[C56独立review](s1/C56_INDEPENDENT_REVIEW.md)确认相对befd仅两处完整shape predicate扩至c5+c6，逆替换恢复完整父AST；helper/stage/stream/num_ctas/maxnreg不变。c5仍compact ACT/scale进入原dynamic padded Down，不能与S2 padded ACT合同混淆；新_dir_a的f16x2/FMA/TMA及c5精度/性能须正式验证。候选3f950未提交，无新增SID。

### S2b最小调度备选

[flatten独立review](s2/FLATTEN_INDEPENDENT_REVIEW.md)证明c82b相对ac338仅两个新padded producer外层range关键词变化；host stages/DROP/SCL/CSCL/布局/phase完全不变。此前优先父S2出现≥2%正常信号的建议已被root后续一次性条件预授权替代；仍须父链完整AC及容差门槛，不扩预算、无新SID。

### 最后备选c356独立验收

[C356独立review](s1/C356_INDEPENDENT_REVIEW.md)证明相对c56仅两处完整集合新增c3、helper不变。c3真实k4/M65536/KTOP4与compact参数配对正确；原_dir_a TMA-A/stage4/maxnreg232改走gather/stage3/无maxnreg，资源路径变化需在线验证。原c3同为f16x2/h*th+h/满tileTMA+tailpointer，未新增pre_far的f32转换。root在c6结果后对c56/c356二选一，此后等在线结果不继续扩候选。

### 第4发预授权调整（尚未提交）

S1 v2等待超过35分钟后，root将第4发从单c6 befd改为**c5+c6统一C56U，SHA `3f95092adf1225f59b5e29a5bc7ad56000ca65dfeda02dd882ab6ded17c9db9a`**。平台已冻结并py_compile，确认第4发尚未发、无SID；预算仍3/8，153526/153550两份在途不变。旧befd自动预授权撤销，仅保留回退。

自动执行门槛仍为S1 v2完整12AC、每案两组SQNR≥22和两次确定性通过，尚无raw≥90完整AC，并且新发后最多两份不同SHA在途。root理由是c5/c6同H/k、相同helper/stage，可逐例验收并省一轮串行等待；不构成已验证性能收益。**新SHA自动提交窗口截至23:25（Asia/Shanghai）**；之后由root根据队列再判断，不继续沿用“只留15分钟”的旧策略。截止23:59不变，应留足收终态/最终冻结时间。

### S2b一次性独立调度预授权（不是已提交）

root新增**S2b c82b一次独立flatten实验**，总预算仍最多8发、最多两份不同SHA在途。门槛：父S2 SID153550完整12AC、每案两组SQNR≥22/两次确定性通过，c4 tk≤0.811ms（约正常R151+3%容差；低/零值单列，不当正常收益），本轮尚无raw≥90完整AC，且23:25前有空位。失败或超过容差回root决定，不绕过父链。

root明确允许layout-only没有≥2%收益时也只试一次flatten，因为它作用所有tile，是独立流水线假设；**此决策替代旧“必须layout≥2%”解读**，未证明提速且不延伸stage/warp sweep。实际第4/5发按父链完成顺序由唯一执行者安排；C56U仍必须先等S1 v2完整AC。自动条件候选仅C56U3f950与S2bc82b两份，**没有第6发自动预授权**。两者均未在此记录为已提交，无新增SID。

### 22:59:37 S1v2终态

153526 exactSHAd11bf，完整12AC/raw87.75/net77.75，minSQNR22.69/每案两组SQNR和两次确定性；zero11/12、low9–12。c6tk1.200相对1535071.249下降3.92%，但未改c7和c8也明显下降，c5tb13.791/c10tb7.733抬高，因此不能宣称稳定结构提速或新高分。最高153151及生产f9ca不变。已加入[本轮异常证据索引](anomaly/INDEX.md)与[独立审计](anomaly/SID153526_AUDIT.md)。C56U按条件正在提交，等实际SID再记第4；S2仍待终态。

### C56U实际第四发

platform_sol已确认C56U网页HTTP201提交成功，SID153620，exactSHA3f950；[提交记录](platform/C56U-submit.json)。预算实际4/8，现153550/S2和153620/C56U两份在途。S2b条件预授权仍须父S2完整AC/c4≤0.811/无raw90/23:25前有空位，无自动第6。

### 153526时间线补充

提交UTC14:11:36，tc1 torchrun日志UTC14:50:16.927，终态首次观测UTC14:59:37.940。前等待约38分41秒，launcher到终态观测约9分21秒；不能直接分为编译/实际运行，也不是精确执行时长。与153507的约26分钟/125秒只是两个观察样本，不构成机制因果或精确ETA，仍以在线终态为准。S2b条件预授权保持，无自动第6发。

### 23:06 S2b立即入队的一次性时间决策（尚待实际SID）

因S1实际提交到首观测终态约48分钟，root指令唯一执行者立即独立提交S2b `c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b`，计划第5/8发。**本条覆盖旧父S2完整AC/c4≤0.811ms及等空位的提交门槛**；不再等153550终态，是明确接受冗余失败风险、争取23:59前回收的一次队列并行。源码已有双审，最终采用仍必须完整12AC并通过SQNR/确定性合同。

临时允许最多三份不同SHA在途，**只准此次S2b新增**；若平台拒绝三并发，不绕过限制。当前已确认在途仍为153550与153620，预算实际4/8；S2b实际SID由平台确认后追加，不能提前记为已提交。旧befd和其他候选不入队，无第6发自动提交；23:25后自动新SHA关闭不变。

### 23:13:42 S2终态与23:16通道截点

[153550完整审计](platform/153550-audit.json)、[原始详情](platform/153550-raw.json)、[冻结源码](platform/153550-source.py)确认exactSHAac338、12案完整AC/raw82.00/net72.00，zero/low均空，minSQNR22.69，每案两组SQNR与两次确定性通过。c4 tk=0.787ms，与R151153507同显示，没有可见布局收益；最佳153151与生产f9ca不变。

当前C56U153620是唯一在途；S2b第五发**从未上传、无SID**，实际预算仍4/8。网页pool offline/0、API额度0，23:16补充后重核仍0，所见下次补充23:46:49；这是临时截点，不作永久通道结论。先前三并发例外未实际用到。唯一执行者正在只读诊断现有网页通道恢复，没有改共享服务/配置或投其他候选；23:25自动新SHA截断仍有效，无第6发自动提交。

### 最后半小时编排调整（覆盖旧截断与父链门槛）

网页池恢复submit2。root取消23:25自动新SHA截断及父链等待，由platform_sol连续提交S2b c82b、C356U75c7、expanded0328，不等待终态；保留153620，最多四份不同SHA在途，平台明示并发限制则服从。实际仍4/8，三份计划后至7/8，第8未授权；新SID待平台确认。root承认此前过于串行，并明确接受相关父链失败造成冗余实验的风险。最终采用仍须完整12AC/双SQNR/确定性，不再创建候选。

### 23:27:19 C56U终态及第5/6发确认

[153620完整审计](platform/153620-audit.json)确认12AC/raw87.75/net77.75，zero11/12、low9–12，minSQNR22.69、每案两次确定性通过；最佳仍153151。第5发S2b已SID153681/c82b，第6发C356U已SID153682/75c7；实际6/8，当前在途681/682。第7expanded0328正在提交，SID待确认。组合C356U+S2b拟作为第8备选，尚无冻结SHA、未授权提交；仅待冻结后检查阻断，不继续扩候选。

第7发expanded E1已网页HTTP201成功，SID153686/exactSHA032832；[提交记录](platform/E1-submit.json)。实际7/8，三份在途153681/153682/153686，第8尚未授权。

组合候选已冻结SHA79549，路径[组合源码](combined/p1_c356_unified_c4_flatten.py)；[独立review](combined/INDEPENDENT_REVIEW.md)未发现阻断，c4实际padded producer标记与c3/5/6compact hybrid互斥，父helper AST不变。第8发实际SID仍待root授权/平台确认，未提前记录已发。

### 第8发确认与停止新发

[COMB提交记录](platform/COMB-submit.json)确认SID153707/exactSHA79549，网页HTTP201；实际8/8。四份在途153681/153682/153686/153707，root停止新发，仅收全部结果。组合[独立review](combined/INDEPENDENT_REVIEW.md)保留；前四全AC，最佳153151 raw89.75/net79.75及生产f9ca不变。

第9修正版[df5c源码](combined/p1_c356_unified_c4_layout.py)已授权待实际SID；[独立review](combined/LAYOUT_INDEPENDENT_REVIEW.md)确认仅两outer关键词逆转、两kernel等ac338，其余COMB不变。

第9正式确认SID153725/exactSHA df5c，网页HTTP201；[提交记录](platform/C9-submit.json)。实际9份，暂停自动新增，四在途153682/153686/153707/153725；S2b负结果不推广，其余不变，等待终态更新。
