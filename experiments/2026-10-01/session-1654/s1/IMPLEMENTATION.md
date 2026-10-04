# S1 c6 首版实现与验收

候选：`p1_s1_c6_v1.py`；SHA256 `05828fef7359fd70159046bc026b414219d7fd28559146cd8793799aaaa15fa7`。
基础：生产 v12 SHA256 `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`，未修改生产文件。

仅当原 producer 进入 token-only Q 支路且 `_GA[0]` 为真、完整 shape 为 `(8192,3584,64,1024,8)` 时启用 `_hybrid`。保留原调用阶段规则，不让 E256 或任何其他 shape 进入。producer设置标记，consumer只按该标记配对。其余调用走原 producer/consumer。

同一 GQ launch仍量化一次 X，每 token 写一次 Q[T,H] 与 SCALE_TOKEN[T]；随后每真实分支按 INV唯一映射写 AH/WI/ACT_SCALE[M]，M=T*k。三数组按当前 A producer 的结合顺序和指数位操作生成。token scale保留原返回意义；DN消费的act_rowscl明确返回ACT_SCALE，不把T行scale当M行scale。

MD保留ORDER//KTOP读取token Q、row mask、B descriptor/MMA、原f16x2 tanh与FP8舍入、compact ACT满tile TMA及尾tile masked store。只将参数计算换为compact AH/WI加载；删除重复SCL写，host不创建sorted weights。MD BM128/BN128/BK128、GROUP_M32、grid132/w8、outer stages2/launch stages3；GQ w4/stages1不变。DN/fin与全部其他原函数AST保持。

关键行：producer分发6453、consumer配对6592；新GQ7722/host7753；新MD7773/host7834。新增helper均追加末尾；旧函数返回合同不变。

检查：`python3 build_s1.py`和`python3 check_s1.py`。compile/AST通过；185个未改原函数AST一致；四个新增函数仅各定义一次，四个调用的全部形参有且仅有一次绑定。CPU覆盖实际c6的65536个branch及257/1 token边界，总67600个compact行的唯一写者与ORDER/INV映射、202800次逐标量FP32位比较均通过。结果见checks.json，完整diff见candidate.diff。

限制：没有导入提交kernel、GPU运行或OJ提交；尚未验证线上编译、全链SQNR/确定性、描述符生命周期及性能。CPU顺序舍入不证明GPU FMA或极端static scale边界。新GQ增加k次标量读写，可能抵消MD收益。首发只看完整AC与正常两对c6整案至少2%信号；异常计时独立记账，不作性能晋升。
