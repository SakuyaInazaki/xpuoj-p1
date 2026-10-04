# S1 c3/c5/c6/c7 扩展备选（未提交）

文件`p1_s1_c3567_vector.py`，SHA256 `0328320d02ba2425fe9904b0817cebea28ebfbbd780efc0729e9e642ca7a987b`。父文件为冻结S1v2 d11bf076…，父文件及生产未改。沿用父helper名字，名字中的c6不再表示单一shape；选择范围由完整shape集合明确限定。

仅原token-only_GA路径的c3(16384,2048,32,2048,4)、c5(8192,3584,64,2560,8)、c6(8192,3584,64,1024,8)、c7(16384,4096,96,2048,3)启用。c4/c8/E256及非tokenQ路径不变；_CALLN/_GA没有改变。

GQ以K_BRANCH_PAD=next_power_of_2(k)生成arange；实际branch索引t*k+j、M=T*k和consumer KTOP均保留真实k。INV/ids/weight/bnorm load以及AH/WI/ACT_SCALE scatter全部mask=j<k；k3的第4lane返回other、无实际读写，不能写row0。Q[T,H]和token scale[T]在vector段之外各写一次。原bound/WI结合顺序和指数运算保持。

四shape原MDg的_gg=(E96且I1024)均假，因此其BM128/BN128/BK128、GROUP_M32、grid132/w8、outer stage2/launch stage3与既有新consumer一致。新MDconsumer和host与父v2完整AST一致；数值/epilogue/ACTcompact/DN/fin不变。相对父仅新增shape集合、GQ KPAD/mask、GQ host KPAD实参。

build_expanded.py生成；check_expanded.py通过compile、参数绑定、decorator、仅允许三个顶层函数改动、完整shape集合及每个branch内存操作mask断言、stage合同。CPU按四shape真实token/专家/k检查245760个branch唯一INV写者/ORDER映射与737280个FP32标量位比较；批量lane与scalar oracle同值，无效lane被排除全部scatter。结果expanded_checks.json；差异expanded.diff。

没有GPU/OJ运行；线上编译、SQNR/确定性、mask lowering及真实收益未知。沿用_GA3–5路径，正式测量覆盖仍未知。只备选，等主agent根据v2/S2结果决定评测。
