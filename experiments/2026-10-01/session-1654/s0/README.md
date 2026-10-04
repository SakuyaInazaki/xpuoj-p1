# S0 c11 独立备选

候选`p1_s0_c11_consti.py`，SHA256 `c56ddab9bb8d6c22e052a287911bf8ea1207db67e5cceb85e5171a41908c6198`。生产v12未改。

仅完整shape `(65536,1024,32,1024,2)`且实际进入现有A/TMA pre consumer时选择专用host；冷fallback、非A路径及其他case保持。kernel+host均追加，GPU helper带`@triton_dist.jit`，未使用函数对象顶层别名。

唯一计算语义变化是kernel参数I加`tl.constexpr`。将名字与I注解归一化后kernel AST与原TMA pre完全一致，专用host AST与旧host完全一致。M/counts/num_tiles仍runtime，BM128/BN128/BK128、GROUP_M8、grid132/w8/stage4/maxnreg232、flatten、DROP、量化/tanh、ACTcompact与DN/fin保持。

`build_check_s0.py`可重建并验证；compile、AST合同及两次调用参数绑定通过。接线6585、新kernel7714、新host7778。checks.json与candidate.diff保留验收证据。未做GPU测试/提交；线上编译、SQNR/确定性和收益未验证。仅正常全AC且c11约≥1.5%收益才值得保留；不能给历史最佳已零计时c11增加积分。

发现已有`experiments/2026-10-01/candidates/p1_S0_c11_constI_v1.py` SHA f51d50968793482b0af597e82f6e83616b0907d87c53df5896b56090bcb219db。它同样只特化I，但直接修改原host使用条件kernel表达式，无专用host/完整shape gate。本地现有notes/reports没有找到其SID或实测结果，不能断言未提交。若平台核实已测相同语义，优先复用结果，不重复评测本备选。
