# c5+c6统一备选（未提交）

候选p1_s1_c56_unified.py，SHA256 `3f95092adf1225f59b5e29a5bc7ad56000ca65dfeda02dd882ab6ded17c9db9a`，父befd19未改。唯一diff是_GA和_dir_a内部两处完整c6 shape predicate扩大到完整c5+c6集合；helper、CALLN/GA边界、cold、其他shape及DN/fin均不变。两案真实k8，未引入KPAD、k3或mask变化。

逆替换两predicate后整个AST等于父；compile、五调用绑定、所有helper AST同父、两案n1/2冷与n>=3统一路径，以及其他10shape×6predicate检查通过。c5原MDg和旧pre_far都是outer2/launch3/w8/GM32；目标合同保持。CPU复用expanded_checks.json的实际c5 T8192/E64/k8 INV唯一写者/FP32合同，helper计算相同，无重复批测。

c5在原_dir_a调用中实际从sortedQ/pre_far转tokenQ/gather hybrid，I runtime→constexpr、f32tan→f16x2、silu h*(1+th)→h*th+h可能FMA、满tile pointer store→TMA（尾tile仍masked pointer）。ACT和scale仍compact；未AC、不是bit等价，GPU编译/SQNR/确定性/TMA生命周期与收益待正式评测。准备而未提交；是否选择由主agent看c6结果决定。
