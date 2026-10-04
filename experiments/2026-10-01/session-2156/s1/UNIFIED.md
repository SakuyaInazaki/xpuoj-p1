# c6统一hybrid备选（未提交）

候选`p1_s1_c6_unified.py`，SHA256 `befd19dcc7a0824ae49c93c91151e051ee6d3d4c8e8da17e2b1fdf40e9a02b68`，父d11bf076…未改。

唯一diff：原GQ的`elif _dir_a`内部按完整c6 shape分派同一hybrid producer，返回Qtoken/tokenS/_hybrid_ah/_wi/_scl并设置_hybrid=True；其他shape走原_gq1p_tm_params。原3–5producer分支完全不变。现MD的if_hybrid优先于elif_dir_a，因此tokenQ只进gather hybrid consumer，不进pre_far。DN仍消费compact ACT及ACT_SCALE。所有helper AST、CALLN/GA/dir_a条件、cold call1/2 fallback与其他shape不变。

build_check_unified.py生成并检查。compile、五个调用参数绑定通过；AST可逆证明：仅将新增_dir_a内部c6分派还原，整个候选AST等于父AST（包含所有重复def及顶层语句）。六个代表n值1/2/3/5/6/100表明c6 1/2仍原cold，3/5原hybrid，6/100现hybrid；其余11shape×6相位选择不变。helper与父逐AST一致，CPU67600branch/202800FP32bit合同复用v2_vector_rechecked.json。

原_dir_a消费的pre_far是f32 tanh，新统一部分将用已有hybrid gather consumer的f16x2；SwiGLU结合顺序也从h*(1+th)变为h*th+h（可能产生FMA收缩），不能仅用tanh精度变化描述数值风险。写回从原pre_far全tile masked pointer store切为新gather consumer满tile TMA store、尾tile masked pointer store，带来不同descriptor/异步写回编译路径。不是严格bit等价，未验证全AC/SQNR/确定性、TMA生命周期或性能。没有GPU/OJ，不作性能结论。没有新增阶段或调用计数识别，也不能因此声称已确定线上计时调用序号。证据unified.diff与unified_checks.json；未覆盖生产或任何旧候选。
