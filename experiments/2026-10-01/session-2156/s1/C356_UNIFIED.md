# c3+c5+c6统一备选（不自动入队）

候选p1_s1_c356_unified.py，SHA256 `75c702478afa8c7d282064fad23d6a542556f30593cae8cb41a201c38da0ea7a`；父3f950 c56未改。仅两处完整shape集合加入c3=(16384,2048,32,2048,4)，没有k3/KPAD/mask/helper改动。选择仅已有_GA与_dir_a生产者分支，cold与CALLN/GA边界保持，其他shape/DN/metadata/fin保持。

build_check_c356.py通过两谓词逆变整AST等于父、所有helper AST不变、compile及五调用绑定。k4合法power2 arange；branch stride/真实k、M=T*k=65536、tokenT=16384、consumer ORDER//KTOP(4)与compact ACT/scale[M]合同保持，metadata完全未改。CPU复用expanded_checks.json实际c3 T16384/E32/k4 INV唯一写者与FP32合同，无重复批测。

新增c3在非_GA旧A路径上：原pre_nf读compact sortedQ的TMA-A，目标改ORDER gather/tokenQ的SIMT load；旧I runtime变constexpr。原outer2不变，但host由stage4/maxnreg232变为目标gather stage3/无maxnreg（w8/GM32/BM128BN128BK128/grid132保持）。可能导致流水深度、寄存器分配和编译资源变化，不是纯c6同构扩展。

数值表达单独核实：c3两consumer原本均f16x2 tanh、h*th+h，未引入c5/c6旧pre_far的f32tan/silu结合差异；c3满tile ACT TMA和尾tile masked pointer也两边原本具备。相同表达不证明FP32 FMA/编译调度下全链bit等价或SQNR已验证。新输入gather/资源变化仍需GPU全AC与正常配对。

未GPU/OJ、未AC/性能结论、不自动入队。仅c6正式信号支持且剩余时间足够时由主agent在c56/c356二选一；证据c356_unified.diff/c356_checks.json。
