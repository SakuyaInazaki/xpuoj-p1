# S1 v2 vector 备用候选

文件`p1_s1_c6_v2_vector.py`，SHA256 `d11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497`。未覆盖v1，未提交。

唯一改动在新GQ helper：将k=8个分支的static_range展开循环换成`j=tl.arange(0,K_BRANCH)`，按8lane批量load INV/FLAT_IDS/FLAT_W/BNORM与scatter AH/WI/ACT_SCALE。token Q和SCALE_TOKEN仍在这段前各写一次；AH的token scalar可广播到8lane。bound和WI的算式AST与逐标量版一致。所有其他顶层节点（含重复def）、consumer、run接线、stage/tile原样相同。

成本理由：提供8lane load/store与bound算式，使编译器可将分支参数操作统一调度，减少8份独立标量指令链；static_range本身已编译展开，不能声称原来在GPU上必然串行或向量版必然更快。向量稀疏gather/scatter不会变成连续内存，也可能有layout广播/转换及寄存器成本。

编译合同：KTOP=8为二次幂，tl.arange合法；INV为置换，同token各lane目标唯一，全token目标也唯一。其他k未接入。无需修改数值结合顺序；没有发现不能保持原算式的具体障碍。GPU降低到何种layout和寄存器分配未验证。

`build_s1_v2_vector.py`生成；`check_s1_v2_vector.py`通过compile、全部新增调用绑定、唯一helper改动/其他顶层AST不变、实际c6及边界67600行INV唯一写者/ORDER映射、202800次原oracle的FP32位比较，并补充每token8lane批量gather/scatter与标量结果逐位一致。证据`v2_vector_checks.json`和`v2_vector.diff`。

仅CPU/静态验证，无GPU/OJ；完整AC、SQNR/确定性和正常性能未知。是否评测由主agent根据v1结果决定。
