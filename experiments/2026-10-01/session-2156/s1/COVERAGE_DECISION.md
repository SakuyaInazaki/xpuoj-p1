# c6 S1覆盖诊断与统一设计（只读）

身份已实算：frozen v12 `08dd08eb51b1602648d31f0694912466d5de4be48b36d3604465216f8f9089f9`；S1v2 `d11bf076b18ff243984aa346b551c3c1bcbe8c0ba65a9bc2c18ad927c8ad3497`；扩展版 `0328320d02ba2425fe9904b0817cebea28ebfbbd780efc0729e9e642ca7a987b`。未改候选、未提交。

下表的n是该模块累计run_kernel调用计数，不是case编号，不是已确认的测量序号。假定本次shape恰为完整c6，代码设置_FL=1。

| n | _GA/_dir_a | frozen08dd数据流 | S1v2/expanded的c6数据流 |
|---|---|---|---|
| 1 | 0/false | BF16 tokens_sorted[M,H]→quant FP8 sortedQ[M,H]；旧fused rowA，ACT BF16 | 完全相同；非hybrid |
| 2 | 0/false | direct `_gq1p_tm`→sortedQ[M,H]；旧`_fgs_tma1_host`，ACT BF16 | 完全相同；非hybrid |
| 3–5 | 1/false | `_gq1p_tok`→Qtoken[T,H]/scale[T]；MDg gather ORDER//k，重复参数与SCL写，ACT FP8compact | 新vector GQ tokenQ[T,H]/tokenS[T]＋compact AH/WI/ACT_SCALE[M]；新gather MD preconsumer，ACT FP8compact |
| ≥6 | 0/true | `_gq1p_tm_params`→sortedQ[M,H]＋compact AH/WI/ACT_SCALE[M]；pre_far consumer，ACT FP8compact | **仍旧sortedQ A链，未启用hybrid** |

c6在expanded中的行为与v2一致。_CALLN全局初始化0，入口自增；现代码没有shape变更reset。已知c6属于_GASET。若module每case新建，n可能按case计；若跨case复用，n按进程累计；现有正常日志未给module导入身份/每次n，因此不能知道c6测量处于3–5还是≥6。

现日志能证明两组SQNR和两组确定性检查通过；部分失败日志公开warmup=1/iters=2/testdata_groups=2。它们没有给oracle、determinism、计时、warmup之间的run_kernel调用排列，不能由这三个数构造计时oracle。SID149972历史失败路径各case有不同torchrun launcher PID，证明确有分case启动历史；不能证明当前完整AC也这样。相同任务ID、整批500秒、相似tk也不证明共用module或未执行hybrid。S1 fullAC只证明评测检查通过，不能分离覆盖不足/净收益为零。

## 最小统一c6方案：正常shape分派，沿用既有相位

以已冻结v2为父，保持四个已验收helper不变，不复制新数学kernel。只修改_run_replicated：

1. 原`elif _GA[0] or (E256...)`中的完整c6 hybrid选择保留。
2. 在原`elif _dir_a`内部增加**完整c6 shape**分派：选择同一新tokenQ/compact参数producer，设置_hybrid=True并填_hybrid_ah/_wi/_scl；其他shape仍旧A producer。不新增_CALLN检查、不修改_GA/_dir_a，也不移动现有冷fallback。
3. 现q8 MD分支已有`if _hybrid`优先于`elif _dir_a`，直接复用新gather preconsumer。当_dir_a为真但_hybrid为真时只消费新compact参数，不读取空的_dir数组；不让tokenQ误入pre_far（pre_far按compact行读Q）。
4. Q仍T行、tokenS仍T行；DN只消费ACTcompact[M,I]与ACT_SCALE[M]。DN/fin、排序、route、metadata、CALLN/GA、NVSHMEM缓存全保持。

这样只把现有≥6 c6 A生产者/消费者也配对为hybrid，原3–5不变、call1/2冷fallback不变；**改变既有相位的实现选择，但没有新阶段识别或改变相位边界**。若要求连call1/2也覆盖，会涉及不同ACT BF16合同与新冷编译，不属于本最小方案。

预计编译风险：不增加JIT函数/参数/tile/stage。新GQ仅已有c6 H3584/K_BRANCH8特化；新MD仅已有c6 I1024/K3584特化，从重复调用角度不应新增维度特化。但代码文本改动可能影响线上改写/编译身份；原A在≥6的q8 pre_far f32 tanh将换为已验收MDg风格f16x2，数值路径变化必须全AC验证，不能称严格逐bit等价。GQ标量gather新增成本可能抵消MD节省；更多次数沿用新链也可能影响编译wall预算。

最小验收：核父SHA；仅_run_replicated diff；helper与v2全部AST不变；对每个c6 n=1/2/3/5/6/大值做不执行kernel的predicate真值表与producer/consumer布局检查（不把表作为计时假设）；其他12shape各n路径不变；确保两处实际producer填同一标记/参数，不能只按shape强制consumer；compile/全部参数绑定；已有branch CPU合同复用；由唯一平台执行者全12案两组SQNR/确定性。正常配对v2/统一版若后者才出现收益，支持覆盖/旧A路径成本差异，但仍不能精确证明测量调用序号；若都无信号，说明已测窗口整案净收益有限，不能仅凭此证明kernel完全没提速。
