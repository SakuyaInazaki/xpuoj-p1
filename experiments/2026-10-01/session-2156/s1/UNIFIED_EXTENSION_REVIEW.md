# expanded后续A路径与统一扩展差异（只读）

三个shape均属于_GASET且(H,I)在_dir_a集合内。expanded仅改_GA token-Q路径：n3–5为hybrid，n≥6时_GA=0/_dir_a=true，仍sorted-Q A。n为模块累计调用，不是已证实的测量序号。冷n1/2不变。

| shape | expanded后续n≥6实际consumer | 若仅补_dir_a统一producer并配对已有hybrid consumer的新增差异 | 保持/阻断 |
|---|---|---|---|
| c3 H2048/I2048/k4 | pre_nf：compact sortedQ、TMA-A；I runtime；outer2/launch4/maxnreg232 | 改tokenQ ORDER gather/SIMT A load，I constexpr；新MDg outer2/launch3/无maxnreg；不是只换Q布局 | 两边f16x2与h*th+h相同；两边满tile TMA ACT/尾pointer；ACTcompact、scalecompact、DN/fin相同。存在实际launch资源变化，若要求统一扩展保持原A launch合同则需新专用host，不能无修改声称stage保持 |
| c5 H3584/I2560/k8 | pre_far：compact sortedQ pointer-A；I runtime；outer2/launch3 | 改tokenQ ORDER gather，I constexpr；f32 tanh→f16x2，h*(1+th)→h*th+h可能FMA；满tile pointer ACT→TMA ACT | BM128/BN128/BK128、GM32/grid132/w8、outer2/launch3保持；ACT/scalecompact、DN/fin相同。数值与TMA生命周期须全AC；暂无布局阻断 |
| c7 H4096/I2048/k3 | pre_far同上；GQ k份sortedQ | 同c5，另新GQ pad4分支mask（expanded已实现） | MD KTOP/M仍真实k3，mask保证第四lane无读写；阶段同c5。ACT/scalecompact、原动态padded DN与INV_PAD/fin保持，不能顺带切static DN；暂无新布局阻断 |

所有统一扩展都应只在原_dir_a producer内部按完整shape选择新GQ与参数，并以实际_hybrid标记优先调用gather consumer；其余shape/cold分支不变，不能只强制consumer读取tokenQ。已有helper可复用，c3/c5/c7都有各自H/I/k特化，若之前已测expanded是否命中编译仍未知。

结论：c5/c7可按c6最小接线方法独立扩展，但需承认同样的数值与满tile TMA变化；c3还改变现有A侧launch4/maxnreg232→MDg launch3，无maxnreg，这是新增资源差异，优先另案验收而非称机械等价。c6正常结果若支持统一版，再按整数门槛优先c3或c7；不因c6 AC宣称其他shape已AC。此处没有生成或修改候选/提交。
