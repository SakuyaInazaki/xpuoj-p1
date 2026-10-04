# C356U + C4 S2b 组合冻结验收

候选 `p1_c356_unified_c4_flatten.py` SHA256：`79549c5bc0151c940d72ee776cdc571f78edf4ab9740fd0a3f6d0d1fb870bd5e`。
父：C356U `75c702478afa8c7d282064fad23d6a542556f30593cae8cb41a201c38da0ea7a`；S2b `c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b`；共同08dd基底。

| 完整shape (T,H,E,I,k) | call 1/2 | 原GA 3–5 | 原dirA 后续 | ACT/DN合同 |
|---|---|---|---|---|
| c3 (16384,2048,32,2048,4) | 原fallback | C356U hybrid | C356U hybrid | compact ACT、原compact输入DN |
| c4 (16384,2048,32,1024,4) | 原fallback | S2b MDg padded | S2b pre_nf padded | compact scale、padded ACT；实际producer flag选择paired DN |
| c5 (8192,3584,64,2560,8) | 原fallback | C356U hybrid | C356U hybrid | compact ACT、原compact输入DN |
| c6 (8192,3584,64,1024,8) | 原fallback | C356U hybrid | C356U hybrid | compact ACT、原compact输入DN |

两个标志分别初始化False。完整shape集合不相交；c4不能进hybrid，c3/5/6不会设置padded标志。c4的S2判定保留inv_pad非None条件，flag仅由实际padded producer三元返回设置。logical M=T*k、padded P、scale长度、DN logical/padded参数完全复用S2b父。

验证：构建脚本断言两父SHA；所有原函数除run与08dd AST一致；4个S1 helper与C356U AST一致，6个S2 helper与S2b AST一致，无名称冲突。run只合4个S2 wiring差异片段，逆去后与C356U run AST一致。11处新增host/kernel调用必需参数完整，无重复参数；launch动态字典仅既有maxnreg。kernel装饰器完整；py_compile通过。CALLN/GA边界、数学、range flatten和host阶段参数均保持父版本。构建、差异和检查见同目录文件。

复用父CPU：expanded c3/c5/c6真实branch/FP32位合同与S2 9histogram/4157tiles/524288rows布局合同，不重复批测。未GPU、未提交。组合在线编译/fullAC/性能仍未知；继承hybrid f32 tanh→f16x2与silu结合顺序/FMA及store路径风险、c3原pre_nf stage4/maxnreg232→gather stage3/无maxnreg风险。outer flatten不保证实际重叠或收益。候选不自动入队。
