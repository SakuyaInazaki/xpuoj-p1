# S1独立提交前审查
候选SHA 05828fef7359fd70159046bc026b414219d7fd28559146cd8793799aaaa15fa7，与root验收值完全匹配；py_compile通过。未修改候选。

- 两个新增GPU kernel均有triton_dist.jit装饰器，四helper各定义一次。
- 新producer仅原_GA token-only支路完整c6 shape触发，consumer由_hybrid配对。
- Q[T,H]/SCALE_TOKEN[T]写一次，AH/WI/ACT_SCALE[T*k]以INV映射写入；consumer ORDER//k gather token Q，同时按compact行读AH/WI。
- ACT仍compact，返回ACT_SCALE[M]给DN，无token/ACT scale混用。
- 参数绑定16+2constexpr、21+5constexpr、6、14正确；原MDg的I constexpr保留。
- row mask、MMA、B descriptor、f16x2 tanh、满tile TMA/尾tile masked store及launch参数与合同一致。

未发现具体提交阻断问题。以上为静态审查；GPU编译、全链SQNR、确定性和性能未验证。等待唯一平台执行者协调，无新增提交。
