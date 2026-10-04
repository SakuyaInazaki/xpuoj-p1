# 组合 layout 修正版

冻结路径：`p1_c356_unified_c4_layout.py`；SHA256 `df5c5f77eadccd9466719d32f8755c381c7686e1dad6d660789d8ba75cac1318`。

从组合79549派生，仅两个新增S2 padded ACT producer的outer `tl.range(..., flatten=True)`改回`tl.range(..., num_stages=2)`。理由是主agent报告S2b153681完整AC但c4 tk1.456ms相对父约0.787ms退化，删除已实测退化的flatten调度。

构建断言两helper AST精确等于原S2 ac338对应定义；除这两helper外所有函数AST与组合79549一致，包括run、其他8新增helper、阶段/数学/调用边界/布局合同。文本差异仅两行，pycompile通过。CPU合同完全复用；未GPU、未提交，组合在线fullAC与性能仍未验证，其他数值/编译风险见VALIDATION.md。原组合和生产均未改。
