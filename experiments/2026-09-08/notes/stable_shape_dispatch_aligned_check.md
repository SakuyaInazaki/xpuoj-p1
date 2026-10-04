# stable shape dispatch aligned check

- 候选：`../candidates/stable_shape_dispatch_aligned.py`
- SHA-256：`1a832e87435d1ce266ce5e8f6d384fdd95a88161c8a1bc2fedf4d815692492af`
- AST（忽略位置、空白与注释）与 `stable_shape_dispatch.py` SHA `673409…` 完全相同。
- 78 个既有 `@jit` 函数的名称、源码文本、decorator/`def` 起始行和结束行均与 `p1/kernel.py` SHA `dd46…` 相同。
- `python3 -m py_compile` 通过；未做 GPU 或平台测试。保持 JIT 布局不表示已经定位或修复历史 TLE。
- 该候选没有修改数学、tile、通信或缓存键；shape-only 权重缓存缺口仍未解决。
