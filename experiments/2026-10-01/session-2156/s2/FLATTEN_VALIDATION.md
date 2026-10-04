# S2b备用冻结（不自动入队）

候选p1_s2_c4_flatten.py，SHA256 `c82b6ee4286c9583851c33c0f715102a0946eaa0247e277060d9453ad3218c4b`；父ac338 layout-only未改。仅两个新增padded ACT producer外层tl.range的keyword列表由num_stages=2变flatten=True，无range-level stages。

build_check_flatten.py通过compile、六调用参数绑定、GPU decorator和AST逆变。将两range keyword列表恢复后，整个文件AST与父完全一致（包括重复定义/顶层语句）。MDg host stage3，pre_nf stage4/maxnreg232，DN stage3；布局、math、inner K loops、DROP/SCL/CSCL/fin、全部阶段分派不变。CPU布局合同复用父9histogram/4157非重叠tile/524288有效行；地址数学未改，没有重复批测。

无GPU/OJ，在线编译、全AC/SQNR/确定性、异步写回/descriptor生命周期、资源与真实收益未知。flatten不保证编译器产生跨tile重叠，也可能增活跃寄存器或破坏原staging；保留DROP/SCL不保证排除编译失败。不自动入队，只有主agent按正式layout-only结果与预算明确选择才交唯一平台执行者。

证据flatten.diff与flatten_checks.json；没有修改父/生产/其他候选或开展stage/warp调参。
