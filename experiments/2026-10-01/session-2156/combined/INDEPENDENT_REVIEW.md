# 组合独立快速验收

结论：未发现提交阻断；未做GPU验证，采用仍需完整12AC/双SQNR/确定性。

独立实算SHA `79549c5bc0151c940d72ee776cdc571f78edf4ab9740fd0a3f6d0d1fb870bd5e` 一致，compile通过。AST对C356U仅`_run_replicated`改变；新增六helper逐一等于S2b父定义（包含装饰器与launch阶段参数）。

c4完整shape的I=1024与hybrid c3/5/6集合互斥。6484行`act_is_padded=False`每call初始化；仅6620/6635附近实际padded producer赋True，显式logical_m=T*k。6599附近hybrid consumer优先，c3/5/6仍compact ACT/scale；c4两producer分别匹配tokenQ及sortedQ，冷路径/其他producer标记保持False。6764附近static DN仅按实际producer标记选择padded ACT读址，原Down/scale/fin保持。

复用两父已有CPU地址合同；没有重复批测或更改候选。在线须分别看c3/5/6数值合同与c4padded/flatten完整AC，不能因局部收益或异常计时提前采用。
