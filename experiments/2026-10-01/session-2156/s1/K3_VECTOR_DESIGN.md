# S1后续k3向量设计（未实现）

c3/c5的k4/k8可直接用二次幂vector，但完整shape分发与对应consumer实际path需独立核验；c7 k3不能用tl.arange(0,3)。若c6有正常信号后扩c7，新增K_BRANCH_PAD=triton.next_power_of_2(k)=4 constexpr，j=tl.arange(0,K_BRANCH_PAD)，branch_mask=j<K_BRANCH。

INV/FLAT_IDS/FLAT_W均使用branch_mask，other=0；BNORM同样branch_mask，other可取1。dest虚拟lane的0不许写入：AH/WI/ACT_SCALE scatter全部mask=branch_mask。token Q与SCALE_TOKEN写入仍在vector之外一次。bound/WI结合顺序、max floors/bit转换均保持；无效lane只做局部无效数学且不产生内存读写副作用。k3每token前三真实lane对应唯一INV写者，第4lane全程不得load实际branch或scatter到row0。

consumer ORDER//KTOP仍用真实k=3，不能用K_BRANCH_PAD；M=T*k、SCL长度T*k、metadata和DN/fin均用真实k。新增mask不改变真实lane算式。CPU需测试真实c7 T16384/E96/k3，确认全49152compact行恰一写者、row0未被虚拟lane覆盖、向量/标量FP32结果逐位同值，再核所有调用stage/量化/tanh未变。GPU编译及SQNR仍须OJ。先等R151/S1v2结果，不生成大范围候选。
