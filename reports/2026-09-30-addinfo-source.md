
ganzhengyu
发表于 
2 周前

@ceerrep 您好，想确认第一题 MegaMoE（Triton-distributed）的通信接口支持范围。提交 141063 在运行前报错：

Import validation failed: Import 'nvshmem.core' is not allowed

请问目前允许使用哪些初始化、对称内存分配/释放和 stream barrier 接口？例如 triton_dist.utils 中的 init_nvshmem_by_torch_process_group、nvshmem_create_tensor、nvshmem_free_tensor_sync、nvshmem_barrier_all_on_stream 是否允许？评测器是否已提前初始化 NVSHMEM？能否提供允许的 import 清单及最小通信示例，以便在规则范围内实现自定义通信？谢谢！


ganzhengyu
发表于 
2 周前

@ceerrep 您好，补充一个第三题 Triton 3.6.0 的确定性现象，想请教排查方向：融合 BF16 投影、FP32 RMS 与系数计算的 64 行分块实现，默认流水线配置提交 141662、num_stages=2 提交 141666，均有多个测试点单次精度通过但 DETERMINISM FAIL；仅将投影的 num_stages 改为 1 的提交 141664 则全部通过。实现没有浮点原子操作；相同候选在 RTX 3090 的独立参考和重复输出检查也通过，Hopper 编译路径使用 WGMMA。目前尚不能确定是实现同步问题还是编译器问题。请问该镜像是否有已知的 WGMMA/流水线相关问题，或推荐的同步、调试方式？若需要最小复现，我可以进一步整理。谢谢！


Neptune
发表于 
2 周前

@ganzhengyu:

@ceerrep 您好，想确认第一题 MegaMoE（Triton-distributed）的通信接口支持范围。提交 141063 在运行前报错：

Import validation failed: Import 'nvshmem.core' is not allowed

请问目前允许使用哪些初始化、对称内存分配/释放和 stream barrier 接口？例如 triton_dist.utils 中的 init_nvshmem_by_torch_process_group、nvshmem_create_tensor、nvshmem_free_tensor_sync、nvshmem_barrier_all_on_stream 是否允许？评测器是否已提前初始化 NVSHMEM？能否提供允许的 import 清单及最小通信示例，以便在规则范围内实现自定义通信？谢谢！

你好，请通过 triton_dist.utils 使用相关功能。以下接口已在当前线上四卡评测环境中验证可用：

from triton_dist.utils import (
    is_shmem_initialized,
    nvshmem_create_tensor,
    nvshmem_create_tensors,
    nvshmem_free_tensor_sync,
    nvshmem_barrier_all_on_stream,
)
评测器在运行提交代码前已经初始化多卡通信和 NVSHMEM。请勿再次调用：

init_nvshmem_by_torch_process_group(...)
initialize_distributed()
finalize_distributed()
当前允许直接导入的主要模块为：

torch
math
triton
triton_dist
最小通信示例

import torch
import torch.distributed as dist

from triton_dist.utils import (
    is_shmem_initialized,
    nvshmem_barrier_all_on_stream,
    nvshmem_create_tensors,
    nvshmem_free_tensor_sync,
)


def communication_example():
    assert dist.is_initialized()
    assert is_shmem_initialized()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    stream = torch.cuda.current_stream()

    # 所有 GPU 必须使用相同的 shape、dtype 和调用顺序。
    peer_buffers = nvshmem_create_tensors(
        (1,), torch.int32, rank, world_size
    )
    local_buffer = peer_buffers[rank]

    # 每张 GPU 写入自己的编号。
    local_buffer.fill_(rank)
    nvshmem_barrier_all_on_stream(stream)
    stream.synchronize()

    # 读取下一张 GPU 写入的值。
    peer_rank = (rank + 1) % world_size
    value = int(peer_buffers[peer_rank][0].item())
    assert value == peer_rank

    # 全部读取完成后再释放。
    nvshmem_barrier_all_on_stream(stream)
    stream.synchronize()
    nvshmem_free_tensor_sync(local_buffer)
请确保所有 GPU 以相同顺序进行分配、同步和释放，否则可能导致程序卡住。正式实现中建议缓存通信缓冲区，避免在每次 run_kernel 调用时重复分配大块内存。


ceerRep
发表于 
2 周前
已编辑
讨论发起者

@ganzhengyu:

@ceerrep 您好，补充一个第三题 Triton 3.6.0 的确定性现象，想请教排查方向：融合 BF16 投影、FP32 RMS 与系数计算的 64 行分块实现，默认流水线配置提交 141662、num_stages=2 提交 141666，均有多个测试点单次精度通过但 DETERMINISM FAIL；仅将投影的 num_stages 改为 1 的提交 141664 则全部通过。实现没有浮点原子操作；相同候选在 RTX 3090 的独立参考和重复输出检查也通过，Hopper 编译路径使用 WGMMA。目前尚不能确定是实现同步问题还是编译器问题。请问该镜像是否有已知的 WGMMA/流水线相关问题，或推荐的同步、调试方式？若需要最小复现，我可以进一步整理。谢谢！

您好，感谢补充提交编号和复现条件。我们已在 H20、Triton 3.6.0 环境中对该实现进行了复现和分析。

已验证现象
该问题可稳定复现，经过简单检查，当前推荐优先使用：

num_stages=1
这是最直接且已经覆盖正式测试形状验证的方案。

如需保留多级流水，可让 RMS 使用独立且不会被编译器合并的 load：

x = tl.load(
    X + m[:, None] * D + kk[None, :],
    (m[:, None] < T) & (kk[None, :] < D),
    0,
)
w = tl.load(...)

acc = tl.dot(x, w, acc)

x_sq = tl.load(
    X + m[:, None] * D + kk[None, :],
    (m[:, None] < T) & (kk[None, :] < D),
    0,
    volatile=True,
)
xf = x_sq.to(tl.float32)
sq += tl.sum(xf * xf, axis=1)
另一种方案是将投影和 RMS 拆成两个 K 循环：

for start in range(...):
    x = tl.load(...)
    w = tl.load(...)
    acc = tl.dot(x, w, acc)

for start in range(...):
    x = tl.load(...)
    xf = x.to(tl.float32)
    sq += tl.sum(xf * xf, axis=1)
该方案在 stage 2 和 stage 3 下均可恢复确定性，但会额外读取一遍输入。

具体原因还在排查。


崎小咪
发表于 
1 周前

您好，我想询问针对评测的几个问题：每道题评分最高就是100分吗？

针对第一题我还想问两个问题：第一，P1 线上环境用的是哪个 Triton / triton-dist 版本？后续是否升级？当前是否支持这些能力：多个 CTA 绑定成一组协同执行（如 num_ctas=2 / CTA cluster）、TMA 数据多播、Gluon 的低层 WGMMA 矩阵乘接口、自动 warp 分工（warp_specialize）、异步任务（async_task）？第二，为什么某个测试点第一次运行 run_kernel、现场编译 Triton 算子时，经常 500 秒超时？平台给首次编译多少时间？超时如何分类？能否提供对应的编译器或 PTXAS 日志？


ceerRep
发表于 
1 周前
讨论发起者

@salt-Star:

您好，我想询问针对评测的几个问题：每道题评分最高就是100分吗？

针对第一题我还想问两个问题：第一，P1 线上环境用的是哪个 Triton / triton-dist 版本？后续是否升级？当前是否支持这些能力：多个 CTA 绑定成一组协同执行（如 num_ctas=2 / CTA cluster）、TMA 数据多播、Gluon 的低层 WGMMA 矩阵乘接口、自动 warp 分工（warp_specialize）、异步任务（async_task）？第二，为什么某个测试点第一次运行 run_kernel、现场编译 Triton 算子时，经常 500 秒超时？平台给首次编译多少时间？超时如何分类？能否提供对应的编译器或 PTXAS 日志？

不一定，100分只是我们给了个上界时间，低于这个时间计分方式将转为对数计分：每高10分说明吞吐快一倍
triton-dist 3.4，对应的triton版本也是 3.4
第一题为了节约torchrun初始化时间，是500秒一口气跑完十个测试点。因此如果500秒没能完成就会超时。
👍
1

ganzhengyu
发表于 
6 天前

@Neptune @ceerrep 谢谢回复87的接口说明。第一题新提交146954已去掉nvshmem.core/bindings及所有初始化调用，只使用triton_dist.utils分配/同步缓冲。现在通过导入检查，但sample在triton_dist/jit.py的make_cubin阶段失败：ptxas --gpu-name=sm_90a 返回255，结果中只有CalledProcessError，没有PTXAS的具体stderr。该版本的派发kernel使用triton_dist.language.symm_at定位对端缓冲。能否提供本次失败的PTXAS诊断，或让提交详情显示编译stderr，以区分代码错误与工具链/链接问题？我也会按您给出的nvshmem_create_tensors示例尝试主机端peer指针映射。谢谢！


ganzhengyu
发表于 
6 天前

@ceerrep 您好，想确认第二题静态参数缓存的失效约定。题面允许预热后复用k/v/sink及切片metadata；公开Tensor.tolist()已在自定义测试验证可用，但Tensor._version被沙箱明确禁止，我们没有绕过该限制。目前实现每次重新读取metadata，能够处理原地修改，但有同步开销。请问跨测试点是否保证静态Tensor对象或存储会更换？按静态Tensor对象身份及全部标量参数缓存派生metadata，是否符合评测约定？若可能在相同Tensor对象上原地更新，是否有允许使用的版本标识或测试边界信号来正确失效缓存？谢谢！


ceerRep
发表于 
4 天前
讨论发起者

@ganzhengyu:

@Neptune @ceerrep 谢谢回复87的接口说明。第一题新提交146954已去掉nvshmem.core/bindings及所有初始化调用，只使用triton_dist.utils分配/同步缓冲。现在通过导入检查，但sample在triton_dist/jit.py的make_cubin阶段失败：ptxas --gpu-name=sm_90a 返回255，结果中只有CalledProcessError，没有PTXAS的具体stderr。该版本的派发kernel使用triton_dist.language.symm_at定位对端缓冲。能否提供本次失败的PTXAS诊断，或让提交详情显示编译stderr，以区分代码错误与工具链/链接问题？我也会按您给出的nvshmem_create_tensors示例尝试主机端peer指针映射。谢谢！

我下午复现一下给你日志。然后相关ptxas诊断也可以用 triton.compile 在本地测试（不需要对应显卡）

相关错误信息的展示会尽快修复


ganzhengyu
发表于 
4 天前

@ceerrep @Neptune 您好，想确认线上评测的具体硬件，便于根据实际设备估算算力、显存带宽和通信代价。题面写第一题为单机4张H800，第二、三题为单张H800；请问三题使用的H800具体型号和显存容量是什么（例如SXM、PCIe或NVL）？第一题四卡之间是NVLink/NVSwitch互联还是PCIe，是否方便说明简化的卡间拓扑？若评测设置了固定功耗或频率上限，也烦请说明。谢谢！


ceerRep
发表于 
4 天前
讨论发起者

@ganzhengyu:

@ceerrep @Neptune 您好，想确认线上评测的具体硬件，便于根据实际设备估算算力、显存带宽和通信代价。题面写第一题为单机4张H800，第二、三题为单张H800；请问三题使用的H800具体型号和显存容量是什么（例如SXM、PCIe或NVL）？第一题四卡之间是NVLink/NVSwitch互联还是PCIe，是否方便说明简化的卡间拓扑？若评测设置了固定功耗或频率上限，也烦请说明。谢谢！

H800 80G SXM，没有设置功耗上限，默认700W

        GPU0    GPU1    GPU2    GPU3    
GPU0     X      NV8     NV8     NV8   
GPU1    NV8      X      NV8     NV8  
GPU2    NV8     NV8      X      NV8    
GPU3    NV8     NV8     NV8      X     

ceerRep
发表于 
3 天前
已编辑
讨论发起者

@ganzhengyu:

@Neptune @ceerrep 谢谢回复87的接口说明。第一题新提交146954已去掉nvshmem.core/bindings及所有初始化调用，只使用triton_dist.utils分配/同步缓冲。现在通过导入检查，但sample在triton_dist/jit.py的make_cubin阶段失败：ptxas --gpu-name=sm_90a 返回255，结果中只有CalledProcessError，没有PTXAS的具体stderr。该版本的派发kernel使用triton_dist.language.symm_at定位对端缓冲。能否提供本次失败的PTXAS诊断，或让提交详情显示编译stderr，以区分代码错误与工具链/链接问题？我也会按您给出的nvshmem_create_tensors示例尝试主机端peer指针映射。谢谢！

检查了一下，在 triton.jit 里用 dist_triton 是不支持的，请用 triton_dist.jit 试试


ganzhengyu
发表于 
2 天前

@Neptune @ceerrep 您好，我们正在按完整算子耗时优化四卡 MegaMoE 和单卡 mHC，但缺少 H800 上各内核或通信阶段的时间分解。当前公开自定义测试只报告完整 tk；尝试在提交中用 torch.cuda.Event 计时会被沙箱拒绝，前端也没有开放 profile 入口。请问赛事方能否在自定义测试或提交详情中提供合法的逐内核耗时、HBM/L2 流量或 NCCL 阶段耗时？如果暂时不能，是否有已允许的阶段计时/性能诊断方式或可公开的官方参考数据？这会帮助我们在每题 200 次软上限内只提交有整链收益证据的结构候选。谢谢！


崎小咪
发表于 
2 天前

提交罚分最高罚10分感觉好残忍🥺放开手优化到最后被罚分罚下来有点搞心态

ganzhengyu
发表于 
1 天前

@Neptune @ceerrep 您好，第三题 mHC 题面允许 CUDA 提交（cuda-h800），但目前平台公布的 customTestModes 中没有 cuda-h800；同题 triton-h800、tilelang-h800 则有 Samples/GeneratedWorkload。请问第三题 CUDA 提交是否已经可以使用自定义测试？若尚未开放，赛前是否会启用 cuda-h800 的 Samples/GeneratedWorkload 入口？我们希望先用不计正式提交次数的自定义测试验证完整算子的正确性。谢谢！


ganzhengyu
发表于 
1 天前

@Neptune @ceerrep 您好，第三题的 CUDA 题面只给出不带 cudaStream_t 参数的 C++ run_kernel 接口。我们发现若实现固定向 stream 0 启动 kernel，在 PyTorch 的非默认 stream CUDA Graph capture 中会产生空图，重放不能写出结果。请问 cuda-h800 评测是否在非默认 stream 或 CUDA Graph 下调用 run_kernel？如果会，官方推荐怎样在这个裸指针 CUDA 接口中取得评测器当前 stream（例如是否提供 PyTorch C10 头文件与链接）？这关系到正确性，感谢说明！


ceerRep
发表于 
1 天前
讨论发起者

@ganzhengyu:

@Neptune @ceerrep 您好，第三题的 CUDA 题面只给出不带 cudaStream_t 参数的 C++ run_kernel 接口。我们发现若实现固定向 stream 0 启动 kernel，在 PyTorch 的非默认 stream CUDA Graph capture 中会产生空图，重放不能写出结果。请问 cuda-h800 评测是否在非默认 stream 或 CUDA Graph 下调用 run_kernel？如果会，官方推荐怎样在这个裸指针 CUDA 接口中取得评测器当前 stream（例如是否提供 PyTorch C10 头文件与链接）？这关系到正确性，感谢说明！

CUDA Custom Test请使用 CUDA 多文件
选手代码没有自行指定的话不会有其他 CUDA stream，也不会有 CUDA Graph
