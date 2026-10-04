import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# L0 of the flatten bisection ladder: dense persistent TMA GEMM (bf16), the same loop shape that
# triton_dist/kernels/nvidia/gemm_reduce_scatter.py compiles WITH flatten=True on this toolchain.
# Sandbox rules: no sys/time/os imports, no module-level non-literal assignment, no fp8 dtype,
# stdout comes back only if the script ends with an exception -> we raise at the end. Keep prints short.

@triton.jit
def gemm_ref(A_DESC, B_DESC, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             GM: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BK)
    for tile_id in range(start_pid, num_tiles, NUM_SMS):
        pid_m, pid_n = tl.swizzle2d(tile_id // num_pid_n, tile_id % num_pid_n, num_pid_m, num_pid_n, GM)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = A_DESC.load([off_m, ki * BK])
            b = B_DESC.load([off_n, ki * BK])
            acc = tl.dot(a, b.T, acc)
        offs_m = off_m + tl.arange(0, BM)
        offs_n = off_n + tl.arange(0, BN)
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


@triton.jit
def gemm_flat(A_DESC, B_DESC, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              GM: tl.constexpr, NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BK)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = tl.swizzle2d(tile_id // num_pid_n, tile_id % num_pid_n, num_pid_m, num_pid_n, GM)
        off_m = pid_m * BM
        off_n = pid_n * BN
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = A_DESC.load([off_m, ki * BK])
            b = B_DESC.load([off_n, ki * BK])
            acc = tl.dot(a, b.T, acc)
        offs_m = off_m + tl.arange(0, BM)
        offs_n = off_n + tl.arange(0, BN)
        tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def run(kern, tag, M, N, K, BM, BN, BK, stages):
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    ad = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
    bd = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
    fn = lambda: kern[(132,)](ad, bd, c, M, N, K, BM=BM, BN=BN, BK=BK, GM=8, NUM_SMS=132,
                              num_warps=8, num_stages=stages)
    k = fn()
    ttg = k.asm["ttgir"]
    ms = triton.testing.do_bench(fn, warmup=5, rep=20)
    tf = 2.0 * M * N * K / ms / 1e9
    print(tag, "M/N/K", M, N, K, "%.4f ms" % ms, "%.1f TF" % tf, "regs", k.n_regs, "spills", k.n_spills,
          "scf.for=", ttg.count("scf.for"))
    return c


def main():
    print("device", torch.cuda.get_device_name())
    # short-K md-like shape (c11: M=T*k=131072, N=2I=2048, K=H=1024); bf16 BK=64 == fp8 BK=128 in bytes
    c0 = run(gemm_ref, "REF  shortK", 131072, 2048, 1024, 128, 256, 64, 3)
    # a fused loop nest shows as ONE scf.for in ttgir (ref shows 2). Native crash here => zero output => L0 dead.
    c1 = run(gemm_flat, "FLAT shortK", 131072, 2048, 1024, 128, 256, 64, 3)
    c2 = run(gemm_ref, "REF  c1like", 32768, 16384, 4096, 128, 256, 64, 3)
    c3 = run(gemm_flat, "FLAT c1like", 32768, 16384, 4096, 128, 256, 64, 3)
    # same seed => same inputs; fused loop must be bitwise identical to the reference


main()
raise RuntimeError("L0 done (see prints above)")
