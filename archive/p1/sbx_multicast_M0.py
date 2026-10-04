import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# M0 of the 2-CTA cluster / TMA-multicast ladder: the SIMPLEST pure-TMA GEMM (bf16), non-persistent,
# compiled with num_ctas=2. Both earlier num_ctas=2 crashes were on non-vanilla kernels (atomics epilogue on
# 08-24; pointer-load synthetic kernel on 09-02). If this compiles, the next rung is the c1/c2 pm kernel shape.
# With num_ctas=2 a program is a 2-CTA cluster: BM=256 -> each CTA owns 128 rows, B tile is multicast.

@triton.jit
def gemm_tma(A_DESC, B_DESC, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    off_m = pid_m * BM
    off_n = pid_n * BN
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, tl.cdiv(K, BK)):
        a = A_DESC.load([off_m, ki * BK])
        b = B_DESC.load([off_n, ki * BK])
        acc = tl.dot(a, b.T, acc)
    offs_m = off_m + tl.arange(0, BM)
    offs_n = off_n + tl.arange(0, BN)
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16))


def run(tag, M, N, K, BM, BN, BK, stages, ctas):
    a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    ad = TensorDescriptor(a, a.shape, a.stride(), [BM, BK])
    bd = TensorDescriptor(b, b.shape, b.stride(), [BN, BK])
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    fn = lambda: gemm_tma[grid](ad, bd, c, M, N, K, BM=BM, BN=BN, BK=BK,
                                num_warps=8, num_stages=stages, num_ctas=ctas)
    k = fn()
    ms = triton.testing.do_bench(fn, warmup=5, rep=20)
    print(tag, "%.4f ms" % ms, "%.1f TF" % (2.0 * M * N * K / ms / 1e9), "regs", k.n_regs, "spills", k.n_spills,
          "multicast_in_ptx=", ("multicast" in k.asm["ptx"]))
    return c


def main():
    print("device", torch.cuda.get_device_name())
    # c1-like: M=32768, N=16384 (=2I), K=4096. Baseline: 1 CTA, per-CTA tile 128x256.
    c1 = run("CTA1 BM128", 32768, 16384, 4096, 128, 256, 64, 3, 1)
    # cluster of 2: program tile 256x256 split over 2 CTAs (128 rows each), B multicast. Native crash => zero output.
    c2 = run("CTA2 BM256", 32768, 16384, 4096, 256, 256, 64, 3, 2)


main()
raise RuntimeError("M0 done (see prints above)")
