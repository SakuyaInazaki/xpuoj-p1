"""CPU numeric screen for the epi_port md-epilogue port (c9/c10 and c1/c2).

Isolates the epilogue: acc_g / acc_u are produced once from a shared FP8
GEMM emulation, then three epilogues run on the SAME accumulators:
  ref   : float64 exact SwiGLU + float64 scaling (no fp8 store)
  base  : the frozen p1/kernel.py arithmetic for that kernel
  new   : the epi_port arithmetic for that kernel
Reported SQNR is measured in the dequantized act domain (q * s).
"""
import torch

torch.manual_seed(0)
FP8 = torch.float8_e4m3fn


def fp8(x):
    return x.to(FP8).to(torch.float32)


def rowquant(x):
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = amax / 448.0
    return fp8(x / s), s.squeeze(-1)


def colquant(w):                      # w [K, N] -> per-output-column scale
    amax = w.abs().amax(0, keepdim=True).clamp_min(1e-30)
    s = amax / 448.0
    return fp8(w / s), s.squeeze(0)


def sqnr(ref, x):
    ref = ref.double(); x = x.double()
    return 10 * torch.log10(ref.pow(2).sum() / (x - ref).pow(2).sum()).item()


def pow2_scale(act):
    """Reproduce the kernel's per-row 2^k scale, calibrated so max|act|/s ~ 32."""
    b = act.abs().amax(-1).clamp_min(1e-30) / 32.0
    bits = b.view(torch.int32)
    bexp = (bits >> 23) & 0xFF
    s = ((bexp) << 23).to(torch.int32).view(torch.float32)
    inv = ((254 - bexp) << 23).to(torch.int32).view(torch.float32)
    return s, inv                      # s * inv == 1 exactly (both 2^k)


def make_case(M, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(M, K, generator=g)
    Wg = torch.randn(K, N, generator=g) / K ** 0.5
    Wu = torch.randn(K, N, generator=g) / K ** 0.5
    aq, a_scale = rowquant(X)
    bgq, bsg = colquant(Wg)
    buq, bsu = colquant(Wu)
    acc_g = (aq.double() @ bgq.double()).float()
    acc_u = (aq.double() @ buq.double()).float()
    w = torch.rand(M, generator=g) * 0.9 + 0.05     # route weight in (0.05, 0.95)
    return acc_g, acc_u, a_scale, bsg, bsu, w


def ref_epi(acc_g, acc_u, a_scale, bsg, bsu, w):
    g = acc_g.double() * a_scale.double()[:, None] * bsg.double()[None, :]
    u = acc_u.double() * a_scale.double()[:, None] * bsu.double()[None, :]
    return (g * torch.sigmoid(g)) * u * w.double()[:, None]


# ---------------- c9/c10 : _fgs_tma1_kernel_gq ----------------
def gq_base(acc_g, acc_u, a_scale, bsg, bsu, w, inv):
    g_scale = a_scale[:, None] * bsg[None, :]
    u_scale = a_scale[:, None] * bsu[None, :]
    g = acc_g * g_scale
    u = acc_u * u_scale
    silu = g / (1.0 + torch.exp2(-g * 1.4426950408889634))
    act = silu * u * w[:, None]
    return fp8(act * inv[:, None])


def gq_new(acc_g, acc_u, a_scale, bsg, bsu, w, inv, mode=1):
    ah = a_scale * 0.5
    h = (acc_g * bsg[None, :]) * ah[:, None]
    u = acc_u * bsu[None, :]
    th = torch.tanh(h) if mode == 2 else torch.tanh(h.to(torch.float16).float()).to(torch.float16).float()
    silu = h * th + h
    wi = (w * a_scale) * inv
    return fp8(silu * u * wi[:, None])


# ---------------- c1/c2 : _fgs_tma2_int_pm_q8_kernel ----------------
def pm_base(acc_g, acc_u, a_scale, bsg, bsu, w, inv):
    g = acc_g * (a_scale[:, None] * bsg[None, :])
    u = acc_u * (a_scale[:, None] * bsu[None, :])
    th = torch.tanh(g * 0.5)                        # tanh.approx.f32
    silu = (g * 0.5) * (1.0 + th)
    act = silu * u * w[:, None]
    return fp8(act * inv[:, None])


def pm_new(acc_g, acc_u, a_scale, bsg, bsu, w, inv, mode=1):
    g = acc_g * (a_scale[:, None] * bsg[None, :])
    u = acc_u * (a_scale[:, None] * bsu[None, :])
    h = g * 0.5
    th = torch.tanh(h) if mode == 2 else torch.tanh(h.to(torch.float16).float()).to(torch.float16).float()
    silu = h * th + h
    act = silu * u * w[:, None]
    return fp8(act * inv[:, None])


CASES = [
    ("c9  E256 H4096 I2048", 4096, 128, 11),
    ("c10 E256 H4096 I1536", 4096, 128, 12),
    ("c1  E8   H4096 I8192", 4096, 128, 13),
    ("c2  E8   H4096 I14336", 4096, 128, 14),
]

print("%-24s %8s %9s %9s %9s %9s  %10s" % ("case", "base dB", "EPP1 dB", "d(EPP1)", "EPP2 dB", "d(EPP2)", "fp8 identical"))
for name, K, N, seed in CASES:
    acc_g, acc_u, a_scale, bsg, bsu, w = make_case(512, K, N, seed)
    ref = ref_epi(acc_g, acc_u, a_scale, bsg, bsu, w)
    s, inv = pow2_scale(ref.float())
    kern = gq_base if name.startswith(("c9", "c10")) else pm_base
    kern_new = gq_new if name.startswith(("c9", "c10")) else pm_new
    qb = kern(acc_g, acc_u, a_scale, bsg, bsu, w, inv)
    q1 = kern_new(acc_g, acc_u, a_scale, bsg, bsu, w, inv, 1)
    q2 = kern_new(acc_g, acc_u, a_scale, bsg, bsu, w, inv, 2)
    db = sqnr(ref, qb * s[:, None])
    d1 = sqnr(ref, q1 * s[:, None])
    d2 = sqnr(ref, q2 * s[:, None])
    same = (qb == q1).double().mean().item()
    print("%-24s %8.3f %9.3f %+9.4f %9.3f %+9.4f  %9.4f%%"
          % (name, db, d1, d1 - db, d2, d2 - db, same * 100))

# row-scale identity: the s/inv expressions are literally unchanged in both
# kernels, so this only re-states the code fact.
print("\nrow-scale s / inv expressions are byte-identical to the baseline in both kernels")
