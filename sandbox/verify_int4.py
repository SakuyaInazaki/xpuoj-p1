"""
verify_int4.py — 离线 SQNR 模拟：权重 fp8(per-row) vs int4/int5/int6(K 方向分组)。

纯 CPU torch，不需要 GPU。忠实复刻 kernel_v723ab.py 里 c9/c10(E=256) 在
第 3~5 次调用走的那条链路：

  md : x 逐 token fp8(amax/448)  ×  gate_up 逐行 fp8   -> fp32 acc
       -> silu(g)*u*w (fp32)
       -> act 量化成 fp8，行尺度 = 2^(bexp+10-127)，bexp 取自
          bound = a_scale^2 * bnorm[e]^2 * |w| 的指数域（_fgs_tma1_kernel_gq）
  dn : act_fp8 × down 逐行 fp8 -> fp32 acc -> *b_scale
       -> epilogue 把 down 输出再量化成 fp8，尺度 = 每 (row,256列块) 一个
          (_dn_tma2_f8_kernel)
  fin: 反量化 -> 按 token 累加 k 个分支 (fp32) -> bf16 写 output

参考实现按题面：BF16 输入 / BF16 中间 / FP32 累加。

唯一被替换的环节是「权重量化器」，其余环节逐位不动，所以 SQNR 的差值就是
换量化器的净代价。
"""
import math
import sys

import torch

torch.manual_seed(20260904)

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def to_fp8(x):
    """fp32 -> e4m3 -> fp32，饱和到 ±448（与 cvt.rn.satfinite 一致）。"""
    return x.clamp(-FP8_MAX, FP8_MAX).to(FP8).float()


def to_bf16(x):
    return x.to(torch.bfloat16).float()


# ---------------------------------------------------------------- 权重量化器

def q_fp8_row(w):
    """现行路径：_quant_weight_fp8，逐行(K 方向全行)一个 fp32 尺度。"""
    amax = w.abs().amax(dim=-1, keepdim=True)
    scale = torch.maximum(amax / FP8_MAX, torch.full_like(amax, 1e-12))
    q = to_fp8(w / scale)
    return q * scale, 8.0


def q_int_group(w, bits=4, group=64, sym8=False, outliers=0, scale_dtype=torch.float16):
    """
    对称整型 + K 方向分组尺度。
      bits   : 4/5/6/8
      group  : 组大小（沿 K）
      sym8   : True  -> 尺度 amax/2^(b-1)，正侧最大码 2^(b-1)-1，顶端会截断
               False -> 尺度 amax/(2^(b-1)-1)，无截断
      outliers: 每组把 |w| 最大的 r 个单独存 fp8（异常值隔离），
                余下的用第 r+1 大定尺度
    返回 (w_hat, bits_per_weight)
    """
    K = w.shape[-1]
    assert K % group == 0
    wg = w.reshape(*w.shape[:-1], K // group, group)

    hi = (1 << (bits - 1)) - 1              # 7 / 15 / 31 / 127
    lo = -(1 << (bits - 1)) if sym8 else -hi
    denom = float(1 << (bits - 1)) if sym8 else float(hi)

    if outliers > 0:
        a = wg.abs()
        thr = a.topk(outliers + 1, dim=-1).values[..., -1:]   # 第 r+1 大
        out_mask = a > thr
        amax = thr
    else:
        out_mask = None
        amax = wg.abs().amax(dim=-1, keepdim=True)

    s = amax / denom
    s = torch.maximum(s, torch.full_like(s, 1e-20))
    s = s.to(scale_dtype).float()                # 尺度按存储精度落一次
    s = torch.maximum(s, torch.full_like(s, 1e-20))

    t = (wg / s).clamp(lo, hi)
    q = torch.floor(t + 0.5)                     # 主机端 (t+8.5).to(uint8) 的等价
    w_hat = q * s

    if out_mask is not None:
        # 异常值原样走 fp8（逐组一个 fp32/fp16 尺度即可，这里直接用 e4m3 相对精度）
        oa = wg.abs().amax(dim=-1, keepdim=True)
        os_ = torch.maximum(oa / FP8_MAX, torch.full_like(oa, 1e-12))
        w_out = to_fp8(wg / os_) * os_
        w_hat = torch.where(out_mask, w_out, w_hat)

    bpw = bits + (16.0 if scale_dtype == torch.float16 else 32.0) / group
    if outliers > 0:
        # 每个异常值：8 bit 值 + log2(group) bit 组内索引
        bpw += outliers * (8.0 + math.log2(group)) / group
    return w_hat.reshape(w.shape), bpw


# ---------------------------------------------------------------- 链路模拟

def silu(x):
    return x / (1.0 + torch.exp(-x))


def build(T, H, I, E, k, heavy=False, dev="cpu"):
    def gen(*shape):
        if heavy:
            # 重尾：t(4) 归一化到单位方差
            z = torch.randn(*shape)
            g = torch.distributions.Chi2(4.0).sample(shape)
            return (z / (g / 4.0).sqrt()) / math.sqrt(2.0)
        return torch.randn(*shape)

    x = gen(T, H) / math.sqrt(H) * 4.0
    Wg = gen(E, I, H) / math.sqrt(H)
    Wu = gen(E, I, H) / math.sqrt(H)
    Wd = gen(E, H, I) / math.sqrt(I)
    gate = torch.randn(E, H) / math.sqrt(H)
    return x, Wg, Wu, Wd, gate


def route(x, gate, k):
    logits = to_bf16(to_bf16(x) @ to_bf16(gate).t())
    probs = torch.softmax(logits, dim=-1)
    w, idx = torch.topk(probs, k, dim=-1)
    w = w / torch.maximum(w.sum(-1, keepdim=True), torch.full_like(w[..., :1], 1e-6))
    return w, idx


def ref_forward(x, Wg, Wu, Wd, rw, idx):
    T, H = x.shape
    k = idx.shape[1]
    out = torch.zeros(T, H, dtype=torch.float32)
    xb = to_bf16(x)
    for j in range(k):
        for e in range(Wg.shape[0]):
            m = idx[:, j] == e
            if not bool(m.any()):
                continue
            xs = xb[m]
            g = to_bf16(xs @ to_bf16(Wg[e]).t()).float()
            u = to_bf16(xs @ to_bf16(Wu[e]).t()).float()
            act = silu(g) * u * rw[m, j:j + 1]
            d = to_bf16(to_bf16(act) @ to_bf16(Wd[e]).t()).float()
            out[m] += d
    return to_bf16(out)


def sub_forward(x, Wg, Wu, Wd, rw, idx, wq, wq_dn=None):
    """wq / wq_dn: gate_up / down 的权重量化函数。其余环节固定。"""
    if wq_dn is None:
        wq_dn = wq
    T, H = x.shape
    E = Wg.shape[0]
    I = Wg.shape[1]
    k = idx.shape[1]

    # --- 权重（静态，缓存） ---
    GU = torch.cat([Wg, Wu], dim=1)                       # (E, 2I, H)
    GU_hat = wq(to_bf16(GU))
    WD_hat = wq_dn(to_bf16(Wd))                              # (E, H, I)
    # bnorm[e] = max_n ||b_{e,n}||_2（用反量化后的真实行）
    bnorm = GU_hat.reshape(E, 2 * I, H).pow(2).sum(-1).sqrt().amax(-1)   # (E,)

    # --- 激活逐 token fp8 ---
    xb = to_bf16(x)
    a_amax = xb.abs().amax(dim=1)
    a_s = torch.maximum(a_amax / FP8_MAX, torch.full_like(a_amax, 1e-12))
    xq = to_fp8(xb / a_s[:, None])                        # 存的是 fp8 码值

    out = torch.zeros(T, H, dtype=torch.float32)
    for j in range(k):
        for e in range(E):
            m = idx[:, j] == e
            if not bool(m.any()):
                continue
            aq = xq[m]
            asc = a_s[m]
            w = rw[m, j]

            bg = GU_hat[e, :I]
            bu = GU_hat[e, I:]
            # md：fp8×fp8 -> fp32 acc；B 的尺度已经乘回 GU_hat 里
            g = (aq @ bg.t()) * asc[:, None]
            u = (aq @ bu.t()) * asc[:, None]
            act = silu(g) * u * w[:, None]

            # --- 中间 act 的 fp8 量化（_fgs_tma1_kernel_gq 的先验上界尺度）---
            bound = (asc * asc * bnorm[e] * bnorm[e] * w.abs()).clamp_min(1e-30)
            bexp = (bound.view(torch.int32) >> 23) & 0xFF
            s_act = ((bexp + 10) << 23).view(torch.float32)
            inv = ((244 - bexp) << 23).view(torch.float32)
            actq = to_fp8(act * inv[:, None])

            # --- dn：fp8×fp8 -> fp32，再把输出量化成 fp8（每 256 列块一个尺度）---
            acc = actq @ WD_hat[e].t()                    # (m, H)
            nch = max(H // 256, 1)
            accv = acc.reshape(-1, nch, min(256, H))
            row_max = accv.abs().amax(dim=-1)             # (m, nch)
            sc = torch.maximum(s_act[:, None] * row_max / FP8_MAX,
                               torch.full_like(row_max, 1e-12))
            q = to_fp8(accv * (s_act[:, None] / sc)[..., None])
            dv = (q * sc[..., None]).reshape(-1, H)
            out[m] += dv
    return to_bf16(out)


def sqnr(ref, y):
    n = (ref - y).float().norm()
    return 20.0 * math.log10(float(ref.float().norm()) / max(float(n), 1e-12))


# ---------------------------------------------------------------- 主流程

def run(name, T, H, I, E, k, heavy=False, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    x, Wg, Wu, Wd, gate = build(T, H, I, E, k, heavy=heavy)
    rw, idx = route(x, gate, k)
    ref = ref_forward(x, Wg, Wu, Wd, rw, idx)

    modes = [
        ("fp8 per-row  (现行基线)",      lambda w: q_fp8_row(w)[0],                                    8.00),
        ("int4 g32  sym7",               lambda w: q_int_group(w, 4, 32)[0],                           4.50),
        ("int4 g64  sym7",               lambda w: q_int_group(w, 4, 64)[0],                           4.25),
        ("int4 g128 sym7",               lambda w: q_int_group(w, 4, 128)[0],                          4.125),
        ("int4 g64  sym8(截顶)",          lambda w: q_int_group(w, 4, 64, sym8=True)[0],                4.25),
        ("int4 g64  +1 异常值",           lambda w: q_int_group(w, 4, 64, outliers=1)[0],               4.47),
        ("int4 g64  +2 异常值",           lambda w: q_int_group(w, 4, 64, outliers=2)[0],               4.69),
        ("int4 g32  +2 异常值",           lambda w: q_int_group(w, 4, 32, outliers=2)[0],               5.31),
        ("int5 g64  sym7",               lambda w: q_int_group(w, 5, 64)[0],                           5.25),
        ("int6 g64  sym7",               lambda w: q_int_group(w, 6, 64)[0],                           6.25),
        ("int6 g128 sym7",               lambda w: q_int_group(w, 6, 128)[0],                          6.125),
        ("int6 g256 sym7",               lambda w: q_int_group(w, 6, 256)[0],                          6.0625),
        ("int8 g64  sym7",                lambda w: q_int_group(w, 8, 64)[0],                           8.25),
    ]
    split = [
        ("md=int4g64 / dn=fp8",  lambda w: q_int_group(w, 4, 64)[0], lambda w: q_fp8_row(w)[0]),
        ("md=fp8 / dn=int4g64",  lambda w: q_fp8_row(w)[0], lambda w: q_int_group(w, 4, 64)[0]),
        ("md=int6g128/dn=int6g128", lambda w: q_int_group(w, 6, 128)[0], lambda w: q_int_group(w, 6, 128)[0]),
    ]

    print(f"\n=== {name}  (T={T} H={H} I={I} E={E} k={k}"
          f"{' 重尾权重' if heavy else ''}) ===")
    print(f"{'权重量化模式':<26}{'bit/权重':>9}{'压缩':>7}{'SQNR(dB)':>10}"
          f"{'Δ vs fp8':>10}{'真机预估':>10}")
    base = None
    for label, fn, bpw in modes:
        y = sub_forward(x, Wg, Wu, Wd, rw, idx, fn)
        s = sqnr(ref, y)
        if base is None:
            base = s
        d = s - base
        proj = 23.12 + d
        flag = "  ✓" if proj >= 22.0 else "  ✗"
        print(f"{label:<26}{bpw:>9.3f}{8.0/bpw:>7.2f}x{s:>10.2f}{d:>10.2f}"
              f"{proj:>10.2f}{flag}")
    for label, fa, fb in split:
        y = sub_forward(x, Wg, Wu, Wd, rw, idx, fa, fb)
        s = sqnr(ref, y)
        d = s - base
        proj = 23.12 + d
        flag = "  ✓" if proj >= 22.0 else "  ✗"
        print(f"{label:<26}{'-':>9}{'-':>8}{s:>10.2f}{d:>10.2f}{proj:>10.2f}{flag}")
    return base


# ---------------------------------------------------------------- 字节布局自检
# 下面两个函数是 p1/kernel_v740_int4w.py 里 _quant_weight_int4/_int6 的逐行
# 复刻（那边 import triton，mac 上跑不起来，所以这里刻意重写一遍）。
# unpack_* 则是内核解包路径的 numpy 等价物，用来钉死字节布局约定。

def host_pack_int4(w, group=64):
    E, N, K = w.shape
    ng = K // group
    wf = w.float().reshape(E, N, ng, group)
    amax = wf.abs().amax(dim=3, keepdim=True)
    sc = torch.maximum(amax * (1.0 / 7.0), torch.full_like(amax, 1e-20))
    sc16 = sc.to(torch.float16)
    t = wf / sc16.float()
    t.clamp_(-7.0, 7.0)
    t += 8.5
    nib = t.to(torch.uint8).reshape(E, N, K)
    q = nib[:, :, 0::2] + nib[:, :, 1::2] * 16
    return q, sc16.reshape(E, N, ng)


def unpack_int4(q, s, group=64):
    """内核侧：lo=(x&15), hi=(x>>4), tl.join(lo,hi).reshape -> 连续 K。"""
    E, N, Kh = q.shape
    K = Kh * 2
    lo = (q & 0x0F).to(torch.int16)
    hi = (q >> 4).to(torch.int16)
    code = torch.stack([lo, hi], dim=-1).reshape(E, N, K)   # == tl.join(...).reshape
    v = (code.float() - 8.0).reshape(E, N, K // group, group)
    return (v * s.float().unsqueeze(-1)).reshape(E, N, K)


def host_pack_int6(w, group=64):
    E, N, K = w.shape
    ng = K // group
    wf = w.float().reshape(E, N, ng, group)
    amax = wf.abs().amax(dim=3, keepdim=True)
    sc = torch.maximum(amax * (1.0 / 31.0), torch.full_like(amax, 1e-20))
    sc16 = sc.to(torch.float16)
    t = wf / sc16.float()
    t.clamp_(-31.0, 31.0)
    t += 32.5
    code = t.to(torch.uint8).reshape(E, N, K)
    hi = (t * 0.25).to(torch.uint8).reshape(E, N, K)
    lo = code - hi * 4
    q4 = hi[:, :, 0::2] + hi[:, :, 1::2] * 16
    q2 = (lo[:, :, 0::4] + lo[:, :, 1::4] * 4
          + lo[:, :, 2::4] * 16 + lo[:, :, 3::4] * 64)
    return q4, q2, sc16.reshape(E, N, ng)


def unpack_int6(q4, q2, s, group=64):
    E, N, Kh = q4.shape
    K = Kh * 2
    h_lo = (q4 & 0x0F).to(torch.int16)
    h_hi = (q4 >> 4).to(torch.int16)
    hi = torch.stack([h_lo, h_hi], dim=-1).reshape(E, N, K)
    l0 = (q2 & 0x03).to(torch.int16)
    l1 = ((q2 >> 2) & 0x03).to(torch.int16)
    l2 = ((q2 >> 4) & 0x03).to(torch.int16)
    l3 = ((q2 >> 6) & 0x03).to(torch.int16)
    lo = torch.stack([l0, l1, l2, l3], dim=-1).reshape(E, N, K)
    code = hi * 4 + lo
    v = (code.float() - 32.0).reshape(E, N, K // group, group)
    return (v * s.float().unsqueeze(-1)).reshape(E, N, K)


def roundtrip():
    print("\n=== 字节布局自检（主机打包 <-> 内核解包） ===")
    torch.manual_seed(3)
    w = to_bf16(torch.randn(3, 9, 1536))          # K=1536：c10 的 dn 归约维
    for group in (32, 64, 128):
        q, s = host_pack_int4(w, group)
        got = unpack_int4(q, s, group)
        want = q_int_group(w.float(), 4, group)[0]
        e = (got - want).abs().max().item()
        assert q.dtype == torch.uint8 and q.shape == (3, 9, 768)
        assert int(q.max()) <= 255 and int(q.min()) >= 0
        print(f"  int4 g{group:<4} 打包字节 {q.numel():>7}  尺度 {s.numel():>6} fp16"
              f"   与参考量化器最大偏差 {e:.3e}   {'OK' if e == 0 else 'MISMATCH'}")
        assert e == 0
        # K 相邻性：字节 j 的低/高半字节必须是 K 的 2j / 2j+1
        assert bool((((q[:, :, 0] & 0x0F).to(torch.int16) - 8).float()
                     * s[:, :, 0].float() == got[:, :, 0]).all())
        assert bool((((q[:, :, 0] >> 4).to(torch.int16) - 8).float()
                     * s[:, :, 0].float() == got[:, :, 1]).all())
    for group in (64, 128):
        q4, q2, s = host_pack_int6(w, group)
        got = unpack_int6(q4, q2, s, group)
        want = q_int_group(w.float(), 6, group)[0]
        e = (got - want).abs().max().item()
        print(f"  int6 g{group:<4} 平面 {q4.numel():>7}+{q2.numel():>6}  尺度 {s.numel():>6} fp16"
              f"   与参考量化器最大偏差 {e:.3e}   {'OK' if e == 0 else 'MISMATCH'}")
        assert e == 0
    print("  字节布局与舍入约定：主机端 (t+0.5).to(uint8) 与内核解包严格互逆 ✓")


if __name__ == "__main__":
    heavy = "--heavy" in sys.argv
    roundtrip()
    run("c9  (4096,4096,256,2048,8) 缩尺", T=128, H=4096, I=2048, E=2, k=2)
    run("c10 (4096,4096,256,1536,8) 缩尺", T=128, H=4096, I=1536, E=2, k=2)
    if heavy:
        run("c9 重尾权重 t(4) 敏感性", T=128, H=4096, I=2048, E=2, k=2, heavy=True)
        run("c9 换种子复现性", T=128, H=4096, I=2048, E=2, k=2, seed=7)
