"""Single-card H800 custom benchmark: current one-hot scatter vs packed-key sort.

Not a P1 submission.  It only scores the per-chunk stable local-ranking scatter.
"""
import json
import torch
import triton
import triton.language as tl


E = 256
BLOCK = 64
N = 32768
C = 512
E_PAD = 256
SHIFT = 6
SENTINEL = 2147483647


@triton.jit
def current_scatter(IDS, BASE, ORDER, INV, N_CONST: tl.constexpr,
                    C_CONST: tl.constexpr, E_PAD: tl.constexpr,
                    BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_CONST
    ids = tl.load(IDS + offs, mask=mask, other=-1)
    e = tl.arange(0, E_PAD)
    eq = (ids[:, None] == e[None, :]).to(tl.int32)
    pre = tl.cumsum(eq, axis=0)
    base = tl.load(BASE + e * C_CONST + pid)
    pos = tl.sum(eq * (base[None, :] + pre - 1), axis=1)
    tl.store(ORDER + pos, offs.to(tl.int64), mask=mask)
    tl.store(INV + offs, pos.to(tl.int64), mask=mask)


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def packed_scatter(IDS, BASE, ORDER, INV, N_CONST: tl.constexpr,
                   C_CONST: tl.constexpr, E_PAD: tl.constexpr,
                   BLOCK: tl.constexpr, SHIFT: tl.constexpr,
                   SENT: tl.constexpr):
    pid = tl.program_id(axis=0)
    idx = tl.arange(0, BLOCK)
    offs = pid * BLOCK + idx
    mask = offs < N_CONST
    ids = tl.load(IDS + offs, mask=mask, other=0)
    key = (ids << SHIFT) + idx
    key = tl.where(mask, key, SENT)
    key = tl.sort(key, dim=0)
    expert = key >> SHIFT
    orig_lane = key & (BLOCK - 1)
    prev_idx = tl.maximum(idx - 1, 0)
    prev_expert = tl.gather(expert, prev_idx, 0)
    is_first = (idx == 0) | (expert > prev_expert)
    start_pos = tl.where(is_first, idx, 0)
    last_start = tl.associative_scan(start_pos, 0, _max_combine)
    rank = idx - last_start
    valid = key < SENT
    base = tl.load(BASE + expert * C_CONST + pid, mask=valid, other=0)
    pos = base + rank
    orig = pid * BLOCK + orig_lane
    tl.store(ORDER + pos, orig.to(tl.int64), mask=valid)
    tl.store(INV + orig, pos.to(tl.int64), mask=valid)


def run():
    dev = 'cuda'
    torch.manual_seed(0)
    flat = torch.randint(0, E, (N,), dtype=torch.int32, device=dev)
    # add skew: make first 4096 rows expert 0 and last 4096 rows expert E-1
    flat[:4096] = 0
    flat[-4096:] = E - 1
    base = torch.zeros((E_PAD * C,), dtype=torch.int32, device=dev)
    # Build BASE exactly as the current offsets kernel would for arbitrary counts.
    counts = torch.bincount(flat.long(), minlength=E)
    hist = torch.zeros((C, E_PAD), dtype=torch.int32, device=dev)
    for c in range(C):
        hist[c] = torch.bincount(flat[c * BLOCK:(c + 1) * BLOCK].long(), minlength=E_PAD)
    incl = hist.cumsum(0)
    per_expert = hist.sum(0)
    prefix = per_expert.cumsum(0) - per_expert
    base_2d = (incl - hist) + prefix[None, :]
    base.copy_(base_2d.t().contiguous().view(-1))

    order_a = torch.empty((N,), dtype=torch.int64, device=dev)
    inv_a = torch.empty((N,), dtype=torch.int64, device=dev)
    order_b = torch.empty((N,), dtype=torch.int64, device=dev)
    inv_b = torch.empty((N,), dtype=torch.int64, device=dev)

    def call_a():
        return current_scatter[(C,)](flat, base, order_a, inv_a, N_CONST=N,
                                     C_CONST=C, E_PAD=E_PAD, BLOCK=BLOCK,
                                     num_warps=8, num_stages=1)

    def call_b():
        return packed_scatter[(C,)](flat, base, order_b, inv_b, N_CONST=N,
                                    C_CONST=C, E_PAD=E_PAD, BLOCK=BLOCK,
                                    SHIFT=SHIFT, SENT=SENTINEL,
                                    num_warps=8, num_stages=1)

    ka = call_a()
    kb = call_b()
    ref = torch.argsort(flat.long(), stable=True)
    ok_order_a = bool(torch.equal(order_a, ref))
    ok_order_b = bool(torch.equal(order_b, ref))
    ok_inv_b = bool(torch.equal(inv_b[order_b], torch.arange(N, device=dev)))
    ok_inv_a = bool(torch.equal(inv_a[order_a], torch.arange(N, device=dev)))
    ms_a = float(triton.testing.do_bench(call_a, warmup=10, rep=50))
    ms_b = float(triton.testing.do_bench(call_b, warmup=10, rep=50))
    warp_times = {}
    warp_res = {}
    for nw in (1, 2, 4, 8, 16):
        def call_w(nw=nw):
            return packed_scatter[(C,)](flat, base, order_b, inv_b, N_CONST=N,
                                        C_CONST=C, E_PAD=E_PAD, BLOCK=BLOCK,
                                        SHIFT=SHIFT, SENT=SENTINEL,
                                        num_warps=nw, num_stages=1)
        k = call_w()
        warp_times[str(nw)] = float(triton.testing.do_bench(call_w, warmup=10, rep=50))
        warp_res[str(nw)] = {'n_regs': k.n_regs, 'n_spills': k.n_spills,
                             'shared': k.metadata.shared}
    report = {
        'kind': 'sort_packed_custom_bench',
        'shape': {'E': E, 'BLOCK': BLOCK, 'N': N, 'C': C},
        'math': {'current_order_exact': ok_order_a, 'packed_order_exact': ok_order_b,
                 'current_inv_exact': ok_inv_a, 'packed_inv_exact': ok_inv_b},
        'timing': {'current_ms': ms_a, 'packed_ms': ms_b,
                   'packed_over_current': ms_b / ms_a,
                   'packed_num_warps_ms': warp_times},
        'resources': {
            'current': {'n_regs': ka.n_regs, 'n_spills': ka.n_spills,
                        'shared': ka.metadata.shared},
            'packed': {'n_regs': kb.n_regs, 'n_spills': kb.n_spills,
                       'shared': kb.metadata.shared},
            'packed_num_warps': warp_res,
        },
    }
    print(json.dumps(report, separators=(',', ':')))


if __name__ == '__main__':
    run()
