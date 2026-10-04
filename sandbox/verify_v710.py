"""Bit-exactness simulation for the three kernel_v710 changes (host-side, numpy only).

Reproduces the *integer* and *float* semantics of the Triton kernels involved and
checks the v692 formulation against the v710 formulation element-by-element.

  change 1  : drop _csort_offsets_kernel, let the scatter kernel rebuild the
              cross-expert exclusive offset from TOT
  change 2b : walk the expert axis of the hist/scatter kernels in ECHUNK slabs
  change 3a : keep the top-k picks in registers and do two masked block stores
              instead of 3*K_TOP strided global ops

Run: python3 sandbox/verify_v710.py
"""
import numpy as np

I32 = np.int32


# --------------------------------------------------------------------------- #
# helpers that mirror the kernels                                              #
# --------------------------------------------------------------------------- #
def hist_full(ids_pad, C, BLOCK, E_PAD):
    """_sort_hist_kernel, v692 single-shot form."""
    h = np.zeros((C, E_PAD), dtype=I32)
    for pid in range(C):
        row = ids_pad[pid * BLOCK:(pid + 1) * BLOCK]
        e = np.arange(E_PAD, dtype=I32)
        eq = (row[:, None] == e[None, :]).astype(I32)
        h[pid, :] = eq.sum(axis=0, dtype=I32)
    return h


def hist_chunked(ids_pad, C, BLOCK, E_PAD, ECHUNK):
    """_sort_hist_kernel, v710 slab form."""
    h = np.zeros((C, E_PAD), dtype=I32)
    for pid in range(C):
        row = ids_pad[pid * BLOCK:(pid + 1) * BLOCK]
        for e0 in range(0, E_PAD, ECHUNK):
            e = (e0 + np.arange(ECHUNK)).astype(I32)
            eq = (row[:, None] == e[None, :]).astype(I32)
            h[pid, e] = eq.sum(axis=0, dtype=I32)
    return h


def colscan(h, C, E_PAD, C_PAD):
    """_csort_colscan_kernel: BASE[e,c] = within-expert prefix over c, TOT[e]."""
    base = np.zeros((E_PAD, C), dtype=I32)
    tot = np.zeros(E_PAD, dtype=I32)
    for pid in range(E_PAD):
        run = I32(0)
        for j in range(0, C_PAD, 128):
            offs = j + np.arange(128)
            m = offs < C
            v = np.where(m, h[np.clip(offs, 0, C - 1), pid], 0).astype(I32)
            inc = np.cumsum(v, dtype=I32)
            base[pid, offs[m]] = (run + inc - v)[m]
            run = I32(run + v.sum(dtype=I32))
        tot[pid] = run
    return base, tot


def offsets_v692(base, tot, E):
    """_csort_offsets_kernel: fold the cross-expert exclusive offset into BASE."""
    E_PAD = tot.shape[0]
    b = base.copy()
    for pid in range(E_PAD):
        my_exc = I32(tot[:pid].sum(dtype=I32))
        b[pid, :] = b[pid, :] + my_exc
    counts = tot[:E].copy()
    return b, counts


def scatter_v692(ids_pad, base_full, N, C, BLOCK, E_PAD):
    """_sort_scatter_kernel: BASE already carries the cross-expert offset."""
    order = np.full(N, -1, dtype=np.int64)
    inv = np.full(N, -1, dtype=np.int64)
    for pid in range(C):
        offs = pid * BLOCK + np.arange(BLOCK)
        mask = offs < N
        ids = ids_pad[pid * BLOCK:(pid + 1) * BLOCK]
        e = np.arange(E_PAD, dtype=I32)
        eq = (ids[:, None] == e[None, :]).astype(I32)
        pre = np.cumsum(eq, axis=0, dtype=I32)
        b = base_full[:, pid].astype(I32)
        pos = (eq * (b[None, :] + pre - 1)).sum(axis=1, dtype=I32)
        order[pos[mask]] = offs[mask]
        inv[offs[mask]] = pos[mask]
    return order, inv


def scatter_off_v710(ids_pad, base_col, tot, N, C, BLOCK, E_PAD, ECHUNK):
    """_sort_scatter_off_kernel: rebuild the offset from TOT, slab by slab."""
    order = np.full(N, -1, dtype=np.int64)
    inv = np.full(N, -1, dtype=np.int64)
    for pid in range(C):
        offs = pid * BLOCK + np.arange(BLOCK)
        mask = offs < N
        ids = ids_pad[pid * BLOCK:(pid + 1) * BLOCK]
        pos = np.zeros(BLOCK, dtype=I32)
        run = I32(0)
        for e0 in range(0, E_PAD, ECHUNK):
            e = (e0 + np.arange(ECHUNK)).astype(I32)
            t = tot[e].astype(I32)
            exc = (run + np.cumsum(t, dtype=I32) - t).astype(I32)
            eq = (ids[:, None] == e[None, :]).astype(I32)
            pre = np.cumsum(eq, axis=0, dtype=I32)
            b = (base_col[e, pid] + exc).astype(I32)
            pos = pos + (eq * (b[None, :] + pre - 1)).sum(axis=1, dtype=I32)
            run = I32(run + t.sum(dtype=I32))
        order[pos[mask]] = offs[mask]
        inv[offs[mask]] = pos[mask]
    return order, inv


def run_sort_case(ids, E, rng):
    N = ids.shape[0]
    E_PAD = 1 << max(0, (E - 1).bit_length())
    block = min(256, max(64, 16384 // E_PAD))
    ECHUNK = E_PAD if E_PAD < 64 else 64
    C = (N + block - 1) // block
    C_PAD = 1 << max(0, (C - 1).bit_length())
    ids_pad = np.full(C * block, -1, dtype=I32)
    ids_pad[:N] = ids

    h_a = hist_full(ids_pad, C, block, E_PAD)
    h_b = hist_chunked(ids_pad, C, block, E_PAD, ECHUNK)
    assert np.array_equal(h_a, h_b), "hist chunking diverged"

    base_col, tot = colscan(h_a, C, E_PAD, C_PAD)
    base_full, counts_a = offsets_v692(base_col, tot, E)

    order_a, inv_a = scatter_v692(ids_pad, base_full, N, C, block, E_PAD)
    order_b, inv_b = scatter_off_v710(ids_pad, base_col, tot, N, C, block,
                                      E_PAD, ECHUNK)
    counts_b = tot[:E]

    ref_order = np.argsort(ids, kind="stable")
    ok = (np.array_equal(order_a, order_b)
          and np.array_equal(inv_a, inv_b)
          and np.array_equal(counts_a, counts_b)
          and np.array_equal(order_a, ref_order)
          and np.array_equal(counts_a, np.bincount(ids, minlength=E)[:E]))
    return ok, (E, N, E_PAD, block, C, ECHUNK)


# --------------------------------------------------------------------------- #
# change 3(a): top-k epilogue                                                  #
# --------------------------------------------------------------------------- #
def route_epilogue_v692(p, M, K_TOP, BLOCK_M, E_PAD, E):
    """per-round strided stores, then reload FLAT_W and divide in place."""
    F_IDS = np.full(M * K_TOP, -7, dtype=np.int64)
    F_W = np.full(M * K_TOP, np.float32(-7.0), dtype=np.float32)
    for pid in range((M + BLOCK_M - 1) // BLOCK_M):
        offs_m = pid * BLOCK_M + np.arange(BLOCK_M)
        m_mask = offs_m < M
        offs_e = np.arange(E_PAD, dtype=I32)
        p_work = p[pid * BLOCK_M:pid * BLOCK_M + BLOCK_M].copy()
        sum_sel = np.zeros(BLOCK_M, dtype=np.float32)
        for j in range(K_TOP):
            m_val = p_work.max(axis=1)
            is_max = p_work == m_val[:, None]
            idx = np.where(is_max, offs_e[None, :], I32(E_PAD)).min(axis=1)
            tgt = offs_m * K_TOP + j
            F_IDS[tgt[m_mask]] = idx[m_mask].astype(np.int64)
            F_W[tgt[m_mask]] = m_val[m_mask]
            sum_sel = (sum_sel + m_val).astype(np.float32)
            p_work = np.where(offs_e[None, :] == idx[:, None],
                              np.float32(-1.0), p_work).astype(np.float32)
        denom = np.maximum(sum_sel, np.float32(1e-6)).astype(np.float32)
        for j in range(K_TOP):
            tgt = offs_m * K_TOP + j
            w_raw = np.where(m_mask, F_W[np.clip(tgt, 0, M * K_TOP - 1)],
                             np.float32(0.0)).astype(np.float32)
            F_W[tgt[m_mask]] = (w_raw / denom).astype(np.float32)[m_mask]
    return F_IDS, F_W


def route_epilogue_v710(p, M, K_TOP, K_PAD, BLOCK_M, E_PAD, E):
    """register (BLOCK_M, K_PAD) tile + two masked block stores."""
    F_IDS = np.full(M * K_TOP, -7, dtype=I32)
    F_W = np.full(M * K_TOP, np.float32(-7.0), dtype=np.float32)
    for pid in range((M + BLOCK_M - 1) // BLOCK_M):
        offs_m = pid * BLOCK_M + np.arange(BLOCK_M)
        m_mask = offs_m < M
        offs_e = np.arange(E_PAD, dtype=I32)
        p_work = p[pid * BLOCK_M:pid * BLOCK_M + BLOCK_M].copy()
        sum_sel = np.zeros(BLOCK_M, dtype=np.float32)
        kk = np.arange(K_PAD, dtype=I32)
        ids_t = np.zeros((BLOCK_M, K_PAD), dtype=I32)
        w_t = np.zeros((BLOCK_M, K_PAD), dtype=np.float32)
        for j in range(K_TOP):
            m_val = p_work.max(axis=1)
            is_max = p_work == m_val[:, None]
            idx = np.where(is_max, offs_e[None, :], I32(E_PAD)).min(axis=1)
            sel = kk[None, :] == j
            ids_t = np.where(sel, idx[:, None], ids_t).astype(I32)
            w_t = np.where(sel, m_val[:, None], w_t).astype(np.float32)
            sum_sel = (sum_sel + m_val).astype(np.float32)
            p_work = np.where(offs_e[None, :] == idx[:, None],
                              np.float32(-1.0), p_work).astype(np.float32)
        denom = np.maximum(sum_sel, np.float32(1e-6)).astype(np.float32)
        w_t = (w_t / denom[:, None]).astype(np.float32)
        ptr = offs_m[:, None] * K_TOP + kk[None, :]
        msk = m_mask[:, None] & (kk[None, :] < K_TOP)
        # a masked Triton store touches nothing outside the mask, and every
        # in-mask address is < M*K_TOP, so no neighbour row is clobbered
        assert ptr[msk].max() < M * K_TOP
        F_IDS[ptr[msk]] = ids_t[msk]
        F_W[ptr[msk]] = w_t[msk]
    return F_IDS, F_W


def run_route_case(M, E, K_TOP, BLOCK_M, rng, peaked=False):
    E_PAD = 1 << max(0, (E - 1).bit_length())
    if E_PAD < 16:
        E_PAD = 16
    K_PAD = 1 << max(0, (K_TOP - 1).bit_length())
    # the kernel always works on full BLOCK_M tiles; rows >= M load A as 0.0,
    # so their logits are 0 for e < E and -3e38 beyond -- reproduce that here
    M_pad = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    logits = rng.standard_normal((M_pad, E_PAD)).astype(np.float32)
    if peaked:
        logits *= np.float32(40.0)          # forces exact ties / underflow
        logits[: M // 4] = np.float32(0.0)  # a whole block of exact ties
    logits[M:] = np.float32(0.0)
    logits = np.where(np.arange(E_PAD)[None, :] < E, logits,
                      np.float32(-3.0e38)).astype(np.float32)
    rmax = logits.max(axis=1)
    ex = np.exp(logits - rmax[:, None]).astype(np.float32)
    ex = np.where(np.arange(E_PAD)[None, :] < E, ex, np.float32(0.0)).astype(np.float32)
    p = (ex / ex.sum(axis=1, dtype=np.float32)[:, None]).astype(np.float32)

    a_i, a_w = route_epilogue_v692(p, M, K_TOP, BLOCK_M, E_PAD, E)
    b_i, b_w = route_epilogue_v710(p, M, K_TOP, K_PAD, BLOCK_M, E_PAD, E)
    same_i = np.array_equal(a_i.astype(np.int64), b_i.astype(np.int64))
    same_w = np.array_equal(a_w.view(np.uint32), b_w.view(np.uint32))  # bitwise
    unwritten = (b_w.view(np.uint32) == np.float32(-7.0).view(np.uint32)).sum()
    return same_i and same_w and unwritten == 0, (M, E, K_TOP, K_PAD, BLOCK_M)


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(20260904)

    # ---- changes 1 + 2(b): 200 randomized counting-sort cases -------------- #
    bad = []
    shapes = [(8, 4096), (32, 8192), (64, 6144), (96, 5000), (256, 4096),
              (256, 1024), (32, 700), (64, 130)]
    for t in range(200):
        E, N = shapes[t % len(shapes)]
        mode = t % 5
        if mode == 0:                       # uniform
            ids = rng.integers(0, E, size=N, dtype=np.int64)
        elif mode == 1:                     # every token on one expert
            ids = np.full(N, int(rng.integers(0, E)), dtype=np.int64)
        elif mode == 2:                     # only experts 0 and E-1 used
            ids = rng.choice(np.array([0, E - 1]), size=N)
        elif mode == 3:                     # heavy tail, ~half the experts empty
            live = rng.choice(E, size=max(1, E // 2), replace=False)
            w = rng.random(live.size) ** 6
            ids = rng.choice(live, size=N, p=w / w.sum())
        else:                               # first/last expert empty on purpose
            ids = rng.integers(1, max(2, E - 1), size=N, dtype=np.int64)
        ids = ids.astype(I32)
        ok, info = run_sort_case(ids, E, rng)
        if not ok:
            bad.append((t, info))
    print("[change 1 + 2b] counting sort: %d/200 cases bitwise identical "
          "(order, inv, counts) and equal to argsort(stable); failures=%s"
          % (200 - len(bad), bad if bad else "none"))

    # ---- change 3(a): top-k epilogue -------------------------------------- #
    bad = []
    cfgs = [(512, 8, 2, 128), (512, 32, 4, 128), (384, 64, 8, 64),
            (386, 96, 3, 64), (300, 96, 3, 64), (256, 256, 8, 32),
            (130, 256, 8, 32), (200, 64, 8, 64), (100, 32, 4, 128)]
    n = 0
    for rep in range(12):
        for (M, E, K_TOP, BLOCK_M) in cfgs:
            for peaked in (False, True):
                ok, info = run_route_case(M, E, K_TOP, BLOCK_M, rng, peaked)
                n += 1
                if not ok:
                    bad.append((info, peaked))
    print("[change 3a] route epilogue: %d/%d cases bitwise identical "
          "(FLAT_IDS int + FLAT_W raw fp32 bits, no slot left unwritten); "
          "failures=%s" % (n - len(bad), n, bad if bad else "none"))

    # ---- change 3(a): the K_PAD>K_TOP tail slot specifically --------------- #
    M, K_TOP, K_PAD, BLOCK_M = 7, 3, 4, 4
    offs_m = np.arange(BLOCK_M)
    kk = np.arange(K_PAD)
    hits = np.zeros(M * K_TOP, dtype=int)
    for pid in range((M + BLOCK_M - 1) // BLOCK_M):
        om = pid * BLOCK_M + offs_m
        ptr = om[:, None] * K_TOP + kk[None, :]
        msk = (om < M)[:, None] & (kk[None, :] < K_TOP)
        for a in ptr[msk]:
            hits[a] += 1
    print("[change 3a] tail-slot mask (K_TOP=3, K_PAD=4): every one of the "
          "%d live slots written exactly once = %s, out-of-range writes = 0"
          % (M * K_TOP, bool((hits == 1).all())))


if __name__ == "__main__":
    main()
