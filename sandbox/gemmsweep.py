"""gemmsweep - MoE-shape GEMM replica + config sweep for the XPUOJ sandbox (customTest / triton-h800).

Two standalone replicas of the live v692 kernels plus a one-axis-at-a-time config sweep.
  md <- p1/kernel_v692.py  _fgs_t1i_mdq_kernel_g (L5198) + _tma_kernel (L5118) + _kernel (L5043)
        merged into one kernel with AMODE picking the A path v692 would really use for that
        shape: 2 = gathered pointer A (the _g variant, shapes in v692's _GASET = c4..c10),
        0 = TMA A (H<=2048 -> c3/c11/c12), 1 = contiguous pointer A (c1/c2).
        Single [BLOCK_M, 2*BLOCK_N] fp32 acc, B always via TensorDescriptor, one dot per K step,
        then reshape/split -> tanh.approx SwiGLU -> per-row pow2 fp8 quant -> TMA store on full
        tiles / masked pointer store on tail tiles.  Epilogue is copied verbatim from v692.
  dn <- p1/kernel_v692.py  _dn_tma2_f8_kernel (L4199), body verbatim; A and B both via
        TensorDescriptor, [BLOCK_M, BLOCK_N] fp32 acc, fp8 out + one fp32 scale per BLOCK_N.

fp8 without the torch fp8 dtype: uint8 storage + triton.reinterpret(t, tl.float8e4nv).  The
TensorWrapper is a legal TensorDescriptor base (post_init only reads .data_ptr() and
.dtype.itemsize, and the launcher only reads .data_ptr()), so descriptors stay fp8-typed and no
in-kernel bitcast is needed -- wgmma still consumes B straight out of shared memory.

Grouped-GEMM metadata is built ENTIRELY IN TRITON: the sandbox has no host-side reduction
at all (host reductions and prefix sums are all refused), and the only torch entry points that
survive are randint / empty / zeros plus dtypes, arithmetic, comparison and indexing.  So
hist_kernel (one-hot + tl.sum + tl.atomic_add, the shape of v692 _expert_hist_kernel L572-585),
pfx_kernel (tl.cumsum, the idiom at v692 L636) and tile_kernel (blocked t >= tcum compare)
produce counts / rbeg / tnum / tcum / per-tile arrays / num_tiles directly into preallocated
device tensors.  E_PAD=256, BLOCK=128, TB=64 are fixed literals so those three kernels compile
once for the whole run instead of once per distinct E.  Expert-per-token is a real multinomial
draw, so tail-block waste matches production.

KNOBS (the two literal tuples below):
  SHAPES_TO_RUN   indices into shape_list(): 0..11 == c1..c12.
  PHASES_TO_RUN   0 baseline (always runs)   1 md stages/BLOCK_K/outer-loop   2 md GROUP_M,
                  BLOCK_N, BLOCK_M, num_warps   3 dn stages/BLOCK_K/GROUP_M   4 dn block shape,
                  flatten   5 dn BN=128 arithmetic-intensity probes that separate the
                  "operand bandwidth" and "clock/issue" explanations of the global derate,
                  including a 2-CTA/SM run at grid 264.  dn config tuples carry their own
                  grid in slot 9, so only that one probe leaves the 132-CTA persistent grid.
                  6 md cross-tile software pipeline: mdflat (compiler flatten, no outer
                  num_stages) plus mdpf / mdpfa (hand-rolled prefetch of the next tile's
                  k=0 operands, issued before this tile's epilogue) via md_pf_kernel.
                  8 SwiGLU tanh probes: tanhoff (MUFU replaced by a multiply, sizes the
                  tanh share of md) and tanhf16 (tanh.approx.f16x2 at pack=2).
                  7 deep-pipeline probes for the memory-bound shapes: BLOCK_K=64 halves
                  per-stage smem, buying num_stages and TMA requests in flight (k64s6 /
                  k64s8 on md and dn), reported as reverse-derived HBM bandwidth.
                  9 weight low-precision STORAGE on dn, at INT6.  The accuracy study
                  settles which compression is allowed: int6 g64 is +0.29 dB and int6 g128
                  -0.02 dB against fp8 per-row, inside the +1.12 dB SQNR margin, while
                  int4 g64 is -8.88 dB (g32 -8.06, g128 -9.60) and outlier isolation
                  recovers only ~2 dB of that.  So int6 (6.125 bit/weight at g128 =
                  49/64 = 0.766 of fp8) is the candidate and int4 is a reference point.
                  c9/c10 dn is weight-BYTE bound (c9: 2.147 GB of B against a 0.818 ms
                  kernel, 2.63 TB/s = 78% of 3.35 on B alone), so 0.766x bytes is worth
                  ~0.19 ms on dn and ~0.5 ms once md follows -- IF the decode Hopper
                  forces on us is cheaper than the bytes it saves.  wgmma takes neither
                  int4 nor int6 AND its B operand must come from shared memory, so a
                  register-side decode costs a compiler-inserted local_alloc round trip
                  (the mechanism that made md_pf_kernel's PFB=1 24% slower -- but that
                  kernel was compute bound and these are not).
                  Everything below is at K=32 granularity because the fp32 accumulator is
                  already 128 reg/thread: a decoded [256,128] B tile would be another 128
                  and spill outright, while a [256,32] chunk is 32 and the whole chain
                  fits in ~190-215 reg.  fp8 wgmma is natively m64nNk32, so four K=32 dots
                  in place of one K=128 dot should cost nothing -- b_q32 tests that rather
                  than assuming it.  Bit planes are loaded COMPACTLY and widened with
                  join+reshape, so the pipeliner stages 24KB/k-step (96KB at 4 stages,
                  plus 64KB of A and 16KB of decoded fp8 = 176KB < 228KB) instead of the
                  96KB/k-step a replicated gather would need.
                  Six dn variants, all selected by dn_lp_kernel's BMODE constexpr:
                    b_ceil   THE CEILING.  fp8, four K=32 dots, but the 32-byte chunk
                             index wraps at 49 of every 64 so exactly 0.765625 of B is
                             ever touched -- "bytes cut to the int6 ratio AND decode free".
                    b_int6   the real thing: hi4 plane (two 4-bit halves per byte) + lo2
                             plane (four 2-bit halves per byte) + one g128 scale byte,
                             10 uint8 ops/element and 3 joins, scale folded into the fp8
                             exponent by a single add.
                    b_q32    full-rate fp8 through the four-K=32 path -> prices the split
                             itself, which is the zero b_ceil and b_int6 are read against.
                    nat      dn_kernel's own K-loop through this kernel = structure control.
                    b_int4   one plane, 6 ops/element -- ALU LOWER BOUND ONLY, it says what
                             happens if the decode gets 1.6x cheaper.  Not accuracy-viable.
                    b_dotsc  tl.dot_scaled with mxfp4 + e8m0 scales.  Triton's formats are
                             {e2m1,e4m3,e5m2,bf16,fp16} -- there is NO 6-bit microscaling
                             format, so this can never express int6; it prices the
                             compiler's own upcast-and-dot lowering on sm90.
                  Numbers are throughput only: every variant is numerically wrong on
                  purpose.  tl.dot_scaled is capability-probed at zero compile cost first
                  and every variant is wrapped in try/except with one retry (reduced
                  num_stages, or the alternate call form for dot_scaled), so no rejection
                  can kill the run.  Compile budget at SHAPES_TO_RUN=(8,): 3 metadata +
                  1 md + 1 dn baseline + 6 variants = 11, at most 13 with retries.
                  Each config is a separate Triton compile (~0.5-1s); phases 1-4 are
                  <=24 configs per shape, phases 5-7 are 3, 3 and 4 each.
                  10 IS THE act ROUND TRIP ON THE CRITICAL PATH?  md writes
                  act[M, I] to HBM and dn reads it straight back; a fused md/dn
                  kernel that does one m-tile's md columns and then that same
                  m-tile's dn columns needs no cross-CTA sync and would turn the
                  round trip into an L2 hit (132 CTAs x 128 x I bytes live = 17 MB
                  at I=1024, inside the 50 MB L2).  That is 2*M*I bytes per case,
                  2.21 GB over c3..c12 = 0.80 ms at 2.75 TB/s.  The calibration
                  model bills each GEMM at max(compute, bytes) and md/dn carry 2-5x
                  memory slack on these shapes, so it scores act as hidden -- but
                  that has never been measured on its own, and it is the whole of
                  the fusion's upside.  Two independent readings:
                    mdpin / dnpin  act store, resp. dn's A load, redirected to the
                             CTA's own 128-row slot of a 132*128*I byte buffer.
                             Same instructions, same tile order, same everything --
                             only the address changes, so the footprint is L2
                             resident and act's DRAM traffic goes to zero.
                             base - pin IS the answer, with no traffic model in
                             between; their sum is the fusion's traffic upside.
                    rep1/3/8/16  the act store repeated into disjoint column bands
                             of a [M, 16*I] buffer, so REP copies really do touch
                             REP x M x I distinct bytes (one act-sized buffer would
                             let L2 swallow the duplicates and flatten the ladder
                             for the wrong reason).  rep3 is the stated gate:
                             >=8% => the store is exposed and the fused kernel is
                             worth writing, <3% => it is hidden by compute and the
                             fusion buys only a launch and a prologue.  rep16 is
                             the positive control: a flat reading there implies a
                             store rate the part cannot deliver, i.e. a broken
                             probe rather than free stores.  mode0/pin0 are the
                             structural controls and must read ~0% vs phase 0.
Baselines for every selected shape run first, then the sweep runs shape by shape, and each line
is printed the moment it is measured, so a timeout kill still leaves the measured rows on stdout.
Host builtins are down to print and range only (list/sorted/min/len/str all removed after
'list' turned out to be banned).  probe() prints p1..p6 markers across the last unverified
host APIs so one run pins down any further ban.  Ends on a deliberate NameError.
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

SHAPES_TO_RUN = (3, 7, 10)
PHASES_TO_RUN = (10,)
DEV = "cuda"
NSM = 132
PEAK = 1979.0
WU = 5
RP = 20
RMAX = 16


def shape_list():
    return ((16384, 4096, 8, 8192, 2), (16384, 4096, 8, 14336, 2),
            (16384, 2048, 32, 2048, 4), (16384, 2048, 32, 1024, 4),
            (8192, 3584, 64, 2560, 8), (8192, 3584, 64, 1024, 8),
            (16384, 4096, 96, 2048, 3), (16384, 4096, 96, 1024, 3),
            (4096, 4096, 256, 2048, 8), (4096, 4096, 256, 1536, 8),
            (65536, 1024, 32, 1024, 2), (65536, 1024, 32, 2048, 2))


def real_pk():
    return (71.9, 73.4, 58.9, 49.7, 64.8, 54.8, 54.6, 46.1, 31.7, 31.6, 41.3, 50.7)


def ga_set():
    return (3, 4, 5, 6, 7, 8, 9)


def P(s):
    print(s, flush=True)


@triton.jit
def md_kernel(A, A_DESC, A_SCALE, BNORM, B_DESC, B_SCALE, W, ORDER, ACT, ACT_DESC, SCL,
              expert_ids, split_size, split_size_cum, tile_num, tile_cum, num_tiles,
              M, I, K: tl.constexpr,
              stride_am, stride_ak, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr, KTOP: tl.constexpr, AMODE: tl.constexpr,
              ONS: tl.constexpr, OFLAT: tl.constexpr, TANH: tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(num_tiles)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid,
                            num_stages=ONS, flatten=OFLAT):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert = tl.load(expert_ids + pid_m)
        n_rows = tl.load(split_size + expert)
        row_begin = tl.load(split_size_cum + pid_m)
        t_num = tl.load(tile_num + pid_m)
        t_cum = tl.load(tile_cum + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < row_begin + n_rows

        a_row = row_begin + local_m * BLOCK_M
        b_row = expert * (2 * I) + pid_n * (2 * BLOCK_N)
        b_sc = tl.load(B_SCALE + expert * (2 * I) + pid_n * 2 * BLOCK_N + tl.arange(0, 2 * BLOCK_N))
        w = tl.load(W + offs_m, mask=row_mask, other=0.0).to(tl.float32)

        if AMODE == 2:
            rows = tl.load(ORDER + offs_m, mask=row_mask, other=0) // KTOP
            a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        elif AMODE == 1:
            a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        else:
            a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)

        acc = tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if AMODE == 0:
                a = A_DESC.load([a_row, k * BLOCK_K])
            else:
                a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0,
                            eviction_policy='evict_last')
                a_ptrs += BLOCK_K * stride_ak
            b = B_DESC.load([b_row, k * BLOCK_K])
            acc = tl.dot(a, b.T, acc)

        scaled = acc * a_scale[:, None] * b_sc[None, :]
        pair = tl.reshape(scaled, (BLOCK_M, BLOCK_N, 2))
        g, u = tl.split(pair)
        if TANH == 1:
            # tanh.approx.f16x2 is one MUFU op for two lanes (PTX 7.0, sm_75+).  With
            # pack=2 Triton packs the two fp16 elements into a single 4-byte register,
            # so the asm takes exactly one input and one output operand -- same
            # convention as the pack=4 uint8 example in tl.inline_asm_elementwise's docs.
            th = tl.inline_asm_elementwise(
                "tanh.approx.f16x2 $0, $1;", "=r,r", [(g * 0.5).to(tl.float16)],
                dtype=tl.float16, is_pure=True, pack=2).to(tl.float32)
        elif TANH == 2:
            # throughput control only: numerically wrong on purpose, it replaces the
            # MUFU with one multiply so the measured delta is the tanh cost itself.
            th = (g * 0.5) * 0.5
        else:
            th = tl.inline_asm_elementwise(
                "tanh.approx.f32 $0, $1;", "=f,f", [g * 0.5],
                dtype=tl.float32, is_pure=True, pack=1)
        silu = (g * 0.5) * (1.0 + th)
        act = silu * u * w[:, None]
        bn = tl.load(BNORM + expert)
        bound = a_scale * a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        if a_row + BLOCK_M <= row_begin + n_rows:
            ACT_DESC.store([a_row, pid_n * BLOCK_N], q)
        else:
            c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy='evict_first')
        tl.store(SCL + offs_m, s, mask=row_mask)


@triton.jit
def dn_kernel(A_DESC, A_SCALE, B_DESC, B_SCALE, C, CSCL,
              expert_ids, split_size, split_size_cum, tile_num, tile_num_cum, num_tiles_total,
              M, N, K: tl.constexpr,
              stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              GROUP_M: tl.constexpr, ONS: tl.constexpr, FLAT: tl.constexpr):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    total_tiles = tl.load(num_tiles_total)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid,
                            num_stages=ONS, flatten=FLAT):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert = tl.load(expert_ids + pid_m)
        n_rows = tl.load(split_size + expert)
        row_begin = tl.load(split_size_cum + pid_m)
        t_num = tl.load(tile_num + pid_m)
        t_cum = tl.load(tile_num_cum + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < row_begin + n_rows

        a_row = row_begin + local_m * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_row = expert * N + pid_n * BLOCK_N
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = A_DESC.load([a_row, k * BLOCK_K])
            b = B_DESC.load([b_row, k * BLOCK_K])
            acc = tl.dot(a, b.T, acc)

        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        b_scale = tl.load(B_SCALE + expert * N + offs_n)
        acc = acc * a_scale[:, None] * b_scale[None, :]
        row_max = tl.max(tl.abs(acc), axis=1)
        s = tl.maximum(row_max / 448.0, 1e-12)
        q = (acc * (1.0 / s)[:, None]).to(tl.float8e4nv)
        tl.store(CSCL + offs_m * num_block_n + pid_n, s, mask=row_mask)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None])


@triton.jit
def dn_lp_kernel(A_DESC, AQ_DESC, A_SCALE, ASC8, B_DESC, BQ_DESC, BP_DESC, BSC8,
                 BH4, BL2, BS6, B_SCALE, C, CSCL,
                 expert_ids, split_size, split_size_cum, tile_num, tile_num_cum, num_tiles_total,
                 M, N, K: tl.constexpr,
                 stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr, ONS: tl.constexpr, FLAT: tl.constexpr,
                 BMODE: tl.constexpr, DSMODE: tl.constexpr):
    # dn_kernel with the STORAGE precision of B swapped out.  Prologue, epilogue and every
    # config axis are byte-identical to dn_kernel; only the K-loop body moves.
    #
    # The target is int6 with a g=128 group scale = 6.125 bit/weight = 49/64 of fp8, the one
    # quantisation the accuracy study clears (+0.29 dB at g64, -0.02 dB at g128, against a
    # +1.12 dB SQNR margin -- int4 loses 7.7-9.6 dB and outlier isolation recovers at most
    # 2 dB of that, so int4 is a reference point here, not a candidate).
    #
    # Everything below K=32 granularity is forced by the register budget: the fp32
    # accumulator is [128,256] = 128 reg/thread, so a decoded [BLOCK_N, BLOCK_K] = [256,128]
    # B tile at 128 elements/thread would spill on its own.  At 32 wide a decoded tile is
    # 32 elements/thread and the whole chain fits in ~190-215 reg.  fp8 wgmma is natively
    # m64nNk32, so splitting one K=128 dot into four K=32 dots costs no MMA efficiency --
    # b_q32 measures that claim directly rather than assuming it.
    #
    # Bit planes are loaded COMPACTLY ([BLOCK_N,16] hi4 and [BLOCK_N,8] lo2 per 32-wide
    # chunk, fully coalesced) and widened in registers with tl.join+tl.reshape, so the
    # pipeliner stages 24KB/k-step instead of the 96KB a replicated gather would stage.
    #
    # Every uint8 mask/shift constant is built with tl.full, never as a Python int literal:
    # Triton promotes a bare int to int32, and an int32 [BLOCK_N,32] intermediate is 32
    # extra registers/thread per live value on top of the accumulator's 128.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    total_tiles = tl.load(num_tiles_total)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid,
                            num_stages=ONS, flatten=FLAT):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert = tl.load(expert_ids + pid_m)
        n_rows = tl.load(split_size + expert)
        row_begin = tl.load(split_size_cum + pid_m)
        t_num = tl.load(tile_num + pid_m)
        t_cum = tl.load(tile_num_cum + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < row_begin + n_rows

        a_row = row_begin + local_m * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_row = expert * N + pid_n * BLOCK_N
        brow = expert * N + offs_n
        ph4 = BH4 + brow[:, None] * (K // 2)
        pl2 = BL2 + brow[:, None] * (K // 4)
        # lane -> (which sub-field of which packed byte).  sh4 picks nibble idx%2 of byte
        # idx//2, sh2 picks 2-bit field idx%4 of byte idx//4; the join pattern below places
        # those bytes so that the shift amount is all that is needed.
        idx = tl.arange(0, 32)
        sh4 = ((idx % 2) * 4).to(tl.uint8)
        sh2 = ((idx % 4) * 2).to(tl.uint8)
        q0f = tl.full((BLOCK_N, 32), 15, tl.uint8)
        q03 = tl.full((BLOCK_N, 32), 3, tl.uint8)
        q02 = tl.full((BLOCK_N, 32), 2, tl.uint8)
        q04 = tl.full((BLOCK_N, 32), 4, tl.uint8)
        q08 = tl.full((BLOCK_N, 32), 8, tl.uint8)
        q20 = tl.full((BLOCK_N, 32), 32, tl.uint8)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if BMODE == 6:
                # nat: dn_kernel's own K-loop, verbatim.  Control for this kernel's extra
                # (unused) arguments -- must read ~0% against the dn baseline.
                acc = tl.dot(A_DESC.load([a_row, k * BLOCK_K]),
                             B_DESC.load([b_row, k * BLOCK_K]).T, acc)
            elif BMODE == 5:
                # b_dotsc: the packed bytes straight to tl.dot_scaled as mxfp4 (e2m1) with
                # e8m0 scales.  NOTE this can only ever be a 4-bit reference: Triton's
                # allowed_formats is {e2m1, e4m3, e5m2, bf16, fp16} -- there is no 6-bit
                # microscaling format, so dot_scaled cannot express int6 at all.  It stays
                # in to price the compiler's own upcast-and-dot lowering on sm90 against
                # the hand-rolled decode below.  bp.T is [BLOCK_K/2, BLOCK_N], the
                # rhs_k_pack=True convention.
                bp = BP_DESC.load([b_row, k * (BLOCK_K // 2)]).to(tl.uint8, bitcast=True)
                bs = tl.load(BSC8 + brow[:, None] * (K // 32)
                             + (k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32))[None, :])
                if DSMODE == 0:
                    acc = tl.dot_scaled(A_DESC.load([a_row, k * BLOCK_K]), None, "e4m3",
                                        bp.T, bs, "e2m1", acc)
                else:
                    asc = tl.load(ASC8 + offs_m[:, None] * (K // 32)
                                  + (k * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32))[None, :],
                                  mask=row_mask[:, None], other=127)
                    acc = tl.dot_scaled(A_DESC.load([a_row, k * BLOCK_K]), asc, "e4m3",
                                        bp.T, bs, "e2m1", acc)
            else:
                # one group scale per k-step: g=128 == BLOCK_K, so a 32-wide chunk always
                # lies inside one group and the scale degenerates to a [BLOCK_N] vector
                # with no selection along K.  It is folded into the fp8 exponent field by
                # a single uint8 add, which is what makes a power-of-two group scale free
                # -- a non-power-of-two scale would have to hit the fp32 accumulator once
                # per group instead, at ~256 cycles/k-step.
                if BMODE == 1:
                    sc = tl.load(BS6 + brow * (K // 128) + k)
                if BMODE == 2:
                    sc = tl.load(BS6 + brow * (K // 128) + k)
                for kk in tl.static_range(4):
                    aq = AQ_DESC.load([a_row, k * BLOCK_K + kk * 32])
                    if BMODE == 7:
                        # b_q32: full-rate fp8, but four K=32 dots instead of one K=128.
                        # Isolates the cost of the 32-wide decomposition itself, so
                        # b_ceil and b_int6 can be read against the right zero.
                        bq = BQ_DESC.load([b_row, k * BLOCK_K + kk * 32])
                    elif BMODE == 0:
                        # b_ceil: THE CEILING.  Identical to b_q32 in every respect except
                        # that the 32-byte chunk index wraps at 49 of every 64, so exactly
                        # 49/64 = 0.765625 of B is ever touched -- the int6 g128 ratio to
                        # the byte.  tiles/expert == 1 on c9/c10 so every byte is read once
                        # and this is a real footprint, not a bound.  Numerically wrong on
                        # purpose: this is "bytes cut to 0.766x AND decode free".
                        bq = BQ_DESC.load([b_row, ((k * 4 + kk) % (((K // 32) * 49) // 64)) * 32])
                    elif BMODE == 2:
                        # b_int4: ALU LOWER BOUND ONLY -- the accuracy study rules int4 out
                        # (-8.9 dB at g64 against a +1.12 dB margin).  Same structure and
                        # the same one join as b_int6, but one plane instead of two and 6
                        # uint8 ops/element instead of 10, so b_int6 - b_int4 prices the
                        # extra plane plus the extra arithmetic on its own.
                        hb16 = tl.load(ph4 + (k * 64 + kk * 16 + tl.arange(0, 16))[None, :])
                        hb = tl.reshape(tl.join(hb16, hb16), (BLOCK_N, 32))
                        h = (hb >> sh4[None, :]) & q0f
                        bq = (((h ^ q08) << q04 | q04) + sc[:, None]).to(tl.float8e4nv,
                                                                        bitcast=True)
                    else:
                        # b_int6: the real thing.  Two bit planes -- hi4 packs two 4-bit
                        # halves per byte, lo2 packs four 2-bit halves per byte -- are
                        # loaded compactly ([BN,16] and [BN,8], coalesced) and widened by
                        # join+reshape so that byte j lands on lanes 2j..2j+1 and 4j..4j+3
                        # respectively; a single shift per lane then extracts the field.
                        # code = (hi4 << 2) | lo2 is the 6-bit weight; the XOR is the
                        # zero-point flip and the final shift/or places it in an e4m3 field
                        # (sign, 4-bit exponent, 1 mantissa bit -- exponent 1111 can never
                        # pair with mantissa 111 here, so no NaN is ever produced).
                        # 10 uint8 ops/element + 3 joins.  Exact arithmetic is a throughput
                        # stand-in for (code-32)*scale at the same instruction count;
                        # numerical correctness is out of scope for this probe.
                        hb16 = tl.load(ph4 + (k * 64 + kk * 16 + tl.arange(0, 16))[None, :])
                        lb8 = tl.load(pl2 + (k * 32 + kk * 8 + tl.arange(0, 8))[None, :])
                        hb = tl.reshape(tl.join(hb16, hb16), (BLOCK_N, 32))
                        lb16 = tl.reshape(tl.join(lb8, lb8), (BLOCK_N, 16))
                        lb = tl.reshape(tl.join(lb16, lb16), (BLOCK_N, 32))
                        h = (hb >> sh4[None, :]) & q0f
                        l = (lb >> sh2[None, :]) & q03
                        code = ((h << q02) | l) ^ q20
                        bq = (((code << q02) | q02) + sc[:, None]).to(tl.float8e4nv,
                                                                     bitcast=True)
                    acc = tl.dot(aq, bq.T, acc)

        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        b_scale = tl.load(B_SCALE + expert * N + offs_n)
        acc = acc * a_scale[:, None] * b_scale[None, :]
        row_max = tl.max(tl.abs(acc), axis=1)
        s = tl.maximum(row_max / 448.0, 1e-12)
        q = (acc * (1.0 / s)[:, None]).to(tl.float8e4nv)
        tl.store(CSCL + offs_m * num_block_n + pid_n, s, mask=row_mask)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None])


@triton.jit
def md_pf_kernel(A, A_SCALE, BNORM, B_DESC, B_SCALE, W, ORDER, ACT, ACT_DESC, SCL,
                 expert_ids, split_size, split_size_cum, tile_num, tile_cum, num_tiles,
                 M, I, K: tl.constexpr,
                 stride_am, stride_ak, stride_cm, stride_cn,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                 GROUP_M: tl.constexpr, KTOP: tl.constexpr, AMODE: tl.constexpr,
                 PFB: tl.constexpr):
    # md with a hand-rolled cross-tile software pipeline.  The k=0 operands of tile t+1
    # are decoded and issued right after tile t's last dot and BEFORE tile t's epilogue,
    # so the K-loop prologue of the next tile overlaps the SwiGLU/quant/store of this one.
    # Each tile is decoded exactly once: the decoded scalars ride the outer loop as
    # carried values (ex/rb/nr/lm/pn), which is why the loop keeps no num_stages of its
    # own - the manual pipeline replaces the compiler one.
    # PFB=1 also register-carries the [2*BLOCK_N, BLOCK_K] B tile; PFB=0 leaves B on the
    # native TMA->smem path and only carries A.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(num_tiles)
    ntile = total_tiles * num_block_n
    offs_k = tl.arange(0, BLOCK_K)

    t0 = tl.minimum(pid, ntile - 1)
    pm0 = t0 // num_block_n
    pn0 = t0 % num_block_n
    ex = tl.load(expert_ids + pm0)
    nr = tl.load(split_size + ex)
    rb = tl.load(split_size_cum + pm0)
    tn0 = tl.load(tile_num + pm0)
    tc0 = tl.load(tile_cum + pm0)
    lm = pm0 - (tc0 - tn0)
    lm, pn = tl.swizzle2d(lm, pn0, tn0, num_block_n, GROUP_M)
    om0 = rb + lm * BLOCK_M + tl.arange(0, BLOCK_M)
    mk0 = om0 < rb + nr
    if AMODE == 2:
        rw0 = tl.load(ORDER + om0, mask=mk0, other=0) // KTOP
        a_nx = tl.load(A + rw0[:, None] * stride_am + offs_k[None, :] * stride_ak,
                       mask=mk0[:, None], other=0.0, eviction_policy='evict_last')
    else:
        a_nx = tl.load(A + om0[:, None] * stride_am + offs_k[None, :] * stride_ak,
                       mask=mk0[:, None], other=0.0, eviction_policy='evict_last')
    if PFB == 1:
        b_nx = B_DESC.load([ex * (2 * I) + pn * (2 * BLOCK_N), 0])

    for tile_id in tl.range(pid, ntile, num_pid):
        cex = ex
        crb = rb
        cnr = nr
        clm = lm
        cpn = pn
        com = crb + clm * BLOCK_M + tl.arange(0, BLOCK_M)
        con = cpn * BLOCK_N + tl.arange(0, BLOCK_N)
        cmask = com < crb + cnr
        carow = crb + clm * BLOCK_M
        cbrow = cex * (2 * I) + cpn * (2 * BLOCK_N)
        b_sc = tl.load(B_SCALE + cex * (2 * I) + cpn * 2 * BLOCK_N + tl.arange(0, 2 * BLOCK_N))
        w = tl.load(W + com, mask=cmask, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.float32)
        if PFB == 1:
            acc = tl.dot(a_nx, b_nx.T, acc)
        else:
            acc = tl.dot(a_nx, B_DESC.load([cbrow, 0]).T, acc)

        if AMODE == 2:
            crw = tl.load(ORDER + com, mask=cmask, other=0) // KTOP
            a_ptrs = A + crw[:, None] * stride_am + (offs_k + BLOCK_K)[None, :] * stride_ak
            a_scale = tl.load(A_SCALE + crw, mask=cmask, other=1.0)
        else:
            a_ptrs = A + com[:, None] * stride_am + (offs_k + BLOCK_K)[None, :] * stride_ak
            a_scale = tl.load(A_SCALE + com, mask=cmask, other=1.0)

        for k in range(1, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=cmask[:, None], other=0.0,
                        eviction_policy='evict_last')
            b = B_DESC.load([cbrow, k * BLOCK_K])
            acc = tl.dot(a, b.T, acc)
            a_ptrs += BLOCK_K * stride_ak

        nt = tl.minimum(tile_id + num_pid, ntile - 1)
        npm = nt // num_block_n
        npn = nt % num_block_n
        ex = tl.load(expert_ids + npm)
        nr = tl.load(split_size + ex)
        rb = tl.load(split_size_cum + npm)
        tnn = tl.load(tile_num + npm)
        tcc = tl.load(tile_cum + npm)
        lm = npm - (tcc - tnn)
        lm, pn = tl.swizzle2d(lm, npn, tnn, num_block_n, GROUP_M)
        nom = rb + lm * BLOCK_M + tl.arange(0, BLOCK_M)
        nmk = nom < rb + nr
        if AMODE == 2:
            nrw = tl.load(ORDER + nom, mask=nmk, other=0) // KTOP
            a_nx = tl.load(A + nrw[:, None] * stride_am + offs_k[None, :] * stride_ak,
                           mask=nmk[:, None], other=0.0, eviction_policy='evict_last')
        else:
            a_nx = tl.load(A + nom[:, None] * stride_am + offs_k[None, :] * stride_ak,
                           mask=nmk[:, None], other=0.0, eviction_policy='evict_last')
        if PFB == 1:
            b_nx = B_DESC.load([ex * (2 * I) + pn * (2 * BLOCK_N), 0])

        scaled = acc * a_scale[:, None] * b_sc[None, :]
        pair = tl.reshape(scaled, (BLOCK_M, BLOCK_N, 2))
        g, u = tl.split(pair)
        th = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;", "=f,f", [g * 0.5],
            dtype=tl.float32, is_pure=True, pack=1)
        silu = (g * 0.5) * (1.0 + th)
        act = silu * u * w[:, None]
        bn = tl.load(BNORM + cex)
        bound = a_scale * a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        sq = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        if carow + BLOCK_M <= crb + cnr:
            ACT_DESC.store([carow, cpn * BLOCK_N], q)
        else:
            c_ptrs = ACT + com[:, None] * stride_cm + con[None, :] * stride_cn
            tl.store(c_ptrs, q, mask=cmask[:, None], eviction_policy='evict_first')
        tl.store(SCL + com, sq, mask=cmask)


@triton.jit
def md_p10_kernel(A, A_DESC, A_SCALE, BNORM, B_DESC, B_SCALE, W, ORDER, ACT, ACT_DESC, SCL,
                  PIN, PIN_DESC, WIDE, WIDE_DESC,
                  expert_ids, split_size, split_size_cum, tile_num, tile_cum, num_tiles,
                  M, I, K: tl.constexpr,
                  stride_am, stride_ak, stride_cm, stride_cn, stride_pm, stride_wm,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, KTOP: tl.constexpr, AMODE: tl.constexpr,
                  ONS: tl.constexpr, OFLAT: tl.constexpr,
                  MODE: tl.constexpr, REP: tl.constexpr):
    # phase 10 -- md_kernel with ONLY the act destination moved.  Prologue, K loop and
    # the whole SwiGLU/quant epilogue are byte-identical to md_kernel (TANH=0 path), so
    # every delta measured here is act STORE TRAFFIC and nothing else.
    #   MODE 0  act -> ACT[M, I], the production destination.  Structural control: it
    #           must read ~0% against the phase-0 md baseline or this kernel is not the
    #           same kernel and no other number in the phase means anything.
    #   MODE 1  act -> PIN[132*128, I] at row pid*BLOCK_M.  Every CTA owns a private
    #           128-row slot, so the whole live act footprint is 132*128*I bytes
    #           (17 MB at I=1024) and stays L2 resident for the entire kernel: the DRAM
    #           write traffic of act goes to ~0 while the instruction stream, the TMA
    #           store, the tail branch and the SCL store all stay exactly as they are.
    #           THIS IS THE MEASUREMENT: mode0 - mode1 = what md/dn fusion can win back
    #           on the write side, with no traffic model in between.
    #   MODE 2  act -> WIDE[M, RMAX*I], written REP times to REP disjoint column bands.
    #           REP copies touch REP x M x I distinct bytes, so DRAM write traffic scales
    #           exactly linearly (a single act-sized buffer would let L2 absorb the
    #           duplicates and the ladder would flatten for the wrong reason).  This is
    #           the sensitivity ladder: it says how much slack md has before the store
    #           becomes visible at all, and REP=16 is the positive control -- at 16x the
    #           implied DRAM rate exceeds what the part can do, so a flat reading there
    #           means the probe itself is broken, not that stores are free.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(I, BLOCK_N)
    total_tiles = tl.load(num_tiles)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid,
                            num_stages=ONS, flatten=OFLAT):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert = tl.load(expert_ids + pid_m)
        n_rows = tl.load(split_size + expert)
        row_begin = tl.load(split_size_cum + pid_m)
        t_num = tl.load(tile_num + pid_m)
        t_cum = tl.load(tile_cum + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = offs_m < row_begin + n_rows

        a_row = row_begin + local_m * BLOCK_M
        b_row = expert * (2 * I) + pid_n * (2 * BLOCK_N)
        b_sc = tl.load(B_SCALE + expert * (2 * I) + pid_n * 2 * BLOCK_N + tl.arange(0, 2 * BLOCK_N))
        w = tl.load(W + offs_m, mask=row_mask, other=0.0).to(tl.float32)

        if AMODE == 2:
            rows = tl.load(ORDER + offs_m, mask=row_mask, other=0) // KTOP
            a_ptrs = A + rows[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_scale = tl.load(A_SCALE + rows, mask=row_mask, other=1.0)
        elif AMODE == 1:
            a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        else:
            a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)

        acc = tl.zeros((BLOCK_M, 2 * BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if AMODE == 0:
                a = A_DESC.load([a_row, k * BLOCK_K])
            else:
                a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0,
                            eviction_policy='evict_last')
                a_ptrs += BLOCK_K * stride_ak
            b = B_DESC.load([b_row, k * BLOCK_K])
            acc = tl.dot(a, b.T, acc)

        scaled = acc * a_scale[:, None] * b_sc[None, :]
        pair = tl.reshape(scaled, (BLOCK_M, BLOCK_N, 2))
        g, u = tl.split(pair)
        th = tl.inline_asm_elementwise(
            "tanh.approx.f32 $0, $1;", "=f,f", [g * 0.5],
            dtype=tl.float32, is_pure=True, pack=1)
        silu = (g * 0.5) * (1.0 + th)
        act = silu * u * w[:, None]
        bn = tl.load(BNORM + expert)
        bound = a_scale * a_scale * bn * bn * tl.abs(w)
        bbits = tl.maximum(bound, 1e-30).to(tl.int32, bitcast=True)
        bexp = (bbits >> 23) & 0xFF
        s = ((bexp + 10) << 23).to(tl.float32, bitcast=True)
        inv = ((244 - bexp) << 23).to(tl.float32, bitcast=True)
        q = (act * inv[:, None]).to(tl.float8e4nv)
        full = a_row + BLOCK_M <= row_begin + n_rows
        if MODE == 0:
            if full:
                ACT_DESC.store([a_row, pid_n * BLOCK_N], q)
            else:
                c_ptrs = ACT + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(c_ptrs, q, mask=row_mask[:, None], eviction_policy='evict_first')
        elif MODE == 1:
            if full:
                PIN_DESC.store([pid * BLOCK_M, pid_n * BLOCK_N], q)
            else:
                p_ptrs = (PIN + (pid * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_pm
                          + offs_n[None, :] * stride_cn)
                tl.store(p_ptrs, q, mask=row_mask[:, None], eviction_policy='evict_first')
        else:
            # the full/tail test is hoisted out of the unrolled copy loop so REP only
            # multiplies stores, never branches.
            if full:
                for r in tl.static_range(REP):
                    WIDE_DESC.store([a_row, (r * num_block_n + pid_n) * BLOCK_N], q)
            else:
                for r in tl.static_range(REP):
                    w_ptrs = (WIDE + offs_m[:, None] * stride_wm
                              + ((r * num_block_n + pid_n) * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_cn)
                    tl.store(w_ptrs, q, mask=row_mask[:, None], eviction_policy='evict_first')
        tl.store(SCL + offs_m, s, mask=row_mask)


@triton.jit
def dn_p10_kernel(A_DESC, PIN_DESC, A_SCALE, B_DESC, B_SCALE, C, CSCL,
                  expert_ids, split_size, split_size_cum, tile_num, tile_num_cum, num_tiles_total,
                  M, N, K: tl.constexpr,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  GROUP_M: tl.constexpr, ONS: tl.constexpr, FLAT: tl.constexpr,
                  PINM: tl.constexpr):
    # phase 10 -- dn_kernel with ONLY the A source moved.  PINM=0 is dn_kernel verbatim
    # (structural control); PINM=1 reads the A tile from the CTA's private 128-row slot
    # of PIN[132*128, K], so the whole A working set is 132*128*K bytes (17 MB at
    # K=I=1024) and every A load after the first pass is an L2 hit.  The dot chain, the
    # B stream, the epilogue and the C store are untouched, so base - pin = the DRAM read
    # traffic of act, which is the read half of what md/dn fusion buys back.
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_block_n = tl.cdiv(N, BLOCK_N)
    total_tiles = tl.load(num_tiles_total)

    for tile_id in tl.range(pid, total_tiles * num_block_n, num_pid,
                            num_stages=ONS, flatten=FLAT):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert = tl.load(expert_ids + pid_m)
        n_rows = tl.load(split_size + expert)
        row_begin = tl.load(split_size_cum + pid_m)
        t_num = tl.load(tile_num + pid_m)
        t_cum = tl.load(tile_num_cum + pid_m)
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = tl.swizzle2d(local_m, pid_n, t_num, num_block_n, GROUP_M)

        offs_m = row_begin + local_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < row_begin + n_rows

        a_row = row_begin + local_m * BLOCK_M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        b_row = expert * N + pid_n * BLOCK_N
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if PINM == 0:
                a = A_DESC.load([a_row, k * BLOCK_K])
            else:
                a = PIN_DESC.load([pid * BLOCK_M, k * BLOCK_K])
            b = B_DESC.load([b_row, k * BLOCK_K])
            acc = tl.dot(a, b.T, acc)

        a_scale = tl.load(A_SCALE + offs_m, mask=row_mask, other=1.0)
        b_scale = tl.load(B_SCALE + expert * N + offs_n)
        acc = acc * a_scale[:, None] * b_scale[None, :]
        row_max = tl.max(tl.abs(acc), axis=1)
        s = tl.maximum(row_max / 448.0, 1e-12)
        q = (acc * (1.0 / s)[:, None]).to(tl.float8e4nv)
        tl.store(CSCL + offs_m * num_block_n + pid_n, s, mask=row_mask)
        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, q, mask=row_mask[:, None])


def u8(rows, cols):
    return torch.randint(0, 60, (rows, cols), dtype=torch.uint8, device=DEV)


def u8p(rows, cols):
    # packed weight planes: the full byte range, so every sub-field covers its range.
    return torch.randint(0, 256, (rows, cols), dtype=torch.uint8, device=DEV)


def u8e(rows, cols):
    # exponent-style scale bytes (int6 group scale, and e8m0 for tl.dot_scaled), kept near
    # the 127 == 2^0 bias so the folded products land in a normal fp8/fp32 range.
    return torch.randint(120, 136, (rows, cols), dtype=torch.uint8, device=DEV)


def f32vec(n):
    return torch.randint(1, 5, (n,), dtype=torch.int32, device=DEV).to(torch.float32) * 0.03125


@triton.jit
def hist_kernel(IDS, OUT, N, E, E_PAD: tl.constexpr, BLOCK: tl.constexpr):
    # verbatim shape of v692 _expert_hist_kernel (kernel_v692.py L572-585)
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ids = tl.load(IDS + offs, mask=mask, other=-1)
    e = tl.arange(0, E_PAD)
    m = (ids[:, None] == e[None, :]).to(tl.int32)
    cnt = tl.sum(m, axis=0)
    tl.atomic_add(OUT + e, cnt, mask=e < E, sem="relaxed")


@triton.jit
def pfx_kernel(COUNTS, RBEG, TNUM, TCUM, NTIL, E, BM, E_PAD: tl.constexpr):
    # counts -> exclusive row prefix sum, tiles per expert, inclusive tile prefix sum.
    # tl.cumsum idiom is the one v692 uses at kernel_v692.py L636.
    e = tl.arange(0, E_PAD)
    m = e < E
    c = tl.load(COUNTS + e, mask=m, other=0)
    rb = tl.cumsum(c, 0) - c
    tn = (c + (BM - 1)) // BM
    tn = tn + (c < 1).to(tl.int32)
    tn = tl.where(m, tn, 0)
    tc = tl.cumsum(tn, 0)
    tl.store(RBEG + e, rb, mask=m)
    tl.store(TNUM + e, tn, mask=m)
    tl.store(TCUM + e, tc, mask=m)
    tot = tl.sum(tl.where(e == (E - 1), tc, 0), axis=0)
    tl.store(NTIL + e, tot, mask=(e == 0))


@triton.jit
def tile_kernel(RBEG, TNUM, TCUM, EIDO, RBEGO, TNUMO, TCUMO, MAXT, E,
                E_PAD: tl.constexpr, TB: tl.constexpr):
    # per-tile expansion: expert of tile t is #{e : t >= tcum[e]}, clamped into [0, E-1]
    # for the padding tail (maxt >= real tile count).  Blocked over t so the
    # [TB, E_PAD] compare matrix stays register-resident.
    pid = tl.program_id(axis=0)
    t = pid * TB + tl.arange(0, TB)
    tm = t < MAXT
    e = tl.arange(0, E_PAD)
    em = e < E
    tc = tl.load(TCUM + e, mask=em, other=0)
    tc = tl.where(em, tc, 2147483647)
    ge = (t[:, None] >= tc[None, :]).to(tl.int32)
    eid = tl.sum(ge, axis=1)
    eid = eid - (eid > (E - 1)).to(tl.int32)
    tl.store(EIDO + t, eid, mask=tm)
    tl.store(RBEGO + t, tl.load(RBEG + eid), mask=tm)
    tl.store(TNUMO + t, tl.load(TNUM + eid), mask=tm)
    tl.store(TCUMO + t, tl.load(TCUM + eid), mask=tm)


def make_counts(r, E, M):
    # the sandbox refuses every host-side reduction and prefix sum, so the histogram
    # has to be a kernel.  E_PAD/BLOCK/TB are fixed literals
    # so hist/pfx/tile each compile exactly once for the whole run.
    counts = torch.zeros(E, dtype=torch.int32, device=DEV)
    hist_kernel[((M + 127) // 128,)](r, counts, M, E, E_PAD=256, BLOCK=128,
                                     num_warps=8, num_stages=1)
    return counts


def make_meta(counts, E, BM, M):
    rbeg = torch.empty(E, dtype=torch.int32, device=DEV)
    tnum = torch.empty(E, dtype=torch.int32, device=DEV)
    tcum = torch.empty(E, dtype=torch.int32, device=DEV)
    ntil = torch.empty(E, dtype=torch.int32, device=DEV)
    pfx_kernel[(1,)](counts, rbeg, tnum, tcum, ntil, E, BM, E_PAD=256,
                     num_warps=4, num_stages=1)
    maxt = M // BM + E + 1
    eids = torch.empty(maxt, dtype=torch.int32, device=DEV)
    rbt = torch.empty(maxt, dtype=torch.int32, device=DEV)
    tnt = torch.empty(maxt, dtype=torch.int32, device=DEV)
    tct = torch.empty(maxt, dtype=torch.int32, device=DEV)
    tile_kernel[((maxt + 63) // 64,)](rbeg, tnum, tcum, eids, rbt, tnt, tct, maxt, E,
                                      E_PAD=256, TB=64, num_warps=8, num_stages=1)
    return (eids, counts, rbt, tnt, tct, ntil)


def build(sh, am):
    T, H, E, I, KT = sh
    M = T * KT
    if am == 2:
        asrc = u8(T, H)
    else:
        asrc = u8(M, H)
    a_s = f32vec(M)
    bnorm = f32vec(E)
    bgu = u8(E * 2 * I, H)
    b_sc = f32vec(E * 2 * I)
    w = f32vec(M)
    order = torch.randint(0, T * KT, (M,), dtype=torch.int32, device=DEV)
    act = torch.empty((M, I), dtype=torch.uint8, device=DEV)
    scl = f32vec(M)
    bdn = u8(E * H, I)
    b_dn_sc = f32vec(E * H)
    # phase 9 only.  bdnp is shared three ways -- the int6 hi4 plane, the int4 plane and
    # tl.dot_scaled's mxfp4 B are all [E*H, I/2] uint8.  On c9 the four new buffers are
    # 1.07 + 0.54 + 0.02 + 0.07 GB on top of the 6.6 GB the baselines already hold.
    bdnp = u8p(E * H, I // 2)
    bl2 = u8p(E * H, I // 4)
    bs6 = u8e(E * H, I // 128)
    bdns = u8e(E * H, I // 32)
    adns = u8e(T * KT, I // 32)
    cout = torch.empty((M, H), dtype=torch.uint8, device=DEV)
    cscl = torch.empty((M, H // 64), dtype=torch.float32, device=DEV)
    rid = torch.randint(0, E, (M,), dtype=torch.int32, device=DEV)
    counts = make_counts(rid, E, M)
    return {"T": T, "H": H, "E": E, "I": I, "KT": KT, "M": M, "AM": am,
            "asrc": asrc, "af8": f8(asrc), "a_s": a_s, "bnorm": bnorm,
            "bgu": bgu, "bguf8": f8(bgu), "b_sc": b_sc, "w": w, "order": order,
            "act": act, "actf8": f8(act), "scl": scl,
            "bdn": bdn, "bdnf8": f8(bdn), "b_dn_sc": b_dn_sc,
            "bdnp": bdnp, "bl2": bl2, "bs6": bs6, "bdns": bdns, "adns": adns,
            "cout": cout, "coutf8": f8(cout), "cscl": cscl,
            "counts": counts, "meta": {}}


def meta_for(bufs, BM):
    cache = bufs["meta"]
    if BM not in cache:
        cache[BM] = make_meta(bufs["counts"], bufs["E"], BM, bufs["M"])
    return cache[BM]


def f8(t):
    return triton.reinterpret(t, tl.float8e4nv)


def md_ms(bufs, cfg, tanh):
    BM, BN, BK, GM, NW, NS, ONS, OFL, AM = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], cfg[8], cfg[9]
    H, I, M, KT = bufs["H"], bufs["I"], bufs["M"], bufs["KT"]
    eids, cnts, rbeg, tnum, tcum, ntil = meta_for(bufs, BM)
    bgu = bufs["bgu"]
    bd = TensorDescriptor(bufs["bguf8"], bgu.shape, bgu.stride(), [2 * BN, BK])
    act = bufs["act"]
    actf8 = bufs["actf8"]
    ad = TensorDescriptor(actf8, act.shape, act.stride(), [BM, BN])
    src = bufs["asrc"]
    aptr = bufs["af8"]
    sam = src.stride(0)
    sak = src.stride(1)
    if AM == 0:
        adesc = TensorDescriptor(aptr, src.shape, src.stride(), [BM, BK])
    else:
        adesc = aptr

    def go():
        md_kernel[(NSM,)](
            aptr, adesc, bufs["a_s"], bufs["bnorm"], bd, bufs["b_sc"], bufs["w"],
            bufs["order"], actf8, ad, bufs["scl"],
            eids, cnts, rbeg, tnum, tcum, ntil,
            M, I, H, sam, sak, act.stride(0), act.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, KTOP=KT,
            AMODE=AM, ONS=ONS, OFLAT=OFL, TANH=tanh, num_warps=NW, num_stages=NS)

    return triton.testing.do_bench(go, warmup=WU, rep=RP)


def dn_ms(bufs, cfg):
    BM, BN, BK, GM, NW, NS, ONS, FL = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], cfg[8]
    GRID = cfg[9]
    H, I, M = bufs["H"], bufs["I"], bufs["M"]
    eids, cnts, rbeg, tnum, tcum, ntil = meta_for(bufs, BM)
    act = bufs["act"]
    bdn = bufs["bdn"]
    cout = bufs["cout"]
    ad = TensorDescriptor(bufs["actf8"], act.shape, act.stride(), [BM, BK])
    bd = TensorDescriptor(bufs["bdnf8"], bdn.shape, bdn.stride(), [BN, BK])
    coutf8 = bufs["coutf8"]

    def go():
        dn_kernel[(GRID,)](
            ad, bufs["scl"], bd, bufs["b_dn_sc"], coutf8, bufs["cscl"],
            eids, cnts, rbeg, tnum, tcum, ntil,
            M, H, I, cout.stride(0), cout.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            ONS=ONS, FLAT=FL, num_warps=NW, num_stages=NS)

    return triton.testing.do_bench(go, warmup=WU, rep=RP)


def dn_lp_ms(bufs, cfg, ns, dsmode):
    BM, BN, BK, GM, NW, FL = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[8]
    GRID, BMODE = cfg[9], cfg[10]
    H, I, M = bufs["H"], bufs["I"], bufs["M"]
    eids, cnts, rbeg, tnum, tcum, ntil = meta_for(bufs, BM)
    act = bufs["act"]
    bdn = bufs["bdn"]
    bdnp = bufs["bdnp"]
    cout = bufs["cout"]
    ad = TensorDescriptor(bufs["actf8"], act.shape, act.stride(), [BM, BK])
    aq = TensorDescriptor(bufs["actf8"], act.shape, act.stride(), [BM, 32])
    bd = TensorDescriptor(bufs["bdnf8"], bdn.shape, bdn.stride(), [BN, BK])
    bq = TensorDescriptor(bufs["bdnf8"], bdn.shape, bdn.stride(), [BN, 32])
    bp = TensorDescriptor(bdnp, bdnp.shape, bdnp.stride(), [BN, BK // 2])
    coutf8 = bufs["coutf8"]

    def go():
        dn_lp_kernel[(GRID,)](
            ad, aq, bufs["scl"], bufs["adns"], bd, bq, bp, bufs["bdns"],
            bdnp, bufs["bl2"], bufs["bs6"], bufs["b_dn_sc"], coutf8, bufs["cscl"],
            eids, cnts, rbeg, tnum, tcum, ntil,
            M, H, I, cout.stride(0), cout.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            ONS=None, FLAT=FL, BMODE=BMODE, DSMODE=dsmode,
            num_warps=NW, num_stages=ns)

    return triton.testing.do_bench(go, warmup=WU, rep=RP)


def dn_lp_cfg(I):
    # slots 1..9 mirror dn_sweep's tuple exactly (BM, BN, BK, GROUP_M, num_warps,
    # num_stages, outer-num_stages, flatten, grid) so nothing but BMODE (slot 10) moves.
    # Order is deliberate: the ceiling and the real candidate run first and the control
    # that makes them readable runs third, so a timeout kill still leaves the verdict.
    b = dn_base_cfg(I)
    gm, ns, fl = b[4], b[6], b[8]
    return (("b_ceil", 128, 256, 128, gm, 8, ns, None, fl, NSM, 0),
            ("b_int6", 128, 256, 128, gm, 8, ns, None, fl, NSM, 1),
            ("b_q32", 128, 256, 128, gm, 8, ns, None, fl, NSM, 7),
            ("nat", 128, 256, 128, gm, 8, ns, None, fl, NSM, 6),
            ("b_int4", 128, 256, 128, gm, 8, ns, None, fl, NSM, 2),
            ("b_dotsc", 128, 256, 128, gm, 8, 3, None, fl, NSM, 5))


def lp_bytes(bmode, dw):
    # HBM bytes this variant reads for B over the whole kernel.  tiles/expert == 1 on
    # c9/c10 so every weight byte is touched exactly once and these are real, not bounds.
    #   b_ceil  49/64            = 0.765625  (exactly the int6 g128 ratio, by construction)
    #   b_int6  1/2 + 1/4 + 1/128 = 0.757812  (hi4 plane + lo2 plane + one scale byte/128)
    #   b_int4  1/2              = 0.5
    #   b_dotsc 1/2 + 1/32       = 0.53125   (mxfp4 + one e8m0 byte per 32)
    if bmode == 6:
        return dw
    if bmode == 7:
        return dw
    if bmode == 0:
        return dw * 0.765625
    if bmode == 1:
        return dw * 0.7578125
    if bmode == 5:
        return dw * 0.53125
    return dw * 0.5


def lp_ops(bmode):
    # decode ALU ops per DECODED element (uint8 lanes; the GPU has no 8-bit ALU so each is
    # one instruction on one element).  joins are counted separately in the report.
    # A [BLOCK_N,32] chunk is 32 elements/thread and one k-step is four such chunks, so
    # 1 op/element over the whole k-step is 128 instructions/thread = 8 warp-instructions
    # per scheduler-cycle-group = ~256 cycles, against ~1024 cycles for one k-step of MMA.
    # int6 at 10 ops/element is therefore ~2.5x the MMA time if none of it overlaps, which
    # is exactly the quantity b_ceil - b_int6 measures.
    if bmode == 1:
        return 10.0
    if bmode == 2:
        return 6.0
    return 0.0


def lp_joins(bmode):
    if bmode == 1:
        return 3.0
    if bmode == 2:
        return 1.0
    return 0.0


def dot_scaled_probe():
    # zero-compile capability probe.  hasattr() is not in the sandbox's builtin whitelist,
    # inspect cannot be imported (import whitelist), and every route to a signature object
    # (__wrapped__ / __doc__ / _semantic) starts with an underscore, which the sandbox
    # rejects STATICALLY -- that would kill the whole run, not just the probe.  So:
    # attribute access in a try/except proves existence, a no-argument call makes the
    # @builtin guard identify itself, and the real signature is read off the compile error
    # of the b_dotsc variant if the call form turns out to be wrong.
    ok = 0
    try:
        probe_fn = tl.dot_scaled
        ok = 1
    except Exception as e:
        P("  cap tl.dot_scaled ABSENT: %s" % ("%s" % e)[:160])
    if ok == 1:
        P("  cap tl.dot_scaled PRESENT (formats are e2m1/e4m3/e5m2/bf16/fp16 -- there is "
          "NO 6-bit microscaling format, so b_dotsc is a 4-bit reference only)")
        try:
            tl.dot_scaled()
            P("  cap dot_scaled() no-arg call did not raise (not a @builtin?)")
        except Exception as e:
            m = "%s" % e
            P("  cap dot_scaled() guard: %s" % m[:220])
    return ok


def try_lp(i, bufs, c):
    # one retry per variant, and every failure prints both ends of the message: a Triton
    # rejection is the cheapest signature/limit documentation available, but the sandbox
    # discards over-long stdout wholesale, so it is clipped to 2 x 150 characters.
    ns = c[6]
    dsm = 0
    ms = 0.0
    try:
        ms = dn_lp_ms(bufs, c, ns, dsm)
    except Exception as e:
        m = "%s" % e
        P("  c%-2d dn %-7s REJECTED ns=%d dsm=%d: %s ||| %s"
          % (i + 1, c[0], ns, dsm, m[:150], m[-150:]))
        if c[10] == 5:
            dsm = 1
        elif ns > 2:
            ns = 2
        else:
            ns = 1
        try:
            ms = dn_lp_ms(bufs, c, ns, dsm)
            P("  c%-2d dn %-7s retry ns=%d dsm=%d COMPILED" % (i + 1, c[0], ns, dsm))
        except Exception as e2:
            m2 = "%s" % e2
            P("  c%-2d dn %-7s REJECTED retry ns=%d dsm=%d: %s ||| %s"
              % (i + 1, c[0], ns, dsm, m2[:150], m2[-150:]))
            ms = 0.0
    return ms


def phase9(i, bufs, dbase):
    # phase 9: weight low-precision STORAGE on dn, at int6 -- the only compression the
    # accuracy study clears (int6 g64 +0.29 dB / g128 -0.02 dB vs fp8, against a +1.12 dB
    # SQNR margin; int4 g64 is -8.88 dB and outlier isolation only recovers ~2 dB of it).
    # c9 dn reads 2.147 GB of B against a 0.818 ms kernel, so cutting B to 49/64 is worth
    # ~0.19 ms on dn alone and ~0.5 ms once md follows.  Throughput only: random nibbles,
    # per-row scales folded into B_SCALE, group scale folded into the fp8 exponent.
    E, H, I = bufs["E"], bufs["H"], bufs["I"]
    dw = wb_dn(E, I, H)
    ds = dot_scaled_probe()
    P("  c%-2d dn base %.4f | B fp8 %.3f GB -> int6 g128 %.3f GB (49/64) -> int4 %.3f GB | base BW %.2f TB/s (%.1f%% of 3.35)"
      % (i + 1, dbase, dw / 1e9, dw * 0.765625 / 1e9, dw / 2e9,
         bwtbs(dw, dbase), bwtbs(dw, dbase) / 3.35 * 100.0))
    res = []
    for c in dn_lp_cfg(I):
        if c[10] == 5 and ds == 0:
            P("  c%-2d dn %-7s SKIPPED: tl.dot_scaled is not on this Triton" % (i + 1, c[0]))
            continue
        ms = try_lp(i, bufs, c)
        if ms > 0.0:
            nb = lp_bytes(c[10], dw)
            P("  c%-2d dn %-7s %.4f %+.1f%%  B %.3f GB (%.1f%%)  decode %.0f op/elem + %.0f join  BW %.2f TB/s (%.1f%%)"
              % (i + 1, c[0], ms, (dbase - ms) / dbase * 100.0, nb / 1e9, nb / dw * 100.0,
                 lp_ops(c[10]), lp_joins(c[10]),
                 bwtbs(nb, ms), bwtbs(nb, ms) / 3.35 * 100.0))
            res.append((c[0], ms))
    nat = 0.0
    q32 = 0.0
    ceil = 0.0
    i6 = 0.0
    i4 = 0.0
    dsc = 0.0
    for r in res:
        if r[0] == "nat":
            nat = r[1]
        if r[0] == "b_q32":
            q32 = r[1]
        if r[0] == "b_ceil":
            ceil = r[1]
        if r[0] == "b_int6":
            i6 = r[1]
        if r[0] == "b_int4":
            i4 = r[1]
        if r[0] == "b_dotsc":
            dsc = r[1]
    natg = 0.0
    if nat > 0.0:
        natg = (dbase - nat) / dbase * 100.0
    q32g = 0.0
    if q32 > 0.0:
        q32g = (dbase - q32) / dbase * 100.0
    ceilg = 0.0
    if ceil > 0.0:
        ceilg = (dbase - ceil) / dbase * 100.0
    i6g = 0.0
    if i6 > 0.0:
        i6g = (dbase - i6) / dbase * 100.0
    i4g = 0.0
    if i4 > 0.0:
        i4g = (dbase - i4) / dbase * 100.0
    dscg = 0.0
    if dsc > 0.0:
        dscg = (dbase - dsc) / dbase * 100.0
    rec = 0.0
    if ceilg > 0.01:
        rec = i6g / ceilg * 100.0
    P("D c%-2d dn base %.4f | nat %.4f %+.1f%% (kernel control, want ~0) | b_q32 %.4f %+.1f%% (cost of the 4xK32 split alone) | b_ceil %.4f %+.1f%% (bytes x0.766, decode free = CEILING)"
      % (i + 1, dbase, nat, natg, q32, q32g, ceil, ceilg))
    if i6 == 0.0:
        P("D c%-2d b_int6 was REJECTED by the compiler -- read the message above, no "
          "throughput verdict from this run" % (i + 1))
    else:
        call = "NO - the decode eats the bytes it saves; int6 weight storage stays off"
        if ceilg < 8.0:
            call = "NO - even a free decode is worth <8%: dn is not weight-byte bound enough"
        if i6g > 8.0:
            call = "YES - int6 weight storage pays; port to the real dn, then md"
        P("D c%-2d b_int6 %.4f %+.1f%% = %.0f%% of the ceiling | b_int4 %+.1f%% (ALU lower bound, 6 vs 10 op/elem, NOT accuracy-viable) | b_dotsc %+.1f%% (4-bit only) | dn saving %.3f ms, c9 md+dn ~%.3f ms | >8%% gate: %s"
          % (i + 1, i6, i6g, rec, i4g, dscg, dbase - i6, (dbase - i6) * 3.0, call))
    return []


def md_pf_ms(bufs, cfg):
    BM, BN, BK, GM, NW, NS, AM, PFB = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], cfg[8]
    H, I, M, KT = bufs["H"], bufs["I"], bufs["M"], bufs["KT"]
    eids, cnts, rbeg, tnum, tcum, ntil = meta_for(bufs, BM)
    bgu = bufs["bgu"]
    bd = TensorDescriptor(bufs["bguf8"], bgu.shape, bgu.stride(), [2 * BN, BK])
    act = bufs["act"]
    actf8 = bufs["actf8"]
    ad = TensorDescriptor(actf8, act.shape, act.stride(), [BM, BN])
    src = bufs["asrc"]
    aptr = bufs["af8"]
    sam = src.stride(0)
    sak = src.stride(1)

    def go():
        md_pf_kernel[(NSM,)](
            aptr, bufs["a_s"], bufs["bnorm"], bd, bufs["b_sc"], bufs["w"],
            bufs["order"], actf8, ad, bufs["scl"],
            eids, cnts, rbeg, tnum, tcum, ntil,
            M, I, H, sam, sak, act.stride(0), act.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, KTOP=KT,
            AMODE=AM, PFB=PFB, num_warps=NW, num_stages=NS)

    return triton.testing.do_bench(go, warmup=WU, rep=RP)


def md_base_cfg(idx, H):
    gm = 8 if H <= 1024 else 32
    if idx in ga_set():
        am = 2
    elif H <= 2048:
        am = 0
    else:
        am = 1
    return ("base", 128, 128, 128, gm, 8, 3, 2, False, am)


def md_probe(idx, H):
    # phase 6 variant 1: compiler cross-tile pipeline.  dn gets this via flatten=FLAT and
    # loses 15-17% without it; md never had it because flatten + an explicit outer
    # num_stages blows up the real kernel, so ONS must be None here (attribute unset),
    # not 1.  Everything else is the baseline config.
    b = md_base_cfg(idx, H)
    gm, am = b[4], b[9]
    return [("mdflat", 128, 128, 128, gm, 8, 3, None, True, am)]


def md_pf_probe(idx, H):
    # variants 2 and 3: hand-rolled cross-tile prefetch.  Register budget per CTA
    # (256 threads x 255 regs = 261KB):
    #   acc [128,256] fp32                        131072 B  (128 reg/thread)
    #   a_nx [128,128] fp8 carried                 16384 B  ( 16 reg/thread)
    #   b_nx [256,128] fp8 carried, PFB=1 only     32768 B  ( 32 reg/thread)
    #   inner a tile [128,128] fp8                 16384 B  ( 16 reg/thread)
    # mdpf ~ 224 reg/thread, mdpfa ~ 192 reg/thread; the inner B tile costs no registers
    # because a descriptor load feeding dot stays in shared memory.
    b = md_base_cfg(idx, H)
    gm, am = b[4], b[9]
    return [("mdpf", 128, 128, 128, gm, 8, 3, am, 1),
            ("mdpfa", 128, 128, 128, gm, 8, 3, am, 0)]


def bwtbs(nb, ms):
    return nb / (ms * 1e-3) / 1e12


def wb_md(E, I, H):
    return 1.0 * E * 2 * I * H


def wb_dn(E, I, H):
    return 1.0 * E * H * I


def smem_md(BK, NS, BM, BN):
    return (BM * BK + 2 * BN * BK) * NS // 1024 + 16


def smem_dn(BK, NS, BM, BN):
    return (BM * BK + BN * BK) * NS // 1024


def md_probe7(idx, H):
    # phase 7: deep pipeline on the memory-bound shapes.  Halving BLOCK_K halves the
    # per-stage smem, which buys num_stages and therefore TMA requests in flight.
    # md baseline is BK=128 ns=3 = 144KB + 16KB TMA-store staging = 160KB, so k64s6 is
    # the SAME 160KB with twice the stages -> a clean constant-smem control, and k64s8
    # (208KB) is the same footprint md would need for BK=128 ns=4.
    b = md_base_cfg(idx, H)
    gm, am = b[4], b[9]
    return [("k64s6", 128, 128, 64, gm, 8, 6, 2, False, am),
            ("k64s8", 128, 128, 64, gm, 8, 8, 2, False, am)]


def dn_probe7(I):
    # dn baseline is BK=128 ns=4 = 192KB (no TMA store), so k64s8 is the constant-smem
    # control at twice the stages and k64s6 (144KB) trades smem back for occupancy.
    b = dn_base_cfg(I)
    gm, fl = b[4], b[8]
    return [("k64s6", 128, 256, 64, gm, 8, 6, None, fl, NSM),
            ("k64s8", 128, 256, 64, gm, 8, 8, None, fl, NSM)]


def dn_base_cfg(I):
    gm = 4 if I == 8192 else 32
    ns = 3 if I == 14336 else 4
    return ("base", 128, 256, 128, gm, 8, ns, None, I <= 2560, NSM)


def gm_alts(gm, tnum):
    # tl.swizzle2d groups along the per-expert tile axis, whose length is t_num,
    # so the effective group size is min(GROUP_M, t_num): every GROUP_M above
    # t_num is the same kernel.  Only enumerate values that actually differ.
    eb = gm
    if tnum < eb:
        eb = tnum
    seen = [eb]
    out = []
    for v in (1, 2, 4, 8, 16, 32, 64):
        ev = v
        if tnum < ev:
            ev = tnum
        if ev not in seen:
            seen.append(ev)
            out.append(v)
    return out[:3]


def md_sweep(idx, H, tnum):
    b = md_base_cfg(idx, H)
    gm, am = b[4], b[9]
    c = [("s2", 128, 128, 128, gm, 8, 2, 2, False, am),
         ("s4", 128, 128, 128, gm, 8, 4, 2, False, am),
         ("k64", 128, 128, 64, gm, 8, 3, 2, False, am),
         ("k64s5", 128, 128, 64, gm, 8, 5, 2, False, am),
         ("o1", 128, 128, 128, gm, 8, 3, 1, False, am),
         ("o3", 128, 128, 128, gm, 8, 3, 3, False, am),
         ("ofl", 128, 128, 128, gm, 8, 3, 1, True, am)]
    for v in gm_alts(gm, tnum):
        c.append(("g%d" % v, 128, 128, 128, v, 8, 3, 2, False, am))
    c.append(("n64", 128, 64, 128, gm, 8, 3, 2, False, am))
    c.append(("m64", 64, 128, 128, gm, 8, 3, 2, False, am))
    c.append(("n64w4", 128, 64, 128, gm, 4, 3, 2, False, am))
    return c


def dn_sweep(I, tnum):
    b = dn_base_cfg(I)
    gm, ns, fl = b[4], b[6], b[8]
    alt = 3 if ns == 4 else 4
    c = [("s2", 128, 256, 128, gm, 8, 2, None, fl, NSM),
         ("s%d" % alt, 128, 256, 128, gm, 8, alt, None, fl, NSM),
         ("k64", 128, 256, 64, gm, 8, ns, None, fl, NSM),
         ("k64s5", 128, 256, 64, gm, 8, 5, None, fl, NSM)]
    for v in gm_alts(gm, tnum):
        c.append(("g%d" % v, 128, 256, 128, v, 8, ns, None, fl, NSM))
    c.append(("m256", 256, 128, 128, gm, 8, ns, None, fl, NSM))
    c.append(("flx", 128, 256, 128, gm, 8, ns, None, not fl, NSM))
    return c


def dn_probe(I):
    # phase 5: decide between "L2->SMEM operand bandwidth" (A) and "clock / issue
    # limited" (B) for the global 1.19 derate.  Halving BN drops the arithmetic
    # intensity 2*BM*BN/(BM+BN) from 170.7 to 128 (operand traffic x1.333) and
    # doubles the tile count; A pays both, B pays only the tile count.
    #   n128         [128,128] acc = 64KB, 8 warps, the A/B discriminator
    #   n128w4       same acc on 128 threads = 128 reg/thread, 1 CTA/SM control
    #   n128w4s2g264 num_stages=2 -> 2*(128*128 + 128*128) = 64KB smem, so two
    #                CTAs really co-reside; grid 264 = 2 per SM.  A big win over
    #                n128w4 means CTA A's epilogue overlaps CTA B's MMA => B.
    b = dn_base_cfg(I)
    gm, ns, fl = b[4], b[6], b[8]
    return [("n128", 128, 128, 128, gm, 8, ns, None, fl, NSM),
            ("n128w4", 128, 128, 128, gm, 4, ns, None, fl, NSM),
            ("n128w4s2g264", 128, 128, 128, gm, 4, 2, None, fl, 264)]


def phase_slice(lst, lo, hi):
    return lst[lo:hi]


def tf(flops, ms):
    return flops / (ms * 1e-3) / 1e12


def probe():
    # one-shot map of the remaining unverified host API, so a single run pins down
    # any further ban instead of costing another slot per discovery.
    z = torch.zeros(8, dtype=torch.int32, device=DEV)
    P("p1 zeros ok")
    t = torch.empty((256, 128), dtype=torch.uint8, device=DEV)
    P("p2 empty ok")
    P("p3 stride/shape ok %d %d" % (t.stride(0), t.shape[0]))
    tf = f8(t)
    P("p4 reinterpret ok")
    d = TensorDescriptor(tf, t.shape, t.stride(), [128, 128])
    P("p5 descriptor ok")
    return z, d


def run_baselines(idxs, store):
    P("== BASELINE (peak fp8 = 1979 TF) ==")
    shs = shape_list()
    rp = real_pk()
    for i in idxs:
        sh = shs[i]
        T, H, E, I, KT = sh
        mb = md_base_cfg(i, H)
        bufs = build(sh, mb[9])
        if i == idxs[0]:
            P("p6 build+meta kernels ok")
        mf = 4.0 * T * KT * H * I
        df = 2.0 * T * KT * H * I
        db = dn_base_cfg(I)
        m = md_ms(bufs, mb, 0)
        d = dn_ms(bufs, db)
        mix = tf(mf + df, m + d) / PEAK * 100.0
        P("B c%-2d T%d H%d E%d I%d k%d A%d | md %.3fms %.0fTF %.1f%% | dn %.3fms %.0fTF %.1f%% | mix %.1f%% real %.1f%%"
          % (i + 1, T, H, E, I, KT, mb[9], m, tf(mf, m), tf(mf, m) / PEAK * 100.0,
             d, tf(df, d), tf(df, d) / PEAK * 100.0, mix, rp[i]))
        store[i] = (0, bufs, m, d, mf, df, 0.0, 0.0)


def run_sweep(i, store, phases):
    if i not in store:
        return
    v0 = store[i]
    bufs, mbase, dbase, mf, df = v0[1], v0[2], v0[3], v0[4], v0[5]
    H, I = bufs["H"], bufs["I"]
    tnum = ((bufs["M"] // bufs["E"]) + 127) // 128
    if tnum < 1:
        tnum = 1
    mcfg = md_sweep(i, H, tnum)
    dcfg = dn_sweep(I, tnum)
    mres = []
    dres = []
    if 1 in phases:
        mres = mres + sweep_group(i, bufs, phase_slice(mcfg, 0, 7), mbase, 1)
    if 2 in phases:
        mres = mres + sweep_group(i, bufs, phase_slice(mcfg, 7, 99), mbase, 1)
    if 3 in phases:
        dres = dres + sweep_group(i, bufs, phase_slice(dcfg, 0, 5), dbase, 0)
    if 4 in phases:
        dres = dres + sweep_group(i, bufs, phase_slice(dcfg, 5, 99), dbase, 0)
    if 5 in phases:
        dres = dres + phase5(i, bufs, dbase)
    if 6 in phases:
        mres = mres + phase6(i, bufs, mbase)
    if 7 in phases:
        r7 = phase7(i, bufs, mbase, dbase)
        mres = mres + r7[0]
        dres = dres + r7[1]
    if 8 in phases:
        mres = mres + phase8(i, bufs, mbase)
    if 9 in phases:
        dres = dres + phase9(i, bufs, dbase)
    if 10 in phases:
        r10 = phase10(i, bufs, mbase, dbase)
        mres = mres + r10[0]
        dres = dres + r10[1]
    bm = top3(i, "md", mres, mbase)
    bd = top3(i, "dn", dres, dbase)
    store[i] = (1, bufs, mbase, dbase, mf, df, bm, bd)


def phase5(i, bufs, dbase):
    res = sweep_group(i, bufs, dn_probe(bufs["I"]), dbase, 0)
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    for r in res:
        if r[0] == "n128":
            r0 = r[1]
        if r[0] == "n128w4":
            r1 = r[1]
        if r[0] == "n128w4s2g264":
            r2 = r[1]
    if r0 > 0.0:
        # ratio = n128/base.  Under B: ratio = 1 + y.  Under A: ratio = 1.3333 + 0.6667*y,
        # where y is the per-tile fixed-cost fraction of the baseline.  Compare the two
        # implied y against the calibration; a ratio below 1.333 falsifies A outright.
        ratio = r0 / dbase
        yb = ratio - 1.0
        ya = (ratio - 1.33333) / 0.66667
        verdict = ""
        if ratio < 1.33333:
            verdict = "  A FALSIFIED (yA<0)"
        if yb > 1.0:
            verdict = "  B FALSIFIED (yB>1)"
        P("D c%-2d n128 ratio %.3f | implied fixed-frac yB %.3f yA %.3f | A needs ratio>=1.333%s"
          % (i + 1, ratio, yb, ya, verdict))
    if r1 > 0.0 and r2 > 0.0:
        P("D c%-2d 2CTA/SM %.4f -> %.4f = %+.1f%% | big gain => B (epilogue/MMA overlap real)"
          % (i + 1, r1, r2, (r1 - r2) / r1 * 100.0))
    return res


def phase6(i, bufs, mbase):
    H = bufs["H"]
    res = sweep_group(i, bufs, md_probe(i, H), mbase, 1)
    am = md_base_cfg(i, H)[9]
    if am == 0:
        P("  c%-2d md pf skipped: baseline uses TMA-A, prefetch variants are pointer-A only"
          % (i + 1))
    else:
        for c in md_pf_probe(i, H):
            ms = md_pf_ms(bufs, c)
            P("  c%-2d md %-6s %.4f %+.1f%%"
              % (i + 1, c[0], ms, (mbase - ms) / mbase * 100.0))
            res.append((c[0], ms))
    bestn = ""
    bestg = -1e30
    for r in res:
        g = (mbase - r[1]) / mbase * 100.0
        if g > bestg:
            bestg = g
            bestn = r[0]
    call = "NO - stay off this line"
    if bestg > 3.0:
        call = "YES - port to the real kernel"
    P("D c%-2d md base %.4f | best cross-tile pipeline %s %+.1f%% | >3%% gate: %s"
      % (i + 1, mbase, bestn, bestg, call))
    return res


def phase7(i, bufs, mbase, dbase):
    H, I, E, M = bufs["H"], bufs["I"], bufs["E"], bufs["M"]
    mw = wb_md(E, I, H)
    dw = wb_dn(E, I, H)
    tpe = ((M // E) + 127) // 128
    # the weight-bytes/time readout is only a true HBM number when each expert owns about
    # one row-tile, otherwise B is re-read once per m-tile and the ratio overstates traffic.
    note = "(B re-read per m-tile: BW is an upper bound)"
    if tpe < 2:
        note = "(weights read once: BW figure is real)"
    P("  c%-2d weights md %.2f GB dn %.2f GB | tiles/expert %d %s"
      % (i + 1, mw / 1e9, dw / 1e9, tpe, note))
    P("  c%-2d md base   %.4f  BW %.2f TB/s (%.1f%% of 3.35) sm%dKB | dn base   %.4f  BW %.2f TB/s (%.1f%%) sm%dKB"
      % (i + 1, mbase, bwtbs(mw, mbase), bwtbs(mw, mbase) / 3.35 * 100.0, smem_md(128, 3, 128, 128),
         dbase, bwtbs(dw, dbase), bwtbs(dw, dbase) / 3.35 * 100.0, smem_dn(128, 4, 128, 256)))
    mr = []
    dr = []
    bm = mbase
    bd = dbase
    for c in md_probe7(i, H):
        ms = md_ms(bufs, c, 0)
        P("  c%-2d md %-6s %.4f %+.1f%%  BW %.2f TB/s (%.1f%% of 3.35) sm%dKB"
          % (i + 1, c[0], ms, (mbase - ms) / mbase * 100.0,
             bwtbs(mw, ms), bwtbs(mw, ms) / 3.35 * 100.0, smem_md(c[3], c[6], c[1], c[2])))
        mr.append((c[0], ms))
        if ms < bm:
            bm = ms
    for c in dn_probe7(I):
        ms = dn_ms(bufs, c)
        P("  c%-2d dn %-6s %.4f %+.1f%%  BW %.2f TB/s (%.1f%% of 3.35) sm%dKB"
          % (i + 1, c[0], ms, (dbase - ms) / dbase * 100.0,
             bwtbs(dw, ms), bwtbs(dw, ms) / 3.35 * 100.0, smem_dn(c[3], c[6], c[1], c[2])))
        dr.append((c[0], ms))
        if ms < bd:
            bd = ms
    gmd = (mbase - bm) / mbase * 100.0
    gdn = (dbase - bd) / dbase * 100.0
    call = "NO - deep pipeline does not buy HBM efficiency here"
    if gmd > 3.0:
        call = "YES - pure config change, port BLOCK_K/num_stages to the real md"
    if gdn > 3.0:
        call = "YES - pure config change, port BLOCK_K/num_stages to the real dn"
    if gmd > 3.0:
        if gdn > 3.0:
            call = "YES - both md and dn, pure config change"
    P("D c%-2d deep-pipe best: md %+.1f%% (BW %.1f%% -> %.1f%%) dn %+.1f%% (BW %.1f%% -> %.1f%%) | >3%% gate: %s"
      % (i + 1, gmd, bwtbs(mw, mbase) / 3.35 * 100.0, bwtbs(mw, bm) / 3.35 * 100.0,
         gdn, bwtbs(dw, dbase) / 3.35 * 100.0, bwtbs(dw, bd) / 3.35 * 100.0, call))
    return mr, dr


def phase8(i, bufs, mbase):
    # phase 8: is the SwiGLU tanh actually worth attacking, and does f16x2 pay?
    # tanhoff runs FIRST because it always compiles: even if tanh.approx.f16x2 is
    # rejected, the tanh share of md is still measured and that alone sizes the ceiling
    # of this whole line of attack.
    H = bufs["H"]
    mb = md_base_cfg(i, H)
    off = md_ms(bufs, mb, 2)
    P("  c%-2d md tanhoff %.4f %+.1f%%  (tanh removed, arithmetic-only control)"
      % (i + 1, off, (mbase - off) / mbase * 100.0))
    f16 = md_ms(bufs, mb, 1)
    P("  c%-2d md tanhf16 %.4f %+.1f%%  (tanh.approx.f16x2, pack=2)"
      % (i + 1, f16, (mbase - f16) / mbase * 100.0))
    cost = mbase - off
    share = cost / mbase * 100.0
    net = (mbase - f16) / mbase * 100.0
    rec = 0.0
    if cost > 1e-9:
        rec = (mbase - f16) / cost * 100.0
    call = "NO - f16x2 does not pay here"
    if net > 2.0:
        call = "YES - pure inline-asm swap, port it"
    if cost <= 1e-9:
        call = "NO - tanh already free, nothing to recover"
    P("D c%-2d md base %.4f | tanhoff %.4f => tanh = %.1f%% of md | tanhf16 %.4f => recovered %.1f%% of tanh, net %+.1f%% | >2%% gate: %s"
      % (i + 1, mbase, off, share, f16, rec, net, call))
    return [("tanhoff", off), ("tanhf16", f16)]


def p10_bufs(bufs):
    # lazily allocated so phases 0-9 pay nothing for them.  pin is 132*128*I bytes
    # (17 MB at I=1024) = exactly the live act footprint a fused md/dn kernel would
    # hold; wide is RMAX x act so the REP ladder touches RMAX*M*I distinct bytes
    # (1.07 GB on c4, 0.80 GB on c8, 2.15 GB on c11).
    if "pin" not in bufs:
        bufs["pin"] = torch.empty((NSM * 128, bufs["I"]), dtype=torch.uint8, device=DEV)
        bufs["wide"] = torch.empty((bufs["M"], RMAX * bufs["I"]), dtype=torch.uint8, device=DEV)
    return bufs["pin"], bufs["wide"]


def md_p10_ms(bufs, cfg, mode, rep):
    BM, BN, BK, GM, NW, NS, ONS, OFL, AM = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], cfg[8], cfg[9]
    H, I, M, KT = bufs["H"], bufs["I"], bufs["M"], bufs["KT"]
    eids, cnts, rbeg, tnum, tcum, ntil = meta_for(bufs, BM)
    pin, wide = p10_bufs(bufs)
    pinf8 = f8(pin)
    widef8 = f8(wide)
    bgu = bufs["bgu"]
    bd = TensorDescriptor(bufs["bguf8"], bgu.shape, bgu.stride(), [2 * BN, BK])
    act = bufs["act"]
    actf8 = bufs["actf8"]
    ad = TensorDescriptor(actf8, act.shape, act.stride(), [BM, BN])
    pd = TensorDescriptor(pinf8, pin.shape, pin.stride(), [BM, BN])
    wd = TensorDescriptor(widef8, wide.shape, wide.stride(), [BM, BN])
    src = bufs["asrc"]
    aptr = bufs["af8"]
    sam = src.stride(0)
    sak = src.stride(1)
    if AM == 0:
        adesc = TensorDescriptor(aptr, src.shape, src.stride(), [BM, BK])
    else:
        adesc = aptr

    def go():
        md_p10_kernel[(NSM,)](
            aptr, adesc, bufs["a_s"], bufs["bnorm"], bd, bufs["b_sc"], bufs["w"],
            bufs["order"], actf8, ad, bufs["scl"],
            pinf8, pd, widef8, wd,
            eids, cnts, rbeg, tnum, tcum, ntil,
            M, I, H, sam, sak, act.stride(0), act.stride(1),
            pin.stride(0), wide.stride(0),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM, KTOP=KT,
            AMODE=AM, ONS=ONS, OFLAT=OFL, MODE=mode, REP=rep,
            num_warps=NW, num_stages=NS)

    return triton.testing.do_bench(go, warmup=WU, rep=RP)


def dn_p10_ms(bufs, cfg, pinm):
    BM, BN, BK, GM, NW, NS, ONS, FL = cfg[1], cfg[2], cfg[3], cfg[4], cfg[5], cfg[6], cfg[7], cfg[8]
    GRID = cfg[9]
    H, I, M = bufs["H"], bufs["I"], bufs["M"]
    eids, cnts, rbeg, tnum, tcum, ntil = meta_for(bufs, BM)
    pin, wide = p10_bufs(bufs)
    act = bufs["act"]
    bdn = bufs["bdn"]
    cout = bufs["cout"]
    ad = TensorDescriptor(bufs["actf8"], act.shape, act.stride(), [BM, BK])
    pd = TensorDescriptor(f8(pin), pin.shape, pin.stride(), [BM, BK])
    bd = TensorDescriptor(bufs["bdnf8"], bdn.shape, bdn.stride(), [BN, BK])
    coutf8 = bufs["coutf8"]

    def go():
        dn_p10_kernel[(GRID,)](
            ad, pd, bufs["scl"], bd, bufs["b_dn_sc"], coutf8, bufs["cscl"],
            eids, cnts, rbeg, tnum, tcum, ntil,
            M, H, I, cout.stride(0), cout.stride(1),
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            ONS=ONS, FLAT=FL, PINM=pinm, num_warps=NW, num_stages=NS)

    return triton.testing.do_bench(go, warmup=WU, rep=RP)


def p10_reps():
    return (1, 3, 8, 16)


def md_p10_try(i, bufs, cfg, mode, rep, tag):
    # every measurement is wrapped: at REP=16 Triton may or may not reuse one TMA-store
    # staging buffer for the 16 identical stores, and 16 x 16KB would blow the 227KB
    # smem budget outright.  A rejection there must not cost the mdpin/dnpin readings,
    # which are the actual answer and are taken first.
    try:
        return md_p10_ms(bufs, cfg, mode, rep)
    except Exception as e:
        m = "%s" % e
        P("  c%-2d md %-7s REJECTED: %s ||| %s" % (i + 1, tag, m[:140], m[-140:]))
        return 0.0


def dn_p10_try(i, bufs, cfg, pinm, tag):
    try:
        return dn_p10_ms(bufs, cfg, pinm)
    except Exception as e:
        m = "%s" % e
        P("  c%-2d dn %-7s REJECTED: %s ||| %s" % (i + 1, tag, m[:140], m[-140:]))
        return 0.0


def phase10(i, bufs, mbase, dbase):
    # phase 10: IS THE act ROUND TRIP ON THE CRITICAL PATH?
    #
    # md writes act[M, I] to HBM and dn reads it straight back.  Fusing the two kernels
    # per m-tile would turn that round trip into an L2 hit (132 CTAs x 128 x I bytes live
    # = 17 MB at I=1024, well inside the 50 MB L2), which is worth 2*M*I bytes per case
    # -- 2.21 GB over c3..c12 = 0.80 ms at 2.75 TB/s.  The calibration model scores act
    # as HIDDEN, because it bills each GEMM at max(compute, bytes) and md/dn have 2-5x
    # of memory slack on these shapes.  That has never been measured on its own.  It is
    # measured here, and it decides whether the fused kernel gets written at all.
    #
    # Three readings, in the order they are printed:
    #   mode0 / pin0     structural controls, must be ~0% against the phase-0 baselines
    #   mdpin / dnpin    THE ANSWER.  base - pin = the DRAM cost of the act write and of
    #                    the act read, measured directly, with no traffic model between
    #                    the number and the conclusion.  Their sum is the fusion's
    #                    traffic upside for this case.
    #   rep1/3/8/16      the sensitivity ladder and the positive control.  rep3 is the
    #                    stated gate (act written three times); rep16 must move or the
    #                    probe is broken, since a flat 16x would imply a store rate no
    #                    H800 can deliver.
    T, E, H, I, M = bufs["T"], bufs["E"], bufs["H"], bufs["I"], bufs["M"]
    mb = md_base_cfg(i, H)
    db = dn_base_cfg(I)
    ab = 1.0 * M * I
    if mb[9] == 2:
        aby = 1.0 * T * H
    else:
        aby = 1.0 * M * H
    mdo = aby + 1.0 * E * 2 * I * H
    full = ab / 2.75e12 * 1e3
    P("  c%-2d act %.0f MB/direction | round trip %.0f MB = %.4f ms at 2.75 TB/s | md base %.4f dn base %.4f | md other bytes %.0f MB"
      % (i + 1, ab / 1e6, 2.0 * ab / 1e6, 2.0 * full, mbase, dbase, mdo / 1e6))
    m0 = md_p10_try(i, bufs, mb, 0, 1, "mode0")
    P("  c%-2d md mode0  %.4f %+.1f%%  (control: act -> ACT, must be ~0%% vs md base)"
      % (i + 1, m0, (mbase - m0) / mbase * 100.0))
    mp = md_p10_try(i, bufs, mb, 1, 1, "mdpin")
    pm = 0.0
    if m0 > 0.0:
        pm = (m0 - mp) / m0 * 100.0
    P("  c%-2d md mdpin  %.4f %+.1f%%  (act -> 17 MB private slot, act write DRAM cost = %.4f ms; full exposure would be %.4f ms)"
      % (i + 1, mp, pm, m0 - mp, full))
    d0 = dn_p10_try(i, bufs, db, 0, "pin0")
    P("  c%-2d dn pin0   %.4f %+.1f%%  (control: A -> act, must be ~0%% vs dn base)"
      % (i + 1, d0, (dbase - d0) / dbase * 100.0))
    dp = dn_p10_try(i, bufs, db, 1, "dnpin")
    pd = 0.0
    if d0 > 0.0:
        pd = (d0 - dp) / d0 * 100.0
    P("  c%-2d dn dnpin  %.4f %+.1f%%  (A -> 17 MB private slot, act read DRAM cost = %.4f ms; full exposure would be %.4f ms)"
      % (i + 1, dp, pd, d0 - dp, full))
    r1 = 0.0
    r3 = 0.0
    r16 = 0.0
    for rp in p10_reps():
        ms = md_p10_try(i, bufs, mb, 2, rp, "rep%d" % rp)
        if ms <= 0.0:
            continue
        if rp == 1:
            r1 = ms
        if rp == 3:
            r3 = ms
        if rp == 16:
            r16 = ms
        add = 0.0
        if r1 > 0.0:
            add = (ms - r1) / r1 * 100.0
        P("  c%-2d md rep%-2d  %.4f %+.1f%% vs rep1  | act bytes %.0f MB, md DRAM %.0f MB => implied %.2f TB/s (cap ~3.35)"
          % (i + 1, rp, ms, add, rp * ab / 1e6, (mdo + rp * ab) / 1e6,
             bwtbs(mdo + rp * ab, ms)))
    # the gate the plan states: act written three times instead of once.
    g3 = 0.0
    if r1 > 0.0:
        g3 = (r3 - r1) / r1 * 100.0
    g16 = 0.0
    if r1 > 0.0:
        g16 = (r16 - r1) / r1 * 100.0
    wr = 0.0
    if m0 > 0.0 and mp > 0.0:
        wr = m0 - mp
    rd = 0.0
    if d0 > 0.0 and dp > 0.0:
        rd = d0 - dp
    up = wr + rd
    ups = 0.0
    if (m0 + d0) > 0.0:
        ups = up / (m0 + d0) * 100.0
    exp = 0.0
    if full > 1e-9:
        exp = up / (2.0 * full) * 100.0
    call = "AMBIGUOUS - 3%..8%, needs a second sample or a bigger case"
    if g3 >= 8.0:
        call = "YES - act write is on the critical path, build the fused kernel"
    if g3 < 3.0:
        call = "NO - act write is hidden by compute; fusion buys only launch+prologue"
    warn = ""
    if g16 < 5.0:
        warn = "  ***PROBE SUSPECT: 16x act traffic moved nothing, check the stores are real***"
    P("D c%-2d GATE rep3 %+.1f%% (>=8%% yes / <3%% no): %s | rep16 %+.1f%% positive control%s"
      % (i + 1, g3, call, g16, warn))
    P("D c%-2d DIRECT  write %.4f ms + read %.4f ms = %.4f ms of md+dn %.4f ms (%.1f%%) | full-DRAM round trip %.4f ms => act is %.0f%% exposed"
      % (i + 1, wr, rd, up, m0 + d0, ups, 2.0 * full, exp))
    return [], []


def sweep_group(i, bufs, cfgs, base, is_md):
    out = []
    for c in cfgs:
        if is_md == 1:
            ms = md_ms(bufs, c, 0)
            tag = "md"
        else:
            ms = dn_ms(bufs, c)
            tag = "dn"
        P("  c%-2d %s %-6s %.4f %+.1f%%" % (i + 1, tag, c[0], ms, (base - ms) / base * 100.0))
        out.append((c[0], ms))
    return out


def top3(i, tag, res, base):
    rem = []
    for r in res:
        rem.append(r)
    if rem == []:
        return base
    line = ""
    best = base
    k = 0
    while k < 3:
        if rem == []:
            break
        bi = 0
        j = 0
        for r in rem:
            if r[1] < rem[bi][1]:
                bi = j
            j = j + 1
        pick = rem[bi]
        line = line + " %s %.4f %+.1f%% /" % (pick[0], pick[1],
                                              (base - pick[1]) / base * 100.0)
        if pick[1] < best:
            best = pick[1]
        rem = rem[:bi] + rem[bi + 1:]
        k = k + 1
    P("T c%-2d %s base %.4f | top3:%s" % (i + 1, tag, base, line))
    return best


def summarize(idxs, store):
    shs = shape_list()
    rp = real_pk()
    tot = 0.0
    bits = ""
    nb = 0
    for i in idxs:
        if i not in store:
            continue
        v = store[i]
        if v[0] == 0:
            continue
        bufs, mbase, dbase, mf, df, bm, bd = v[1], v[2], v[3], v[4], v[5], v[6], v[7]
        T, H, E, I, KT = shs[i]
        case_ms = (6.0 * T * KT * H * I) / (rp[i] / 100.0 * PEAK * 1e12) * 1e3
        gain = 1.0 - (bm + bd) / (mbase + dbase)
        sv = 0.75 * case_ms * gain
        tot = tot + sv
        nb = nb + 1
        bits = bits + " c%d %.3f" % (i + 1, sv)
    P("S best-config saving over %d shapes: %.3f ms total |%s" % (nb, tot, bits))


def main():
    P("gemmsweep: shapes=%s phases=%s" % (SHAPES_TO_RUN, PHASES_TO_RUN))
    idxs = [q for q in SHAPES_TO_RUN]
    phases = [q for q in PHASES_TO_RUN]
    store = {}
    probe()
    run_baselines(idxs, store)
    for i in idxs:
        run_sweep(i, store, phases)
    summarize(idxs, store)
    P("=== END ===")


main()
end_of_script_marker_raise_name_error()
