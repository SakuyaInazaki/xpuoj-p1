"""V732 (a_scale 预排) 的逐比特等价证明 + V733 (s/inv 外提) 的主机端位运算镜像。

Part 1 -- V732.  完整镜像 _fgs_t1i_mdq_kernel_g 的 tile->(offs_m, row_mask) 映射
          (含真实的 tl.swizzle2d 置换与 128 行 m 瓦片切分)，对每一条 lane 比较
              base : a_s[ (ORDER[offs_m] // KTOP) ]        , masked -> 1.0
              v732 : (a_s[order // KTOP])[ offs_m ]        , masked -> 1.0
          断言两者的 float32 位模式完全相同，并统计散射/连续读的 32B 扇区数。
          200 组随机路由 + 手工边界(空专家 / 恰好 128 行 / 只差 1 行的尾瓦片)。

Part 2 -- V733.  200 组随机 bound 上，验证主机端的「指数掩码」写法
              bits = f32.view(int32) & 0x7F800000
              s    = (bits + (10 << 23)).view(f32)
              inv  = ((244 << 23) - bits).view(f32)
          与核内的移位链
              bexp = (f32.view(int32) >> 23) & 0xFF
              s    = ((bexp + 10) << 23).view(f32)
              inv  = ((244 - bexp) << 23).view(f32)
          在 int32 环绕语义下逐比特相同(含 bexp=0/255 的溢出端)。

Run:  python3 sandbox/verify_v732.py
"""
import numpy as np

BM = 128
BN = 128


# ---------------------------------------------------------------------------
# helpers: 与内核完全同构的映射
# ---------------------------------------------------------------------------
def swizzle2d(i, j, size_i, size_j, size_g):
    """tl.swizzle2d 的标量镜像 (triton/language/standard.py)。"""
    ij = i * size_j + j
    size_gj = size_g * size_j
    group_id = ij // size_gj
    off_i = group_id * size_g
    size_g_actual = min(size_i - off_i, size_g)
    new_i = off_i + (ij % size_g_actual)
    new_j = (ij % size_gj) // size_g_actual
    return new_i, new_j


def build_meta(counts):
    """mirror of build_block_row_idx_info_kernel 的语义输出(每 m 瓦片一项)。"""
    counts = np.asarray(counts, dtype=np.int64)
    tiles_per_e = -(-counts // BM)                       # ceil
    row_off_e = np.concatenate([[0], np.cumsum(counts)[:-1]])
    expert_ids, row_begin, tile_num, tile_cum = [], [], [], []
    cum = 0
    for e, t in enumerate(tiles_per_e):
        cum += int(t)
        for _ in range(int(t)):
            expert_ids.append(e)
            row_begin.append(int(row_off_e[e]))
            tile_num.append(int(t))
            tile_cum.append(cum)
    return (np.array(expert_ids, dtype=np.int64), np.array(row_begin, dtype=np.int64),
            np.array(tile_num, dtype=np.int64), np.array(tile_cum, dtype=np.int64))


def sectors(idx):
    """把 4B 元素下标折算成不同的 32B 扇区数(= L1 wavefront 数的下界)。"""
    return len(np.unique(idx // 8))


# ---------------------------------------------------------------------------
# Part 1: V732 逐比特等价
# ---------------------------------------------------------------------------
def check_one(T, E, I, ktop, group_m, rng, force=None):
    """按真实链路构造: flat_ids -> 稳定计数排序 order/counts -> 元数据 -> 瓦片映射。"""
    M = T * ktop
    flat_ids = rng.integers(0, E, size=M).astype(np.int64)
    if force is not None:                                # 手工边界: 指定某专家的行数
        e, want = force
        idx = rng.permutation(M)
        flat_ids[idx[:want]] = e
        flat_ids[idx[want:]] = rng.integers(0, E - 1, size=M - want)
        flat_ids[idx[want:]] += (flat_ids[idx[want:]] >= e)
    order = np.argsort(flat_ids, kind="stable").astype(np.int64)   # == _counting_sort_order
    counts = np.bincount(flat_ids, minlength=E).astype(np.int64)
    a_s = np.abs(rng.standard_normal(T).astype(np.float32) * 3.0 + 5.0) + 1e-6

    expert_ids, row_begin_t, tile_num, tile_cum = build_meta(counts)
    n_m_tiles = len(expert_ids)
    num_block_n = -(-I // BN)

    a_s_sorted = a_s[order // ktop]                      # <- V732 主机端预排
    assert a_s_sorted.shape == (M,)

    scat_sec = 0
    cont_sec = 0
    covered = np.zeros(M, dtype=np.int64)
    for tile_id in range(n_m_tiles * num_block_n):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n
        expert = expert_ids[pid_m]
        n_rows = counts[expert]
        row_begin = row_begin_t[pid_m]
        t_num = tile_num[pid_m]
        t_cum = tile_cum[pid_m]
        local_m = pid_m - (t_cum - t_num)
        local_m, pid_n = swizzle2d(local_m, pid_n, t_num, num_block_n, group_m)

        offs_m = row_begin + local_m * BM + np.arange(BM, dtype=np.int64)
        row_mask = offs_m < row_begin + n_rows

        # --- baseline: ORDER 连续读 -> //KTOP -> A_SCALE 散射读 ---------------
        ordv = np.where(row_mask, order[np.clip(offs_m, 0, M - 1)], 0)
        rows = ordv // ktop
        base = np.where(row_mask, a_s[rows], np.float32(1.0)).astype(np.float32)

        # --- v732: A_SCALE 连续读 -------------------------------------------
        new = np.where(row_mask,
                       a_s_sorted[np.clip(offs_m, 0, M - 1)],
                       np.float32(1.0)).astype(np.float32)

        assert base.view(np.uint32).tolist() == new.view(np.uint32).tolist(), \
            f"bit mismatch at tile {tile_id}"

        # 未被 mask 的 lane 必须落在 [0, M) —— 连续读不会越界
        assert (offs_m[row_mask] >= 0).all() and (offs_m[row_mask] < M).all()

        scat_sec += sectors(rows[row_mask])
        cont_sec += sectors(offs_m[row_mask])
        covered[offs_m[row_mask]] += 1

    # 每一行都被覆盖，且恰好被 num_block_n 个 CTA 各算一遍(冗余倍数)
    assert (covered == num_block_n).all(), "row coverage broken"
    return scat_sec, cont_sec, n_m_tiles * num_block_n


def part1():
    rng = np.random.default_rng(20260904)
    tot_s = tot_c = tot_t = 0
    for trial in range(200):
        E = int(rng.choice([8, 16, 32, 64]))
        ktop = int(rng.choice([2, 3, 4, 8]))
        T = int(rng.choice([512, 1024, 2048]))
        I = int(rng.choice([1024, 1536, 2048, 2560]))
        group_m = int(rng.choice([8, 32]))
        force = None
        if trial % 7 == 0:
            force = (int(rng.integers(0, E)), 0)            # 空专家
        elif trial % 5 == 0:
            force = (int(rng.integers(0, E)), BM)           # 整除边界
        elif trial % 11 == 0:
            force = (int(rng.integers(0, E)), BM + 1)       # 只差 1 行的尾瓦片
        s, c, nt = check_one(T, E, I, ktop, group_m, rng, force)
        tot_s += s
        tot_c += c
        tot_t += nt
    print("Part1  200/200 真实路由(稳定计数排序 order) 逐比特一致 "
          "(含空专家/整除边界/+1 尾瓦片)")
    print(f"       A_SCALE 的 32B 扇区数: 散射 {tot_s / tot_t:.1f} / 瓦片 -> "
          f"连续 {tot_c / tot_t:.1f} / 瓦片  (削 {100.0 * (1 - tot_c / tot_s):.1f}%,"
          f" 共 {tot_t} 个瓦片)")


def part1b():
    """真实的 6 个判题形状(c3~c8, 即 _GASET 中走 _fgs_t1i_mdq_kernel_g 的 E<=96 案)。"""
    cases = [
        ("c3", 16384, 2048, 32, 2048, 4), ("c4", 16384, 2048, 32, 1024, 4),
        ("c5",  8192, 3584, 64, 2560, 8), ("c6",  8192, 3584, 64, 1024, 8),
        ("c7", 16384, 4096, 96, 2048, 3), ("c8", 16384, 4096, 96, 1024, 3),
    ]
    rng = np.random.default_rng(11)
    print("Part1b 判题形状上的 A_SCALE 读取代价 (32B 扇区 = L1 wavefront 下界)")
    tot_saved = 0
    for name, T, H, E, I, k in cases:
        group_m = 8 if H <= 1024 else 32
        sc, cc, nt = check_one(T, E, I, k, group_m, rng)
        tot_saved += sc - cc
        print(f"       {name}  M={T*k:6d} nbn={I//BN:2d} 瓦片={nt:6d}  "
              f"散射 {sc/nt:6.1f} -> 连续 {cc/nt:5.1f} 扇区/瓦片   "
              f"整案省 {(sc-cc)/1000.0:8.1f}K 扇区 ({(sc-cc)*32/1e6:5.1f} MB L1<->L2)")
    print(f"       c3~c8 合计省 {tot_saved/1000.0:.0f}K 个 L1 wavefront "
          f"/ {tot_saved*32/1e6:.0f} MB L1<->L2 流量")


# ---------------------------------------------------------------------------
# Part 2: V733 主机端位运算镜像
# ---------------------------------------------------------------------------
def kernel_chain(bound):
    """核内 L5304-5307 的原样镜像(Triton int32 语义 = numpy int32 环绕)。"""
    b = np.maximum(bound.astype(np.float32), np.float32(1e-30))
    bbits = b.view(np.int32)
    bexp = np.right_shift(bbits, 23) & np.int32(0xFF)
    s = ((bexp + np.int32(10)) << np.int32(23)).view(np.float32)
    inv = ((np.int32(244) - bexp) << np.int32(23)).view(np.float32)
    return s, inv


def host_chain(bound):
    """主机端 torch 侧的等价写法(只用 &, +, -, .view(dtype))。"""
    b = np.maximum(bound.astype(np.float32), np.float32(1e-30))
    bits = b.view(np.int32) & np.int32(0x7F800000)
    s = (bits + np.int32(10 << 23)).view(np.float32)
    inv = (np.int32(244 << 23) - bits).view(np.float32)
    return s, inv


def part2():
    rng = np.random.default_rng(7)
    ok = 0
    for _ in range(200):
        n = 4096
        # 覆盖真实量级 + 极端: 次正规、0、极大、指数域两端
        mode = rng.integers(0, 4)
        if mode == 0:
            v = np.exp(rng.uniform(-60, 25, n)).astype(np.float32)
        elif mode == 1:
            v = (rng.standard_normal(n).astype(np.float32) ** 2) * np.float32(1e-3)
        elif mode == 2:
            v = np.ldexp(np.float32(1.0), rng.integers(-140, 128, n)).astype(np.float32)
        else:
            v = rng.choice(np.array([0.0, 1e-45, 1e-38, 1.0, 3.4e38, np.inf],
                                    dtype=np.float32), n)
        with np.errstate(over='ignore', invalid='ignore'):
            s1, i1 = kernel_chain(v)
            s2, i2 = host_chain(v)
        assert s1.view(np.uint32).tobytes() == s2.view(np.uint32).tobytes(), "s mismatch"
        assert i1.view(np.uint32).tobytes() == i2.view(np.uint32).tobytes(), "inv mismatch"
        ok += 1
    # 全指数域穷举(bexp = 0..255, 尾数取几个代表值)
    exps = np.arange(256, dtype=np.int32)
    for mant in (0, 1, 0x400000, 0x7FFFFF):
        bits = (exps << 23) | np.int32(mant)
        v = bits.view(np.float32)
        with np.errstate(over='ignore', invalid='ignore'):
            s1, i1 = kernel_chain(v)
            s2, i2 = host_chain(v)
        assert s1.view(np.uint32).tobytes() == s2.view(np.uint32).tobytes()
        assert i1.view(np.uint32).tobytes() == i2.view(np.uint32).tobytes()
    print(f"Part2  {ok}/200 随机 + 全 256 个指数档 x 4 个尾数: "
          f"指数掩码写法与核内移位链逐比特相同(含 bexp=0/255 环绕端)")


if __name__ == "__main__":
    part1()
    part1b()
    part2()
