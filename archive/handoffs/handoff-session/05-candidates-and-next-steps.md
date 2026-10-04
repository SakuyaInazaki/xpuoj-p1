# 05 候选文件清单与下一步

## 当前 canonical

- `p1/kernel.py` = `p1/kernel_sorted_v3_hybrid.py` = 114706 / 52.50。
- SHA-256：`262b1eb050f5fe72f38a4e948cd96b17b23b29961618170008a27a5c623b2fca`。

## 本会话候选文件清单（按时间）

| 文件 | 提交 | 分数 | 状态 / 建议 |
|---|---:|---:|---|
| `kernel_c128_packe8_nometa96.py` | 114627 | 48.08 | 已被后续 sorted dispatch 取代。 |
| `kernel_c128_packe8_case2_nometa.py` | 114629 | 49.67 | case2 pack 方向来源。 |
| `kernel_fastag_k4gt.py` | 114631/114635 | WA | 官方 fast AG 失败，勿再提交。 |
| `kernel_directag_k4gt.py` | 114651 | WA | v1 写法失败，保留作反例。 |
| `kernel_directag_k4gt_v2.py` | 114658 | 50.67 | direct AG v2 基础。 |
| `kernel_c128_packe8e96_nometa.py` | 114663 | 47.83 | E8/E96 全 pack，未采用。 |
| `kernel_directag_v2_case2pack.py` | 114668 | 51.58 | 上一 canonical，有备份 `kernel_114668_backup.py`。 |
| `kernel_directag_c64_case2pack.py` | 114673 | 48.83 | direct AG c64，差于 c128。 |
| `kernel_directag_c256_case2pack.py` | 114676 | 48.50 | direct AG c256，差于 c128。 |
| `kernel_directag_v2_case2pack_e96pack.py` | 114677 | 49.42 | E96 pack 未采用。 |
| `kernel_directag_v2_packe8.py` | 114679 | 49.17 | E8 全 pack 未采用。 |
| `kernel_directag_v2_case2pack_diag.py` | 114681 | WA(预期) | 二调 raise 的 phase diag。 |
| `kernel_sorted_dispatch.py` | 114684 | WA | v1 slot bug，反例。 |
| `kernel_sorted_dispatch_v2.py` | 114692 | 50.17 | sorted dispatch 第一个正确版。 |
| `kernel_sorted_dispatch_e32only.py` | 114693 | 49.17 | E32 sorted c8 证据。 |
| `kernel_sorted_e8case1_e32.py` | 114694 | 49.67 | E32+E8case1 组合。 |
| `kernel_sorted_e96only_c4.py` | 114697 | 52.17 | E96 c4 关键中间版。 |
| `kernel_sorted_hybrid_v1.py` | 114701 | 50.75 | v3 之前的 hybrid。 |
| `kernel_sorted_e32_c4.py` | 114702 | 49.33 | E32 c4 反例。 |
| `kernel_sorted_e96c4_e32c8.py` | 114704 | 47.25 | 噪声样本。 |
| `kernel_sorted_v3_hybrid.py` | 114706 | **52.50** | **当前 canonical**。 |
| `kernel_sorted_v3_e96c2.py` | 114711 | 49.67 | E96 c2 反例。 |
| `kernel_sorted_v3_e8c64.py` | 114712 | 49.67 | case1 14.74 单点信号。 |
| `kernel_sorted_v3_e8c128.py` | 114713 | 52.33 | 接近但未超过。 |
| `kernel_sorted_v4_e32pack.py` | 114718 | 51.67 | E32 weight/meta 打包，未采用。 |
| `kernel_sorted_v4_e8c64_e32pack.py` | 114721 | 50.33 | 组合，未采用。 |
| `kernel_sorted_select_ag.py` | 114727 | 48.42 | 单次 token gather 变体。 |
| `kernel_sorted_v3_e8c64_e32c16.py` | 114728 | 49.75 | 参数组合，未采用。 |
| `kernel_sorted_select_ag_e8c64.py` | 114730 | 47.42 | 噪声样本。 |
| `kernel_sorted_e8both_c64.py` | 114732 | 47.83 | 本轮 case4 异常。 |
| `kernel_sorted_e8both_c64_selectsort.py` | 114734 | 51.00 | 组合候选。 |
| `kernel_directag_routes.py` | 114741 | 52.33 | **最接近 canonical 的候选**。 |
| `kernel_directag_routes_onebarrier.py` | 114743 | WA | 单 barrier 反例。 |
| `kernel_directag_routes_nonpack.py` | 114752 | 51.50 | case6 有重复改善。 |
| `kernel_directag_routes_selectsort.py` | 114755 | 48.25 | 组合，噪声大。 |
| `kernel_directag_routes_case6.py` | 114758 | 49.75 | **case6 route direct AG 三次改善**，值得平稳时重试。 |

## 最值得重试 / 合并的方向

1. **case6 route direct AG**：`kernel_directag_routes_case6.py`
   - 114741：case6 6.54
   - 114752：case6 7.34
   - 114758：case6 6.84
   - canonical 114706：case6 7.872
   - 三次重复改善，建议评测机平稳时提交或合并；只影响 case6，风险小。

2. **k>4 单次 token gather**：`kernel_sorted_select_ag.py`
   - 理论少一次约 200-450MB 的 gather；连续几轮被波动压住。
   - 可与 1 合并成新候选。

3. **E8 case1 chunks c64**：单点 case1 14.74 出现过，但整体不稳定；
   可与 case2 旧 path 组合继续观察。

4. **不要继续**：
   - 官方 `fast_allgather`。
   - 多 AG 单 barrier。
   - E32 sorted c4 / E96 sorted c2、c8。
   - try/except。

## 长期方向（需要本地 H800）

- 本地 4×H800 + NCU/SASS profile。
- 输出 BF16 的高吞吐 FP8 grouped GEMM（case1/2/12 大 I 的唯一数量级机会）。
- dispatch 与 grouped GEMM 的 persistent/pipeline 重叠。
- 若继续远程盲调，优先做“单点逐 case 有重复信号”的小改动，不要随机大扫参数。

## 提交前检查

```bash
cd /home/sakimi26/xpuoj-p1
python -m py_compile p1/<candidate>.py
sha256sum p1/kernel.py p1/kernel_sorted_v3_hybrid.py   # canonical 不能意外改动
python scripts/submit.py p1/<candidate>.py --poll --interval 10 --timeout 1800 2>&1 | tee logs/submit_<candidate>.log
```
