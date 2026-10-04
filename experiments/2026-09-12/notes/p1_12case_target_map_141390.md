# P1 12-case target map at SID 141390

All cases use `world_size=4`. Shapes are `(T,H,E,I,topk)` in testcase order.
The `tb`, `tk`, and integer score columns are the saved SID 141390 values.
Every known shape currently matches one of the six top-level replicated
dispatch guards in `p1/kernel.py`; none of these 12 cases reaches the common
path.

| case | T | H | E | I | topk | world | active path | tb ms | tk ms | integer score | saved stage profile |
|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|---:|:---|
| c1 | 16384 | 4096 | 8 | 8192 | 2 | 4 | replicated | 17.908 | 4.746 | 79 | — |
| c2 | 16384 | 4096 | 8 | 14336 | 2 | 4 | replicated | 29.295 | 8.193 | 78 | — |
| c3 | 16384 | 2048 | 32 | 2048 | 4 | 4 | replicated | 6.956 | 1.421 | 83 | — |
| c4 | 16384 | 2048 | 32 | 1024 | 4 | 4 | replicated | 4.917 | 0.852 | 85 | — |
| c5 | 8192 | 3584 | 64 | 2560 | 8 | 4 | replicated | 12.733 | 2.880 | 81 | MD 1.748–1.750 ms; DN 0.861–0.864 ms |
| c6 | 8192 | 3584 | 64 | 1024 | 8 | 4 | replicated | 7.518 | 1.348 | 84 | — |
| c7 | 16384 | 4096 | 96 | 2048 | 3 | 4 | replicated | 10.164 | 2.328 | 81 | MD 1.310–1.312 ms; DN 0.651–0.653 ms |
| c8 | 16384 | 4096 | 96 | 1024 | 3 | 4 | replicated | 6.975 | 1.362 | 83 | — |
| c9 | 4096 | 4096 | 256 | 2048 | 8 | 4 | replicated | 8.322 | 2.634 | 75 | no MD/DN elapsed split |
| c10 | 4096 | 4096 | 256 | 1536 | 8 | 4 | replicated | 6.680 | 2.017 | 76 | no MD/DN elapsed split |
| c11 | 65536 | 1024 | 32 | 1024 | 2 | 4 | replicated | 5.610 | 1.009 | 84 | — |
| c12 | 65536 | 1024 | 32 | 2048 | 2 | 4 | replicated | 8.156 | 1.646 | 83 | — |

Outside c9/c10, the largest saved kernel times are c2, c1, c5, and c7 in that
order. Only c5/c7 have saved MD/DN elapsed-time coverage: SID 141410 repeated MD
once and SID 141411 repeated DN once on the SID 141408 diagnostic base. Those
increments are stage estimates rather than a decomposition measured in SID
141390 itself. The bounded notes contain no equivalent elapsed-stage profile
for the other cases.

Sources:

- `1-full.md`: four-H800/world-size contract and shape meanings.
- `p1/kernel.py:18-25`: ordered `_KNOWN12` shape tuples.
- `p1/kernel.py:5967-6014` and `docs/CODE_MAP.md`: six replicated dispatch guards.
- `experiments/2026-09-08/results/stable_c5_c7_dispatch_141408_vs_141390.json`: saved SID 141390 per-case `tb`, `tk`, and integer scores.
- `experiments/2026-09-08/notes/stable_profile_c5_c7.md` and `experiments/2026-09-08/results/c57_stage_profiles_141410_141411_vs_141408.json`: stage-profile coverage and corrected increments.
