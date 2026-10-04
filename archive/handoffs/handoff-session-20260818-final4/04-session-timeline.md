# 04 本会话完整时间线（submission 116594-116897，本 agent 共 101 次提交）

## 阶段概览

1. **116594-116646：接手后第一轮：order-derived gather 与 persistent fused gateup**
   - 同窗口对照发现旧 116310 在该窗口实际为 44.6ms；v109/v110/v111 分别有效。
   - v115 = v110 + v111，成为第一个新 base（116627，timeUsed 42753）。
   - case2 full-fused 再次确认 TLE（116646）。
2. **116663-116745：relaxed atomics、s4、int64 topk_ids**
   - v129：所有 activation amax atomic 改 `sem="relaxed"`（116695）。
   - v130：fused gateup `num_stages=4`（116707）。
   - v136：topk_ids 保持 int64（116735）。
3. **116747-116788：orderW 主线**
   - v140：fused gateup 内直接 `W[ORDER[offs_m]]`，省掉 `flat_weights[order]`（116754）。
   - v142：case2 SwiGLU 也 orderW（116767）。
   - v143：fused GROUP_M=16（116773）。
4. **116792-116897：case2 gather 与 route E8 微调**
   - v147：case2 token gather 改 custom order-derived gather（116808）。
   - v155：route E<=16 BM64/BN16/w4（116858）。
   - v159：route E8 BLOCK_K=128（116882，当前 base）。

## 全部提交表

| id | 状态 | raw | timeUsed | 日志 | 说明 |
|---:|---|---:|---:|---|---|
| 116594 | Accepted | 74.42 | 43018 | submit_v109 order+weight gather.log | order-derived token row + weights 融合进 gather（非 persistent） |
| 116600 | Accepted | 74.5 | 42952 | submit_v110 order token only.log | 仅 order-derived token row（非 persistent） |
| 116605 | WrongAnswer | 0 | 160409958 | submit_diag v98 phases.log | v98 phase CUDA-event 诊断（WA，读 userError） |
| 116607 | Accepted | 74.17 | 43610 | submit_v111 fused persist tiles.log | fused gateup persistent tile-loop 全量 |
| 116612 | Accepted | 73.42 | 44539 | submit_v112 gather M-only.log | token gather M-only（每 program 循环全部 H） |
| 116616 | Accepted | 73 | 44566 | submit_v114 persist case1.log | 仅 case1 persistent fused |
| 116620 | Accepted | 73.25 | 44630 | submit_base control.log | 旧 116310 同窗口对照 |
| 116622 | Accepted | 73.92 | 44086 | submit_v111 r2.log | fused persistent tile-loop 复测 |
| 116626 | Accepted | 74 | 43462 | submit_v110 r2.log | order token only 复测 |
| 116627 | Accepted | 75 | 42753 | submit_v115 NEW BASE.log | persistent fused + order token；新 base |
| 116629 | Accepted | 74.17 | 43180 | submit_v116 order+weight fused.log | v115 + weights 融合进 gather |
| 116633 | Accepted | 74.33 | 43220 | submit_v115 r2.log | 新 base 复测 |
| 116637 | Accepted | 73.08 | 44625 | submit_base control 2.log | 旧 116310 同窗口对照 2 |
| 116639 | Accepted | 73.83 | 43725 | submit_v117 case2 order gather.log | v115 + case2 custom order gather |
| 116643 | Accepted | 73.5 | 44377 | submit_v118 case2 order div.log | v115 + case2 `order // k` |
| 116646 | TimeLimitExceeded | 0 | 473140376 | submit_v119 case2 fused rowA.log | case2 persistent fused rowA；TLE |
| 116663 | WrongAnswer | 0 |  | submit_diag v115 phases.log | v115 phase 诊断 |
| 116666 | Accepted | 69.25 | 53467 | submit_v120 fused persist major fixed.log | 修正 v106 M-major persistent |
| 116667 | Accepted | 74 | 43583 | submit_v121 cache TMA desc.log | TMA descriptor 缓存 |
| 116668 | Accepted | 73.58 | 43906 | submit_v115 control r3.log | 对照 |
| 116672 | Accepted | 73.75 | 43969 | submit_v121 r2.log | descriptor 缓存复测 |
| 116675 | Accepted | 75.25 | 44304 | submit_v122 meta num_sms132.log | metadata num_sms=132 |
| 116676 | Accepted | 69.75 | 52407 | submit_v123 fused grid160.log | fused grid=160 |
| 116679 | WrongAnswer | 0 | 55486627 | submit_v125 buffer cache.log | 中间 tensor buffer cache；WA/TLE |
| 116683 | WrongAnswer | 0 | 47590395 | submit_v126 packed sort method.log | tensor.sort 打包排序；sandbox guard |
| 116685 | Accepted | 73.92 | 43256 | submit_v127 binned amax case2.log | case2 amax 分 bin |
| 116688 | Accepted | 73.5 | 44361 | submit_v115 control r4.log | 对照 |
| 116690 | Accepted | 73 | 44393 | submit_v127 r2.log | binned amax 复测 |
| 116694 | WrongAnswer | 0 | 48302782 | submit_v128 packed sort functional.log | functional torch.sort；WA guard |
| 116695 | Accepted | 74.33 | 42664 | submit_v129 NEW BASE.log | 所有 activation amax atomic sem=relaxed |
| 116696 | Accepted | 74 | 43537 | submit_v129 r2.log | relaxed 复测 |
| 116701 | Accepted | 73.83 | 42987 | submit_v130 fused s4.log | + fused num_stages=4 |
| 116702 | Accepted | 73.83 | 43539 | submit_v129 control r3.log | 对照 |
| 116707 | Accepted | 74.25 | 42985 | submit_v130 r2 NEW BASE.log | v130 复测 |
| 116708 | Accepted | 73.67 | 43587 | submit_v129 control r4.log | 对照 |
| 116711 | WrongAnswer | 6.08 | 69982555 | submit_v131 fused s5.log | num_stages=5；OutOfResources |
| 116716 | Accepted | 75.83 | 43541 | submit_v132 grid128 s4.log | fused grid=128；raw 异常 |
| 116718 | Accepted | 73.67 | 43809 | submit_v133 case2 row relaxed s4.log | case2 row SwiGLU/down + relaxed + s4 |
| 116724 | Accepted | 74.08 | 43005 | submit_v134 desc cache s4.log | v130 + TMA descriptor cache |
| 116727 | Accepted | 73.5 | 44852 | submit_v135 fused w16 s4.log | fused num_warps=16 |
| 116730 | Accepted | 74.08 | 43300 | submit_v136 no id conversions.log | topk_ids 保持 int64 |
| 116735 | Accepted | 74.25 | 43212 | submit_v136 r2 NEW BASE.log | no id conversions 复测 |
| 116736 | Accepted | 73.5 | 43924 | submit_v130 control r5.log | 对照 |
| 116745 | Accepted | 73.83 | 43280 | submit_v137 inplace weight norm.log | topk_weights /= |
| 116747 | Accepted | 74.42 | 42830 | submit_v138 fused GM16 s4.log | v136 + fused GM16（无 orderW） |
| 116749 | Accepted | 74.17 | 43234 | submit_v136 control r6.log | 对照 |
| 116751 | Accepted | 74.5 | 42343 | submit_v140 fused orderW.log | fused gateup 内按 order 读原始 W |
| 116753 | Accepted | 74.33 | 43826 | submit_v141 orderW not I8192.log | orderW 排除 I=8192 |
| 116754 | Accepted | 74.92 | 41808 | submit_v140 r2 NEW BASE.log | fused orderW 复测 |
| 116758 | Accepted | 74.67 | 42762 | submit_v142 case2 swiglu orderW.log | case2 SwiGLU 也 orderW |
| 116760 | Accepted | 73.5 | 43809 | submit_v140 control r3.log | 对照 |
| 116767 | Accepted | 74.33 | 43160 | submit_v142 r2 NEW BASE.log | case2 SwiGLU orderW 复测 |
| 116768 | Accepted | 73.58 | 43823 | submit_v140 control r4.log | 对照 |
| 116771 | Accepted | 74.08 | 42945 | submit_v143 orderW GM16.log | + fused GROUP_M=16 |
| 116772 | Accepted | 74.33 | 43299 | submit_v142 control r3.log | 对照 |
| 116773 | Accepted | 74.17 | 42869 | submit_v143 r2 NEW BASE.log | orderW GM16 复测 |
| 116774 | Accepted | 73.67 | 43804 | submit_v142 control r4.log | 对照 |
| 116783 | Accepted | 74.17 | 42972 | submit_v144 orderW GM4.log | GM4 扫描 |
| 116784 | Accepted | 74.08 | 43778 | submit_v143 control r5.log | 对照 |
| 116787 | Accepted | 73.92 | 43388 | submit_v144 r2.log | GM4 复测 |
| 116788 | Accepted | 73.5 | 43811 | submit_v143 control r6.log | 对照 |
| 116792 | Accepted | 76 | 43123 | submit_v145 order32.log | order 转 int32；raw 异常成 scoreboard best |
| 116794 | Accepted | 74 | 43230 | submit_v143 control r7.log | 对照 |
| 116795 | Accepted | 73.75 | 43790 | submit_v146 order32 not case1/2.log | order32 排除 case1/2 |
| 116798 | Accepted | 74.42 | 42659 | submit_v147 case2 gather order.log | case2 custom order-derived gather |
| 116799 | Accepted | 74.42 | 43812 | submit_v143 control r8.log | 对照 |
| 116808 | Accepted | 73.92 | 43232 | submit_v147 r2 NEW BASE.log | case2 custom gather 复测 |
| 116810 | Accepted | 73.75 | 43763 | submit_v143 control r9.log | 对照 |
| 116812 | Accepted | 73.92 | 43163 | submit_v148 case2 gather BM256.log | case2 custom gather BM256 |
| 116813 | Accepted | 74.08 | 43219 | submit_v147 control r10.log | 对照 |
| 116816 | Accepted | 73.83 | 43122 | submit_v149 case2 gather+amax.log | case2 gather + scalar amax |
| 116817 | Accepted | 74.08 | 43204 | submit_v147 control r11.log | 对照 |
| 116823 | WrongAnswer | 0 | 156308539 | submit_diag v147 phases.log | v147 phase 诊断 |
| 116828 | Accepted | 74.08 | 43117 | submit_v150 case2 gather partial amax.log | case2 gather + partial-store amax |
| 116830 | Accepted | 74.08 | 43190 | submit_v147 control r12.log | 对照 |
| 116833 | Accepted | 74.17 | 42756 | submit_v151 orderW GM24.log | fused GM24 |
| 116834 | Accepted | 73.83 | 43731 | submit_v147 control r13.log | 对照 |
| 116837 | Accepted | 74 | 43286 | submit_v151 r2.log | GM24 复测 |
| 116839 | Accepted | 76 | 43222 | submit_v147 control r14.log | 对照 |
| 116841 | WrongAnswer | 56.17 | 5510909 | submit_v153 inv sort side stream.log | inv argsort 放 side stream；determinism fail |
| 116842 | Accepted | 74.08 | 43274 | submit_v147 control r15.log | 对照 |
| 116844 | Accepted | 73.92 | 43253 | submit_v154 case2 quant BM256.log | case2 token quant BM256 |
| 116845 | Accepted | 74.08 | 43203 | submit_v147 control r16.log | 对照 |
| 116847 | Accepted | 74.17 | 43174 | submit_v155 route E8 BM64.log | route E<=16 改 BM64/BN16/w4 |
| 116848 | Accepted | 73.5 | 43747 | submit_v147 control r17.log | 对照 |
| 116858 | Accepted | 74 | 43315 | submit_v155 r2 NEW BASE.log | route E8 BM64 复测 |
| 116859 | Accepted | 73.67 | 43746 | submit_v147 control r18.log | 对照 |
| 116863 | Accepted | 73.83 | 43763 | submit_v156 route E8 BM64 w8.log | route E8 BM64 + w8 |
| 116864 | Accepted | 74.17 | 43218 | submit_v155 control r19.log | 对照 |
| 116868 | Accepted | 74 | 43184 | submit_v158 route N32 BM64.log | route N<=32 BM64 |
| 116869 | Accepted | 73.67 | 43665 | submit_v155 control r20.log | 对照 |
| 116879 | Accepted | 74.17 | 43198 | submit_v159 route E8 BK128.log | route E8 BLOCK_K=128 |
| 116880 | Accepted | 73.75 | 43714 | submit_v155 control r21.log | 对照 |
| 116882 | Accepted | 74.08 | 43131 | submit_v159 r2 NEW BASE.log | route E8 BK128 复测；当前 base |
| 116883 | Accepted | 73.58 | 43777 | submit_v155 control r22.log | 对照 |
| 116888 | Accepted | 75.75 | 43210 | submit_v160 route N64 BM64.log | route N<=64 BM64 |
| 116889 | Accepted | 73.92 | 43742 | submit_v159 control r23.log | 对照 |
| 116892 | Accepted | 73.75 | 43767 | submit_v160 r2.log | route N64 BM64 复测 |
| 116893 | Accepted | 74 | 43358 | submit_v159 control r24.log | 对照 |
| 116896 | Accepted | 73.83 | 43804 | submit_v161 route E8 BK256.log | route E8 BK256 |
| 116897 | Accepted | 73.92 | 43285 | submit_v159 control r25.log | 对照 |
