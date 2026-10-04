# 04 续接时间线（final5，更新到 submission 117244）

## 关键阶段

1. final gather tiling：v162 -> v172 -> 当前 case1/2 BT8，其余 BT32。
2. route 细化：E32/E64 BK128，E96 BM64/BK128/w4，E32 BM64。
3. case2 SwiGLU：BN256 -> BN128 -> BN64。
4. fused gateup：GM16 -> GM32/s4。
5. case2 token gather：BM128/BH128 -> BM64/BH128 -> BM64/BH256。

## 有效提交速览

| id | 说明 | timeUsed |
|---:|---|---:|
| 117157 | fused GM32 首次 | 42966 |
| 117207 | route E32 BM64 首次 | 42980 |
| 117218 | case2 gather BM64 首次 | 42903 |
| 117234 | case2 gather BM64/BH256 首次 | 42961 |
| 117237 | 117234 复测 | 43233 |

## 负结果速览

- 117083 down w16：43831 vs 43598。
- 117166 down w4：108642，明显慢。
- 117193 fused grid264：43711 vs 43189。
- 117199 gateup BK256/s2：44978 vs 43704。
- 117212 case2 dual normal：case2 10.543 vs 10.376。
- 117222 case2 gather BM32：case2 10.357 vs 10.205。
- 117228/117231 token quant BM64：两窗互斥。
- 117241/117244 non-case2 gather BM64/BH256：两窗互斥。

## 仍值得观察

- 117195/117198 final case9/10 BT16：case9/10 两窗互斥，未采纳。
- 平台 Pending 偶发；不要死等。
