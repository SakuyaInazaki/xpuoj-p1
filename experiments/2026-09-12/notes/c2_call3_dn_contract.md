# c2 call>=3 MD/DN/final-sum contract

Scope: the active replicated c2 shape only: `T=16384, H=4096, E=8,
I=14336, topk=2`, hence expanded `M=32768`.  This is a bounded source audit
of the active MD callsite, `_dn_tma2_f8_{kernel,host}`, and the final FP8
branch-sum consumer.  It does not propose a kernel change.

| Stage | Inputs and indexing | Output contract | Active launch geometry |
|---|---|---|---|
| MD at `_CALLN>=3 && E==8` | Sorted FP8 A descriptor plus `fp8_tokens_s`; converted GU `gu_qi` and `gu_si`; `_gu_bnorm(gu_q,gu_s)`; sorted route weights `_wsort=flat_weights[order]`; grouping metadata and `order`. | Produces `act_q8[M,I]` FP8 and `act_rowscl[M]` FP32.  Route weight and GU/A dequantization are already folded into this quantized activation contract. | One persistent launch, grid `(132,)`, `BM=BN=BK=128`, `GROUP_M=8` for c2, warps 8, stages 4, `maxnreg=168`. |
| DN | `A=act_q8`, `A_SCALE=act_rowscl`; `B=dn_q[E,H,I]`, `B_SCALE=dn_s[E,H]`; metadata maps each sorted row tile to expert.  DN does **not** reread GU tensors/scales, route weights, or `order`. | `down[M,H]` FP8 and `dscl[M,H/256]` FP32.  `down[r,n]` uses the normal row-major `[r,n]` index. `dscl[r,p]` is stored at `r*(H/256)+p`, one scale per 256-column output chunk. | One persistent launch, grid `(132,)`, `BM=128, BN=256, BK=128`, `GROUP_M=32`, `FLAT=False`, warps 8, stages 3.  A descriptor block `[128,128]`; flattened DN descriptor block `[256,128]`.  For c2 there are `H/256=16` N chunks. |
| Final consumer | For token `t` and branch `j`, sorted source row is `r=inv_order[t*2+j]`.  It reads `down[r,h]` and `dscl[r,h//256]`. | FP32 accumulates the two dequantized branches in `j=0,1` order, then stores BF16 output `[T,H]`.  There is no later route-weight multiply. | One launch with grid `(ceil(T/32),ceil(H/256))=(512,16)`, `BLOCK_T=32`, `BLOCK_H=256`, warps 8, stages 1. |

For an oracle, for sorted row `r`, its metadata-selected expert `e`, output
chunk `p`, and `n=256p+c`, reproduce the DN epilogue exactly as:

```text
d[r,n] = dot_fp8_to_f32(act_q8[r,:], dn_q[e,n,:])
z[r,n] = d[r,n] * dn_s[e,n]
row_max[r,p] = max_{n in chunk p} abs(z[r,n])
s[r,p] = max(act_rowscl[r] * row_max[r,p] / 448, 1e-12)
down[r,n] = fp8_e4m3fn(z[r,n] * act_rowscl[r] / s[r,p])
dscl[r,p] = s[r,p]
```

The reconstructed DN value is therefore `float(down[r,n])*dscl[r,p]`.
The end-to-end oracle must then use `inv_order[t*2+j]`, sum both reconstructed
branches in FP32 in the same static branch order, and round the result to BF16.
Because `dn_s[e,n]` varies with `n`, it must be applied before the 256-column
`row_max`; pulling it outside that reduction changes the quantizer.

Source anchors: `p1/kernel.py:4223-4297` (DN), `p1/kernel.py:5670-5688` and
`p1/kernel.py:5814-5823` (active c2 callsite), and `p1/kernel.py:4300-4359`
(final consumer).  No standalone c2 MD/DN/sum stage timing is established by
this source audit; those three measurements remain the diagnostic's job.
