# Replicated route, counting-sort, and metadata facts for c9/c10

Scope is the active E256 replicated path in `p1/kernel.py` for c9/c10 (`T=4096`, `topk=8`, hence `M=32768`). This is a source-level dependency and launch audit of route generation, `_counting_sort_order`, `_prepare_moe_metadata`, and their `_run_replicated` call points. It does not select an optimization or claim platform bitwise behavior.

## Launch table

`_CALLN` is the process-global call counter incremented by `run_kernel`, rather than a counter local to c9/c10.

| `_CALLN` branch for E256 | Route stage | `_counting_sort_order` | Metadata | Explicit launches visible in this scope |
|---|---|---|---|---:|
| `1` | `_route_gemm_softmax_kernel`: 1 Triton launch on grid 128; `torch.topk`: backend launch count opaque at Python level; `_topk_renorm_kernel`: 1 Triton launch on grid 16 | `_sort_hist_kernel`: 1 launch on grid 512; eager Torch transpose/contiguous/cumsum/subtract/cast/sum/slice operations have an opaque launch count; `_sort_scatter_kernel`: 1 launch on grid 512 | `build_block_row_idx_info_kernel`: 1 launch on grid 132 | **5 explicit Triton launches**, plus `torch.topk` and eager Torch work of unknown launch count |
| `2` | Same route path as call 1 | Histogram grid 512, `_csort_colscan_kernel` grid 256, `_csort_offsets_kernel` grid 256, scatter grid 512 | One grid-132 metadata launch | **7 explicit Triton launches**, plus `torch.topk` of unknown launch count |
| `>=3` | `_route_full_kernel`: 1 Triton launch on grid 128; it fuses route GEMM, BF16 logit rounding, softmax, top-k, and renormalization | Same four Triton launches as call 2 | One grid-132 metadata launch | **6 explicit Triton launches** |

The totals exclude input/weight preparation, GQ, MD, DN, and final branch gathering. No stored c9/c10 measurement isolates elapsed time for route, counting sort, or metadata. Their stage time is therefore **unknown**; source comments about reducing launches are not timing evidence.

## Tensor and purpose table

| Step | Inputs and intermediate tensors at c9/c10 | Downstream dependency | What the step establishes |
|---|---|---|---|
| Route, calls 1–2 | `_route_gemm_softmax` allocates `probs[T,E]` FP32 (4 MiB). `torch.topk` produces `topk_ids[T,8]` and `topk_weights[T,8]`; `flat_ids` is a reshape view and `_topk_renorm_flat` allocates `flat_weights[M]` FP32 (128 KiB). | `flat_ids` feeds counting sort. `flat_weights` is later paired with each branch through `ORDER`. | Expert selection and normalized route weights. It does not group rows by expert. |
| Route, calls >=3 | `_route_full` directly allocates `flat_ids[M]` int64 (256 KiB) and `flat_weights[M]` FP32 (128 KiB). | Same consumers as above. | Same logical route products, produced by one fused kernel. |
| Histogram | `hist[C,E_PAD]` int32 with `C=512`, `E_PAD=256` (512 KiB). | Supplies either the eager first-call prefix path or the kernelized call>=2 prefix path. | Per-chunk expert counts. It is needed to form contiguous expert groups; it does not itself impose a row order. |
| Prefix/base construction, call 1 | `flat_t=hist.T.contiguous().reshape(-1)`, `incl=flat_t.cumsum(0)`, `base=(incl-flat_t)` int32, and `counts[E]` int32. | `base` feeds scatter; `counts` feeds metadata and all grouped MD/DN kernels. | Computes each expert/chunk destination base. Chunk-major prefixes are also part of the chosen stable placement. |
| Prefix/base construction, calls >=2 | `base[E_PAD*C]` int32 (512 KiB), `tot[E_PAD]` int32, and `counts[E]` int32. Column scan computes within-expert chunk prefixes; offsets add preceding-expert totals. | Same as call 1. | Computes contiguous expert ranges and stable chunk bases without eager host operations. |
| Stable scatter | `order[M]` and `inv_order[M]`, both int64 (256 KiB each). Within a chunk, `tl.cumsum` assigns increasing positions; chunk bases preserve chunk order. | `order[s]=r` identifies original flat branch `r` and weight `flat_weights[r]`. Depending on the active GQ path, token `r//8` is either placed into sorted row `s` beforehand through `inv_order`, or selected directly by MD through `order[s]//8`. `inv_order` is also used by final gather. | Materializes expert grouping. Stability makes equal-expert rows follow original flat order and matches the documented stable-argsort layout; determinism/layout compatibility is the extra property beyond grouping. |
| Metadata | With the imported group-GEMM block size 128, `M_grid=ceil(32768/128)+256=512`. It allocates `split_size_cum_per_expert[E]`, scratch `expert_idx_to_tile_offset[E]`, four int32 arrays of length `M_grid`, and `num_tiles_total[1]`. | Six returned tensors plus `counts` schedule the grouped MD and DN kernels. Scratch `expert_idx_to_tile_offset` is not returned. | Converts expert counts into row starts and persistent tile scheduling. It reads neither `flat_ids` nor `order/inv_order`; stable within-expert order is irrelevant to this step. |

## Effect of changing `ORDER`

Let original flattened branch index be `r=t*8+j`, sorted row be `s`, and a valid permutation satisfy `order[s]=r` and `inv_order[r]=s`.

- MD is row independent. On the direct per-token GQ variant it uses `order[s]//8` to select token `t`; on the sorted-input variants, the preceding gather or `_gq1p_tm` has already put that token into row `s`. The route weight is paired through `flat_weights[order[s]]`. Reordering rows within the same expert therefore only permutes MD output rows when the mapping remains consistent throughout the selected path.
- DN is also row independent and consumes the same expert-grouped row and metadata range. A within-expert permutation only carries that row's activation to another position in the same expert range.
- Final gather iterates original branch slots `j=0..7`, loads `src_row=inv_order[t*8+j]`, and adds those rows in original top-k slot order. Thus the final association and addition order are preserved by any bijective expert-grouping permutation with its exact inverse.

Consequently, stable order is not a mathematical requirement for per-row MD/DN or the final top-k sum. Contiguous grouping by expert, correct route-weight pairing, and an exact inverse are required. Stability supplies a deterministic layout equivalent to the documented stable argsort and can matter to regression expectations or unstated implementation assumptions. This audit does not claim that replacing it would pass platform bitwise checks.

## Source anchors

- Route selection and helper calls: `p1/kernel.py:5556-5565`, `5577-5582`, `5630-5633`, `5941-5946`.
- Fused and split route helpers: `p1/kernel.py:914-1013`, `1769-1867`.
- Stable counting sort: `p1/kernel.py:605-789`.
- Active metadata helper: `p1/kernel.py:3821-3865`.
- MD token and weight lookup: `p1/kernel.py:1492-1564`.
- Direct GQ inverse use: `p1/kernel.py:793-867`.
- Final BF16 and FP8 gathers: `p1/kernel.py:388-413`, `4301-4335`.
