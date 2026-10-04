# c9 FP8 bulk two-leg communication probe

Date: 2026-09-05  
Candidate: `p1/codex_bulk_c9_comm_probe.py`  
Baseline copied verbatim: current `p1/kernel.py` (v760a)

## Purpose and isolation

This is a c9-only diagnostic, not an EP implementation. It measures a realistic bulk input all-gather, an owner-sized FP8 output exchange, and deterministic owner aggregation while leaving the model result entirely to the original `_run_replicated` path.

The exact guard is `(T,H,E,I,k,world,Ep)=(4096,4096,256,2048,8,4,64)`. c10 and every unknown shape execute the copied `kernel.py` code without entering the probe. Every rank sees the same shape predicate; there is no call-count, seed, validation-phase, or timing-phase gate.

The probe has no `output` argument. It does not mutate `hidden_states`. All remote writes land in newly allocated symmetric scratch, and the fixed-order combine writes another scratch tensor. After the probe returns, the original `_run_replicated(...)` call runs unchanged and alone writes the submitted output.

The verifier mechanically removes the one added definition block and the one exact guarded call, then requires byte-for-byte equality with the current `kernel.py`. It also compares the `_run_replicated` AST between the two files. The candidate therefore carries none of v811d's q6 numerical changes.

## Communication layout

Input representation uses the already established `_gq1p_tok`: one FP8 E4M3 row and one FP32 scale for each current-invocation input row.

`DCH=4` divides each peer's 4096 rows into four 1024-row chunks:

- FP8 payload per chunk: `1024*4096 = 4 MiB`;
- scale payload per chunk: `1024*4 = 4 KiB`;
- each CTA issues one 4 MiB `putmem_nbi_block`, fences, then issues one 4 KiB `putmem_signal_nbi_block`;
- each peer therefore receives four large payload puts and four scale/signal puts per leg;
- 16 CTAs/rank/leg cover four peers; 12 CTAs target remote peers and four perform the symmetric self-copy;
- the receiver waits on one unique `(source rank, chunk)` signal slot for all 16 chunks.

After the first wait, `xq_all/xs_all` contains a complete `[world,T,H]` 64 MiB FP8 tensor plus `[world,T]` scales made from this invocation's inputs. This scratch is used as the full-sized simulated local EP partial. In the second leg, rank `s` sends block `owner` to `owner`'s receive block `s`. The second wait therefore leaves every owner with four `[T,H]` contributions, one from each source rank.

The combine kernel reads contribution ranks in `0,1,2,3` order, dequantizes with the transmitted row scales, accumulates in FP32, and writes BF16 symmetric scratch. This store prevents dead-code elimination and models the fixed-order owner reduction without touching the real result.

There are exactly two synchronization layers using the same Accepted V701 pattern: payload put, `fence`, scale `putmem_signal`, then `signal_wait_until` acquire. Monotonic epochs allow safe scratch reuse without clearing signal slots between calls. Symmetric buffers are created in identical order on all ranks and cached only by `(world,T,H,device)`; they hold no reusable model input or output.

## Byte accounting

Per rank, per leg:

| Traffic | Remote bytes |
|---|---:|
| FP8 rows | `3*4096*4096 = 50,331,648` (48 MiB) |
| FP32 row scales | `3*4096*4 = 49,152` (48 KiB) |
| total | `50,380,800` |

Two legs transfer `100,761,600` remote bytes/rank (`96.09375 MiB`), whose ideal one-direction floor is about `.504 ms` at 200 GB/s. The probe also pays two signal-wait launches, input quantization, and the fixed combine. Its measured delta is therefore a conservative full communication-scaffold cost, not a pure wire-only number.

Route ids and normalized top-k weights are intentionally omitted. Their remote payload is only `3*T*k*(4+4)=786,432` bytes (`0.75 MiB`), a `.0039 ms` wire floor at 200 GB/s. Computing routing again inside the additive probe would duplicate the baseline's much larger routing work and make the communication lower bound misleading. A future full EP implementation must exchange the actual current-call route data.

## Memory and index bounds

Persistent symmetric scratch per rank is approximately 160.125 MiB:

- input gather FP8 + scales: 64 MiB + 64 KiB;
- owner receive FP8 + scales: 64 MiB + 64 KiB;
- BF16 fixed-combine sink: 32 MiB;
- two 16-slot int64 signal arrays: 256 bytes.

The temporary local quantized input is another 16 MiB + 16 KiB. CPU enumeration verifies every chunk covers exactly 1024 rows, every destination receives each `(source,row)` once, every owner receives each `(source,chunk)` once, signal slots do not collide, and all source/destination offsets remain within `world*T*H` and `world*T` capacities.

## Local validation

Run:

```text
python3 p1/codex_bulk_c9_comm_probe_verify.py
python3 -m py_compile p1/codex_bulk_c9_comm_probe.py p1/codex_bulk_c9_comm_probe_verify.py
```

Both pass. The verifier reports:

```text
bulk c9 communication probe static/CPU checks: PASS
remote bytes/rank: 50,380,800 per leg; 100,761,600 for two legs
CTAs/rank: 16 per leg (12 remote); operations/peer/leg: 8; sync layers: 2
```

This workstation has no runnable Triton/CUDA/NVSHMEM target. Python compilation and CPU index checks do not establish Triton 3.4 JIT compatibility, device-side signal progress, numerical oracle results, or performance.

## Platform interpretation and stop rule

The first platform run should verify compile and completion before interpreting speed. Because the output path is unchanged, any SQNR or determinism difference indicates memory corruption, synchronization failure, or an unintended baseline delta and rejects the probe.

If it is correct, compare c9 against a contemporaneous unchanged `kernel.py` anchor. Record the additive total as `delta_total`. The probe deliberately includes input FP8 quantization and fixed-order FP32/BF16 owner combine, so the architectural quantity is `delta_comm = delta_total - delta_quant - delta_combine`. The clean local-only control retains `_gq1p_tok`, the two symmetric self-block copies, and `_bulk_c9_fixed_combine_kernel`, but omits all three remote peers; it directly measures `delta_quant + delta_combine + local-copy overhead` with the same buffers and launches.

V702 measured comparable c9 input quantization and final combine at about `.099 + .070 = .169 ms`; use this only as a provisional estimate because the new combine layout differs. The first timing can therefore be classified as follows:

- `delta_total < .65 ms`: communication necessarily passes the `.65 ms` stop line because the local work is nonnegative;
- `delta_total >= .82 ms`: provisionally fails after subtracting the historical `.17 ms` local-work estimate;
- `.65 <= delta_total < .82 ms`: indeterminate; run the local-only control and use its paired median rather than guessing.

Stop bulk EP when the resulting paired median `delta_comm >= .65 ms`, or when variance cannot resolve about `.1 ms`. A lower communication delta supports one full c9-only implementation; it does not support enabling c10.

One measurement caveat is deliberate: the additive probe quantizes and communicates before the baseline MoE, so its delta includes allocator/cache perturbation and fixed-combine cost. This makes the stopping decision conservative and prevents a communication design from passing based on wire bytes alone.
