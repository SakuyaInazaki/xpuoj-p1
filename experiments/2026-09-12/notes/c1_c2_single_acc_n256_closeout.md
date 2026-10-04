# c1/c2 single-accumulator N256 closeout

Date: 2026-09-12. Historical/source audit only; no new platform action.

- V348/V348b tested the c1/c2 E8 PM path with the existing `[g128|u128]` block-interleaved weights, one `[256,128]` B descriptor load and one `[BM,256]` accumulator. BM128/BK128/w8/s4/`maxnreg=168` stayed fixed. Correct splitting used `reshape(BM,2,BN) -> trans(0,2,1) -> split`.
- The clean 132577/132578 comparison recorded an E8 PM regression of approximately **+0.33 ms**, so that path was rolled back (`archive/handoffs/handoff-session-20260821/session-notes.md:1063-1082`).
- V494 retested the same single-accumulator PM structure with element-interleaved weights and direct `reshape/split`; SID 134832/134833 versus 134834 showed c1 approximately **2.5% slower after drift correction** (`reports/codex_handoff_audit_20260905.md:172`).
- Keeping block interleave requires the logical transpose for correct G/U mapping, but it is not a material runtime cross-warp cost: V354's paired transpose-removal test was neutral, and the later epilogue profile measured split at about 27 cycles (`session-notes.md:1188-1199,1418-1424`).

Decision: close the c1/c2 single-accumulator N256 route. The PM/TMA-A core has two independent negative results, and split/reordering is not the missing benefit.
