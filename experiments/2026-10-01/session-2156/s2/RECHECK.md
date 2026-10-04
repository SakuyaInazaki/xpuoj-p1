# S2 five-minute recheck (2026-10-01 22:00 Asia/Shanghai)

Production was rehashed as `f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a`; old README08dd is stale. Frozen S2 remains `ac338895e390139faaaebcaad98fba9ae1d5a3d55e378b2965f90da5431f2ea8`, file `../../session-1654/s2/p1_s2_c4_layout_only.py`. No production or candidate edits, no submission.

## New measurements and duplicate search

The post17:20 notes and newly collected structured audits show only153250(c8 regular-A extension, AC82.17),153278(c67 rename, AC81.92),153322(c67 line shift, AC82.08). These are not padded ACT. Local SHA scanning found only the frozen S2 source matching ac338; ACT padded-input/store markers appear only in that candidate and its own build/check scripts. No local submission audit has ac338 SHA. Platform executor independently checked latest20 submissions: no exact ac338, no in-flight and no padded candidate file evidence. This establishes no known prior test, not proof of absence from all historical/platform records.

## Candidate remains reviewable

Python compile passed. Original full CPU/AST checker was rerun against the frozen v12 base (not today's changed production) with outputs redirected here: 4157 full tiles nonoverlapping,524288 valid rows uniquely covered and compact scale paired; swizzle/persistent enumeration, final helper definitions, calls, unchanged math/launch arguments passed. See `contract-recheck.json` and `local-evidence.json`. The existing checker script itself was not edited.

Both actual c4 producers are covered: token-Q MDg and sorted-Q pre_nf. ACT is padded to69632 rows (68MiB rather than64MiB), while AH/WI/SCL stay compact65536 rows. Actual new producer returns the padded flag; matching static DN changes only input ACT read row; Down/DSCL/INV_PAD/final gather remain unchanged. Cold/other producer paths stay compact. Existing stage/warp/tile/DROP controls remain unchanged.

Risks for first formal test: new Triton kernels must compile within the full500-second batch budget; GPU TMA stores/loads, descriptor/sandbox behavior and SQNR have not been tested. Existing static identity/cache and phase assumptions are inherited, not repaired. S2 is derived from **v12 08dd**, not current153151/f9ca; it does not retain the c67 equivalent-copy layout or claim its anomalous timing signature. The mathematical c6/c7 lineage was equivalent, but byte/layout/compiler identities differ. Treat any new whole-vector changes separately from c4 performance.

## Potential score increase is small without anomalies

At153151 c4 `tb4.996,tk0.781,q86`. Holding tb fixed, reachingq87 requires tk≤0.746529ms (~4.41% reduction),q88≤0.681273(~12.77%),q89≤0.617483(~20.94%). Each integer case point adds1/12≈0.08333raw. c4 alone would need three integers to move that *conditional same-SID*1077 total to1080; these percentages are arithmetic, not expected gains. The older normal median c4 needs~7.09% reduction for its next integer, showing tb sensitivity. A2% c4 improvement may be measurable but earns no additional integer immediately.

The layout change removes masked tail-store branches and pairs full padded TMA reads with those stores; each nonempty expert has at most one tail, so the affected tail count is≤32 versus roughly512–543 row tiles (~≤6% of row tiles). This is a small concrete cost target, not a promise of4–21% full-chain gain. C4 is suitable for an independent first correctness/performance test if root has budget after collecting the new high-score same-SHA retest. Do not project normal c4 points onto a new source's future anomaly score.
