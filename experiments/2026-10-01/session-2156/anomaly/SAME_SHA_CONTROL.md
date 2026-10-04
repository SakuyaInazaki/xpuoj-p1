# Same-SHA anomalous/normal control:153151 vs153507

Limited to the two complete audits, original result files and frozen source fingerprints. No historical cohort scan, new probe/launch or timing manipulation. Sources:[153151 audit](../../session-1654/platform/153151-audit.json),[153151 raw](../../session-1654/platform/153151-raw.json),[153507 audit](../platform/153507-audit.json),[153507 raw](../platform/153507-raw.json).

**New control fact:** exact sourceSHA `f9ca009609bc2a50c320cfe5e912961a99cbd0f53c58510482f43438784e7c6a` again passed all12cases but changed fromraw89.75 to81.92. It restoredc8–c12normaltk andc1normaltb. This directly refutes assigning the historical zero/low vector orc1baseline inflation as deterministic performance of that source; it does not prove zero values can never recur.

## Accuracy and recorded diagnostics

Every case has exactly the same two displayedSQNR values between runs; all24SQNR values match, minimum22.69dB. Each case in each run has two determinism passes andpass=true/Accepted. This supports consistent judged output quality across timing outcomes, not full output-byte identity between independent submissions or correctness on untested inputs.

Both audits contain13log records, including12case metric/check summaries plus an additionaltc1 launcher/check record. OmittedLength totals0 in both. Neither set contains anyprofiler orCUPTI warning/token in its captured log text. Both show identical NVSHMEM discovery path `/opt/venvs/python3.10-torch/lib/python3.10/site-packages/nvidia/nvshmem` and the same OMP_NUM_THREADS=1 torch.distributed.run warning atline851. This reveals aPython3.10environment path, not exactPyTorch/CUDA/Triton/CUPTIversions or GPU model/frequency/topology. No hardware/version diagnostic differentiates the pair.

Onlytc1has the verbose launcher excerpt in these successful logs. Othercase records do not expose launcher timestamps/rank diagnostics. Per-SID task ids/PIDs/run-log directories naturally differ; this is not mechanism evidence. The returned case/log dictionary order is not a validated execution timeline.

153151 submitUTC08:51:24,tc1 launcher09:06:53.808,first recorded terminal audit北京时间17:11:44;153507 submitUTC13:59:28,tc1 launcher14:25:41.740,terminal first observed14:27:46.817. Submission-to-visible-tc1 wait is roughly15m30s vs26m14s; these intervals may include queue/startup and cannot be categorized from current fields. Launcher-to-terminal observation includes polling latency and is not exact totalbatchwalltime. Neither long wait proves OJfailure or describes aCUPTI/profiler initialization state.500seconds applies to stated batch execution budget, not total submission wait.

## Recorded timing/score differences

Each table cell lists153151 /153507; alltimesms. Platformcaseq is authoritative.

| c | tk | tb | q |
|---|---|---|---|
| 1 | 4.632 / 4.601 | 40.455 / 17.658 | 89 / 79 |
| 2 | 7.881 / 7.838 | 28.652 / 28.769 | 78 / 78 |
| 3 | 1.339 / 1.347 | 6.964 / 6.968 | 83 / 83 |
| 4 | 0.781 / 0.787 | 4.996 / 4.997 | 86 / 86 |
| 5 | 2.744 / 2.761 | 12.110 / 12.085 | 81 / 81 |
| 6 | 1.235 / 1.249 | 7.430 / 7.557 | 85 / 85 |
| 7 | 2.097 / 2.183 | 10.263 / 10.265 | 83 / 82 |
| 8 | 0.466 / 1.231 | 7.112 / 7.174 | 93 / 85 |
| 9 | 0.063 / 2.449 | 7.762 / 7.762 | 99 / 76 |
| 10 | 0.000 / 1.875 | 6.707 / 6.693 | 100 / 78 |
| 11 | 0.000 / 0.858 | 5.649 / 5.685 | 100 / 86 |
| 12 | 0.000 / 1.447 | 8.141 / 8.149 | 100 / 84 |

153151zero c10–c12 /lowc8–c12;153507zero/lowempty. c7tk2.097→2.183 also changesq83→82, without qualifying as the historicallowthreshold. qtotal1077→983, difference94integers: c1+10,c7+1,c8+8,c9+23,c10+22,c11+14,c12+16. Allremainingcaseq match. The large c1tb40.455vs17.658(~2.29x) adds10integers independently of the後半tk anomaly. Othercasebaseline changes are comparatively small; no baseline timing provenance/phase decomposition is exposed.

## What remains unsupported

No trace activities/start-endtimestamp/droppedrecord counters, profiler schedules/read timing, truecachehit/coldcompile status, rewrittenruntimecode identity, GPUidentity orallranktimeline appears. Exact submittedSHA does not guarantee the same compiled artifact/cache/machine; it likewise does not justify a specific cold-cache cause. Profiler coverage/aggregation omission, CUPTI timestamp loss, hostrewrite/cache differences orpath/caching problems remain hypotheses. This pair cannot choose among them, confirm finitebuffer exhaustion or produce a stable trigger. No warning difference supports attaching the earliercycle-clearing warning to this anomaly.

The useful addition is a fullyAC same-source negative control with identical displayedaccuracy but normalized timing/baseline, plus visible queue/startup-delay evidence. **Cause cannot be localized from these logs.** Preserve the anomalous historicalscore as aresult, judge ordinary optimizations on normal pairedcase times, and avoid attributing highscore to addedmath or a guaranteed timingmechanism.
