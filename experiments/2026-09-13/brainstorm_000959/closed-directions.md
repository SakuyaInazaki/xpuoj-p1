# Closed directions and changed-premise rule

Do not propose another parameter setting inside these closed families: E>=128 BM64 with unchanged B traffic; the measured modern bulk A2A design; tested dual-CTA occupancy configurations; deep small-K tile/stage sweeps; manual imprecise FP8 accumulation; in-kernel int4 unpack; near-uniform route-weight threshold dropping; single-pass activation quantization that inserts scale multiplication through dot K blocks; the measured side-stream metadata overlap; tested warp specialization/num_ctas=2; the measured quant-chain fusion/fat tile; direct unsorted token gather. Source: `docs/HISTORICAL_NO_REPEAT.md:6-18`.

Also closed in current state: c11 fusion; c5/c7 MD BN64 dual CTA; short-K DN dual CTA; full-shape fixed call3; c9 bulk/complete EP; FP6 packed MD; native INT6 packed/unpacked MD; E256 L2-FP16acc BM256; E256 sorted-A dual-TMA; E256 BM64 MD; c2 num_ctas=2; c2 BM64/s2/r128. Source: `docs/STATE.md:47-61`.

A candidate may touch the same broad component only if its mechanism changes the failed premise, such as eliminating the measured traffic/decode/materialization cost rather than retuning its tile. It must name that changed premise and use the mature FP8 path as contemporaneous control.
