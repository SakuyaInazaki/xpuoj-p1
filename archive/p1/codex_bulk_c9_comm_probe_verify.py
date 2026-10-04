"""Static and CPU index checks for codex_bulk_c9_comm_probe.py.

This machine has no runnable Triton/CUDA target.  These checks validate the
mechanical baseline delta and communication layout; they do not claim JIT,
NVSHMEM, numerical-oracle, or performance validation.
"""

from __future__ import annotations

import ast
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / "kernel.py"
CAND = HERE / "codex_bulk_c9_comm_probe.py"

T = 4096
H = 4096
WORLD = 4
DCH = 4


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("missing function: " + name)


def check_mechanical_delta() -> None:
    base = BASE.read_text()
    cand = CAND.read_text()

    added_begin = cand.index("# c9-only communication lower-bound probe.")
    added_end = cand.index("def _shifted_cumsum", added_begin)
    stripped = cand[:added_begin] + cand[added_end:]

    injected = """        if (expert_gate_proj.shape[0] * world_size == 256
                and expert_gate_proj.shape[0] == 64
                and expert_gate_proj.shape[1] == 2048
                and world_size == 4):
            _bulk_c9_comm_probe(hidden_states, rank, world_size)
"""
    assert stripped.count(injected) == 1
    stripped = stripped.replace(injected, "")
    assert stripped == base, "candidate differs from baseline outside probe+guard"

    bt = ast.parse(base)
    ct = ast.parse(cand)
    assert ast.dump(function(bt, "_run_replicated")) == ast.dump(
        function(ct, "_run_replicated")
    )
    probe = function(ct, "_bulk_c9_comm_probe")
    assert [a.arg for a in probe.args.args] == ["x", "rank", "world"]
    assert not any(
        isinstance(n, ast.Name) and n.id == "output" for n in ast.walk(probe)
    )


def check_chunk_coverage() -> None:
    rows = (T + DCH - 1) // DCH
    assert rows == 1024
    covered = []
    for chunk in range(DCH):
        row0 = chunk * rows
        nrows = min(T - row0, rows)
        assert nrows > 0
        covered.extend(range(row0, row0 + nrows))
        assert (row0 + nrows) * H <= T * H
    assert covered == list(range(T))


def check_allgather_mapping() -> None:
    # Destination rank -> source block -> rows.  Every [source,T,H] block is
    # covered exactly once, including the local symmetric self-copy.
    seen = set()
    rows = T // DCH
    for src_rank in range(WORLD):
        for dst_rank in range(WORLD):
            for chunk in range(DCH):
                row0 = chunk * rows
                for row in range(row0, row0 + rows):
                    key = (dst_rank, src_rank, row)
                    assert key not in seen
                    seen.add(key)
    assert len(seen) == WORLD * WORLD * T


def check_owner_mapping_and_signals() -> None:
    # Sending rank s reads source block owner and writes owner.partial[s].
    seen = set()
    signal_slots = {dst: set() for dst in range(WORLD)}
    rows = T // DCH
    for send_rank in range(WORLD):
        for owner in range(WORLD):
            for chunk in range(DCH):
                src0 = (owner * T + chunk * rows) * H
                src1 = src0 + rows * H
                dst0 = (send_rank * T + chunk * rows) * H
                dst1 = dst0 + rows * H
                assert 0 <= src0 < src1 <= WORLD * T * H
                assert 0 <= dst0 < dst1 <= WORLD * T * H
                key = (owner, send_rank, chunk)
                assert key not in seen
                seen.add(key)
                slot = send_rank * DCH + chunk
                assert slot not in signal_slots[owner]
                signal_slots[owner].add(slot)
    assert len(seen) == WORLD * WORLD * DCH
    assert all(len(v) == WORLD * DCH for v in signal_slots.values())


def check_capacity_and_accounting() -> None:
    fp8_full = WORLD * T * H
    scale_full = WORLD * T * 4
    sink = T * H * 2
    signals = 2 * WORLD * DCH * 8
    assert fp8_full == 64 * 1024**2
    assert scale_full == 64 * 1024
    assert sink == 32 * 1024**2
    assert signals == 256

    remote_per_leg = (WORLD - 1) * T * H + (WORLD - 1) * T * 4
    assert remote_per_leg == 48 * 1024**2 + 48 * 1024
    assert 2 * remote_per_leg == 100_761_600

    ctas_per_leg = WORLD * DCH
    remote_ctas_per_leg = (WORLD - 1) * DCH
    assert (ctas_per_leg, remote_ctas_per_leg) == (16, 12)
    assert 2 * DCH == 8  # four payload puts + four scale/signal puts per peer/leg


def main() -> None:
    check_mechanical_delta()
    check_chunk_coverage()
    check_allgather_mapping()
    check_owner_mapping_and_signals()
    check_capacity_and_accounting()
    print("bulk c9 communication probe static/CPU checks: PASS")
    print("remote bytes/rank: 50,380,800 per leg; 100,761,600 for two legs")
    print("CTAs/rank: 16 per leg (12 remote); operations/peer/leg: 8; sync layers: 2")


if __name__ == "__main__":
    main()
