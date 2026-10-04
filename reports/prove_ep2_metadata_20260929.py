"""CPU oracle for the EP2 per-expert/per-tile argument mismatch. No GPU claim."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/2026-09-27/candidates/v866_c9c10_pair_ep2_halfweights.py"
OUT = ROOT / "experiments/2026-09-29/candidates"


def metadata(counts):
    starts, experts, tile_starts, tile_nums, tile_cums = [], [], [], [], []
    row = tiles = 0
    for e, n in enumerate(counts):
        starts.append(row)
        nt = (n+127)//128
        tiles += nt
        experts.extend([e]*nt)
        tile_starts.extend([row]*nt)
        tile_nums.extend([nt]*nt)
        tile_cums.extend([tiles]*nt)
        row += n
    return starts, experts, tile_starts, tile_nums, tile_cums


def check(counts):
    starts, experts, tile_starts, tile_nums, tile_cums = metadata(counts)
    seen = [0]*sum(counts)
    wrong = oob = 0
    for pid, e in enumerate(experts):
        local_m = pid-(tile_cums[pid]-tile_nums[pid])
        begin = tile_starts[pid]
        assert begin == sum(counts[:e])
        for lane in range(128):
            row = begin+local_m*128+lane
            if row < begin+counts[e]:
                assert 0 <= row < sum(counts)
                seen[row] += 1
        if pid >= len(starts):
            oob += 1
        elif starts[pid] != begin:
            wrong += 1
    assert all(n == 1 for n in seen)
    assert len(experts) <= (sum(counts)+127)//128+len(counts)
    return {"experts": len(counts), "rows": sum(counts), "tiles": len(experts),
            "old_wrong_in_bounds_tile_reads": wrong, "old_oob_tile_reads": oob}


def bad_calls(path):
    t = ast.parse(path.read_text())
    matches = []
    for fn in t.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name != "_run_pair_replicated_c9c10":
            continue
        for n in ast.walk(fn):
            if not isinstance(n, ast.Call):
                continue
            names = [x.id if isinstance(x, ast.Name) else None for x in n.args]
            for j in range(len(names)-2):
                if names[j:j+3] == ["meta_expert_ids", "counts", "meta_split_cum"]:
                    matches.append({"line": n.lineno, "callee": ast.unparse(n.func)})
    return matches


def main():
    rng = random.Random(20260929)
    vectors = [[129, 1, 0, 256], [256]*128, [0]*128, [65536]+[0]*127,
               [0]*127+[65536], [127, 128, 129, 0]*32]
    for _ in range(24):
        counts = [0]*128
        for _ in range(32768):
            counts[rng.randrange(128)] += 1
        vectors.append(counts)
    checks = [check(x) for x in vectors]
    original = BASE.read_text()
    old = "meta_expert_ids, counts, meta_split_cum,"
    new = "meta_expert_ids, counts, meta_tile_split,"
    assert original.count(old) == 2
    assert len(bad_calls(BASE)) == 2
    fixed = original.replace(old, new)
    ast.parse(fixed)
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = OUT / "ep2_v866_metadata_fix.py"
    candidate.write_text(fixed)
    assert not bad_calls(candidate)
    patch = "".join(difflib.unified_diff(original.splitlines(True), fixed.splitlines(True),
                    fromfile=str(BASE.relative_to(ROOT)), tofile=str(candidate.relative_to(ROOT))))
    (OUT / "ep2_v866_metadata_fix.patch").write_text(patch)
    affected = {}
    for p in sorted((ROOT / "experiments/2026-09-27/candidates").glob("v8*.py")):
        calls = bad_calls(p)
        if calls:
            affected[str(p.relative_to(ROOT))] = calls
    result = dict(scope="Static call-site proof and CPU metadata simulation only; no remote failure-causality or GPU validation.",
                  base=str(BASE.relative_to(ROOT)), base_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  candidate=str(candidate.relative_to(ROOT)), candidate_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
                  patched_calls=2, production_unchanged_sha256=hashlib.sha256((ROOT/'p1/kernel.py').read_bytes()).hexdigest(),
                  cases=checks, total_checked_rows=sum(c['rows'] for c in checks),
                  affected_files=affected,
                  small_counterexample={"counts": vectors[0], "old_per_expert_starts": metadata(vectors[0])[0],
                                        "required_per_tile_starts": metadata(vectors[0])[2]})
    (ROOT / "reports/2026-09-29-ep2-metadata-proof.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','affected_files')}, ensure_ascii=False, indent=2))
    print(f"CPU checks: {len(checks)} distributions, {result['total_checked_rows']} rows; affected historical files: {len(affected)}")


if __name__ == "__main__":
    main()
