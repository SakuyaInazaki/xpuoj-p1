"""Prepare bounded, unsubmitted source-layout controls from exact SID 152976.

No extra GPU work, call-count branch, arithmetic change, or profiler API is added.
Each factor swaps one adjacent JIT kernel / ordinary host function pair. The
function bodies and other top-level statements stay unchanged. This can control
submitted line locations, not the judge's rewritten JIT source or cache state.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/2026-10-01/guide-candidates"
BASE_SHA = "88361f262dddcb58986b23b9d54d9219c1645529b98752e8b768e19fc2ace673"
PAIRS = {
    "J1": ("_fgs_t1i_mdq_kernel_g", "_fgs_tma1_intq_host"),
    "J2": ("_fgs_t1i_mdq_tma_pre_kernel", "_fgs_tma1_intq_host_pre"),
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def start(node):
    return min([node.lineno] + [d.lineno for d in node.decorator_list])


def structure(source):
    tree = ast.parse(source)
    funcs = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs[node.name].append(node)
    return (
        {k: [ast.dump(n, include_attributes=False) for n in nodes] for k, nodes in funcs.items()},
        [ast.dump(n, include_attributes=False) for n in tree.body if not isinstance(n, ast.FunctionDef)],
        {(k if len(nodes) == 1 else f"{k}#{i}"): start(n) for k, nodes in funcs.items() for i, n in enumerate(nodes)
         if any("jit" in ast.unparse(d) for d in n.decorator_list)},
    )


def swap(source, kernel, host):
    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert all(sum(isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body) == 1 for name in (kernel, host))
    a, b = functions[kernel], functions[host]
    idx = tree.body.index(a)
    assert tree.body[idx + 1] is b, "Functions must be adjacent top-level statements"
    assert a.decorator_list and not b.decorator_list
    assert not b.args.defaults, "No definition-time host argument evaluation permitted"
    lines = source.splitlines(keepends=True)
    lo, middle = start(a)-1, start(b)-1
    hi = start(tree.body[idx+2])-1 if idx+2 < len(tree.body) and isinstance(tree.body[idx+2], ast.FunctionDef) else b.end_lineno
    # hi includes the blank separator after the host, before the next decorator.
    result = "".join(lines[:lo] + lines[middle:hi] + lines[lo:middle] + lines[hi:])
    assert len(result.splitlines()) == len(lines)
    return result


def main():
    source_path = ROOT / "experiments/2026-10-01/candidates/p1_r1_all_pad_split_mid_v17_tiledshift.py"
    raw = source_path.read_bytes()
    assert sha(raw) == BASE_SHA, "The live candidate changed; recover exact SID source before proceeding"
    source = raw.decode()
    base_funcs, base_other, base_lines = structure(source)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "anchor_sid152976_88361f262ddd.py").write_bytes(raw)
    result = {"base_sid": 152976, "base_sha256": BASE_SHA, "submitted_by_this_script": False, "candidates": []}
    for name, factors in (("J1", ["J1"]), ("J2", ["J2"]), ("J12", ["J1", "J2"])):
        changed = source
        for factor in factors:
            changed = swap(changed, *PAIRS[factor])
        funcs, other, lines = structure(changed)
        assert funcs == base_funcs and other == base_other
        shifted = {k: {"before": base_lines[k], "after": v} for k, v in lines.items() if v != base_lines[k]}
        assert set(shifted) == {PAIRS[f][0] for f in factors}, shifted
        file = DEST / f"p1_layout_{name}_unmeasured.py"
        compile(changed, str(file), "exec")  # Parse/compile Python only; no imports/GPU execution.
        file.write_text(changed)
        result["candidates"].append({"name": name, "path": str(file.relative_to(ROOT)), "sha256": sha(changed.encode()),
                                     "same_function_bodies_and_other_top_level_ast": True,
                                     "changed_submitted_jit_start_lines": shifted, "gpu_status": "UNTESTED"})
    result["limits"] = ["Submitted source control only; runtime source rewriting and cache identity are unobserved.",
                         "No correctness/performance/anomaly success claim without official full-AC results.",
                         "No causal estimate is possible from three nonrandom first-SHA trials."]
    (OUT := ROOT / "reports/2026-10-01-controlled-layout-probes.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
