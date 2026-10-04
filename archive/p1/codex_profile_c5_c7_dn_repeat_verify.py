"""Static verifier for the c5/c7 down-projection replay diagnostic.

The judge-only Triton/NVSHMEM modules are unavailable locally, so this checks
source structure and mechanically removes the diagnostic AST before comparing
it with the current baseline.
"""

import ast
import copy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "kernel.py"
CAND = ROOT / "codex_profile_c5_c7_dn_repeat.py"


def _call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return None


def _find_fn(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _strip_probe(tree):
    tree = copy.deepcopy(tree)
    host = _find_fn(tree, "_dn_tma2_f8_host")
    assert host.args.args[-1].arg == "repeat_dn"
    assert len(host.args.defaults) and isinstance(host.args.defaults[-1], ast.Constant)
    assert host.args.defaults[-1].value is False
    host.args.args.pop()
    host.args.defaults.pop()
    repeat_ifs = [
        n for n in host.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "repeat_dn"
    ]
    assert len(repeat_ifs) == 1
    assert len(repeat_ifs[0].body) == 1 and isinstance(repeat_ifs[0].body[0], ast.Expr)
    host.body.remove(repeat_ifs[0])

    run = _find_fn(tree, "_run_replicated")
    assigns = [
        n for n in ast.walk(run)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "profile_repeat_dn"
    ]
    assert len(assigns) == 1
    parent = next(
        n for n in ast.walk(run)
        if hasattr(n, "body") and isinstance(n.body, list) and assigns[0] in n.body
    )
    parent.body.remove(assigns[0])

    calls = [
        n for n in ast.walk(run)
        if isinstance(n, ast.Call) and _call_name(n.func) == "_dn_tma2_f8_host"
        and any(k.arg == "repeat_dn" for k in n.keywords)
    ]
    assert len(calls) == 1
    calls[0].keywords = [k for k in calls[0].keywords if k.arg != "repeat_dn"]
    ast.fix_missing_locations(tree)
    return tree


def main():
    base_source = BASE.read_text()
    cand_source = CAND.read_text()
    base = ast.parse(base_source)
    cand = ast.parse(cand_source)

    host = _find_fn(cand, "_dn_tma2_f8_host")
    launches = [
        n for n in ast.walk(host)
        if isinstance(n, ast.Call) and _call_name(n.func) == "_dn_tma2_f8_kernel"
    ]
    assert len(launches) == 2
    assert ast.dump(launches[0], include_attributes=False) == ast.dump(
        launches[1], include_attributes=False
    )

    kernel = _find_fn(cand, "_dn_tma2_f8_kernel")
    assert not any(
        isinstance(n, ast.Call) and (_call_name(n.func) or "").startswith("atomic")
        for n in ast.walk(kernel)
    )
    assert sum(
        isinstance(n, ast.Call) and _call_name(n.func) == "store"
        for n in ast.walk(kernel)
    ) == 2

    run = _find_fn(cand, "_run_replicated")
    guard = next(
        n.value for n in ast.walk(run)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == "profile_repeat_dn"
    )
    guard_text = ast.unparse(guard)
    assert "_CALLN" not in guard_text
    for literal in ("T == 8192", "H == 3584", "E == 64", "I == 2560", "k == 8",
                    "T == 16384", "H == 4096", "E == 96", "I == 2048", "k == 3"):
        assert literal in guard_text

    stripped = _strip_probe(cand)
    assert ast.dump(stripped, include_attributes=False) == ast.dump(base, include_attributes=False)

    # No diagnostic-only timing or synchronization machinery may enter the delta.
    added_markers = ("torch.cuda.Event", "synchronize(", "barrier", "nvshmem_create_tensor")
    for marker in added_markers:
        assert cand_source.count(marker) == base_source.count(marker)

    print("PASS: exact duplicate dn launch, no atomics, shape-only guard, stripped AST == kernel.py")


if __name__ == "__main__":
    main()
