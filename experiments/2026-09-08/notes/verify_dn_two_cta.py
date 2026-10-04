import ast
import math
import random
from pathlib import Path


BASE = Path("p1/kernel.py")
CAND = Path("experiments/2026-09-08/candidates/dn_two_cta_shortk.py")
TARGETS = {
    (16384, 2048, 32, 1024, 4),
    (8192, 3584, 64, 1024, 8),
    (16384, 4096, 96, 1024, 3),
    (65536, 1024, 32, 1024, 2),
}


def defs(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


base = ast.parse(BASE.read_text())
cand = ast.parse(CAND.read_text())
cdefs = defs(cand)

target_assign = next(
    node for node in cand.body
    if isinstance(node, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "_DN_TWO_CTA_SET" for t in node.targets)
)
assert set(ast.literal_eval(target_assign.value)) == TARGETS

host = cdefs["_dn_tma2_f8_host"]
assert host.args.args[-1].arg == "two_cta"
assert isinstance(host.args.defaults[-1], ast.Constant) and host.args.defaults[-1].value is False
guard = next(node for node in host.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "two_cta")
launch = next(node for node in ast.walk(guard) if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript))
assert ast.literal_eval(launch.func.slice) == (264,)
keywords = {kw.arg: ast.literal_eval(kw.value) for kw in launch.keywords}
assert keywords == {
    "BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 32,
    "FLAT": True, "num_warps": 8, "num_stages": 2, "maxnreg": 128,
}
descs = [node for node in ast.walk(guard) if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "TensorDescriptor"]
assert len(descs) == 1 and ast.literal_eval(descs[0].args[-1]) == [64, 128]

kernel = cdefs["_dn_tma2_f8_kernel"]
num_pid = next(
    node for node in kernel.body if isinstance(node, ast.Assign)
    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "num_pid"
)
assert isinstance(num_pid.value, ast.Call) and isinstance(num_pid.value.func, ast.Attribute)
assert num_pid.value.func.attr == "num_programs"
ranges = [node for node in ast.walk(kernel) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "range"]
assert any(any(isinstance(arg, ast.Name) and arg.id == "num_pid" for arg in call.args) for call in ranges)

meta = cdefs["_prepare_moe_metadata_bm64"]
meta_consts = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in meta.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)
}
assert meta_consts["num_sms"] == 132 and meta_consts["block_m"] == 64

run = cdefs["_run_replicated"]
flag_assign = next(node for node in run.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "use_dn_two_cta")
assert isinstance(flag_assign.value, ast.Compare)
flagged_calls = [
    node for node in ast.walk(run) if isinstance(node, ast.Call)
    and getattr(node.func, "id", None) == "_dn_tma2_f8_host"
    and any(kw.arg == "two_cta" for kw in node.keywords)
]
assert len(flagged_calls) == 1


class Strip(ast.NodeTransformer):
    def visit_Module(self, node):
        node.body = [
            child for child in node.body
            if not (isinstance(child, ast.FunctionDef) and child.name == "_prepare_moe_metadata_bm64")
            and not (
                isinstance(child, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "_DN_TWO_CTA_SET" for t in child.targets)
            )
        ]
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "_dn_tma2_f8_host":
            node.args.args = node.args.args[:-1]
            node.args.defaults = node.args.defaults[:-1]
            node.body = [
                child for child in node.body
                if not (isinstance(child, ast.If) and isinstance(child.test, ast.Name) and child.test.id == "two_cta")
            ]
        elif node.name == "_run_replicated":
            node.body = [
                child for child in node.body
                if not (
                    isinstance(child, ast.Assign) and isinstance(child.targets[0], ast.Name)
                    and child.targets[0].id == "use_dn_two_cta"
                )
            ]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if getattr(node.func, "id", None) == "_dn_tma2_f8_host":
            node.keywords = [kw for kw in node.keywords if kw.arg != "two_cta"]
        return node


stripped = ast.fix_missing_locations(Strip().visit(cand))
assert ast.dump(stripped, include_attributes=False) == ast.dump(base, include_attributes=False)


def verify_counts(counts, nblocks):
    total = sum(counts)
    capacity = math.ceil(total / 64) + len(counts)
    tile_count = sum(math.ceil(c / 64) for c in counts)
    assert tile_count <= capacity
    seen = set()
    prefix = 0
    for expert, count in enumerate(counts):
        for tile in range(math.ceil(count / 64)):
            row0 = prefix + tile * 64
            for col in range(nblocks):
                for row in range(row0, min(row0 + 64, prefix + count)):
                    key = (row, col)
                    assert key not in seen
                    seen.add(key)
        prefix += count
    assert prefix == total
    assert len(seen) == total * nblocks


rng = random.Random(20260908)
for T, H, E, _I, topk in TARGETS:
    assert 16 <= E <= 96 and _I == 1024
    M = T * topk
    edge = [0, 1, 63, 64, 65] + [0] * (E - 5)
    edge[-1] += M - sum(edge)
    verify_counts(edge, H // 256)
    cuts = sorted(rng.randrange(M + 1) for _ in range(E - 1))
    points = [0, *cuts, M]
    counts = [points[i + 1] - points[i] for i in range(E)]
    verify_counts(counts, H // 256)

print("dn two-CTA static/CPU checks: PASS")
