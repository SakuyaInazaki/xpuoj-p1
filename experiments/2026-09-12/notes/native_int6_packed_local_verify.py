"""Bounded CPU verification for the frozen packed native-INT6 diagnostic."""

import ast
import hashlib
import json
import random
import symtable
from pathlib import Path


CANDIDATE = Path(__file__).parents[1] / "candidates" / "native_int6_md_custom_bench_packed.py"
EXPECTED_SHA256 = "4077d317ac20b4446b26645c744b33a34dfef1fb6cbf1b4bd8aa10ec36107d57"


def pack(values):
    codes = [value & 0x3F for value in values]
    low = sum((code & 0xF) << (4 * slot) for slot, code in enumerate(codes))
    high = sum(((code >> 4) & 0x3) << (2 * slot) for slot, code in enumerate(codes))
    return low, high


def unpack_swar(low, high):
    low = (low | (low << 8)) & 0x00FF00FF
    low = (low | (low << 4)) & 0x0F0F0F0F
    high = (high | (high << 12)) & 0x000F000F
    high = (high | (high << 6)) & 0x03030303
    word = low | (high << 4)
    word |= (word & 0x20202020) * 6
    raw = [(word >> (8 * slot)) & 0xFF for slot in range(4)]
    return [value - 256 if value >= 128 else value for value in raw]


def jit_free_globals(source):
    tree = ast.parse(source)
    table = symtable.symtable(source, str(CANDIDATE), "exec")
    children = {child.get_name(): child for child in table.get_children() if child.get_type() == "function"}
    names = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        is_jit = any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            and decorator.attr == "jit"
            for decorator in node.decorator_list
        )
        if is_jit:
            child = children[node.name]
            globals_used = sorted(
                symbol.get_name()
                for symbol in child.get_symbols()
                if symbol.is_global() and symbol.is_referenced()
            )
            names.append({"kernel": node.name, "free_globals": globals_used})
    return names


def main():
    source = CANDIDATE.read_text()
    actual_sha = hashlib.sha256(source.encode()).hexdigest()
    assert actual_sha == EXPECTED_SHA256

    exhaustive = 0
    for slot in range(4):
        for code in range(64):
            values = [0, 0, 0, 0]
            values[slot] = code if code < 32 else code - 64
            assert unpack_swar(*pack(values)) == values
            exhaustive += 1

    rng = random.Random(9122026)
    random_groups = 10000
    for _ in range(random_groups):
        values = [rng.randrange(-32, 32) for _ in range(4)]
        assert unpack_swar(*pack(values)) == values

    audit = jit_free_globals(source)
    assert len(audit) == 8
    assert all(set(item["free_globals"]) <= {"tl", "range"} for item in audit)
    compile(source, str(CANDIDATE), "exec", ast.PyCF_ONLY_AST)
    print(json.dumps({
        "candidate_sha256": actual_sha,
        "swar_exhaustive_slot_codes": exhaustive,
        "swar_random_seed": 9122026,
        "swar_random_groups": random_groups,
        "swar_passed": True,
        "ast_parse_passed": True,
        "jit_free_global_audit": audit,
    }, indent=2))


if __name__ == "__main__":
    main()
