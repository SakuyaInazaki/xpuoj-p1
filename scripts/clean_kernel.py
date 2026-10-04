#!/usr/bin/env python3
"""Prune unreachable module functions without importing or running GPU code.

Keep every non-function module statement and the transitive dependencies of
run_kernel. Resolve repeated definitions using Python's final module binding.
The report verifies the retained AST and records the original source hash.
"""

import argparse
import ast
import hashlib
import io
import json
import tokenize
from pathlib import Path


FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def loaded_names(node):
    return {
        n.id for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def analyze(source):
    tree = ast.parse(source)
    definitions = {}
    for node in tree.body:
        if isinstance(node, FUNCTIONS):
            definitions.setdefault(node.name, []).append(node)
    roots = {"run_kernel"}
    for node in tree.body:
        if not isinstance(node, FUNCTIONS):
            roots.update(loaded_names(node))
    # No early aliases/defaults may capture a superseded function object.
    duplicates = {k for k, nodes in definitions.items() if len(nodes) > 1}
    assert not (roots & duplicates), "Early reference to an overwritten function"
    for nodes in definitions.values():
        for node in nodes:
            early_nodes = node.decorator_list + node.args.defaults
            early_nodes += [n for n in node.args.kw_defaults if n is not None]
            assert not any(loaded_names(n) & duplicates for n in early_nodes)
    reachable = set()
    pending = list(roots & definitions.keys())
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend((loaded_names(definitions[name][-1]) & definitions.keys()) - reachable)
    keep = [n for n in tree.body if not isinstance(n, FUNCTIONS)
            or (n.name in reachable and n is definitions[n.name][-1])]
    return tree, definitions, reachable, keep


def clean(source):
    tree, definitions, reachable, keep = analyze(source)
    removed = [n for n in tree.body if isinstance(n, FUNCTIONS) and n not in keep]
    lines = source.splitlines(keepends=True)
    skip = set()
    for node in removed:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        skip.update(range(start, node.end_lineno + 1))
    text = "".join(line for i, line in enumerate(lines, 1) if i not in skip)
    lines = text.splitlines(keepends=True)
    # Token locations distinguish real comments from strings and inline PTX.
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            lines[row - 1] = lines[row - 1][:column].rstrip() + "\n"
    compact = []
    blank = 0
    for line in lines:
        if line.strip():
            blank = 0
            compact.append(line.rstrip())
        else:
            blank += 1
            if blank <= 1:
                compact.append("")
    result = "\n".join(compact).strip() + "\n"
    result = (
        "# v926 source cleanup; frozen measured source: references/kernel_v926_measured.py\n"
        "# Math, launch configurations, call dispatch and cache behavior are retained.\n"
        "# See ../docs/OPTIMIZATION_GUIDE_V926_2026-09-30.md for known contract gaps.\n\n"
        + result
    )
    expected = ast.Module(body=keep, type_ignores=[])
    actual = ast.parse(result)
    assert ast.dump(expected, include_attributes=False) == ast.dump(actual, include_attributes=False)
    new_defs = {n.name: n for n in actual.body if isinstance(n, FUNCTIONS)}
    report = {
        "method": "Conservative final-definition dependency closure; all non-function statements retained",
        "source_sha256": digest(source.encode()),
        "clean_sha256": digest(result.encode()),
        "original_lines": len(source.splitlines()),
        "clean_lines": len(result.splitlines()),
        "original_functions": sum(map(len, definitions.values())),
        "retained_functions": len(new_defs),
        "retained_ast_equal": True,
        "gpu_tested": False,
        "removed": [{"name": n.name, "line": n.lineno, "end_line": n.end_lineno,
                     "reason": "overwritten" if n.name in reachable else "unreachable"}
                    for n in removed],
        "functions": [{"name": name, "old_line": definitions[name][-1].lineno,
                       "new_line": node.lineno, "ast_equal":
                       ast.dump(definitions[name][-1], include_attributes=False)
                       == ast.dump(node, include_attributes=False)}
                      for name, node in new_defs.items()],
    }
    assert all(row["ast_equal"] for row in report["functions"])
    return result, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    assert args.source.resolve() != args.output.resolve(), "Keep the input snapshot immutable"
    result, report = clean(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("removed", "functions")}, indent=2))


if __name__ == "__main__":
    main()
