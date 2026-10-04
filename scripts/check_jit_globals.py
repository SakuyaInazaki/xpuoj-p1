#!/usr/bin/env python3
"""Find plain module constants read directly by @triton.jit functions.

This is a narrow, source-only preflight check.  It parses Python with ``ast``;
it never imports or executes the files being checked.
"""

from __future__ import annotations

import argparse
import ast
import builtins
from pathlib import Path
import sys


def is_plain_constant(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex, str, bytes)) and not isinstance(
            node.value, bool
        )
    if isinstance(node, ast.Tuple):
        return all(is_plain_constant(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return is_plain_constant(node.operand)
    return False


def dotted_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(target_names(item) for item in node.elts))
    if isinstance(node, ast.Starred):
        return target_names(node.value)
    return set()


class FunctionNames(ast.NodeVisitor):
    def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.root = root
        self.bound = {
            arg.arg
            for arg in (
                root.args.posonlyargs + root.args.args + root.args.kwonlyargs
                + ([root.args.vararg] if root.args.vararg else [])
                + ([root.args.kwarg] if root.args.kwarg else [])
            )
        }
        self.loads: list[ast.Name] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self.root:
            for statement in node.body:
                self.visit(statement)
        else:
            self.bound.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # A lambda is a separate lexical scope and irrelevant to the JIT body.
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loads.append(node)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bound.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.bound.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.bound.update(alias.asname or alias.name for alias in node.names)


def analyze(path: Path) -> list[tuple[int, str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    triton_names = {"triton"}
    tl_names = {"tl"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "triton":
                    triton_names.add(alias.asname or "triton")
                elif alias.name == "triton.language":
                    tl_names.add(alias.asname or "triton.language")
        elif isinstance(node, ast.ImportFrom) and node.module == "triton":
            for alias in node.names:
                if alias.name == "language":
                    tl_names.add(alias.asname or "language")

    constants: dict[str, bool] = {}
    jit_helpers: set[str] = set()
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
            if any(dotted_name(dec.func if isinstance(dec, ast.Call) else dec) in
                   {f"{name}.jit" for name in triton_names} for dec in node.decorator_list):
                jit_helpers.add(node.name)
        elif isinstance(node, ast.Assign) and is_plain_constant(node.value):
            for target in node.targets:
                for name in target_names(target):
                    constants[name] = False
        elif isinstance(node, ast.AnnAssign) and node.value is not None and is_plain_constant(node.value):
            for name in target_names(node.target):
                constants[name] = dotted_name(node.annotation) in {
                    f"{tl_name}.constexpr" for tl_name in tl_names
                }

    allowed_globals = set(dir(builtins)) | triton_names | tl_names | jit_helpers
    findings: list[tuple[int, str, str]] = []
    for function in functions:
        if function.name not in jit_helpers:
            continue
        names = FunctionNames(function)
        names.visit(function)
        first: dict[str, int] = {}
        for load in names.loads:
            if (load.id in constants and not constants[load.id] and load.id not in names.bound
                    and load.id not in allowed_globals):
                first.setdefault(load.id, load.lineno)
        findings.extend((line, function.name, name) for name, line in first.items())
    return sorted(findings)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AST preflight for plain module constants captured by @triton.jit functions"
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    count = 0
    for path in args.paths:
        try:
            findings = analyze(path)
        except (OSError, UnicodeError, SyntaxError) as exc:
            print(f"{path}: error: {exc}", file=sys.stderr)
            return 2
        for line, function, variable in findings:
            print(f"{path}:{line}: {function}: module constant {variable} is read directly")
        count += len(findings)
    if count:
        print(f"check_jit_globals: found {count} direct module-constant reference(s)")
        return 1
    print(f"check_jit_globals: OK ({len(args.paths)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
