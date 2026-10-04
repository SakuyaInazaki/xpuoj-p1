# -*- coding: utf-8 -*-
"""Constant-fold p1/kernel_v780_ragged.py under `_RAG_EN[0] == 0` and prove the
result is *identical* to the accepted baseline p1/kernel_v760a_tanh.py.

Why the folding rules below are sound when `_RAG_EN[0] == 0`:

  R1 (gate)  `_rag_on` is a pure-Python `and`-chain whose first term is
             `_RAG_EN[0] == 1`.  With `_RAG_EN = [0]` it is statically False,
             so the `if _rag_on:` arm is dead and `_rag_tail` is bound to
             `None` and never re-bound.  The live statement is exactly the
             baseline line.
  R2 (args)  The only two uses of `_rag_tail` are the keyword arguments
             `tail=_rag_tail`; with `_rag_tail is None` they are equal to the
             parameter default `tail=None`, so removing the argument *and* the
             `tail=None` parameter cannot change any call.
  R3 (tails) Both `if tail is not None:` bodies are then unreachable.
  R4 (defs)  `_prepare_moe_metadata_rag` / `_rag_meta_kernel` are then never
             referenced.  `triton_dist.jit` compiles lazily (see
             work/repo/python/triton_dist/jit.py:372 -> triton.jit), so an
             un-launched kernel costs zero ptxas.
  => everything the process can execute is the baseline text, verbatim.

Run:  python3 sandbox/fold_v780.py
"""
import ast
import difflib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAND = os.path.join(ROOT, "p1", "kernel_v780_ragged.py")
OFF = os.path.join(ROOT, "p1", "kernel_v780a_ragoff.py")
BASE = os.path.join(ROOT, "p1", "kernel_v760a_tanh.py")

BASE_GATE = "    metadata = _prepare_moe_metadata(expert_counts, E, T * k)"


def fold(text):
    lines = text.split("\n")
    out = []
    i = 0
    removed = {"definitions": 0, "dn tail": 0, "md tail": 0, "gate": 0,
               "param": 0, "arg": 0}
    while i < len(lines):
        ln = lines[i]
        if "V780 RAGGED BEGIN" in ln:
            tag = ln.split("BEGIN (")[1].split(")")[0]
            module_level = not ln.startswith(" ")
            j = i
            while "V780 RAGGED END" not in lines[j]:
                j += 1
            removed[tag] = j - i + 1
            if tag == "gate":                       # R1
                out.append(BASE_GATE)
            i = j + 1
            if module_level:                        # the 2 blank lines we added
                while i < len(lines) and lines[i] == "":
                    i += 1
            continue
        if ln.endswith("):  # V780 RAGGED PARAM"):  # R2
            out.append(ln.replace(", tail=None):  # V780 RAGGED PARAM", "):"))
            removed["param"] += 1
            i += 1
            continue
        if ln.strip() == "tail=_rag_tail,  # V780 RAGGED ARG":   # R2
            removed["arg"] += 1
            i += 1
            continue
        out.append(ln)
        i += 1
    return "\n".join(out), removed


def main():
    base = io.open(BASE, encoding="utf-8").read()
    for name, path in (("kernel_v780_ragged.py", CAND),
                       ("kernel_v780a_ragoff.py", OFF)):
        cand = io.open(path, encoding="utf-8").read()
        folded, rm = fold(cand)
        d = list(difflib.unified_diff(base.split("\n"), folded.split("\n"),
                                      "v760a", name + " @ _RAG_EN=0", n=0))
        same_ast = ast.dump(ast.parse(base)) == ast.dump(ast.parse(folded))
        print("== %s" % name)
        print("   folded away: %s" % rm)
        print("   text  diff vs v760a : %d lines" % len(d))
        print("   AST   diff vs v760a : %s" % ("0 (identical)" if same_ast else "NON-EMPTY"))
        for l in d[:40]:
            print("     " + l)
        assert not d and same_ast, "fold did NOT reproduce the baseline"

    # And: the two shipped files differ only by the master switch.
    cand = io.open(CAND, encoding="utf-8").read()
    off = io.open(OFF, encoding="utf-8").read()
    d = [l for l in difflib.unified_diff(cand.split("\n"), off.split("\n"), n=0)
         if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    print("== kernel_v780_ragged.py vs kernel_v780a_ragoff.py : %s" % d)
    assert d == ["-_RAG_EN = [1]", "+_RAG_EN = [0]"]

    # Live-diff summary of the candidate against the baseline.
    d = list(difflib.unified_diff(base.split("\n"), cand.split("\n"), n=0))
    print("== kernel_v780_ragged.py vs v760a : +%d / -%d lines, %d hunks" % (
        sum(1 for l in d if l.startswith("+") and not l.startswith("+++")),
        sum(1 for l in d if l.startswith("-") and not l.startswith("---")),
        sum(1 for l in d if l.startswith("@@"))))
    print("ALL FOLD PROOFS PASS")


main()
