"""_EPP[0]==0 fold proof for epi_port.py.

Four mechanical AST transforms, then ast.dump equality against the frozen
baseline p1/kernel.py:

  1) drop every new TOP-LEVEL definition/assignment whose name does not exist
     in the baseline (_EPP);
  2) drop every function parameter named EPP (the tl.constexpr gate handed to
     the two touched kernels);
  3) drop every call keyword named EPP (the two launch sites);
  4) replace every `If` whose test mentions EPP by its `orelse`
     (= the path actually taken when EPP is falsy).

Equality of the dumps => with _EPP[0] = 0 the candidate is the baseline.
The candidate adds no @triton_dist.jit / @triton.jit source, so the jit-source
count is checked separately and must be unchanged.

usage: python3 experiments/2026-09-13/candidates/epi_port_astproof.py
"""
import ast
import sys

BASE = "p1/kernel.py"
NEW = "experiments/2026-09-13/candidates/epi_port.py"
GATE = "EPP"
TOPGATE = "_EPP"


def toplevel_names(tree):
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
    return out


def mentions_gate(node):
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and x.id == GATE:
            return True
    return False


class Fold(ast.NodeTransformer):
    def __init__(self):
        self.dropped_arg = 0
        self.dropped_kw = 0
        self.folded_if = 0
        self.deleted_if = 0

    def visit_arguments(self, node):
        self.generic_visit(node)
        keep = [a for a in node.args if a.arg != GATE]
        self.dropped_arg += len(node.args) - len(keep)
        node.args = keep
        keep = [a for a in node.kwonlyargs if a.arg != GATE]
        self.dropped_arg += len(node.kwonlyargs) - len(keep)
        if len(keep) != len(node.kwonlyargs):
            node.kw_defaults = [d for a, d in zip(node.kwonlyargs, node.kw_defaults) if a.arg != GATE]
        node.kwonlyargs = keep
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        keep = [k for k in node.keywords if k.arg != GATE]
        self.dropped_kw += len(node.keywords) - len(keep)
        node.keywords = keep
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if mentions_gate(node.test):
            if node.orelse:
                self.folded_if += 1
                return node.orelse
            self.deleted_if += 1
            return None
        return node


def jit_source_count(tree):
    n_dist = 0
    n_plain = 0
    for x in ast.walk(tree):
        if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in x.decorator_list:
            s = ast.unparse(d)
            if s == "triton_dist.jit":
                n_dist += 1
            elif s == "triton.jit":
                n_plain += 1
    return n_dist, n_plain


def main():
    base = ast.parse(open(BASE).read())
    new = ast.parse(open(NEW).read())

    jb = jit_source_count(base)
    jn = jit_source_count(new)
    print("jit sources  base triton_dist/triton = %d/%d ; candidate = %d/%d" % (jb + jn))
    if jb != jn:
        print("JIT SOURCE COUNT CHANGED -- candidate adds a kernel source")
        return 1

    bn = toplevel_names(base)
    added, kept = [], []
    for n in new.body:
        nm = None
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nm = n.name
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            nm = n.targets[0].id
        if nm is not None and nm not in bn:
            added.append(nm)
            continue
        kept.append(n)
    new.body = kept
    if added != [TOPGATE]:
        print("unexpected new top-level names: %s" % added)
        return 1

    f = Fold()
    new = f.visit(new)
    ast.fix_missing_locations(new)
    db, dn = ast.dump(base), ast.dump(new)
    print("dropped top-level names (%d): %s" % (len(added), ", ".join(added)))
    print("dropped %s params: %d ; dropped %s call keywords: %d ; If folded to orelse: %d ; If deleted: %d"
          % (GATE, f.dropped_arg, GATE, f.dropped_kw, f.folded_if, f.deleted_if))
    if db == dn:
        print("AST IDENTICAL after folding %s[0] -> 0.  PROOF OK" % TOPGATE)
        return 0
    print("AST DIFFERS -- residual is not the baseline")
    for i in range(min(len(db), len(dn))):
        if db[i] != dn[i]:
            print("first divergence at char %d" % i)
            print("base: ..." + db[max(0, i - 300):i + 300])
            print("new : ..." + dn[max(0, i - 300):i + 300])
            break
    return 1


sys.exit(main())
