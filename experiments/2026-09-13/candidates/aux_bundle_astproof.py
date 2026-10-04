"""_AXB[0]==0 fold proof for aux_bundle.py.

Three mechanical AST transforms, then ast.dump equality against the frozen
baseline p1/kernel.py:

  1) drop every new TOP-LEVEL definition/assignment whose name does not exist
     in the baseline (_AXB, _AXGSET, _AXG, _route_fq_kernel, _route_fq);
  2) drop every `Assign` whose single target is one of the gate-only scratch
     locals (_axq, _axs) -- with the gate off nothing reads them;
  3) replace every `If` whose test mentions `_AXB` by its `orelse`
     (= the path actually taken when _AXB[0] is falsy), or delete it when it
     has no `orelse`.

Equality of the dumps => with _AXB[0] = 0 the candidate is the baseline.

usage: python3 experiments/2026-09-13/candidates/aux_bundle_astproof.py
"""
import ast
import sys

BASE = "p1/kernel.py"
NEW = "experiments/2026-09-13/candidates/aux_bundle.py"
GATE = "_AXB"
SCRATCH = ("_axq", "_axs")


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
        self.dropped_assign = 0
        self.folded_if = 0
        self.deleted_if = 0

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in SCRATCH:
                self.dropped_assign += 1
                return None
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == GATE:
                self.dropped_assign += 1
                return None
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


def main():
    base = ast.parse(open(BASE).read())
    new = ast.parse(open(NEW).read())
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
    f = Fold()
    new = f.visit(new)
    ast.fix_missing_locations(new)
    db, dn = ast.dump(base), ast.dump(new)
    print("dropped top-level names (%d): %s" % (len(added), ", ".join(added)))
    print("dropped scratch assigns: %d ; If folded to orelse: %d ; If deleted: %d"
          % (f.dropped_assign, f.folded_if, f.deleted_if))
    if db == dn:
        print("AST IDENTICAL after folding %s[0] -> 0.  PROOF OK" % GATE)
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
