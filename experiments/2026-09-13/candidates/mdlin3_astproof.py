"""Fold proof for the three "md prologue sinking + cross-tile pipelining" probes
that push the already-promoted v830 transform onto the remaining three md kernels.

Baseline is the frozen production file p1/kernel.py (= v830, which already carries
`_MDL` / `LIN` for `_fgs_t1i_mdq_tma_kernel`; those two names exist in the baseline
and are therefore NOT touched here).

Per candidate the gate pair is (module gate, tl.constexpr parameter):

  mdlin_g_c38.py     _MDLG / LNG   -> _fgs_t1i_mdq_kernel_g      (c3..c8)
  mdlin_gq_c910.py   _MDLQ / LNQ   -> _fgs_tma1_kernel_gq        (c9/c10)
  mdlin_q8_c12.py    _MDLP / LNP   -> _fgs_tma2_int_pm_q8_kernel (c1/c2)

Four mechanical AST transforms (same shape as mdlin_astproof.py, with the If rule
widened to also match the module gate so that the duplicated host launch site in
mdlin_q8_c12.py folds back as well):

  1) drop every new TOP-LEVEL definition/assignment whose name does not exist in
     the baseline (must be exactly the module gate);
  2) drop every function parameter named <PARAM>;
  3) drop every call keyword named <PARAM>;
  4) replace every `If` whose test mentions <PARAM> or <MODULE GATE> by its
     `orelse` (= the path actually taken when the gate is falsy).

Equality of the two ast.dump strings => with <MODULE GATE>[0] = 0 the candidate IS
the baseline, including the c1/c2 `maxnreg=168` launch keyword.  The @triton_dist.jit
/ @triton.jit source counts are compared too and must be unchanged.

usage: python3 experiments/2026-09-13/candidates/mdlin3_astproof.py [candidate.py ...]
"""
import ast
import sys

BASE = "p1/kernel.py"
GATES = {
    "mdlin_g_c38.py": ("_MDLG", "LNG"),
    "mdlin_gq_c910.py": ("_MDLQ", "LNQ"),
    "mdlin_q8_c12.py": ("_MDLP", "LNP"),
}
DEFAULT = ["experiments/2026-09-13/candidates/" + n for n in
           ("mdlin_g_c38.py", "mdlin_gq_c910.py", "mdlin_q8_c12.py")]


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


class Fold(ast.NodeTransformer):
    def __init__(self, param, topgate):
        self.param = param
        self.names = {param, topgate}
        self.dropped_arg = 0
        self.dropped_kw = 0
        self.folded_if = 0
        self.deleted_if = 0

    def mentions(self, node):
        for x in ast.walk(node):
            if isinstance(x, ast.Name) and x.id in self.names:
                return True
        return False

    def visit_arguments(self, node):
        self.generic_visit(node)
        keep = [a for a in node.args if a.arg != self.param]
        self.dropped_arg += len(node.args) - len(keep)
        node.args = keep
        keep = [a for a in node.kwonlyargs if a.arg != self.param]
        self.dropped_arg += len(node.kwonlyargs) - len(keep)
        if len(keep) != len(node.kwonlyargs):
            node.kw_defaults = [d for a, d in zip(node.kwonlyargs, node.kw_defaults)
                                if a.arg != self.param]
        node.kwonlyargs = keep
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        keep = [k for k in node.keywords if k.arg != self.param]
        self.dropped_kw += len(node.keywords) - len(keep)
        node.keywords = keep
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self.mentions(node.test):
            if node.orelse:
                self.folded_if += 1
                return node.orelse
            self.deleted_if += 1
            return None
        return node


def jit_source_count(tree):
    n_dist = n_plain = 0
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


def check(path, base_src):
    topgate, param = GATES[path.rsplit("/", 1)[-1]]
    base = ast.parse(base_src)
    new = ast.parse(open(path).read())
    print("--- %s   gate %s / param %s" % (path, topgate, param))

    jb, jn = jit_source_count(base), jit_source_count(new)
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
    if added != [topgate]:
        print("unexpected new top-level names: %s" % added)
        return 1

    f = Fold(param, topgate)
    new = f.visit(new)
    ast.fix_missing_locations(new)
    db, dn = ast.dump(base), ast.dump(new)
    print("dropped top-level names (%d): %s" % (len(added), ", ".join(added)))
    print("dropped %s params: %d ; dropped %s call keywords: %d ; If folded to orelse: %d ; If deleted: %d"
          % (param, f.dropped_arg, param, f.dropped_kw, f.folded_if, f.deleted_if))
    if db == dn:
        print("AST IDENTICAL after folding %s[0] -> 0.  PROOF OK" % topgate)
        return 0
    print("AST DIFFERS -- residual is not the baseline")
    for i in range(min(len(db), len(dn))):
        if db[i] != dn[i]:
            print("first divergence at char %d" % i)
            print("base: ..." + db[max(0, i - 300):i + 300])
            print("new : ..." + dn[max(0, i - 300):i + 300])
            break
    return 1


def main():
    paths = sys.argv[1:] or DEFAULT
    base_src = open(BASE).read()
    rc = 0
    for p in paths:
        rc |= check(p, base_src)
    return rc


sys.exit(main())
