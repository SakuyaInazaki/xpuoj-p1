"""_FU[0]==0 折叠证明。

把 kernel_v800_fuse.py 的 AST 做两步机械变换：
  1) 删掉所有新增的顶层定义/赋值（名字在基线里根本不存在）；
  2) 把每个 test 里出现 `_FU` 的 If 节点替换成它的 orelse（= `_FU[0]` 恒 0 时的实际执行路径）。
然后与 kernel_v760a_tanh.py 的 AST 逐字比较 ast.dump。相等 ⇒ 闸门关掉后两份文件语义完全一致。

用法: python3 scripts/fu_astproof.py
"""
import ast
import sys


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


def mentions_fu(node):
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and x.id == "_FU":
            return True
    return False


class Fold(ast.NodeTransformer):
    def visit_Assign(self, node):
        # 3) 删掉「只写 _FU」的赋值语句（_FU 在基线里不存在，闸门关掉后它的值无人读取）
        self.generic_visit(node)
        if len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "_FU":
                return None
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if mentions_fu(node.test):
            if node.orelse:
                return node.orelse
            return None
        return node


def main():
    base = ast.parse(open("archive/p1/kernel_v760a_tanh.py").read())
    new = ast.parse(open("archive/p1/kernel_v800_fuse.py").read())
    bn = toplevel_names(base)
    added = []
    kept = []
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
    new = Fold().visit(new)
    ast.fix_missing_locations(new)
    db = ast.dump(base)
    dn = ast.dump(new)
    print("new top-level names dropped (%d): %s" % (len(added), ", ".join(added)))
    if db == dn:
        print("AST IDENTICAL after folding _FU[0] -> 0.  PROOF OK")
        return 0
    print("AST DIFFERS -- residual is not the baseline")
    for i in range(min(len(db), len(dn))):
        if db[i] != dn[i]:
            print("first divergence at char %d" % i)
            print("base: ..." + db[max(0, i - 200):i + 200])
            print("new : ..." + dn[max(0, i - 200):i + 200])
            break
    return 1


sys.exit(main())
