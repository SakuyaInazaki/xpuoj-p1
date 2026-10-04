from pathlib import Path
import ast,hashlib,difflib
OUT=Path(__file__).resolve().parent
p=OUT/'p1_s1_c6_v1.py'; s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='05828fef7359fd70159046bc026b414219d7fd28559146cd8793799aaaa15fa7'
t=ast.parse(s); fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_s1_c6_gq_tok_params_kernel')
loop=next(n for n in fn.body if isinstance(n,ast.For))
old=ast.get_source_segment(s,loop)
# Indent preserved around the exact loop, with every branch arithmetic AST unchanged.
lines=s.splitlines(True)
block='    j = tl.arange(0, K_BRANCH)\n'+''.join(line[4:] for line in lines[loop.lineno:loop.end_lineno])
new=''.join(lines[:loop.lineno-1])+block+''.join(lines[loop.end_lineno:])
a=ast.parse(new); fs={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)}
for name,n in {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}.items():
 if name!=fn.name:assert ast.dump(n)==ast.dump(fs[name]),name
newfn=fs[fn.name]; newfn.body=newfn.body[:-len(loop.body)-1]+[loop]
assert ast.dump(fn)==ast.dump(newfn)
compile(new,str(OUT/'p1_s1_c6_v2_vector.py'),'exec')
(OUT/'p1_s1_c6_v2_vector.py').write_text(new)
(OUT/'v2_vector.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='p1_s1_c6_v1.py',tofile='p1_s1_c6_v2_vector.py')))
print(hashlib.sha256(new.encode()).hexdigest())
