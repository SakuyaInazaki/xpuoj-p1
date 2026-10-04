"""cross.py <case> <target_pts> <cand:anchor,...>  — did each run reach target_pts with its OWN tb?"""
import sys, json, os
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'knobs6_data')
db = json.load(open(os.path.join(H, 'tk.json')))
case, tgt = sys.argv[1], int(sys.argv[2])
print(f"c{case}: reaching {tgt} pts needs tk <= tb*{100-tgt}/{tgt}")
for pair in sys.argv[3].split(','):
    cd, an = pair.split(':')
    row = []
    for tag, s in (('cand', cd), ('anch', an)):
        if s not in db: row.append(f"{tag} n/a"); continue
        tk, tb = db[s][case][0], db[s][case][1]
        pts = int(100*tb/(tb+tk)); need = tb*(100-tgt)/tgt
        row.append(f"{tag} {s} tk={tk:.4f} tb={tb:.3f} pts={pts} need<={need:.4f} {'REACHED' if tk<=need else 'no'}")
    print("  " + " | ".join(row))
