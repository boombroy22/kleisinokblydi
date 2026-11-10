"""Print per-group bounds of an OBJ file (used to measure the R6 mannequin)."""
import sys

path = sys.argv[1]
groups, gface, verts = {}, {}, []
cur = None
for line in open(path):
    p = line.split()
    if not p:
        continue
    if p[0] == "g":
        cur = p[1]
        groups.setdefault(cur, [])
        gface.setdefault(cur, 0)
    elif p[0] == "v":
        verts.append(tuple(map(float, p[1:4])))
        groups[cur].append(len(verts) - 1)
    elif p[0] == "f":
        gface[cur] += 1

for axis, name in enumerate("xyz"):
    vals = [v[axis] for v in verts]
    print(f"ALL {name}: [{min(vals):.3f}, {max(vals):.3f}]")
for g, idx in groups.items():
    b = []
    for axis in range(3):
        vals = [verts[i][axis] for i in idx]
        b.append((min(vals), max(vals)))
    size = " ".join(f"{hi - lo:.3f}" for lo, hi in b)
    rng = " ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in b)
    print(f"{g}: verts={len(idx)} faces={gface[g]} range={rng} size={size}")
