"""Full geometry preview: cloth + decorations with flat colours, budget and clearance report."""
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloak import blender_io as bio  # noqa: E402
from cloak import checks  # noqa: E402
from cloak.cloth import build_cloth  # noqa: E402
from cloak.drape import Drape  # noqa: E402
from cloak.layout import Layout  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANNEQUIN = os.path.join(ROOT, ".hoplite/attachments/art_upload_1dd2865d39204b0593ed52ac7ad16914/R6 character Maneken.obj")
OUT = "/tmp/previews"
os.makedirs(OUT, exist_ok=True)
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
views = [a for a in args if a in bio.VIEWS] or ["front", "side", "back"]

t0 = time.time()
bio.clear_scene()
d = Drape()
layers, cloth_parts = build_cloth(d)
lay = Layout(d)
deco = lay.build_all()
parts = cloth_parts + deco

cat = defaultdict(int)
for p in parts:
    key = p.name.rstrip("0123456789")
    for pref in ("Cape", "Pauldron", "Crest", "Crescent", "Barnacle", "Seaweed", "Net", "Chain", "Shell", "Hook",
                 "Lantern", "Starfish"):
        if p.name.startswith(pref) or pref in p.name:
            key = pref
            break
    cat[key] += p.tri_count()
total = sum(cat.values())
for k, v in sorted(cat.items(), key=lambda kv: -kv[1]):
    print(f"  {k:14s} {v:6d}")
print("TOTAL TRIS:", total)

print("BODY PENETRATION:", checks.body_report(parts) or "none")
print("CLOTH PENETRATION:", checks.cloth_report(deco, d, layers, min_clear=0.002) or "none")

col = bio.get_collection("Cloak")
mcol = bio.get_collection("Mannequin")
palette = {
    "cloth_navy": (0.03, 0.045, 0.075), "cloth_cream": (0.62, 0.55, 0.42), "bone": (0.66, 0.6, 0.48),
    "skull": (0.7, 0.64, 0.52), "barnacle": (0.5, 0.48, 0.44), "rope": (0.35, 0.25, 0.14),
    "iron": (0.25, 0.1, 0.04), "iron_anchor": (0.32, 0.13, 0.05), "shell": (0.75, 0.68, 0.58),
    "lantern_glass": (0.1, 0.6, 0.6), "compass": (0.5, 0.38, 0.15), "starfish": (0.6, 0.4, 0.22),
    "seaweed": (0.2, 0.24, 0.05), "net": (0.25, 0.18, 0.1),
}
mats = {}
for p in parts:
    key = p.material.replace("_outer", "").replace("_inner", "")
    if key not in mats:
        c = palette.get(key, (1, 0, 1))
        mats[key] = bio.simple_material("pv_" + key, c, emission=(0.2, 1.0, 1.0) if key == "lantern_glass" else None)
    bio.make_object(p, col, mats[key])
grey = bio.simple_material("mannequin", (0.55, 0.55, 0.55), 0.6)
for ob in bio.load_mannequin(MANNEQUIN, mcol):
    ob.data.materials.append(grey)
print("scene built in %.1fs" % (time.time() - t0))

bio.setup_render(res=(560, 700), samples=16)
import bpy  # noqa: E402

bpy.context.scene.render.film_transparent = True
for v in views:
    bio.render_view(v, os.path.join(OUT, f"full_{v}.png"))
print("done in %.1fs" % (time.time() - t0))
