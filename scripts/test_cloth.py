"""Quick geometry preview for the cloth layers (run inside Blender)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloak import blender_io as bio  # noqa: E402
from cloak.cloth import build_cloth  # noqa: E402
from cloak.drape import Drape  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANNEQUIN = os.path.join(ROOT, ".hoplite/attachments/art_upload_1dd2865d39204b0593ed52ac7ad16914/R6 character Maneken.obj")
OUT = "/tmp/previews"
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
bio.clear_scene()
d = Drape()
print("U =", d.U)
layers, parts = build_cloth(d)
total = 0
for p in parts:
    print(f"{p.name:24s} verts={len(p.verts):6d} tris={p.tri_count():6d}")
    total += p.tri_count()
print("cloth total tris:", total, "build time %.1fs" % (time.time() - t0))

col = bio.get_collection("Cloak")
mcol = bio.get_collection("Mannequin")
mats = {
    "cloth_navy_outer": bio.simple_material("navy_o", (0.035, 0.05, 0.085)),
    "cloth_navy_inner": bio.simple_material("navy_i", (0.02, 0.03, 0.05)),
    "cloth_cream_outer": bio.simple_material("cream_o", (0.62, 0.55, 0.42)),
    "cloth_cream_inner": bio.simple_material("cream_i", (0.4, 0.35, 0.27)),
}
for p in parts:
    bio.make_object(p, col, mats[p.material])
grey = bio.simple_material("mannequin", (0.55, 0.55, 0.55), 0.6)
for ob in bio.load_mannequin(MANNEQUIN, mcol):
    ob.data.materials.append(grey)

bio.setup_render(res=(480, 600), samples=12)
for v in ("front", "side", "back"):
    bio.render_view(v, os.path.join(OUT, f"cloth_{v}.png"))
print("done in %.1fs" % (time.time() - t0))
