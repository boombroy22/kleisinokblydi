"""End-to-end build of the Shipwreck Leviathan Cloak (run inside Blender 4.5).

blender -b --factory-startup -P scripts/build.py -- [--size 2048] [--reuse-bake] [--quick]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cloak import bake as BK  # noqa: E402
from cloak import blender_io as bio  # noqa: E402
from cloak import checks  # noqa: E402
from cloak.cloth import build_cloth  # noqa: E402
from cloak.drape import Drape  # noqa: E402
from cloak.layout import Layout  # noqa: E402
from cloak.rig import CapeRig  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANNEQUIN = os.path.join(ROOT, ".hoplite/attachments/art_upload_1dd2865d39204b0593ed52ac7ad16914/R6 character Maneken.obj")
args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


SIZE = opt("--size", 2048)
QUICK = "--quick" in args
REUSE = "--reuse-bake" in args
OUT = opt("--out", os.path.join(ROOT, "output"))
CACHE = "/tmp/cloak_cache"
TEX = os.path.join(OUT, "Textures")
PREV = os.path.join(OUT, "Previews")
for p in (OUT, TEX, PREV, CACHE):
    os.makedirs(p, exist_ok=True)
NAME = "ShipwreckLeviathanCloak"
report = {}
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ geometry
bio.clear_scene()
bpy.context.scene.unit_settings.system = "NONE"  # 1 Blender unit = 1 stud (Roblox docs)
d = Drape()
layers, cloth_parts = build_cloth(d)
lay = Layout(d)
deco = lay.build_all()
parts = cloth_parts + deco
report["triangles_by_part"] = {p.name: p.tri_count() for p in parts}
report["triangles_total"] = int(sum(report["triangles_by_part"].values()))
log("triangles:", report["triangles_total"])
report["body_penetration"] = checks.body_report(parts)
report["cloth_penetration"] = checks.cloth_report(deco, d, layers, min_clear=0.002)
log("body penetration:", report["body_penetration"] or "none")
log("cloth penetration:", report["cloth_penetration"] or "none")

col = bio.get_collection("Cloak")
mcol = bio.get_collection("Mannequin")
objs, rim_info, outer_by_layer = [], {}, {}
for p in parts:
    ob = bio.make_object(p, col)
    objs.append(ob)
    if p.name.endswith("_rim"):
        rim_info[ob.name] = p.props["rim_tri"]
    if p.name.endswith("_outer"):
        outer_by_layer[p.props["layer"]] = ob
mann = bio.load_mannequin(MANNEQUIN, mcol)
grey = bio.simple_material("Mannequin", (0.62, 0.62, 0.62), 0.55)
for ob in mann:
    ob.data.materials.append(grey)
log("objects:", len(objs))

# ------------------------------------------------------------- UV + bake
mats = BK.build_materials(objs, layers)
bake_objs = [o for o in objs if o["bake"]]
rim_objs = [o for o in objs if not o["bake"]]
BK.pack_atlas(bake_objs, cloth_cuts={"CapeNavy": [2.25, d.U / 2, d.U - 2.25]})
BK.assign_rim_uvs(rim_objs, outer_by_layer, rim_info)
log("atlas packed")
cache_ok = REUSE and all(os.path.exists(os.path.join(CACHE, f"bake_{c}.npy")) for c in BK.CHANNELS)
if cache_ok:
    res = {c: np.load(os.path.join(CACHE, f"bake_{c}.npy")) for c in BK.CHANNELS}
    cache_ok = res["color"].shape[0] == SIZE
if not cache_ok:
    smp = {"color": 3, "rme": 1, "normal": 2, "ao": 16} if QUICK else None
    res = BK.bake_all(bake_objs, list(mats.values()), SIZE, CACHE, samples=smp, log=log)
mask = BK.island_mask(SIZE, bake_objs + rim_objs)
res["rough"] = res["rme"][..., 0:1]
res["metal"] = res["rme"][..., 1:2]
emit_mask = np.clip(res["rme"][..., 2], 0.0, 1.0)


# -------------------------------------------------------------- textures
def masked_blur(img, m, sigma):
    from scipy.ndimage import gaussian_filter

    w = gaussian_filter(m.astype(np.float32), sigma)
    out = gaussian_filter(img * m, sigma) / np.maximum(w, 1e-4)
    return np.where(m, out, img)


def to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def save(arr, path, size=None, mode="RGB", normal=False):
    a = np.clip(arr[::-1], 0, 1)  # Blender rows are bottom-up
    im = Image.fromarray((a * 255 + 0.5).astype(np.uint8), mode)
    if size and size != im.size[0]:
        im = im.resize((size, size), Image.LANCZOS)
        if normal:
            n = np.asarray(im).astype(np.float32) / 255 * 2 - 1
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
            im = Image.fromarray(((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8), "RGB")
    im.save(path, optimize=True)
    return path


ao = masked_blur(res["ao"][..., 0], mask, 1.1)
col_lin = res["color"]
shade = (0.38 + 0.62 * np.power(np.clip(ao, 0, 1), 1.1))[..., None]
final_lin = col_lin * shade
final_lin = final_lin * (1 - emit_mask[..., None]) + col_lin * emit_mask[..., None]
color_srgb = to_srgb(final_lin)
glow_srgb = to_srgb(final_lin + col_lin * emit_mask[..., None] * 0.6)  # TextureID: lantern reads as glowing
files = {
    "color_2048": save(glow_srgb, os.path.join(TEX, f"{NAME}_TextureID_2048.png"), SIZE),
    "color_1024": save(color_srgb, os.path.join(TEX, f"{NAME}_ColorMap_1024.png"), 1024),
    "normal_1024": save(res["normal"], os.path.join(TEX, f"{NAME}_NormalMap_1024.png"), 1024, normal=True),
    "rough_1024": save(res["rough"][..., 0], os.path.join(TEX, f"{NAME}_RoughnessMap_1024.png"), 1024, "L"),
    "metal_1024": save(res["metal"][..., 0], os.path.join(TEX, f"{NAME}_MetalnessMap_1024.png"), 1024, "L"),
    "emit_1024": save(emit_mask, os.path.join(TEX, f"{NAME}_EmissiveMask_1024.png"), 1024, "L"),
}
if SIZE > 1024:
    files["color_full"] = save(color_srgb, os.path.join(TEX, f"{NAME}_ColorMap_{SIZE}.png"), SIZE)
    files["normal_full"] = save(res["normal"], os.path.join(TEX, f"{NAME}_NormalMap_{SIZE}.png"), SIZE)
report["textures"] = {k: os.path.relpath(v, OUT) for k, v in files.items()}
report["atlas_coverage"] = float(mask.mean())
log("textures saved, atlas coverage %.1f%%" % (100 * mask.mean()))

# ----------------------------------------------------------- final material
fm = bpy.data.materials.new("ShipwreckLeviathanCloak_Mat")
fm.use_nodes = True
nt = fm.node_tree
bsdf = nt.nodes["Principled BSDF"]


def tex(path, non_color=False):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(path)
    if non_color:
        n.image.colorspace_settings.name = "Non-Color"
    return n


tc = tex(files["color_full"] if SIZE > 1024 else files["color_1024"])
nt.links.new(tc.outputs["Color"], bsdf.inputs["Base Color"])
tr = tex(files["rough_1024"], True)
nt.links.new(tr.outputs["Color"], bsdf.inputs["Roughness"])
tm = tex(files["metal_1024"], True)
nt.links.new(tm.outputs["Color"], bsdf.inputs["Metallic"])
tn = tex(files["normal_full"] if SIZE > 1024 else files["normal_1024"], True)
nm = nt.nodes.new("ShaderNodeNormalMap")
nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
te = tex(files["emit_1024"], True)
mul = nt.nodes.new("ShaderNodeMix")
mul.data_type = "RGBA"
mul.blend_type = "MULTIPLY"
mul.inputs[0].default_value = 1.0
nt.links.new(tc.outputs["Color"], mul.inputs[6])
nt.links.new(te.outputs["Color"], mul.inputs[7])
nt.links.new(mul.outputs[2], bsdf.inputs["Emission Color"])
bsdf.inputs["Emission Strength"].default_value = 3.0

# ------------------------------------------------------------------- rig
cloth_uv = []
for p in cloth_parts:
    if p.name.endswith("_outer"):
        cloth_uv.append(np.stack([p.attrs["pat_u"], p.attrs["pat_v"]], 1))
for p in deco:
    if p.weight_mode == "cloth":
        uvp, _ = d.project(p.world_verts())
        cloth_uv.append(uvp)
cloth_uv = np.concatenate(cloth_uv)
cloth_uv = cloth_uv[cloth_uv[:, 1] > 0]
rig = CapeRig(d, cloth_uv, lay.points["lantern_top"], lay.points["lantern_bottom"],
              lay.points["anchor_top"], lay.points["anchor_crown"])
arm = rig.build_armature(col)
for ob in objs:
    rig.skin_object(ob, d)
report["chains"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "joints"}
                    for c in rig.cols]
log("skinned; chains:", [(c["name"], c["n"], round(c["ext"], 2)) for c in rig.cols])

# ------------------------------------------------------------------ join
for o in bpy.context.view_layer.objects:
    o.select_set(False)
for ob in objs:
    ob.select_set(True)
    ob.data.materials.clear()
    ob.data.materials.append(fm)
bpy.context.view_layer.objects.active = objs[0]
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.object.join()
cloak = bpy.context.view_layer.objects.active
cloak.name = NAME
cloak.data.name = NAME
me = cloak.data
for name in [a.name for a in me.attributes if a.name in ("pat_u", "pat_v", "t")]:
    me.attributes.remove(me.attributes[name])
for uvl in [u.name for u in me.uv_layers if u.name != "UVMap"]:
    me.uv_layers.remove(me.uv_layers[uvl])
bm = bmesh.new()
bm.from_mesh(me)
bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
bm.to_mesh(me)
bm.free()
me.update()
for k in list(cloak.keys()):
    del cloak[k]
mod = cloak.modifiers.new("Armature", "ARMATURE")
mod.object = arm
cloak.parent = arm

# ------------------------------------------------------------- validation
tris = len(me.polygons)
gnames = {g.index: g.name for g in cloak.vertex_groups}
infl = np.zeros(len(me.vertices), dtype=np.int32)
wsum = np.zeros(len(me.vertices))
root_w = 0
for v in me.vertices:
    ws = [g for g in v.groups if g.weight > 0]
    infl[v.index] = len(ws)
    wsum[v.index] = sum(g.weight for g in ws)
    root_w += sum(1 for g in ws if gnames[g.group] == "Root")
bones = [b.name for b in arm.data.bones]
missing = [n for n in gnames.values() if n not in bones]
dims = cloak.dimensions
bb = np.array([cloak.matrix_world @ __import__("mathutils").Vector(c) for c in cloak.bound_box])
report.update({
    "final_triangles": tris, "vertices": len(me.vertices), "bones": len(bones), "bone_names": bones,
    "max_influences": int(infl.max()), "unweighted_vertices": int((infl == 0).sum()),
    "weight_sum_min": float(wsum.min()), "weight_sum_max": float(wsum.max()), "root_weighted_vertices": root_w,
    "groups_without_bone": missing, "bbox_min": bb.min(0).round(3).tolist(), "bbox_max": bb.max(0).round(3).tolist(),
    "dimensions_studs": [round(x, 3) for x in dims],
})
log("final tris:", tris, "verts:", len(me.vertices), "bones:", len(bones), "max infl:", infl.max(),
    "unweighted:", int((infl == 0).sum()), "wsum: %.4f..%.4f" % (wsum.min(), wsum.max()))
assert tris <= 20000, tris
assert infl.max() <= 4 and (infl == 0).sum() == 0 and root_w == 0 and not missing

# ----------------------------------------------------------------- export
bpy.ops.object.select_all(action="DESELECT")
cloak.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
fbx_path = os.path.join(OUT, f"{NAME}.fbx")
bpy.ops.export_scene.fbx(filepath=fbx_path, use_selection=True, object_types={"ARMATURE", "MESH"},
                         apply_unit_scale=True, apply_scale_options="FBX_SCALE_UNITS", axis_forward="Z", axis_up="Y",
                         use_space_transform=True, bake_space_transform=False, use_mesh_modifiers=False,
                         mesh_smooth_type="FACE", use_triangles=True, add_leaf_bones=False, primary_bone_axis="Y",
                         secondary_bone_axis="X", use_armature_deform_only=False, armature_nodetype="NULL",
                         bake_anim=False, path_mode="COPY", embed_textures=True, use_custom_props=False)
glb_path = os.path.join(OUT, f"{NAME}.glb")
bpy.ops.export_scene.gltf(filepath=glb_path, use_selection=True, export_format="GLB", export_skins=True,
                          export_animations=False, export_yup=True, export_apply=False)
report["files"] = {"fbx": os.path.relpath(fbx_path, OUT), "glb": os.path.relpath(glb_path, OUT)}
log("exported FBX + GLB")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{NAME}.blend"), compress=True)
log("saved .blend")
with open(os.path.join(OUT, "build_report.json"), "w") as f:
    json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

# ---------------------------------------------------------------- previews
for ob in mann:
    ob.hide_render = False
bio.setup_render(res=(900, 1100) if not QUICK else (560, 700), samples=48 if not QUICK else 16)
bpy.context.scene.render.film_transparent = True
for v in ("front", "side", "back", "front34", "back34"):
    bio.render_view(v, os.path.join(PREV, f"{v}.png"))
# deformation test: swing every chain outward progressively (like wind) and render
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    if pb.name.startswith("Cape_") and not pb.name.endswith("Tip"):
        k = int(pb.name.split("_")[-1])
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (np.radians(5 + 2.5 * k), 0.0, np.radians(2.0 * np.sin(k)))
    if pb.name == "Lantern":
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (np.radians(14), 0, np.radians(10))
bpy.ops.object.mode_set(mode="OBJECT")
for v in ("side", "back34"):
    bio.render_view(v, os.path.join(PREV, f"pose_test_{v}.png"))
log("previews rendered; done")
