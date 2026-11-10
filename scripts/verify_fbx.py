"""Independent check of the exported files: re-import the FBX/GLB into an empty scene.

blender -b --factory-startup -P scripts/verify_fbx.py -- output/ShipwreckLeviathanCloak.fbx
"""
import json
import os
import sys

import bpy
import numpy as np

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path, axis_forward="Z", axis_up="Y")
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
out = {"file": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1e6, 2)}
assert len(meshes) == 1 and len(arms) == 1, (len(meshes), len(arms))
me_ob, arm = meshes[0], arms[0]
me = me_ob.data
dg = bpy.context.evaluated_depsgraph_get()
tris = sum(len(p.vertices) - 2 for p in me.polygons)
out.update(mesh=me_ob.name, armature=arm.name, triangles=tris, vertices=len(me.vertices),
           uv_layers=[u.name for u in me.uv_layers], materials=[m.name for m in me.materials])
imgs = [n.image.name for m in me.materials if m and m.use_nodes for n in m.node_tree.nodes
        if n.type == "TEX_IMAGE" and n.image]
out["embedded_textures"] = imgs
bones = arm.data.bones
out["bones"] = len(bones)
roots = [b.name for b in bones if b.parent is None]
out["root_bones"] = roots
W = arm.matrix_world
root = bones[roots[0]]
out["root_head_world"] = [round(x, 4) for x in (W @ root.head_local)]
gname = {g.index: g.name for g in me_ob.vertex_groups}
infl, wsum, root_w = [], [], 0
for v in me.vertices:
    ws = [g for g in v.groups if g.weight > 1e-6]
    infl.append(len(ws))
    wsum.append(sum(g.weight for g in ws))
    root_w += sum(1 for g in ws if gname[g.group] == "Root")
infl, wsum = np.array(infl), np.array(wsum)
out.update(max_influences=int(infl.max()), unweighted=int((infl == 0).sum()),
           weight_sum_range=[round(float(wsum.min()), 4), round(float(wsum.max()), 4)], root_weighted=root_w)
co = np.array([me_ob.matrix_world @ v.co for v in me.vertices])
out["bbox_min"] = co.min(0).round(3).tolist()
out["bbox_max"] = co.max(0).round(3).tolist()
# chains: walk the hierarchy and list every hanging chain
chains = {}
for b in bones:
    if b.name.startswith("Cape_") and not b.name.endswith("_Tip"):
        col = b.name.split("_")[1]
        chains.setdefault(col, []).append(b.name)
out["chains"] = {k: len(v) for k, v in sorted(chains.items())}
# deformation sanity: pose every chain bone and make sure vertices move smoothly (no exploding verts)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    if pb.name.startswith("Cape_") and not pb.name.endswith("_Tip"):
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (np.radians(12), 0, 0)
bpy.ops.object.mode_set(mode="OBJECT")
dg = bpy.context.evaluated_depsgraph_get()
ev = me_ob.evaluated_get(dg).to_mesh()
co2 = np.array([me_ob.matrix_world @ v.co for v in ev.vertices])
disp = np.linalg.norm(co2 - co, axis=1)
out["posed_max_displacement"] = round(float(disp.max()), 3)
out["posed_nan"] = bool(np.isnan(co2).any())
# neighbouring vertices must stay neighbours: compare edge-length change
e = np.array([ed.vertices[:] for ed in me.edges])
l0 = np.linalg.norm(co[e[:, 0]] - co[e[:, 1]], axis=1)
l1 = np.linalg.norm(co2[e[:, 0]] - co2[e[:, 1]], axis=1)
out["posed_max_edge_stretch"] = round(float(np.max(l1 / np.maximum(l0, 1e-6))), 3)
big = l0 > 0.03
ratio = l1[big] / l0[big]
dl = np.abs(l1 - l0)
worst = int(np.argmax(dl))
out["posed_edge_ratio_p999_over3cm"] = round(float(np.percentile(ratio, 99.9)), 3)
out["posed_edge_ratio_max_over3cm"] = round(float(ratio.max()), 3)
out["posed_max_abs_edge_change"] = round(float(dl.max()), 4)
out["posed_worst_edge"] = {"len0": round(float(l0[worst]), 4), "len1": round(float(l1[worst]), 4),
                           "at": co[e[worst, 0]].round(3).tolist()}
tiny = np.argmax(l1 / np.maximum(l0, 1e-6))
out["posed_max_ratio_edge"] = {"len0": round(float(l0[tiny]), 5), "len1": round(float(l1[tiny]), 5),
                               "at": co[e[tiny, 0]].round(3).tolist()}
# weight smoothness: change of the weight vector across each edge per stud of edge length
gidx = {n: i for i, n in enumerate(gname.values())}
Wm = np.zeros((len(me.vertices), len(gname)))
for v in me.vertices:
    for g in v.groups:
        Wm[v.index, g.group] = g.weight
grad = np.abs(Wm[e[:, 0]] - Wm[e[:, 1]]).sum(1) / np.maximum(l0, 1e-6)
out["weight_gradient_p999_per_stud"] = round(float(np.percentile(grad[big], 99.9)), 3)
out["weight_gradient_max_per_stud"] = round(float(grad[big].max()), 3)
for ei in np.nonzero(big)[0][np.argsort(-grad[big])][:4]:
    a, b = e[ei]
    print("GRAD_EDGE", round(float(grad[ei]), 2), "len0", round(float(l0[ei]), 3), co[a].round(2).tolist(),
          {gname[i]: round(Wm[a, i], 2) for i in np.nonzero(Wm[a] > 0.01)[0]},
          {gname[i]: round(Wm[b, i], 2) for i in np.nonzero(Wm[b] > 0.01)[0]})
print("VERIFY", json.dumps(out, indent=1))

# coherent "wind from the front" pose: every chain bone turns about the same WORLD axis (+X),
# so neighbouring chains move together -> any big stretch here would mean a weighting defect
from mathutils import Vector  # noqa: E402

bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
for pb in arm.pose.bones:
    pb.rotation_mode = "AXIS_ANGLE"
    pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
    if pb.name.startswith("Cape_") and not pb.name.endswith("_Tip"):
        axis_local = (pb.bone.matrix_local.to_3x3().inverted() @ (arm.matrix_world.to_3x3().inverted() @ Vector((1, 0, 0)))).normalized()
        pb.rotation_axis_angle = (np.radians(10.0), *axis_local)
bpy.ops.object.mode_set(mode="OBJECT")
dg = bpy.context.evaluated_depsgraph_get()
ev = me_ob.evaluated_get(dg).to_mesh()
co3 = np.array([me_ob.matrix_world @ v.co for v in ev.vertices])
l3 = np.linalg.norm(co3[e[:, 0]] - co3[e[:, 1]], axis=1)
r3 = l3[big] / l0[big]
wind = {"max_displacement": round(float(np.linalg.norm(co3 - co, axis=1).max()), 3),
        "edge_ratio_p999_over3cm": round(float(np.percentile(r3, 99.9)), 3),
        "edge_ratio_max_over3cm": round(float(r3.max()), 3),
        "edge_ratio_min_over3cm": round(float(r3.min()), 3)}
print("VERIFY_WIND", json.dumps(wind))
idx_big = np.nonzero(big)[0]
order = idx_big[np.argsort(-r3)][:6]


def vw(vi):
    return {gname[g.group]: round(g.weight, 2) for g in me.vertices[vi].groups if g.weight > 0.01}


for ei in order:
    a, b = e[ei]
    print("WORST_EDGE", round(float(l3[ei] / l0[ei]), 2), "len0", round(float(l0[ei]), 3),
          co[a].round(2).tolist(), vw(a), co[b].round(2).tolist(), vw(b))
wsum_ok = bool(abs(wsum.min() - 1) < 1e-3 and abs(wsum.max() - 1) < 1e-3)
print("VERIFY_SUMMARY", json.dumps({"weights_normalised": wsum_ok, "max_influences": int(infl.max()),
                                    "triangles": tris, "bones": len(bones)}))

glb = path[:-4] + ".glb"
if os.path.exists(glb):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    gm = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.users_collection
          and not o.name.lower().startswith("icosphere")]  # importer's bone-shape helper is not part of the file
    ga = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    gt = sum(len(p.vertices) - 2 for o in gm for p in o.data.polygons)
    print("VERIFY_GLB", json.dumps({"meshes": [o.name for o in gm], "armatures": len(ga), "triangles": gt,
                                    "bones": len(ga[0].data.bones) if ga else 0,
                                    "all_objects": [o.name for o in bpy.context.scene.objects]}))
