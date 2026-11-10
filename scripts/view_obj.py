"""Render an OBJ from several angles (inspection helper).

blender -b --factory-startup -P scripts/view_obj.py -- in.obj out_prefix [scale]
"""
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
path, prefix = args[0], args[1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Z", up_axis="Y")
obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
mat = bpy.data.materials.new("clay")
mat.use_nodes = True
mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1)
for o in obs:
    o.data.materials.clear()
    o.data.materials.append(mat)
    for p in o.data.polygons:
        p.use_smooth = False
pts = [o.matrix_world @ Vector(c) for o in obs for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
centre = (lo + hi) / 2
size = max(hi - lo)
sc = bpy.context.scene
sc.render.engine = "CYCLES"  # Workbench/EEVEE need EGL, which this headless box lacks
sc.cycles.device = "CPU"
sc.cycles.samples = 12
sc.cycles.use_denoising = True
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sc.collection.objects.link(sun)
sc.render.resolution_x = sc.render.resolution_y = 900
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = size * float(args[2]) if len(args) > 2 else size * 1.15
views = {"front": (0, -1, 0), "front34": (-0.7, -0.7, 0.25), "side": (1, 0, 0), "top": (0, -0.15, 1),
         "below": (0, -0.8, -0.6)}
for name, d in views.items():
    dv = Vector(d).normalized()
    cam.location = centre + dv * size * 4
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    sun.rotation_euler = (centre - (cam.location + Vector((0.3, 0.0, 1.5)) * size * 3)).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True)
print("bounds", tuple(round(x, 3) for x in lo), tuple(round(x, 3) for x in hi), "objects", len(obs))
