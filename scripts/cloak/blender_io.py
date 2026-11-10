"""bpy helpers: Part -> Blender object, mannequin import, preview rendering."""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def get_collection(name):
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def make_object(part, collection, material=None):
    me = bpy.data.meshes.new(part.name)
    me.from_pydata(part.verts.tolist(), [], part.faces)
    me.update()
    nloops = len(me.loops)
    vi = np.empty(nloops, dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    if part.uv is not None:
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", part.uv[vi].astype(np.float32).ravel())
    else:
        layer = me.uv_layers.new(name="UVMap")
    if part.local_uv is not None:
        layer2 = me.uv_layers.new(name="LocalUV")
        layer2.data.foreach_set("uv", part.local_uv[vi].astype(np.float32).ravel())
    me.uv_layers.active = me.uv_layers["UVMap"]
    for k, arr in part.attrs.items():
        a = me.attributes.new(k, "FLOAT", "POINT")
        a.data.foreach_set("value", np.asarray(arr, dtype=np.float32))
    if part.merge:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bm.to_mesh(me)
        bm.free()
    me.polygons.foreach_set("use_smooth", [part.smooth] * len(me.polygons))
    me.update()
    ob = bpy.data.objects.new(part.name, me)
    collection.objects.link(ob)
    ob.matrix_world = Matrix(part.matrix.tolist())
    if material is not None:
        me.materials.append(material)
    ob["weight_mode"] = part.weight_mode
    if part.anchor is not None:
        ob["anchor"] = [float(x) for x in part.anchor]
    ob["bake"] = bool(part.bake)
    ob["uv_priority"] = float(part.uv_priority)
    ob["mat_key"] = part.material
    if "layer" in part.props:
        ob["layer"] = part.props["layer"]
    if "blend" in part.props:
        ob["blend_start"] = [float(x) for x in part.props["blend"][0]]
        ob["blend_end"] = [float(x) for x in part.props["blend"][1]]
    return ob


def load_mannequin(path, collection):
    """Parse the Roblox-exported R6 OBJ and convert to the Blender torso-centred frame.

    Roblox (x, y, z) world -> torso-centred (x+9, y-8, z+8) -> Blender (-x, z, y).
    """
    verts, groups, cur = [], {}, None
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "g":
            cur = p[1]
            groups.setdefault(cur, [])
        elif p[0] == "v":
            x, y, z = map(float, p[1:4])
            rx, ry, rz = x + 9.0, y - 8.0, z + 8.0
            verts.append((-rx, rz, ry))
        elif p[0] == "f":
            idx = [int(t.split("/")[0]) - 1 for t in p[1:]]
            groups[cur].append(idx)
    names = {"R61": "LeftLeg?", "R62": "Leg2", "R63": "Arm1", "R64": "Arm2", "R65": "Torso", "R66": "Head"}
    objs = []
    for g, faces in groups.items():
        used = sorted({i for f in faces for i in f})
        remap = {o: n for n, o in enumerate(used)}
        me = bpy.data.meshes.new("Mannequin_" + g)
        me.from_pydata([verts[i] for i in used], [], [[remap[i] for i in f] for f in faces])
        me.update()
        me.polygons.foreach_set("use_smooth", [g == "R66"] * len(me.polygons))
        ob = bpy.data.objects.new("Mannequin_" + names.get(g, g), me)
        collection.objects.link(ob)
        objs.append(ob)
    return objs


def simple_material(name, color, roughness=0.8, emission=None):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
        m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 3.0
    return m


# ------------------------------------------------------------------ render
VIEWS = {
    "front": ((0.0, -30.0), (90.0, 0.0, 0.0)),
    "side": ((30.0, 0.0), (90.0, 0.0, 90.0)),
    "back": ((0.0, 30.0), (90.0, 0.0, 180.0)),
    "front34": ((-21.0, -21.0), (80.0, 0.0, -45.0)),
    "back34": ((21.0, 21.0), (80.0, 0.0, 135.0)),
}


def setup_render(res=(640, 800), samples=24, ortho_scale=6.3, zc=-0.42):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except TypeError:
        pass
    sc.cycles.max_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.get("PreviewWorld") or bpy.data.worlds.new("PreviewWorld")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs[1].default_value = 0.42
    cam_data = bpy.data.cameras.get("PreviewCam") or bpy.data.cameras.new("PreviewCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam = bpy.data.objects.get("PreviewCam") or bpy.data.objects.new("PreviewCam", cam_data)
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    sc.camera = cam
    sun_data = bpy.data.lights.get("KeySun") or bpy.data.lights.new("KeySun", "SUN")
    sun_data.energy = 2.4
    sun_data.angle = math.radians(12)
    sun = bpy.data.objects.get("KeySun") or bpy.data.objects.new("KeySun", sun_data)
    if sun.name not in sc.collection.objects:
        sc.collection.objects.link(sun)
    sc["zc"] = zc
    return cam, sun


def render_view(view, path, cam=None, sun=None, persp=False):
    sc = bpy.context.scene
    cam = cam or bpy.data.objects["PreviewCam"]
    sun = sun or bpy.data.objects["KeySun"]
    (x, y), rot = VIEWS[view]
    zc = sc.get("zc", -0.42)
    target = Vector((0.0, 0.0, zc))
    if view.endswith("34"):
        loc = target + Vector((x, y, 9.0)).normalized() * 30.0
    else:
        loc = Vector((x, y, zc))
    cam.location = loc
    cam.rotation_euler = (target - loc).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "PERSP" if persp else "ORTHO"
    if persp:
        cam.data.lens = 160
    view_dir = (target - Vector(cam.location)).normalized()
    right = view_dir.cross(Vector((0, 0, 1))).normalized()
    d = (view_dir + Vector((0, 0, -0.9)) - right * 0.55).normalized()
    sun.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path
