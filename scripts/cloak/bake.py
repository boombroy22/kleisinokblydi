"""UV atlas packing + Cycles baking of all procedural channels into one texture set."""
import os

import bpy
import numpy as np

from . import materials as M


# ------------------------------------------------------------------ materials
def build_materials(objs, layers):
    """Create one procedural material per material key and assign it."""
    cache = {}
    edge_imgs = {}
    for L in layers:
        sd, rinfo = L.parts[0].props["edge_raster"]
        H, W = sd.shape
        img = bpy.data.images.new(f"EdgeSD_{L.name}", W, H, alpha=False, float_buffer=True)
        img.colorspace_settings.name = "Non-Color"
        rgba = np.ones((H, W, 4), dtype=np.float32)
        rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = np.clip(sd, -0.3, 0.6)
        img.pixels.foreach_set(rgba.ravel())
        img.pack()
        edge_imgs[L.name] = (img, rinfo, "navy" if "Navy" in L.name else "cream")
    makers = {
        "bone": lambda: M.bone_material("bone"),
        "skull": lambda: M.bone_material("skull", skull=True),
        "barnacle": M.barnacle_material,
        "rope": M.rope_material,
        "iron": lambda: M.iron_material("iron"),
        "iron_anchor": lambda: M.iron_material("iron_anchor", tint="#9a4e25", verdigris=0.55),
        "shell": M.shell_material,
        "lantern_glass": M.glass_material,
        "compass": M.compass_material,
        "starfish": M.starfish_material,
        "seaweed": M.seaweed_material,
        "net": M.net_material,
    }
    for ob in objs:
        key = ob["mat_key"]
        layer = ob.get("layer")
        if layer:
            inner = key.endswith("_inner")
            mkey = f"{layer}_{'inner' if inner else 'outer'}"
            if mkey not in cache:
                img, rinfo, kind = edge_imgs[layer]
                cache[mkey] = M.cloth_material(mkey, kind, inner, img, rinfo)
            mat = cache[mkey]
        else:
            if key not in cache:
                cache[key] = makers[key]()
            mat = cache[key]
        ob.data.materials.clear()
        ob.data.materials.append(mat)
    return cache


# --------------------------------------------------------------------- UV pack
def _select_only(objs, active=None):
    bpy.ops.object.mode_set(mode="OBJECT") if bpy.context.object and bpy.context.object.mode != "OBJECT" else None
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or objs[0]


def _poly_arrays(me):
    ls = np.empty(len(me.polygons), dtype=np.int32)
    lt = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    vi = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    loop_face = np.repeat(np.arange(len(ls)), lt)
    return ls, lt, vi, loop_face


def split_cloth_islands(ob, cuts):
    """Cut a big cloth UV island into chunks along pattern-u so the atlas packs tighter."""
    me = ob.data
    pu = np.empty(len(me.vertices), dtype=np.float32)
    me.attributes["pat_u"].data.foreach_get("value", pu)
    ls, lt, vi, loop_face = _poly_arrays(me)
    fu = np.bincount(loop_face, weights=pu[vi]) / np.maximum(np.bincount(loop_face), 1)
    chunk = np.searchsorted(np.asarray(cuts), fu)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    layer = me.uv_layers["UVMap"]
    layer.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    uv[:, 0] += chunk[loop_face] * 60.0
    layer.data.foreach_set("uv", uv.ravel())


def pack_atlas(bake_objs, margin=0.0035, cloth_cuts=None):
    for ob in bake_objs:
        pr = float(ob.get("uv_priority", 1.0))
        uv = ob.data.uv_layers["UVMap"]
        arr = np.empty(len(uv.data) * 2, dtype=np.float32)
        uv.data.foreach_get("uv", arr)
        uv.data.foreach_set("uv", arr * pr)
        ob.data.uv_layers.active = uv
        if cloth_cuts and ob.get("layer") in cloth_cuts:
            split_cloth_islands(ob, cloth_cuts[ob["layer"]])
    _select_only(bake_objs)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="ANY", scale=True,
                            merge_overlap=False, margin_method="FRACTION", margin=margin, pin=False,
                            shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_by_vertex(ob):
    me = ob.data
    vi = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    uv = np.empty(len(me.loops) * 2, dtype=np.float32)
    me.uv_layers["UVMap"].data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    out = np.zeros((len(me.vertices), 2), dtype=np.float32)
    out[vi] = uv
    return out


def assign_rim_uvs(rim_objs, outer_by_layer, rim_info, inset=0.0009):
    """Rim strips sample the boundary texels of the outer triangle that owns each edge."""
    for ob in rim_objs:
        outer = outer_by_layer[ob["layer"]]
        ome = outer.data
        ols, olt, ovi, _ = _poly_arrays(ome)
        ouv = np.empty(len(ome.loops) * 2, dtype=np.float32)
        ome.uv_layers["UVMap"].data.foreach_get("uv", ouv)
        ouv = ouv.reshape(-1, 2)
        tri_info = rim_info[ob.name]
        me = ob.data
        rls, rlt, _, _ = _poly_arrays(me)
        ruv = np.zeros((len(me.loops), 2), dtype=np.float32)
        for j, (ti, a, b, c) in enumerate(tri_info):
            loops = range(ols[ti], ols[ti] + olt[ti])
            uvs = {int(ovi[k]): ouv[k] for k in loops}
            ua, ub, uc = uvs[int(a)], uvs[int(b)], uvs[int(c)]

            def toward(p, q):
                dv = q - p
                n = float(np.linalg.norm(dv))
                return p + dv * min(0.45, inset / max(n, 1e-9))

            s = rls[j]
            ruv[s:s + 4] = (ua, toward(ua, uc), toward(ub, uc), ub)
        me.uv_layers["UVMap"].data.foreach_set("uv", ruv.ravel())


# ----------------------------------------------------------------------- bake
def _new_image(name, size, non_color=True):
    img = bpy.data.images.get(name)
    if img:
        bpy.data.images.remove(img)
    img = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True)
    img.colorspace_settings.name = "Non-Color" if non_color else "Linear Rec.709"
    return img


def _set_target(mats, img):
    for m in mats:
        nt = m.node_tree
        n = nt.nodes.get("BAKE_TARGET")
        if n is None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.name = "BAKE_TARGET"
        n.image = img
        nt.nodes.active = n
        n.select = True


def _route(mats, channel):
    """channel None -> normal shading; 'rme' -> (rough, metal, emissive mask) packed in RGB;
    otherwise emit CH_<channel> through an Emission shader."""
    for m in mats:
        nt = m.node_tree
        out = nt.nodes["OUT"]
        if channel is None:
            nt.links.new(nt.nodes["BSDF"].outputs["BSDF"], out.inputs["Surface"])
            continue
        em = nt.nodes.get("BAKE_EMIT") or nt.nodes.new("ShaderNodeEmission")
        em.name = "BAKE_EMIT"
        em.inputs["Strength"].default_value = 1.0
        if channel == "rme":
            cc = nt.nodes.get("BAKE_PACK") or nt.nodes.new("ShaderNodeCombineColor")
            cc.name = "BAKE_PACK"
            for i, ch in enumerate(("rough", "metal", "emask")):
                nt.links.new(nt.nodes[f"CH_{ch}"].outputs[0], cc.inputs[i])
            nt.links.new(cc.outputs[0], em.inputs["Color"])
        else:
            nt.links.new(nt.nodes[f"CH_{channel}"].outputs[0], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])


CHANNELS = ("color", "rme", "normal", "ao")


def bake_all(bake_objs, mats, size, out_dir, samples=None, log=print):
    samples = samples or {"color": 6, "rme": 1, "normal": 3, "ao": 32}
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.render.bake.margin = max(8, size // 128)
    sc.render.bake.margin_type = "EXTEND"
    sc.render.bake.use_clear = True
    sc.render.bake.use_selected_to_active = False
    if sc.world is None:
        sc.world = bpy.data.worlds.new("BakeWorld")
    sc.world.light_settings.distance = 0.35
    _select_only(bake_objs)
    results = {}
    import time

    for ch in CHANNELS:
        t0 = time.time()
        img = _new_image(f"BAKE_{ch}", size, non_color=(ch != "color"))
        _set_target(mats, img)
        sc.cycles.samples = samples[ch]
        if ch in ("normal", "ao"):
            _route(mats, None)
            if ch == "normal":
                sc.render.bake.normal_space = "TANGENT"
                sc.render.bake.normal_r, sc.render.bake.normal_g, sc.render.bake.normal_b = "POS_X", "POS_Y", "POS_Z"
                bpy.ops.object.bake(type="NORMAL")
            else:
                bpy.ops.object.bake(type="AO")
        else:
            _route(mats, ch)
            bpy.ops.object.bake(type="EMIT")
        px = np.empty(size * size * 4, dtype=np.float32)
        img.pixels.foreach_get(px)
        results[ch] = px.reshape(size, size, 4)[..., :3].copy()
        np.save(os.path.join(out_dir, f"bake_{ch}.npy"), results[ch])
        log(f"  baked {ch:6s} in {time.time() - t0:5.1f}s")
    _route(mats, None)
    return results


def island_mask(size, bake_objs):
    """Raster coverage of all UV triangles (for AO denoising and padding checks)."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (size, size), 0)
    dr = ImageDraw.Draw(img)
    for ob in bake_objs:
        me = ob.data
        uv = np.empty(len(me.loops) * 2, dtype=np.float32)
        me.uv_layers["UVMap"].data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2) * size
        for poly in me.polygons:
            pts = [(float(uv[i, 0]), float(size - uv[i, 1])) for i in poly.loop_indices]
            dr.polygon(pts, fill=255)
    return np.asarray(img)[::-1] > 0  # flip to Blender's bottom-up row order
