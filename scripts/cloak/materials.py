"""Procedural Cycles materials for every surface of the cloak.

Each material exposes its channels through reroute nodes named CH_color, CH_rough,
CH_metal and CH_emit so the baker can route any channel into an Emission shader.
"""
import bpy
import numpy as np


def srgb(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c) + (1.0,)


class NB:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def node(self, kind, **props):
        n = self.nt.nodes.new(kind)
        for k, v in props.items():
            setattr(n, k, v)
        n.location = (self.x, 0)
        self.x += 30
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)

    def _in(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.link(v, sock)
        elif isinstance(v, str):
            sock.default_value = srgb(v)
        elif isinstance(v, (tuple, list)):
            sock.default_value = v
        else:
            sock.default_value = v

    # ---- sources
    def coord(self, kind="Object"):
        return self.node("ShaderNodeTexCoord").outputs[kind]

    def uvmap(self, name):
        return self.node("ShaderNodeUVMap", uv_map=name).outputs["UV"]

    def attr(self, name):
        return self.node("ShaderNodeAttribute", attribute_name=name).outputs["Fac"]

    def geom(self, out):
        return self.node("ShaderNodeNewGeometry").outputs[out]

    def ao(self, dist=0.1, samples=8, only_local=False):
        n = self.node("ShaderNodeAmbientOcclusion", samples=samples, only_local=only_local)
        n.inputs["Distance"].default_value = dist
        return n.outputs["AO"]

    def value(self, v):
        n = self.node("ShaderNodeValue")
        n.outputs[0].default_value = v
        return n.outputs[0]

    def image(self, img, vec):
        n = self.node("ShaderNodeTexImage", image=img, interpolation="Linear", extension="EXTEND")
        self.link(vec, n.inputs["Vector"])
        return n.outputs["Color"]

    # ---- textures
    def noise(self, vec, scale, detail=4.0, rough=0.55, dist=0.0, kind="Fac", dims="3D"):
        n = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        self._in(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = dist
        return n.outputs[kind]

    def voronoi(self, vec, scale, feature="F1", out="Distance", rand=1.0, dims="3D"):
        n = self.node("ShaderNodeTexVoronoi", feature=feature, voronoi_dimensions=dims)
        self._in(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Randomness"].default_value = rand
        return n.outputs[out]

    def wave(self, vec, scale, dist=0.0, detail=0.0, kind="BANDS", direction="X", profile="SIN"):
        n = self.node("ShaderNodeTexWave", wave_type=kind, bands_direction=direction, wave_profile=profile)
        self._in(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Distortion"].default_value = dist
        n.inputs["Detail"].default_value = detail
        return n.outputs["Fac"]

    # ---- maths
    def math(self, op, a, b=0.0, c=0.0, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        self._in(n.inputs[2], c)
        return n.outputs[0]

    def mr(self, v, a, b, c=0.0, d=1.0, clamp=True, interp="LINEAR"):
        n = self.node("ShaderNodeMapRange", clamp=clamp, interpolation_type=interp)
        self._in(n.inputs["Value"], v)
        self._in(n.inputs["From Min"], a)
        self._in(n.inputs["From Max"], b)
        self._in(n.inputs["To Min"], c)
        self._in(n.inputs["To Max"], d)
        return n.outputs["Result"]

    def sstep(self, v, a, b):
        return self.mr(v, a, b, 0.0, 1.0, interp="SMOOTHSTEP")

    def vmath(self, op, a, b=(0, 0, 0), out=0):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        return n.outputs[out]

    def sep(self, v):
        n = self.node("ShaderNodeSeparateXYZ")
        self.link(v, n.inputs[0])
        return n.outputs

    def comb(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        for i, c in enumerate((x, y, z)):
            self._in(n.inputs[i], c)
        return n.outputs[0]

    def mapping(self, vec, loc=(0, 0, 0), scale=(1, 1, 1), rot=(0, 0, 0)):
        n = self.node("ShaderNodeMapping", vector_type="POINT")
        self.link(vec, n.inputs["Vector"])
        n.inputs["Location"].default_value = loc
        n.inputs["Scale"].default_value = scale
        n.inputs["Rotation"].default_value = rot
        return n.outputs["Vector"]

    def ramp(self, fac, stops, interp="LINEAR"):
        n = self.node("ShaderNodeValToRGB")
        self._in(n.inputs["Fac"], fac)
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) > 1:
            cr.elements.remove(cr.elements[-1])
        for i, (pos, col) in enumerate(stops):
            el = cr.elements[0] if i == 0 else cr.elements.new(pos)
            el.position = pos
            el.color = srgb(col) if isinstance(col, str) else (col, col, col, 1.0)
        return n.outputs["Color"]

    @staticmethod
    def _sock(sockets, ident):
        for s in sockets:
            if s.identifier == ident:
                return s
        raise KeyError(ident)

    def mix(self, fac, a, b, blend="MIX", clamp=True):
        n = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend, clamp_result=clamp)
        self._in(self._sock(n.inputs, "Factor_Float"), fac)
        self._in(self._sock(n.inputs, "A_Color"), a)
        self._in(self._sock(n.inputs, "B_Color"), b)
        return self._sock(n.outputs, "Result_Color")

    def mixf(self, fac, a, b):
        n = self.node("ShaderNodeMix", data_type="FLOAT")
        self._in(self._sock(n.inputs, "Factor_Float"), fac)
        self._in(self._sock(n.inputs, "A_Float"), a)
        self._in(self._sock(n.inputs, "B_Float"), b)
        return self._sock(n.outputs, "Result_Float")

    def bump(self, height, strength=0.3, distance=0.02, normal=None):
        n = self.node("ShaderNodeBump")
        self._in(n.inputs["Height"], height)
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = distance
        if normal is not None:
            self.link(normal, n.inputs["Normal"])
        return n.outputs["Normal"]

    def channel(self, name, sock):
        if not isinstance(sock, bpy.types.NodeSocket):
            if isinstance(sock, (tuple, list, str)):
                n = self.node("ShaderNodeRGB")
                n.outputs[0].default_value = srgb(sock) if isinstance(sock, str) else tuple(sock)
            else:
                n = self.node("ShaderNodeValue")
                n.outputs[0].default_value = float(sock)
            sock = n.outputs[0]
        r = self.node("NodeReroute")
        r.name = name
        r.label = name
        self.link(sock, r.inputs[0])
        return r

    def finish(self, color, rough, metal=0.0, normal=None, emit=None, emit_strength=0.0, emask=0.0):
        self.channel("CH_color", color)
        self.channel("CH_rough", rough)
        self.channel("CH_metal", metal)
        self.channel("CH_emask", emask)
        self.channel("CH_emit", emit if emit is not None else (0.0, 0.0, 0.0, 1.0))
        p = self.node("ShaderNodeBsdfPrincipled")
        p.name = "BSDF"
        self._in(p.inputs["Base Color"], color)
        self._in(p.inputs["Roughness"], rough)
        self._in(p.inputs["Metallic"], metal)
        if normal is not None:
            self.link(normal, p.inputs["Normal"])
        if emit is not None:
            self._in(p.inputs["Emission Color"], emit)
            p.inputs["Emission Strength"].default_value = emit_strength
        out = self.node("ShaderNodeOutputMaterial")
        out.name = "OUT"
        self.link(p.outputs["BSDF"], out.inputs["Surface"])
        return p


# --------------------------------------------------------------------- helpers
def _height_mix(nb, parts):
    """Sum of (socket, weight) pairs -> scalar height."""
    acc = None
    for s, w in parts:
        term = nb.math("MULTIPLY", s, w)
        acc = term if acc is None else nb.math("ADD", acc, term)
    return acc


def _painted_holes(nb, luv_vec, density, scale=7.0, size=0.11):
    """Little burn/tear holes painted into the cloth: returns (hole mask, frayed ring mask)."""
    dist = nb.noise(luv_vec, 6.0, 2.0, 0.5, kind="Color")
    warped = nb.vmath("ADD", luv_vec, nb.vmath("MULTIPLY", dist, (0.04, 0.04, 0.0)))
    vd = nb.voronoi(warped, scale, "F1", "Distance", dims="2D")
    rnd = nb.voronoi(warped, scale, "F1", "Color", dims="2D")
    rsep = nb.sep(rnd)
    keep = nb.math("LESS_THAN", rsep[0], density)
    radius = nb.math("MULTIPLY", nb.math("ADD", rsep[1], 0.4), size)
    hole = nb.math("MULTIPLY", nb.math("LESS_THAN", vd, radius), keep)
    ring = nb.math("MULTIPLY", nb.math("SUBTRACT", nb.math("LESS_THAN", vd, nb.math("MULTIPLY", radius, 1.7)), hole), keep)
    return hole, ring


# ----------------------------------------------------------------------- cloth
def cloth_material(name, kind, inner, edge_img, rinfo):
    """kind: 'navy' | 'cream'.  rinfo = (u0, v0, W, H) of the edge-distance raster."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    u0, v0, W, H = rinfo
    img_uv = nb.mapping(luv, loc=(-u0 / W, -v0 / H, 0.0), scale=(1.0 / W, 1.0 / H, 1.0))
    sd = nb.sep(nb.image(edge_img, img_uv))[0]  # signed distance to the torn edge (studs)
    luv_sep = nb.sep(luv)
    vcoord = luv_sep[1]
    lower = nb.sstep(vcoord, 1.2, 3.8)

    n_big = nb.noise(obj, 1.3, 5.0, 0.6)
    n_mid = nb.noise(obj, 4.5, 6.0, 0.6)
    n_fine = nb.noise(luv, 38.0, 3.0, 0.6)
    weave_u = nb.wave(luv, 34.0, 0.0, 0.0, "BANDS", "X")
    weave_v = nb.wave(luv, 34.0, 0.0, 0.0, "BANDS", "Y")
    weave = nb.math("MULTIPLY", weave_u, weave_v)
    salt = nb.sstep(nb.noise(obj, 2.4, 8.0, 0.65, dist=0.6), 0.56, 0.72)
    hole, ring = _painted_holes(nb, luv, nb.mixf(lower, 0.10, 0.55))
    wear = nb.sstep(sd, 0.10, 0.0)          # close to the torn edge
    fray = nb.math("MULTIPLY", nb.sstep(sd, 0.035, 0.0), nb.sstep(nb.noise(luv, 60.0, 2.0), 0.35, 0.6))

    if kind == "navy":
        base = nb.ramp(n_mid, [(0.30, "#141a24"), (0.55, "#1f2734"), (0.75, "#2b3443")])
        faded = nb.sstep(n_big, 0.55, 0.75)
        base = nb.mix(nb.math("MULTIPLY", faded, 0.5), base, "#3b4452")
        base = nb.mix(nb.math("MULTIPLY", weave, 0.07), base, "#4a5262", "SCREEN")
        base = nb.mix(nb.math("MULTIPLY", salt, nb.mixf(lower, 0.22, 0.55)), base, "#86888a")
        base = nb.mix(nb.math("MULTIPLY", lower, 0.5), base, "#2a2117", "MULTIPLY")
        base = nb.mix(nb.math("MULTIPLY", wear, 0.6), base, "#3b3226")
        base = nb.mix(fray, base, "#7b7462")
        base = nb.mix(nb.math("MULTIPLY", ring, 0.4), base, "#5a5448")
        base = nb.mix(hole, base, "#07080b")
        rough = nb.mixf(salt, 0.86, 0.95)
    else:
        base = nb.ramp(n_mid, [(0.30, "#c8b893"), (0.55, "#dccfb0"), (0.78, "#ebe2cb")])
        stain_n = nb.noise(obj, 3.2, 6.0, 0.6, dist=0.4)
        stain = nb.sstep(stain_n, 0.56, 0.63)
        stain_rim = nb.math("MULTIPLY", nb.sstep(stain_n, 0.54, 0.58), nb.sstep(stain_n, 0.64, 0.58))
        base = nb.mix(nb.math("MULTIPLY", stain, 0.28), base, "#9c8058")
        base = nb.mix(nb.math("MULTIPLY", stain_rim, 0.45), base, "#6e5232")
        mold = nb.math("LESS_THAN", nb.voronoi(obj, 55.0, "F1", "Distance"), 0.08)
        mold = nb.math("MULTIPLY", mold, nb.sstep(n_big, 0.55, 0.72))
        base = nb.mix(nb.math("MULTIPLY", mold, 0.7), base, "#4d4a33")
        base = nb.mix(nb.math("MULTIPLY", weave, 0.06), base, "#f2ead6", "SCREEN")
        base = nb.mix(nb.math("MULTIPLY", lower, 0.45), base, "#7a6446", "MULTIPLY")
        base = nb.mix(nb.math("MULTIPLY", wear, 0.65), base, "#5e4a30")
        base = nb.mix(nb.math("MULTIPLY", fray, 0.6), base, "#3c3022")
        base = nb.mix(nb.math("MULTIPLY", ring, 0.8), base, "#5a4529")
        base = nb.mix(hole, base, "#120d08")
        rough = nb.mixf(stain, 0.9, 0.82)
    if inner:
        base = nb.mix(0.42, base, "#0d0d0f", "MULTIPLY") if kind == "navy" else nb.mix(0.5, base, "#3a3328", "MULTIPLY")
    height = _height_mix(nb, [(weave, 0.15), (n_fine, 0.25), (n_mid, 0.4), (hole, -0.6), (wear, -0.15)])
    normal = nb.bump(height, 0.35, 0.01)
    nb.finish(base, rough, 0.0, normal)
    return mat


# ------------------------------------------------------------------------ bone
def bone_material(name="bone", skull=False):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    obj = nb.coord("Object")
    t = nb.attr("t")
    n1 = nb.noise(obj, 3.0 if not skull else 1.5, 6.0, 0.6)
    n2 = nb.noise(obj, 14.0 if not skull else 7.0, 4.0, 0.6)
    base = nb.ramp(n1, [(0.3, "#b9a986"), (0.55, "#d4c8aa"), (0.8, "#e6ddc6")])
    streak = nb.wave(obj, 2.2, 6.0, 3.0, "BANDS", "Z")
    base = nb.mix(nb.math("MULTIPLY", streak, 0.18), base, "#9b8763", "MULTIPLY")
    ao = nb.ao(0.12 if not skull else 0.06, 8)
    cav = nb.math("POWER", nb.math("SUBTRACT", 1.0, ao), 1.3)
    base = nb.mix(nb.math("MULTIPLY", cav, 0.9), base, "#4a3a25")
    pores_d = nb.voronoi(obj, 38.0 if not skull else 18.0, "F1", "Distance")
    pores = nb.math("MULTIPLY", nb.math("LESS_THAN", pores_d, 0.12), nb.sstep(n2, 0.45, 0.6))
    base = nb.mix(nb.math("MULTIPLY", pores, 0.85), base, "#2e241a")
    crack_d = nb.voronoi(nb.vmath("ADD", obj, nb.vmath("MULTIPLY", nb.noise(obj, 6.0, 2.0, kind="Color"), (0.15, 0.15, 0.15))),
                         9.0 if not skull else 4.5, "DISTANCE_TO_EDGE", "Distance")
    crack = nb.math("MULTIPLY", nb.math("LESS_THAN", crack_d, 0.025), nb.sstep(n1, 0.5, 0.7))
    base = nb.mix(nb.math("MULTIPLY", crack, 0.7), base, "#5a4630")
    edge = nb.sstep(nb.geom("Pointiness"), 0.52, 0.62)
    base = nb.mix(nb.math("MULTIPLY", edge, 0.35), base, "#f3ecda")
    if not skull:
        # darker roots on claws (t = 0 at the base)
        base = nb.mix(nb.math("MULTIPLY", nb.sstep(t, 0.35, 0.0), 0.45), base, "#6b5638", "MULTIPLY")
    else:
        # eye sockets, nose and teeth in skull object space (+Y face, +Z up, unit radius)
        xyz = nb.sep(obj)
        front = nb.sstep(xyz[1], 0.25, 0.55)
        ex = nb.math("SUBTRACT", nb.math("ABSOLUTE", xyz[0]), 0.33)
        ez = nb.math("SUBTRACT", xyz[2], 0.06)
        eye = nb.math("ADD", nb.math("MULTIPLY", ex, ex), nb.math("MULTIPLY", ez, ez))
        eye = nb.math("MULTIPLY", nb.sstep(eye, 0.05, 0.022), front)
        nx = nb.math("ABSOLUTE", xyz[0])
        nose = nb.math("MULTIPLY", nb.math("LESS_THAN", nx, nb.mr(xyz[2], -0.36, -0.12, 0.10, 0.0)),
                       nb.math("MULTIPLY", nb.math("GREATER_THAN", xyz[2], -0.36), nb.math("LESS_THAN", xyz[2], -0.10)))
        nose = nb.math("MULTIPLY", nose, front)
        teeth_band = nb.math("MULTIPLY", nb.math("GREATER_THAN", xyz[2], -0.62), nb.math("LESS_THAN", xyz[2], -0.44))
        gaps = nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", xyz[0], 26.0))), 0.25)
        teeth = nb.math("MULTIPLY", nb.math("MULTIPLY", teeth_band, gaps), front)
        dark = nb.math("MAXIMUM", nb.math("MAXIMUM", eye, nose), nb.math("MULTIPLY", teeth, 0.8))
        base = nb.mix(dark, base, "#120c08")
        jaw = nb.math("MULTIPLY", nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("ADD", xyz[2], 0.43)), 0.025), front)
        base = nb.mix(nb.math("MULTIPLY", jaw, 0.6), base, "#3b2c1c")
    rough = nb.mixf(cav, 0.55, 0.85)
    height = _height_mix(nb, [(n2, 0.4), (pores, -0.5), (crack, -0.4), (streak, 0.15)])
    nb.finish(base, rough, 0.0, nb.bump(height, 0.45, 0.01))
    return mat


def barnacle_material(name="barnacle"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    s = nb.sep(luv)
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", s[0], s[0]), nb.math("MULTIPLY", s[1], s[1])))
    ang = nb.math("ARCTAN2", s[1], s[0])
    ridges = nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", ang, 9.0)))
    n1 = nb.noise(obj, 30.0, 4.0)
    base = nb.ramp(n1, [(0.3, "#8f897d"), (0.6, "#bdb6a8"), (0.85, "#d7d0c2")])
    base = nb.mix(nb.math("MULTIPLY", ridges, 0.35), base, "#6d675c", "MULTIPLY")
    crater = nb.sstep(r, 0.42, 0.30)
    base = nb.mix(crater, base, "#1c1813")
    foot = nb.sstep(r, 0.75, 1.0)
    base = nb.mix(nb.math("MULTIPLY", foot, 0.6), base, "#4f5a3a")
    nb.finish(base, 0.8, 0.0, nb.bump(_height_mix(nb, [(ridges, 0.4), (n1, 0.3)]), 0.4, 0.005))
    return mat


# ------------------------------------------------------------------------ rope
def rope_material(name="rope", lay=0.42):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    s = nb.sep(luv)
    q = nb.math("MULTIPLY", nb.math("SUBTRACT", s[0], nb.math("DIVIDE", s[1], lay)), 3.0)
    f = nb.math("FRACT", q)
    strand = nb.math("SINE", nb.math("MULTIPLY", f, np.pi))
    fib_coord = nb.comb(nb.math("MULTIPLY", q, 9.0), nb.math("MULTIPLY", s[1], 4.0), 0.0)
    fibers = nb.noise(fib_coord, 3.0, 3.0, 0.6, dims="3D")
    base = nb.ramp(strand, [(0.0, "#2e2114"), (0.35, "#6e5434"), (0.8, "#9c7f55"), (1.0, "#b0946a")])
    base = nb.mix(nb.math("MULTIPLY", fibers, 0.4), base, "#3e2e1c", "MULTIPLY")
    algae = nb.sstep(nb.noise(obj, 3.5, 5.0, 0.6), 0.6, 0.72)
    base = nb.mix(nb.math("MULTIPLY", algae, 0.6), base, "#4d5226")
    base = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.ao(0.08, 6)), 0.8), base, "#1d150c")
    nb.finish(base, 0.92, 0.0, nb.bump(_height_mix(nb, [(strand, 0.7), (fibers, 0.25)]), 0.6, 0.01))
    return mat


# ----------------------------------------------------------------------- metal
def iron_material(name="iron", tint="#8a4320", verdigris=0.0):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    obj = nb.coord("Object")
    n1 = nb.noise(obj, 9.0, 8.0, 0.65)
    n2 = nb.noise(obj, 35.0, 4.0, 0.6)
    base = nb.ramp(n1, [(0.25, "#2a1a12"), (0.45, "#4a2a18"), (0.62, tint), (0.8, "#b4632c")])
    pits = nb.math("MULTIPLY", nb.math("LESS_THAN", nb.voronoi(obj, 60.0, "F1", "Distance"), 0.14), nb.sstep(n2, 0.5, 0.6))
    base = nb.mix(nb.math("MULTIPLY", pits, 0.8), base, "#160d08")
    edge = nb.sstep(nb.geom("Pointiness"), 0.53, 0.6)
    worn = nb.math("MULTIPLY", edge, nb.sstep(n2, 0.4, 0.55))
    base = nb.mix(worn, base, "#5d5752")
    if verdigris > 0:
        vg = nb.sstep(nb.noise(obj, 5.0, 6.0, 0.6), 0.64, 0.74)
        base = nb.mix(nb.math("MULTIPLY", vg, verdigris), base, "#4f7d6c")
    base = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.ao(0.06, 6)), 0.85), base, "#120a06")
    metal = nb.math("MULTIPLY", worn, 0.75)
    rough = nb.mixf(worn, 0.86, 0.42)
    nb.finish(base, rough, metal, nb.bump(_height_mix(nb, [(n2, 0.4), (pits, -0.5), (n1, 0.2)]), 0.5, 0.008))
    return mat


def compass_material(name="compass"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    s = nb.sep(luv)
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", s[0], s[0]), nb.math("MULTIPLY", s[1], s[1])))
    ang = nb.math("ARCTAN2", s[0], s[1])
    is_dial = nb.math("LESS_THAN", r, 0.81)
    # 8-point compass rose: long cardinal points, short intercardinal ones
    a8 = nb.math("MULTIPLY", ang, 4.0 / np.pi)
    k = nb.math("ROUND", a8)
    off = nb.math("ABSOLUTE", nb.math("SUBTRACT", a8, k))           # 0 at a point axis, 0.5 between
    card = nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SUBTRACT", nb.math("MODULO", nb.math("ADD", k, 8.0), 2.0), 0.0)), 0.5)
    tip_r = nb.mixf(card, 0.48, 0.74)
    star_r = nb.math("MULTIPLY", tip_r, nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", off, 1.75)))
    star = nb.math("LESS_THAN", r, star_r)
    side = nb.math("GREATER_THAN", nb.math("SUBTRACT", a8, k), 0.0)
    paper = nb.ramp(nb.noise(obj, 30.0, 4.0), [(0.3, "#b9a77d"), (0.7, "#ddd0ab")])
    rose = nb.mix(side, "#2a1d12", "#8c6a3d")
    dial = nb.mix(star, paper, rose)
    ring1 = nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SUBTRACT", r, 0.62)), 0.012)
    ring2 = nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SUBTRACT", r, 0.76)), 0.015)
    ticks = nb.math("MULTIPLY", nb.math("LESS_THAN", nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", ang, 18.0))), 0.12),
                    nb.math("MULTIPLY", nb.math("GREATER_THAN", r, 0.66), nb.math("LESS_THAN", r, 0.75)))
    lines = nb.math("MAXIMUM", nb.math("MAXIMUM", ring1, ring2), ticks)
    dial = nb.mix(lines, dial, "#24180e")
    centre = nb.math("LESS_THAN", r, 0.06)
    dial = nb.mix(centre, dial, "#6b4a22")
    brass = nb.ramp(nb.noise(obj, 25.0, 5.0), [(0.3, "#5b4219"), (0.6, "#8f6d2f"), (0.85, "#b18f48")])
    vg = nb.sstep(nb.noise(obj, 18.0, 4.0), 0.62, 0.72)
    brass = nb.mix(nb.math("MULTIPLY", vg, 0.7), brass, "#4a7a68")
    base = nb.mix(is_dial, brass, dial)
    stains = nb.sstep(nb.noise(obj, 9.0, 5.0), 0.58, 0.7)
    base = nb.mix(nb.math("MULTIPLY", stains, 0.35), base, "#4a3a22", "MULTIPLY")
    metal = nb.mixf(is_dial, 0.8, 0.0)
    rough = nb.mixf(is_dial, 0.42, 0.2)
    nb.finish(base, rough, metal, nb.bump(_height_mix(nb, [(lines, -0.3), (star, 0.15)]), 0.3, 0.004))
    return mat


# ----------------------------------------------------------------------- misc
def shell_material(name="shell"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    t = nb.attr("t")
    s = nb.sep(luv)
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", s[0], s[0]), nb.math("MULTIPLY", s[1], s[1])))
    ang = nb.math("ARCTAN2", s[0], nb.math("MULTIPLY", s[1], -1.0))
    ribs = nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", ang, 9.0)))
    growth = nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", r, 42.0)))
    bands = nb.noise(nb.comb(nb.math("MULTIPLY", r, 6.0), 0.0, 0.0), 2.0, 2.0)
    base = nb.ramp(bands, [(0.35, "#c9b49a"), (0.5, "#ece2cf"), (0.65, "#b8957a")])
    base = nb.mix(nb.math("MULTIPLY", ribs, 0.35), base, "#7d6a58", "MULTIPLY")
    base = nb.mix(nb.math("MULTIPLY", growth, 0.12), base, "#8f7b66", "MULTIPLY")
    base = nb.mix(nb.math("MULTIPLY", t, 0.5), base, "#6f655a", "MULTIPLY")
    base = nb.mix(nb.math("MULTIPLY", nb.math("SUBTRACT", 1.0, nb.ao(0.05, 6)), 0.7), base, "#2b241d")
    nb.finish(base, 0.5, 0.0, nb.bump(_height_mix(nb, [(ribs, 0.5), (growth, 0.1), (nb.noise(obj, 40.0), 0.2)]), 0.4, 0.004))
    return mat


def starfish_material(name="starfish"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    s = nb.sep(luv)
    r = nb.math("SQRT", nb.math("ADD", nb.math("MULTIPLY", s[0], s[0]), nb.math("MULTIPLY", s[1], s[1])))
    bumps = nb.math("LESS_THAN", nb.voronoi(luv, 14.0, "F1", "Distance", dims="2D"), 0.22)
    base = nb.ramp(r, [(0.0, "#8a5a33"), (0.5, "#c28a55"), (1.0, "#d9a777")])
    base = nb.mix(nb.math("MULTIPLY", bumps, 0.6), base, "#ead2a8")
    base = nb.mix(nb.math("MULTIPLY", nb.sstep(nb.noise(obj, 20.0), 0.55, 0.7), 0.4), base, "#5a3a20", "MULTIPLY")
    nb.finish(base, 0.8, 0.0, nb.bump(_height_mix(nb, [(bumps, 0.6)]), 0.5, 0.004))
    return mat


def seaweed_material(name="seaweed"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    t = nb.attr("t")
    s = nb.sep(luv)
    du = nb.math("ABSOLUTE", nb.math("SUBTRACT", s[0], 0.5))
    veins = nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", nb.math("SUBTRACT", nb.math("MULTIPLY", s[1], 14.0),
                                                                          nb.math("MULTIPLY", du, 5.0)), np.pi)))
    base = nb.ramp(du, [(0.0, "#36421a"), (0.07, "#4e5c22"), (0.3, "#66722b"), (0.5, "#9c9c42")])
    base = nb.mix(nb.math("MULTIPLY", nb.sstep(veins, 0.85, 1.0), 0.35), base, "#3a4518")
    base = nb.mix(nb.math("MULTIPLY", nb.sstep(t, 0.7, 1.0), 0.6), base, "#6e5a2a")
    spots = nb.math("LESS_THAN", nb.voronoi(obj, 45.0, "F1", "Distance"), 0.1)
    base = nb.mix(nb.math("MULTIPLY", spots, 0.5), base, "#2b2f12")
    nb.finish(base, 0.55, 0.0, nb.bump(_height_mix(nb, [(veins, 0.2), (nb.math("SUBTRACT", 0.5, du), 0.5)]), 0.4, 0.004))
    return mat


def net_material(name="net"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    luv = nb.uvmap("LocalUV")
    obj = nb.coord("Object")
    s = nb.sep(luv)
    tw = nb.math("ABSOLUTE", nb.math("SINE", nb.math("MULTIPLY", nb.math("ADD", nb.math("MULTIPLY", s[1], 9.0), s[0]), np.pi)))
    base = nb.ramp(tw, [(0.0, "#2b2016"), (1.0, "#7a5d3b")])
    base = nb.mix(nb.math("MULTIPLY", nb.sstep(nb.noise(obj, 6.0), 0.55, 0.7), 0.5), base, "#4a4a26")
    nb.finish(base, 0.9, 0.0, nb.bump(tw, 0.4, 0.003))
    return mat


def glass_material(name="lantern_glass"):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nb = NB(mat)
    obj = nb.coord("Object")
    xyz = nb.sep(obj)
    # flame silhouette in lantern object space (glass spans z in [-0.3, 0.3], radius ~0.19)
    zf = nb.mr(xyz[2], -0.16, 0.20, 0.0, 1.0)
    width = nb.math("MULTIPLY", nb.math("POWER", nb.math("SUBTRACT", 1.0, zf), 0.7), 0.105)
    wob = nb.math("MULTIPLY", nb.noise(nb.comb(xyz[2], 0.0, 0.0), 9.0, 3.0, kind="Fac", dims="3D"), 0.05)
    dx = nb.math("ABSOLUTE", nb.math("ADD", xyz[0], nb.math("SUBTRACT", wob, 0.025)))
    flame = nb.math("MULTIPLY", nb.sstep(dx, width, nb.math("MULTIPLY", width, 0.35)),
                    nb.math("MULTIPLY", nb.math("GREATER_THAN", xyz[2], -0.17), nb.math("LESS_THAN", xyz[2], 0.21)))
    core = nb.sstep(nb.math("ADD", nb.math("MULTIPLY", xyz[0], xyz[0]),
                            nb.math("MULTIPLY", nb.math("ADD", xyz[2], 0.07), nb.math("ADD", xyz[2], 0.07))), 0.012, 0.0)
    glow = nb.math("MAXIMUM", nb.math("MULTIPLY", flame, 0.85), core)
    rim = nb.sstep(nb.math("ABSOLUTE", xyz[2]), 0.18, 0.3)
    emit = nb.ramp(glow, [(0.0, "#0f6f78"), (0.6, "#3fe6ea"), (1.0, "#e8ffff")])
    emit = nb.mix(nb.math("MULTIPLY", rim, 0.7), emit, "#06343a")
    base = nb.mix(0.35, emit, "#1b7d84")
    emask = nb.math("MULTIPLY", nb.math("ADD", nb.math("MULTIPLY", glow, 0.55), 0.45),
                    nb.math("SUBTRACT", 1.0, nb.math("MULTIPLY", rim, 0.6)))
    nb.finish(base, 0.12, 0.0, None, emit, 4.0, emask=emask)
    return mat
