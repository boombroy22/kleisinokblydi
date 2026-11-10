"""Torn cloth layers: 2D patterns (shapely) -> triangulation -> draped, thick 3D shells."""
import numpy as np
import triangle as tr
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .drape import Drape
from .meshdata import Part
from .noise import fbm1, smoothstep

THICK = 0.022  # cloth thickness in studs


# ---------------------------------------------------------------- 2D helpers
def resample(pts, step):
    pts = np.asarray(pts, dtype=np.float64)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], 1), t


def tatter(pts, rng, seed, amount=1.0, tooth_w=(0.10, 0.28), tooth_len=(0.07, 0.30),
           long_p=0.14, long_len=(0.38, 0.62), notch=(0.0, 0.05), deep_p=0.18,
           deep=(0.10, 0.30), jitter=0.012, step=0.016, ramp=0.08, sharp=(0.6, 1.0)):
    """Turn a polyline (part of a CCW outline) into a torn edge.

    Teeth point along the outward (right-hand) normal of the polyline.
    """
    P, s = resample(pts, step)
    T = np.gradient(P, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-12)[:, None]
    Nrm = np.stack([T[:, 1], -T[:, 0]], 1)
    L = s[-1]
    prof = np.zeros(len(s))
    pos = 0.0
    while pos < L - 1e-6:
        long_tooth = rng.random() < long_p
        w = rng.uniform(*tooth_w) * (1.25 if long_tooth else 1.0)
        a, b = pos, min(pos + w, L)
        tl = rng.uniform(*long_len) if long_tooth else rng.uniform(*tooth_len)
        tip = a + (b - a) * rng.uniform(0.3, 0.7)
        nd = rng.uniform(*deep) if rng.random() < deep_p else rng.uniform(*notch)
        m = (s >= a) & (s <= b)
        left = (s - a) / max(tip - a, 1e-6)
        right = (b - s) / max(b - tip, 1e-6)
        shape = np.clip(np.where(s <= tip, left, right), 0.0, 1.0) ** rng.uniform(*sharp)
        # torn strips: broad body, ragged pointed end
        prof[m] = tl * shape[m] - nd * (1.0 - shape[m]) ** 2
        pos = b
    jit = jitter * fbm1(s * 30.0, seed, 3) + 0.4 * jitter * fbm1(s * 80.0, seed + 3, 2)
    env = smoothstep(0.0, ramp, s) * smoothstep(0.0, ramp, L - s)
    # long strips only hang downwards: teeth on near-vertical edges stay short
    hang = 0.3 + 0.7 * np.clip(Nrm[:, 1], 0.0, 1.0) ** 2
    off = (prof * amount * hang + jit) * env
    return P + Nrm * off[:, None]


def ragged_hole(cu, cv, r, rng, aspect=None):
    n = int(np.clip(8 + r * 50, 8, 16))
    th = np.sort(rng.uniform(0.0, 2 * np.pi, n))
    rr = r * (1.0 + 0.32 * rng.uniform(-1, 1, n))
    rr *= np.where(np.arange(n) % 2 == 0, 1.0, rng.uniform(0.6, 0.9, n))
    ax = aspect if aspect is not None else rng.uniform(0.7, 1.15)  # holes tear vertically
    rot = rng.uniform(-0.4, 0.4)
    x = rr * np.cos(th) * ax
    y = rr * np.sin(th)
    c, s_ = np.cos(rot), np.sin(rot)
    pts = np.stack([cu + c * x - s_ * y, cv + s_ * x + c * y], 1)
    return Polygon(pts).buffer(0)


def largest(geom):
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom


def triangulate(poly, max_area, min_angle=22, u_scale=1.9):
    """Constrained Delaunay in a u-stretched space -> triangles elongated along v
    (vertical folds need more resolution across than along)."""
    sp = affinity.scale(poly, xfact=u_scale, yfact=1.0, origin=(0, 0))
    verts, segs, holes = [], [], []

    def add_ring(coords):
        c = np.asarray(coords)[:-1]
        base = len(verts)
        verts.extend(c.tolist())
        n = len(c)
        segs.extend([(base + i, base + (i + 1) % n) for i in range(n)])

    add_ring(sp.exterior.coords)
    for ring in sp.interiors:
        add_ring(ring.coords)
        holes.append(Polygon(ring).representative_point().coords[0])
    A = dict(vertices=np.asarray(verts), segments=np.asarray(segs))
    if holes:
        A["holes"] = np.asarray(holes)
    B = tr.triangulate(A, f"pq{min_angle}a{max_area:.6f}")
    V = B["vertices"].copy()
    V[:, 0] /= u_scale
    return V, B["triangles"]


def edge_raster(poly, ppu=220.0, pad=0.08):
    """Signed distance to the pattern outline (studs, + inside) on a raster in pattern space."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import distance_transform_edt

    u0, v0, u1, v1 = poly.bounds
    u0, v0, u1, v1 = u0 - pad, v0 - pad, u1 + pad, v1 + pad
    W = int(np.ceil((u1 - u0) * ppu))
    H = int(np.ceil((v1 - v0) * ppu))
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)

    def px(coords):
        return [((x - u0) * ppu, (y - v0) * ppu) for x, y in coords]

    polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
    for p in polys:
        dr.polygon(px(p.exterior.coords), fill=255)
        for ring in p.interiors:
            dr.polygon(px(ring.coords), fill=0)
    mask = np.asarray(img) > 127
    sd = distance_transform_edt(mask) / ppu - distance_transform_edt(~mask) / ppu
    return sd.astype(np.float32), (u0, v0, u1 - u0, v1 - v0)


# ------------------------------------------------------------ cloth layers
class ClothLayer:
    def __init__(self, name, drape, poly, material, offset=0.0, max_area=0.04):
        self.name = name
        self.drape = drape
        self.poly = poly
        self.material = material
        self.offset = offset
        self.max_area = max_area

    def build(self):
        uv2, tris = triangulate(self.poly, self.max_area)
        u, v = uv2[:, 0], uv2[:, 1]
        d = self.drape
        mid = d.point(u, v, offset=self.offset)
        nrm = d.normal(u, v)
        tris = np.asarray(tris)
        a, b, c = mid[tris[:, 0]], mid[tris[:, 1]], mid[tris[:, 2]]
        fn = np.cross(b - a, c - a)
        cen_n = nrm[tris[:, 0]] + nrm[tris[:, 1]] + nrm[tris[:, 2]]
        flip = np.einsum("ij,ij->i", fn, cen_n) < 0
        outer_tris = np.where(flip[:, None], tris[:, ::-1], tris)

        outer = mid + nrm * (THICK / 2)
        inner = mid - nrm * (THICK / 2)
        local_uv = np.stack([u, v], 1)
        sd, rinfo = edge_raster(self.poly)
        attrs = {"pat_u": u, "pat_v": v}

        p_out = Part(f"{self.name}_outer", outer, outer_tris.tolist(), self.material + "_outer",
                     uv=np.stack([u, -v], 1), local_uv=local_uv, attrs=dict(attrs),
                     weight_mode="cloth", uv_priority=1.15)
        # the underside of the part lying flat on the shoulders is pressed against the body and never
        # visible: skip it to save triangles (rim strips still close the edges)
        keep_in = ~np.all(v[outer_tris] < -0.04, axis=1)
        in_tris = outer_tris[keep_in][:, ::-1]
        used = np.unique(in_tris)
        remap = np.full(len(u), -1)
        remap[used] = np.arange(len(used))
        p_in = Part(f"{self.name}_inner", inner[used], remap[in_tris].tolist(), self.material + "_inner",
                    uv=np.stack([u, v], 1)[used], local_uv=local_uv[used],
                    attrs={"pat_u": u[used], "pat_v": v[used]}, weight_mode="cloth", uv_priority=0.5)

        # rim strips along every boundary edge (outer winding: interior on the left)
        edges = {}
        for ti, t in enumerate(outer_tris):
            for i in range(3):
                edges[(t[i], t[(i + 1) % 3])] = (ti, t[(i + 2) % 3])
        bnd = [e for e in edges if (e[1], e[0]) not in edges]
        rim_v, rim_f, rim_luv, rim_src, rim_side, rim_tri = [], [], [], [], [], []
        index = {}

        def rv(i, side):
            key = (i, side)
            if key not in index:
                index[key] = len(rim_v)
                rim_v.append(outer[i] if side == 0 else inner[i])
                rim_luv.append(local_uv[i])
                rim_src.append(i)
                rim_side.append(side)
            return index[key]

        for a_, b_ in bnd:
            rim_f.append((rv(a_, 0), rv(a_, 1), rv(b_, 1), rv(b_, 0)))
            ti, c_ = edges[(a_, b_)]
            rim_tri.append((ti, a_, b_, c_))  # owning outer triangle -> consistent rim UVs
        rim_src = np.asarray(rim_src)
        p_rim = Part(f"{self.name}_rim", np.asarray(rim_v), rim_f, self.material + "_outer",
                     local_uv=np.asarray(rim_luv), attrs={"pat_u": u[rim_src], "pat_v": v[rim_src]},
                     weight_mode="cloth", bake=False)
        p_rim.props["rim_src"] = rim_src  # index into the outer part's vertices
        p_rim.props["rim_side"] = np.asarray(rim_side)
        p_rim.props["rim_tri"] = np.asarray(rim_tri)
        for p in (p_out, p_in, p_rim):
            p.props["edge_raster"] = (sd, rinfo)
            p.props["layer"] = self.name
            p.merge = False  # vertex order must stay 1:1 with the pattern (rim UV lookup, weights)
        self.parts = (p_out, p_in, p_rim)
        self.n_bnd = len(bnd)
        return self.parts


# ---------------------------------------------------------------- patterns
def _outline(top_pts, right_pts, hem_pts, left_pts):
    """Assemble a CCW outline (u right, v up) from four ordered pieces."""
    pts = np.concatenate([top_pts, right_pts[1:], hem_pts[1:], left_pts[1:-1]])
    return Polygon(pts).buffer(0)


def _finish(poly, holes, tol=0.011):
    if holes:
        poly = poly.difference(unary_union(holes))
    poly = largest(poly.buffer(0))
    # drop slivers thinner than ~1.5 cm that would only produce degenerate triangles
    poly = largest(poly.buffer(-0.006).buffer(0.006).buffer(0))
    return poly.simplify(tol, preserve_topology=True)


def navy_hem_table(d):
    wb, ws, wf, wF, wE = d.W_BACK_CORNER, d.W_SIDE, d.W_FRONT_CORNER, d.W_FRONT, d.W_END
    return [(0.0, 3.66), (0.45, 3.58), (0.78, 3.15), (0.95, 2.45), (1.12, 2.30), (1.42, 2.42), (1.62, 2.95),
            (wb - 0.05, 3.40), (wb + 0.28, 3.52), (wb + 0.42, 2.80), (ws + 0.03, 1.40), (ws + 0.25, 1.18),
            (wf, 1.14), (wF, 1.20), (wF + 0.6, 1.22), (wE, 1.04)]


def navy_pattern(d: Drape, seed=11):
    rng = np.random.default_rng(seed)
    U = d.U
    tab = navy_hem_table(d)
    us = np.linspace(0.0, U, 900)
    w = np.abs(us - U / 2)
    hem = np.interp(w, [p[0] for p in tab], [p[1] for p in tab]) + 0.09 * fbm1(us * 1.7, seed, 3)

    top = np.stack([us, -d.top_len(us)], 1)
    hem_line = np.stack([us[::-1], hem[::-1]], 1)
    right = np.stack([np.full(40, U), np.linspace(top[-1, 1], hem[-1], 40)], 1)
    left = np.stack([np.zeros(40), np.linspace(hem[0], top[0, 1], 40)], 1)

    hem_t = tatter(hem_line, rng, seed + 1)
    kw = dict(amount=0.4, tooth_len=(0.01, 0.05), long_p=0.0, deep_p=0.06, deep=(0.04, 0.10))
    right_t = tatter(right, rng, seed + 2, **kw)
    left_t = tatter(left, rng, seed + 3, **kw)
    poly = _outline(top, right_t, hem_t, left_t)

    holes = []
    hv_of = lambda cu: np.interp(abs(cu - U / 2), [p[0] for p in tab], [p[1] for p in tab])  # noqa: E731
    # a few holes in the front capelets (visible from the front)
    for cu, cv, r in ((0.55, 0.78, 0.055), (1.25, 0.95, 0.07), (U - 0.62, 0.72, 0.06), (U - 1.35, 0.98, 0.065)):
        holes.append(ragged_hole(cu, cv, r, rng))
    # back: mostly in the lower half, some eating into the hem
    while len(holes) < 22:
        cu = rng.uniform(2.2, U - 2.2)
        hv = hv_of(cu)
        cv = rng.uniform(0.9, hv + 0.05)
        if rng.random() > 0.2 + 0.8 * smoothstep(1.0, hv, cv):
            continue
        r = 0.05 + 0.11 * rng.random() ** 1.6
        holes.append(ragged_hole(cu, cv, r, rng))
    return _finish(poly, holes)


def cream_back_pattern(d: Drape, u0, u1, seed, hem_base, top_v=2.25):
    rng = np.random.default_rng(seed)
    us = np.linspace(u0, u1, 200)
    hem = hem_base + 0.14 * fbm1(us * 2.1, seed, 3) + 0.10 * np.sin((us - u0) / (u1 - u0) * np.pi)
    top = np.stack([us, np.full_like(us, top_v)], 1)
    hem_line = np.stack([us[::-1], hem[::-1]], 1)
    right = np.stack([np.full(40, u1), np.linspace(top_v, hem[-1], 40)], 1)
    left = np.stack([np.full(40, u0), np.linspace(hem[0], top_v, 40)], 1)
    hem_t = tatter(hem_line, rng, seed + 1, tooth_w=(0.08, 0.22), tooth_len=(0.08, 0.36), long_p=0.18)
    kw = dict(amount=0.7, tooth_len=(0.03, 0.12), long_p=0.06)
    right_t = tatter(right, rng, seed + 2, **kw)
    left_t = tatter(left, rng, seed + 3, **kw)
    poly = _outline(top, right_t, hem_t, left_t)
    holes = []
    for _ in range(400):
        if len(holes) >= 6:
            break
        cu = rng.uniform(u0 + 0.15, u1 - 0.15)
        cv = rng.uniform(top_v + 0.45, hem_base + 0.05)
        holes.append(ragged_hole(cu, cv, 0.06 + 0.09 * rng.random() ** 1.4, rng))
    return _finish(poly, holes)


def cream_front_pattern(d: Drape, side, seed=31):
    """side=+1: left front panel (u near 0); side=-1: right front panel (u near U)."""
    rng = np.random.default_rng(seed)
    U = d.U
    width = 0.60
    u0, u1 = (0.0, width) if side > 0 else (U - width, U)
    vtop = 0.30
    us = np.linspace(u0, u1, 120)
    hem = 3.02 + 0.16 * fbm1(us * 2.3, seed, 3) + (0.1 if side > 0 else -0.04)
    top = np.stack([us, np.full_like(us, vtop)], 1)
    hem_line = np.stack([us[::-1], hem[::-1]], 1)
    right = np.stack([np.full(60, u1), np.linspace(vtop, hem[-1], 60)], 1)
    left = np.stack([np.full(60, u0), np.linspace(hem[0], vtop, 60)], 1)
    hem_t = tatter(hem_line, rng, seed + 1, tooth_w=(0.07, 0.18), tooth_len=(0.10, 0.38), long_p=0.3)

    def fade_torn(line, torn_seed):
        torn = tatter(line, rng, torn_seed, amount=0.7, tooth_w=(0.10, 0.24), tooth_len=(0.015, 0.06),
                      long_p=0.05, deep_p=0.08)
        o, _ = resample(line, 0.016)  # same step as tatter() so both polylines align point-for-point
        k = min(len(o), len(torn))
        amt = smoothstep(1.0, 1.6, o[:k, 1])[:, None]  # untorn while hidden under the capelet
        return o[:k] * (1 - amt) + torn[:k] * amt

    right_t = fade_torn(right, seed + 2)
    left_t = fade_torn(left, seed + 3)
    poly = _outline(top, right_t, hem_t, left_t)
    holes = [ragged_hole(rng.uniform(u0 + 0.15, u1 - 0.15), rng.uniform(1.5, 2.8), 0.05 + 0.06 * rng.random(), rng)
             for _ in range(3)]
    return _finish(poly, holes)


def build_cloth(d: Drape, navy_area=0.056, cream_area=0.066):
    U = d.U
    layers = [
        ClothLayer("CapeNavy", d, navy_pattern(d), "cloth_navy", offset=0.0, max_area=navy_area),
        ClothLayer("CapeCreamBackL", d, cream_back_pattern(d, U / 2 - 1.66, U / 2 - 0.74, 21, 3.72, top_v=1.95),
                   "cloth_cream", offset=-0.034, max_area=cream_area),
        ClothLayer("CapeCreamBackR", d, cream_back_pattern(d, U / 2 + 0.76, U / 2 + 1.62, 27, 3.62, top_v=1.95),
                   "cloth_cream", offset=-0.034, max_area=cream_area),
        ClothLayer("CapeCreamFrontL", d, cream_front_pattern(d, +1), "cloth_cream", offset=-0.034,
                   max_area=cream_area),
        ClothLayer("CapeCreamFrontR", d, cream_front_pattern(d, -1, seed=37), "cloth_cream", offset=-0.034,
                   max_area=cream_area),
    ]
    parts = []
    for L in layers:
        parts.extend(L.build())
    return layers, parts
