"""Geometry of the face head: shell with eye sockets, eyelids, lash strips, brows, irises, mouth/blush planes.

Every piece carries a 2D parameter per vertex (usually face coordinates (s, z)) that is later mapped into
its UV island, so textures can be painted analytically in the same coordinates.
"""
import numpy as np
import triangle as tr
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import design as D
from .surface import HH, R, ZS, head_normal, head_point

N_AROUND = 96
DS = 2 * np.pi * R / N_AROUND
K_B = 19                       # front band spans s in [-K_B*DS, K_B*DS]
S_B = K_B * DS
Z_ROWS = np.linspace(-ZS, ZS, 13)


class Piece:
    def __init__(self, name, obj, verts, faces, param, island, weights, normal_mode="surface", sz=None):
        self.name = name
        self.obj = obj                      # FaceHead | FaceEyes | FaceOverlay
        self.verts = np.asarray(verts, dtype=np.float64)
        self.faces = [tuple(int(i) for i in f) for f in faces]
        self.param = np.asarray(param, dtype=np.float64)    # UV-island parameter per vertex
        self.island = island
        self.weights = {k: np.broadcast_to(np.asarray(v, dtype=np.float64), (len(self.verts),)).copy()
                        for k, v in weights.items()}
        self.normal_mode = normal_mode      # 'surface' -> analytic head normal at sz, 'flat' -> face normals
        self.sz = self.param if sz is None else np.asarray(sz, dtype=np.float64)

    def tri_count(self):
        return sum(len(f) - 2 for f in self.faces)


# ------------------------------------------------------------------ helpers
def _triangulate(outer, holes=(), flags="pq28a0.0006Y"):
    verts, segs, hole_pts = [], [], []

    def ring(pts):
        pts = np.asarray(pts, dtype=np.float64)
        b = len(verts)
        verts.extend(pts.tolist())
        n = len(pts)
        segs.extend([(b + i, b + (i + 1) % n) for i in range(n)])

    ring(outer)
    for h in holes:
        ring(h)
        hole_pts.append(Polygon(h).representative_point().coords[0])
    A = {"vertices": np.asarray(verts), "segments": np.asarray(segs)}
    if hole_pts:
        A["holes"] = np.asarray(hole_pts)
    B = tr.triangulate(A, flags)
    return B["vertices"], B["triangles"]


def _orient(verts, faces, want_normals):
    """Flip faces whose geometric normal disagrees with the wanted (per-vertex) normal."""
    out = []
    for f in faces:
        p = verts[list(f)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if len(f) == 4:
            n = n + np.cross(p[2] - p[0], p[3] - p[0])
        w = want_normals[list(f)].sum(0)
        out.append(tuple(f) if np.dot(n, w) >= 0 else tuple(f[::-1]))
    return out


def _grid_faces(nc, nr):
    """Quads of an (nr x nc) vertex grid stored row-major."""
    return [(j * nc + i, j * nc + i + 1, (j + 1) * nc + i + 1, (j + 1) * nc + i)
            for j in range(nr - 1) for i in range(nc - 1)]


def _surface_piece(name, obj, sz, faces, off, island, weights, param=None):
    sz = np.asarray(sz, dtype=np.float64)
    V = head_point(sz[:, 0], sz[:, 1], off)
    N = head_normal(sz[:, 0], sz[:, 1])
    faces = _orient(V, faces, N)
    return Piece(name, obj, V, faces, sz if param is None else param, island, weights, "surface", sz=sz)


def _slab(name, poly, off_top, island, weights_fn, flags="pq20a0.00003Y"):
    """Raised flat strip: triangulated top face on the surface + side walls down below the shell."""
    poly = poly.buffer(0)
    ext = np.asarray(poly.exterior.coords)[:-1]
    if Polygon(ext).exterior.is_ccw is False:
        ext = ext[::-1]
    v2, tris = _triangulate(ext, [], flags)
    top = _surface_piece(name + "_top", "FaceHead", v2, tris.tolist(), off_top, island, weights_fn(v2))
    # side walls along the boundary (first len(ext) vertices of the triangulation are the input ring)
    n = len(ext)
    b = ext
    Vt = head_point(b[:, 0], b[:, 1], off_top)
    Vb = head_point(b[:, 0], b[:, 1], D.OFF_STRIP_BOTTOM)
    V = np.vstack([Vt, Vb])
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    # walls face away from the strip's interior
    cen = Polygon(ext).representative_point().coords[0]
    cen3 = head_point(cen[0], cen[1], off_top)
    fixed = []
    for f in faces:
        p = V[list(f)]
        nrm = np.cross(p[1] - p[0], p[3] - p[0])
        fixed.append(f if np.dot(nrm, p.mean(0) - cen3) >= 0 else f[::-1])
    sz = np.vstack([b, b])
    side = Piece(name + "_side", "FaceHead", V, fixed, sz, island, weights_fn(sz), "flat", sz=sz)
    return [top, side]


def _mirror(sz, side):
    sz = np.array(sz, dtype=np.float64)
    if side < 0:
        sz[:, 0] *= -1
    return sz


# ------------------------------------------------------------------ shell
def shell_pieces():
    pieces = []
    # front band with the two eye openings
    sk = np.arange(-K_B, K_B + 1) * DS
    outer = ([(s, -ZS) for s in sk] + [(S_B, z) for z in Z_ROWS[1:]] +
             [(s, ZS) for s in sk[::-1][1:]] + [(-S_B, z) for z in Z_ROWS[::-1][1:-1]])
    hole_l = D.opening_polygon()
    hole_r = _mirror(hole_l, -1)[::-1]
    v2, tris = _triangulate(outer, [hole_l, hole_r], "pq28a0.0006Y")
    w = {"FaceBase": 1.0}
    pieces.append(_surface_piece("ShellFront", "FaceHead", v2, tris.tolist(), 0.0, "band_front", w))
    # back band (regular grid, wraps around the back of the head)
    ks = np.arange(K_B, N_AROUND - K_B + 1)
    S, Z = np.meshgrid(ks * DS, Z_ROWS)
    sz = np.stack([S.ravel(), Z.ravel()], 1)
    pieces.append(_surface_piece("ShellBack", "FaceHead", sz, _grid_faces(len(ks), len(Z_ROWS)), 0.0,
                                 "band_back", w))
    # fillets (profile quarter circles) - param (s, z)
    phis = np.linspace(0.0, np.pi / 2, 7)
    for name, sign in (("FilletTop", 1.0), ("FilletBot", -1.0)):
        zs = sign * (ZS + (HH - ZS) * np.sin(phis))
        zs = np.sort(zs)
        for part, ks in (("Front", np.arange(-K_B, K_B + 1)), ("Back", np.arange(K_B, N_AROUND - K_B + 1))):
            S, Z = np.meshgrid(ks * DS, zs)
            sz = np.stack([S.ravel(), Z.ravel()], 1)
            pieces.append(_surface_piece(name + part, "FaceHead", sz, _grid_faces(len(ks), len(zs)), 0.0,
                                         (name + part).lower(), w))
    # caps (flat discs of radius 0.3) - param = planar (x, y)
    for name, zc in (("CapTop", HH), ("CapBot", -HH)):
        th = np.arange(N_AROUND) * DS / R
        rings = [0.3, 0.2, 0.1]
        V = [(0.0, 0.0, zc)]
        for r in rings:
            V += [(r * np.sin(t), -r * np.cos(t), zc) for t in th]
        V = np.asarray(V)
        faces = []
        n = N_AROUND
        for ri in range(len(rings) - 1):
            a0, b0 = 1 + ri * n, 1 + (ri + 1) * n
            faces += [(a0 + i, a0 + (i + 1) % n, b0 + (i + 1) % n, b0 + i) for i in range(n)]
        last = 1 + (len(rings) - 1) * n
        faces += [(last + i, last + (i + 1) % n, 0) for i in range(n)]
        N = np.tile([0.0, 0.0, np.sign(zc)], (len(V), 1))
        faces = _orient(V, faces, N)
        p = Piece(name, "FaceHead", V, faces, V[:, :2], name.lower(), w, "surface",
                  sz=np.stack([np.zeros(len(V)), np.full(len(V), zc)], 1))
        p.cap_normal = N
        pieces.append(p)
    return pieces


# ------------------------------------------------------------------ eyes
def socket_pieces(side):
    S = "L" if side > 0 else "R"
    hole = D.opening_polygon()
    hole = _mirror(hole, side)
    if side < 0:
        hole = hole[::-1]
    pieces = []
    v2, tris = _triangulate(hole, [], "pq28a0.00004Y")
    pieces.append(_surface_piece(f"Sclera_{S}", "FaceHead", v2, tris.tolist(), D.OFF_SCLERA, f"sclera_{S}",
                                 {"FaceBase": 1.0}))
    # walls between the shell opening and the eye white
    n = len(hole)
    Vt = head_point(hole[:, 0], hole[:, 1], 0.0)
    Vb = head_point(hole[:, 0], hole[:, 1], D.OFF_SCLERA)
    V = np.vstack([Vt, Vb])
    seg = np.linalg.norm(np.diff(np.vstack([hole, hole[:1]]), axis=0), axis=1)
    per = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    cen = Polygon(hole).representative_point().coords[0]
    cen3 = head_point(cen[0], cen[1], D.OFF_SCLERA / 2)
    faces = []
    for i in range(n):
        f = (i, (i + 1) % n, n + (i + 1) % n, n + i)
        p = V[list(f)]
        nrm = np.cross(p[1] - p[0], p[3] - p[0])
        faces.append(f if np.dot(nrm, cen3 - p.mean(0)) >= 0 else f[::-1])  # walls face into the socket
    param = np.vstack([np.stack([per, np.zeros(n)], 1), np.stack([per, np.full(n, -D.OFF_SCLERA)], 1)])
    pieces.append(Piece(f"SocketWall_{S}", "FaceHead", V, faces, param, f"wall_{S}", {"FaceBase": 1.0}, "flat",
                        sz=np.vstack([hole, hole])))
    return pieces


def lid_pieces(side):
    S = "L" if side > 0 else "R"
    xs = np.linspace(-0.06, 1.06, 41)
    w = D.lid_weight(xs)
    out = []
    for kind, edge, sign, off in (("Up", D.y_up, 1.0, D.OFF_LID_UP), ("Low", D.y_lo, -1.0, D.OFF_LID_LO)):
        vs = np.array([0.0, 0.012, 0.03, 0.06, 0.11, 0.18, 0.27, 0.40, 0.55, 0.75, 1.0]) * 0.22
        X, Vv = np.meshgrid(xs, vs)
        s = D.eye_s(X) * side
        z = edge(np.clip(X, 0, 1)) + sign * Vv
        sz = np.stack([s.ravel(), z.ravel()], 1)
        wv = np.tile(w, len(vs))
        bone = f"Lid_{kind}_{S}"
        out.append(_surface_piece(f"Lid{kind}_{S}", "FaceHead", sz, _grid_faces(len(xs), len(vs)), off,
                                  f"lid{kind.lower()}_{S}", {bone: wv, "FaceBase": 1.0 - wv}))
    return out


def _curve_poly(C, below, above):
    """Polygon around centre line C (M,2) with per-point extents below/above along the curve normal."""
    T = np.gradient(C, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    Nn = np.stack([-T[:, 1], T[:, 0]], 1)
    Nn[Nn[:, 1] < 0] *= -1  # normal points up (+z)
    up = C + Nn * above[:, None]
    dn = C - Nn * below[:, None]
    return np.vstack([dn, up[::-1]]), up, Nn


def lash_up_polygon():
    t = np.linspace(0, 1, 90)
    x = (1 - np.cos(np.pi * t)) / 2
    C = np.stack([D.eye_s(x), D.y_up(x)], 1)
    th = D.lash_up_thickness(x)
    corner = C[-1]
    wing = corner + D.WING
    wth = th[-1] * np.array([0.85, 0.5, 0.0])
    C2 = np.vstack([C, wing])
    th2 = np.concatenate([th, wth])
    poly_pts, up, Nn = _curve_poly(C2, D.LASH_UP_BELOW * th2, (1 - D.LASH_UP_BELOW) * th2)
    shapes = [Polygon(poly_pts).buffer(0)]
    for xs, length, base, tilt in D.LASH_SPIKES:
        i = int(np.argmin(np.abs(x - xs)))
        p = up[i]
        tang = np.gradient(C, axis=0)[i]
        tang /= np.linalg.norm(tang)
        direc = Nn[i] * np.cos(tilt) + tang * np.sin(tilt)
        a = p - tang * base / 2 - Nn[i] * 0.004
        b = p + tang * base / 2 - Nn[i] * 0.004
        apex = p + direc * length
        shapes.append(Polygon([a, b, apex]).buffer(0))
    return unary_union(shapes).buffer(0.0006).buffer(-0.0006)


def lash_lo_polygon():
    t = np.linspace(0, 1, 60)
    x = D.LASH_LO_X0 + (1 - D.LASH_LO_X0) * t
    C = np.stack([D.eye_s(x), D.y_lo(x)], 1)
    th = D.lash_lo_thickness(x)
    th[0] = 0.0
    poly_pts, _, _ = _curve_poly(C, th * 0.5, th * 0.5)
    return Polygon(poly_pts).buffer(0)


def brow_polygon():
    from scipy.interpolate import PchipInterpolator

    t = np.linspace(0, 1, 60)
    fs = PchipInterpolator(np.linspace(0, 1, len(D.BROW_S)), D.BROW_S)
    fz = PchipInterpolator(np.linspace(0, 1, len(D.BROW_S)), D.BROW_Z)
    fw = PchipInterpolator(np.linspace(0, 1, len(D.BROW_S)), D.BROW_W)
    C = np.stack([fs(t), fz(t)], 1)
    w = fw(t)
    poly_pts, up, _ = _curve_poly(C, w * 0.5, w * 0.5)
    # slanted inner cut: move the top-inner corner outwards
    n = len(C)
    poly_pts[-1, 0] += 0.007
    return Polygon(poly_pts).buffer(0)


def strip_pieces(side):
    S = "L" if side > 0 else "R"

    def mir(poly):
        if side > 0:
            return poly
        from shapely.affinity import scale

        return scale(poly, xfact=-1, yfact=1, origin=(0, 0))

    def lid_w(bone):
        def fn(sz):
            x = D.eye_x(sz[:, 0])
            wv = D.lid_weight(x)
            return {bone: wv, "FaceBase": 1.0 - wv}
        return fn

    out = []
    out += _slab(f"LashUp_{S}", mir(lash_up_polygon()), D.OFF_LASH_UP, f"lashup_{S}", lid_w(f"Lid_Up_{S}"))
    out += _slab(f"LashLow_{S}", mir(lash_lo_polygon()), D.OFF_LASH_LO, f"lashlow_{S}", lid_w(f"Lid_Low_{S}"))
    out += _slab(f"Brow_{S}", mir(brow_polygon()), D.OFF_BROW, f"brow_{S}", lambda sz: {f"Brow_{S}": 1.0})
    return out


def iris_piece(side):
    S = "L" if side > 0 else "R"
    rings = np.array([0.0, 0.3, 0.55, 0.78, 1.0])
    ang = np.linspace(0, 2 * np.pi, 41)[:-1]
    uv = [(0.0, 0.0)]
    for r in rings[1:]:
        uv += [(r * np.cos(a), r * np.sin(a)) for a in ang]
    uv = np.asarray(uv)
    n = len(ang)
    faces = [(0, 1 + i, 1 + (i + 1) % n) for i in range(n)]
    for ri in range(len(rings) - 2):
        a0, b0 = 1 + ri * n, 1 + (ri + 1) * n
        faces += [(a0 + i, b0 + i, b0 + (i + 1) % n, a0 + (i + 1) % n) for i in range(n)]
    sz = np.stack([side * D.IRIS_S + D.IRIS_A * uv[:, 0], D.IRIS_Z + D.IRIS_B * uv[:, 1]], 1)
    return _surface_piece(f"Iris_{S}", "FaceEyes", sz, faces, D.OFF_IRIS, "iris", {f"Eye_{S}": 1.0}, param=uv)


# ------------------------------------------------------------------ overlay planes
def _plane(name, sc, zc, w, h, nc, nr, off, island, bone, hidden):
    S, Z = np.meshgrid(np.linspace(sc - w / 2, sc + w / 2, nc), np.linspace(zc - h / 2, zc + h / 2, nr))
    sz = np.stack([S.ravel(), Z.ravel()], 1)
    p = _surface_piece(name, "FaceOverlay", sz, _grid_faces(nc, nr), off, island, {bone: 1.0})
    if hidden:
        p.verts = p.verts + np.array([0.0, D.HIDE_SHIFT, 0.0])
    return p


def overlay_pieces():
    out = []
    for k, nm in enumerate(D.MOUTH_NAMES):
        out.append(_plane(f"Mouth{nm}", D.MOUTH_S, D.MOUTH_Z, D.MOUTH_W, D.MOUTH_H, 13, 7, D.OFF_MOUTH,
                          f"mouth_{k:02d}", f"Mouth_{nm}", hidden=(k != 0)))
    for side, S in ((1, "L"), (-1, "R")):
        out.append(_plane(f"Blush_{S}", side * D.BLUSH_S, D.BLUSH_Z, D.BLUSH_W, D.BLUSH_H, 9, 5, D.OFF_BLUSH,
                          f"blush_{S}", "Blush", hidden=True))
    return out


def all_pieces():
    pieces = shell_pieces()
    for side in (1, -1):
        pieces += socket_pieces(side) + lid_pieces(side) + strip_pieces(side) + [iris_piece(side)]
    pieces += overlay_pieces()
    return pieces
