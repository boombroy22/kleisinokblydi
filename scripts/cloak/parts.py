"""Procedural low-poly generators for the cloak decorations (numpy only).

Every generator returns a Part (or a MeshBuilder) with per-vertex UVs duplicated along
seams (welded later in Blender while keeping the UV split), a "real size" atlas basis
UV and a procedural LocalUV.  All faces are wound outward.
"""
import numpy as np

from .meshdata import Part, frame_matrix
from .noise import smoothstep

TAU = 2 * np.pi


def nrm(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def rotate(v, axis, ang):
    k = nrm(axis)
    v = np.asarray(v, dtype=np.float64)
    return v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * np.dot(k, v) * (1 - np.cos(ang))


# ------------------------------------------------------------------ curves
def catmull(ctrl, n_per=10, closed=False):
    P = np.asarray(ctrl, dtype=np.float64)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1:i + 3]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed:
        out.append(P[-2])
    return np.asarray(out)


def resample_path(P, step=None, n=None, closed=False):
    P = np.asarray(P, dtype=np.float64)
    if closed:
        P = np.vstack([P, P[:1]])
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if n is None:
        n = max(2, int(round(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n + (1 if closed else 0))
    Q = np.stack([np.interp(t, s, P[:, k]) for k in range(3)], 1)
    if closed:
        Q, t = Q[:-1], t[:-1]
    return Q, t


def rmf(P, up=None):
    """Rotation-minimising frames (double reflection). Returns T, N, B with T x N = B."""
    P = np.asarray(P, dtype=np.float64)
    T = nrm(np.gradient(P, axis=0))
    if up is None:
        up = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N0 = np.cross(np.cross(T[0], up), T[0])
    if np.linalg.norm(N0) < 1e-6:
        N0 = np.cross(T[0], [1.0, 0.0, 0.0])
    N = [nrm(N0)]
    for i in range(len(P) - 1):
        v1 = P[i + 1] - P[i]
        c1 = v1 @ v1
        if c1 < 1e-14:
            N.append(N[-1])
            continue
        rL = N[i] - (2 / c1) * (v1 @ N[i]) * v1
        tL = T[i] - (2 / c1) * (v1 @ T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        Ni = rL - (2 / c2) * (v2 @ rL) * v2 if c2 > 1e-14 else rL
        N.append(nrm(Ni))
    N = np.asarray(N)
    B = np.cross(T, N)
    return T, N, B


# ------------------------------------------------------------------ meshes
class MeshBuilder:
    """Accumulates verts/faces/uv/local_uv/t from several sub-shapes."""

    def __init__(self):
        self.v, self.f, self.uv, self.luv, self.t = [], [], [], [], []

    def nverts(self):
        return sum(len(x) for x in self.v)

    def add(self, verts, faces, uv, luv=None, t=None):
        base = self.nverts()
        verts = np.asarray(verts, dtype=np.float64)
        self.v.append(verts)
        self.f.extend([tuple(int(i) + base for i in fc) for fc in faces])
        self.uv.append(np.asarray(uv, dtype=np.float64))
        self.luv.append(np.asarray(luv if luv is not None else uv, dtype=np.float64))
        self.t.append(np.zeros(len(verts)) if t is None else np.asarray(t, dtype=np.float64))
        return self

    def merge(self, other):
        base = self.nverts()
        self.v.extend(other.v)
        self.uv.extend(other.uv)
        self.luv.extend(other.luv)
        self.t.extend(other.t)
        self.f.extend([tuple(i + base for i in fc) for fc in other.f])
        return self

    def add_part_world(self, p):
        self.add(p.world_verts(), p.faces, p.uv, p.local_uv, p.attrs.get("t"))
        return self

    def part(self, name, material, **kw):
        attrs = dict(kw.pop("attrs", {}))
        attrs.setdefault("t", np.concatenate(self.t))
        return Part(name, np.concatenate(self.v), self.f, material, uv=np.concatenate(self.uv),
                    local_uv=np.concatenate(self.luv), attrs=attrs, **kw)


def tube(P, radius, sides=8, frames=None, cap0="flat", cap1="flat", profile=None, closed=False):
    """Sweep a (possibly elliptical) circle along P.

    radius : scalar, (M,) or (M,2) [radius along N, radius along B]
    cap    : 'flat' | 'tip' | None ('tip' turns the end point into a cone apex)
    profile: f(theta (S+1,), s, i) -> radial multiplier
    """
    P = np.asarray(P, dtype=np.float64)
    M = len(P)
    T, N, B = frames if frames is not None else rmf(P)
    r = np.asarray(radius, dtype=np.float64)
    if r.ndim == 0:
        r = np.full((M, 2), float(r))
    elif r.ndim == 1:
        r = np.stack([r, r], 1)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_total = s[-1] + (np.linalg.norm(P[0] - P[-1]) if closed else 0.0)
    rpos = r[r.max(1) > 0]
    circ = TAU * (float(np.mean(rpos)) if len(rpos) else 0.01)
    th = np.linspace(0.0, TAU, sides + 1)
    W = sides + 1
    i0 = 1 if cap0 == "tip" else 0
    i1 = M - 1 if cap1 == "tip" else M
    idx = list(range(i0, i1))
    rings = []
    for i in idx:
        k = profile(th, s[i], i) if profile is not None else 1.0
        rings.append(P[i] + (r[i, 0] * k * np.cos(th))[:, None] * N[i] + (r[i, 1] * k * np.sin(th))[:, None] * B[i])
    if closed:
        rings.append(rings[0].copy())
    nr = len(rings)
    svals = [s[i] for i in idx] + ([s_total] if closed else [])
    V = [np.concatenate(rings)]
    UV = [np.stack([np.tile(th / TAU * circ, nr), np.repeat(svals, W)], 1)]
    LUV = [np.stack([np.tile(th / TAU, nr), np.repeat(svals, W)], 1)]
    TT = [np.repeat(np.asarray(svals) / max(s_total, 1e-9), W)]
    F = []
    for a_ in range(nr - 1):
        for j in range(sides):
            F.append((a_ * W + j, a_ * W + j + 1, (a_ + 1) * W + j + 1, (a_ + 1) * W + j))
    nv = nr * W
    mid = (np.arange(sides) + 0.5) / sides
    if cap0 == "tip":
        V.append(np.repeat(P[0][None], sides, 0))
        UV.append(np.stack([mid * circ, np.full(sides, s[0])], 1))
        LUV.append(np.stack([mid, np.full(sides, s[0])], 1))
        TT.append(np.zeros(sides))
        F += [(nv + j, j + 1, j) for j in range(sides)]
        nv += sides
    if cap1 == "tip":
        last = (nr - 1) * W
        V.append(np.repeat(P[-1][None], sides, 0))
        UV.append(np.stack([mid * circ, np.full(sides, s[-1])], 1))
        LUV.append(np.stack([mid, np.full(sides, s[-1])], 1))
        TT.append(np.ones(sides))
        F += [(last + j, last + j + 1, nv + j) for j in range(sides)]
        nv += sides
    mb = MeshBuilder()
    mb.add(np.concatenate(V), F, np.concatenate(UV), np.concatenate(LUV), np.concatenate(TT))
    if not closed:
        for cap, ring, ci, sign in ((cap0, rings[0], i0, -1.0), (cap1, rings[-1], i1 - 1, 1.0)):
            if cap != "flat":
                continue
            rr = float(np.mean(r[ci]))
            cuv = np.stack([np.cos(th[:-1]), np.sin(th[:-1])], 1) * rr
            verts = np.vstack([P[ci][None], ring[:-1]])
            uvs = np.vstack([[0.0, 0.0], cuv])
            if sign > 0:
                faces = [(0, 1 + j, 1 + (j + 1) % sides) for j in range(sides)]
            else:
                faces = [(0, 1 + (j + 1) % sides, 1 + j) for j in range(sides)]
            mb.add(verts, faces, uvs, uvs / max(rr, 1e-9), np.full(sides + 1, 0.0 if sign < 0 else 1.0))
    return mb


def planar_loop(center, axis_a, axis_b, half_a, half_b, wire, major=6, minor=4, squash=0.62):
    """Closed tube around a rounded rectangle in the plane (axis_a, axis_b): chain link / ring."""
    ang = np.linspace(0.0, TAU, major, endpoint=False) + (np.pi / major if major % 2 == 0 else 0.0)
    ca, sa = np.cos(ang), np.sin(ang)
    pa = np.sign(ca) * np.abs(ca) ** squash * half_a
    pb = np.sign(sa) * np.abs(sa) ** squash * half_b
    a, b = nrm(axis_a), nrm(axis_b - np.dot(axis_b, nrm(axis_a)) * nrm(axis_a))
    center = np.asarray(center, dtype=np.float64)
    C = center + pa[:, None] * a + pb[:, None] * b
    plane_n = nrm(np.cross(a, b))
    T = nrm(np.roll(C, -1, 0) - np.roll(C, 1, 0))
    Nout = nrm(np.cross(T, plane_n))
    flip = np.einsum("ij,ij->i", Nout, C - center) < 0
    Nout[flip] *= -1
    B = np.cross(T, Nout)
    return tube(C, wire, sides=minor, frames=(T, Nout, B), cap0=None, cap1=None, closed=True)


def revolve(profile_rz, sides, name, material, matrix=None, uv_priority=1.0):
    """Revolve an (r, z) profile ordered top -> bottom around local Z (r = 0 ends become apexes)."""
    prof = np.asarray(profile_rz, dtype=np.float64)
    M = len(prof)
    P = np.stack([np.zeros(M), np.zeros(M), prof[:, 1]], 1)
    T = np.repeat([[0.0, 0.0, -1.0]], M, 0)
    N = np.repeat([[1.0, 0.0, 0.0]], M, 0)
    B = np.cross(T, N)
    cap0 = "tip" if prof[0, 0] <= 1e-9 else "flat"
    cap1 = "tip" if prof[-1, 0] <= 1e-9 else "flat"
    # tube() measures s along P; use the profile length instead so UVs are not squashed
    mb = tube(P, prof[:, 0], sides=sides, frames=(T, N, B), cap0=cap0, cap1=cap1)
    return mb.part(name, material, matrix=matrix, uv_priority=uv_priority)


# ------------------------------------------------------------- decorations
def claw(root, out_dir, length, r0, curl_axis, curl=2.0, sides=8, segs=7, flat=0.78, name="Claw",
         material="bone"):
    d = nrm(out_dir)
    pts = [np.asarray(root, dtype=np.float64)]
    for _ in range(segs):
        d = rotate(d, curl_axis, curl / segs)
        pts.append(pts[-1] + d * length / segs)
    P = np.asarray(pts)
    t = np.linspace(0, 1, len(P))
    rr = r0 * (1 + 0.18 * np.sin(np.pi * t)) * (1 - t) ** 0.9
    Bv = nrm(curl_axis)
    T = nrm(np.gradient(P, axis=0))
    N = nrm(np.cross(Bv, T))
    B = np.cross(T, N)
    mb = tube(P, np.stack([rr, rr * flat], 1), sides=sides, frames=(T, N, B), cap0="flat", cap1="tip")
    return mb.part(name, material, uv_priority=1.2)


def long_bone(p0, p1, bow, r_mid, r0, r1, sides=8, segs=8, seed=0, name="Bone", material="bone", path=None):
    if path is None:
        c = (np.asarray(p0, dtype=np.float64) + np.asarray(p1)) / 2 + np.asarray(bow)
        t = np.linspace(0, 1, segs + 1)[:, None]
        P = (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * c + t ** 2 * np.asarray(p1)
    else:
        P, _ = resample_path(catmull(path, 12), n=segs + 1)
    t = np.linspace(0, 1, len(P))
    rr = r_mid + (r0 - r_mid) * np.exp(-(t / 0.14) ** 2) + (r1 - r_mid) * np.exp(-((1 - t) / 0.14) ** 2)
    endw = np.exp(-(t / 0.16) ** 2) + np.exp(-((1 - t) / 0.16) ** 2)
    T0, T1 = nrm(P[1] - P[0]), nrm(P[-1] - P[-2])
    P = np.vstack([P[0] - T0 * r0 * 0.80, P[0] - T0 * r0 * 0.50, P, P[-1] + T1 * r1 * 0.50, P[-1] + T1 * r1 * 0.80])
    rr = np.concatenate([[0.0, r0 * 0.72], rr, [r1 * 0.72, 0.0]])
    endw = np.concatenate([[1.0, 1.0], endw, [1.0, 1.0]])
    ph = np.random.default_rng(seed).uniform(0, TAU, 3)

    def prof(th, s, i):
        return 1 + 0.07 * np.sin(3 * th + ph[0] + s * 5) + 0.20 * endw[i] * np.cos(2 * th + ph[1])

    mb = tube(P, rr, sides=sides, cap0="tip", cap1="tip", profile=prof)
    return mb.part(name, material, uv_priority=1.15)


def skull(origin, facing, up, size, name="Skull", lon=10, lat=7):
    """Low-poly skull in local space (+Y face, +Z up), placed by a frame matrix."""
    phis = np.linspace(0, TAU, lon + 1) + np.pi / 2 + np.pi / lon
    thetas = np.linspace(0, np.pi, lat + 1)[1:-1]
    verts, uv = [], []
    for it, th in enumerate(thetas):
        for ip, ph in enumerate(phis):
            verts.append((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
            uv.append((ip / lon, 1 - (it + 1) / lat))
    verts = np.asarray(verts)
    top = len(verts)
    verts = np.vstack([verts, np.repeat([[0, 0, 1.0]], lon, 0), np.repeat([[0, 0, -1.0]], lon, 0)])
    uv = np.vstack([uv, np.stack([(np.arange(lon) + 0.5) / lon, np.ones(lon)], 1),
                    np.stack([(np.arange(lon) + 0.5) / lon, np.zeros(lon)], 1)])
    x, y, z = verts[:, 0].copy(), verts[:, 1].copy(), verts[:, 2].copy()
    x *= 0.80
    z *= 0.86
    front = smoothstep(0.1, 0.8, y)
    lower = smoothstep(0.05, -0.6, z)
    x *= 1 - 0.32 * lower * front
    y += 0.10 * lower * front
    z -= 0.10 * lower
    y = np.where(y > 0.78, 0.78 + (y - 0.78) * 0.4, y)
    z += 0.08 * smoothstep(0.0, -0.8, y) * smoothstep(-0.2, 0.6, z)
    for sx in (-1, 1):
        dd = (x - sx * 0.33) ** 2 + (z - 0.06) ** 2
        y -= 0.20 * np.exp(-dd / 0.045) * front
    verts = np.stack([x, y, z], 1)  # unit skull; size lives in the matrix so the shader sees unit coords
    W = lon + 1
    F = []
    for it in range(len(thetas) - 1):
        for ip in range(lon):
            a, b = it * W + ip, it * W + ip + 1
            c, d = (it + 1) * W + ip + 1, (it + 1) * W + ip
            F.append((a, d, c, b))
    last = (len(thetas) - 1) * W
    for ip in range(lon):
        F.append((top + ip, ip, ip + 1))
        F.append((top + lon + ip, last + ip + 1, last + ip))
    M = frame_matrix(origin, y_axis=facing, z_axis=up, scale=size / 2.0)
    p = Part(name, verts, F, "skull", uv=uv * np.array([TAU * size * 0.42, np.pi * size * 0.42]),
             local_uv=uv, matrix=M, uv_priority=1.4)
    p.attrs["t"] = np.zeros(len(verts))
    return p


def barnacle(origin, normal, size, rng, name="Barnacle", sides=6):
    j = rng.uniform(0.85, 1.15, sides)
    th = np.linspace(0, TAU, sides, endpoint=False) + rng.uniform(0, TAU)
    rings = [(1.0, -0.18), (0.62, 0.60), (0.40, 0.80)]
    verts, uv = [], []
    for rr, zz in rings:
        for k in range(sides):
            verts.append((rr * j[k] * np.cos(th[k]), rr * j[k] * np.sin(th[k]), zz))
            uv.append((rr * np.cos(th[k]), rr * np.sin(th[k])))
    verts.append((0, 0, 0.48))
    uv.append((0, 0))
    verts = np.asarray(verts) * size
    F = []
    for ri in range(2):
        for k in range(sides):
            a, b = ri * sides + k, ri * sides + (k + 1) % sides
            F.append((a, b, b + sides, a + sides))
    c = 3 * sides
    for k in range(sides):
        F.append((2 * sides + k, 2 * sides + (k + 1) % sides, c))
    n = nrm(normal)
    helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    M = frame_matrix(origin, z_axis=n, y_axis=np.cross(n, np.cross(helper, n)))
    uv = np.asarray(uv)
    p = Part(name, verts, F, "barnacle", uv=uv * size, local_uv=uv, matrix=M, uv_priority=1.3)
    p.attrs["t"] = np.asarray([0.0] * sides + [0.5] * sides + [1.0] * sides + [1.0])
    return p


def rope(ctrl, radius, sides=8, seg_len=0.1, closed=False, name="Rope"):
    P = catmull(ctrl, 16, closed=closed)
    P, _ = resample_path(P, step=seg_len, closed=closed)
    mb = tube(P, radius, sides=sides, cap0=None if closed else "flat", cap1=None if closed else "flat",
              closed=closed)
    return mb.part(name, "rope", uv_priority=1.0)


def chain(path, link_len=0.20, link_w=0.125, wire=0.026, name="Chain", twist0=0.0, major=6, max_links=None,
          minor=4):
    """Links along a polyline, alternating 90 degrees."""
    P = np.asarray(path, dtype=np.float64)
    pitch = link_len - 2.2 * wire
    Q, s = resample_path(P, step=pitch / 3.0)
    n = max(1, int(s[-1] / pitch))
    if max_links:
        n = min(n, max_links)
    T, N, B = rmf(Q)
    mb = MeshBuilder()
    ends = []
    for i in range(n):
        sc = (i + 0.5) * pitch
        k = int(np.clip(np.searchsorted(s, sc), 1, len(s) - 1))
        a2 = N[k] if (i % 2 == 0) else B[k]
        a2 = rotate(a2, T[k], twist0 + 0.18 * np.sin(i * 1.7))
        mb.merge(planar_loop(Q[k], T[k], a2, link_len / 2 - wire, link_w / 2 - wire, wire, major=major,
                             minor=minor))
        ends.append(Q[k] + T[k] * (link_len / 2 - wire))
    p = mb.part(name, "iron", uv_priority=0.75)
    p.props["end"] = ends[-1]
    p.props["end_dir"] = T[min(len(T) - 1, int(np.searchsorted(s, (n - 0.5) * pitch)))]
    return p


def anchor(top, size=1.0, name="Anchor"):
    """Ship anchor hanging from its ring at `top`, drawn in the XZ plane, facing -Y."""
    ex, ey, ez = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    top = np.asarray(top, dtype=np.float64)
    s = size
    mb = MeshBuilder()
    ring_r = 0.075 * s
    rc = top - ez * ring_r
    mb.merge(planar_loop(rc, ex, ez, ring_r, ring_r, 0.022 * s, major=10, minor=4, squash=1.0))
    shank_top = rc - ez * (ring_r + 0.005 * s)
    crown = top - ez * 0.93 * s
    P = np.linspace(shank_top, crown, 5)
    rr = np.linspace(0.032, 0.042, 5) * s
    Tn = np.repeat(-ez[None], 5, 0)
    Nn = np.repeat(ex[None], 5, 0)
    mb.merge(tube(P, np.stack([rr, rr * 0.8], 1), sides=6, frames=(Tn, Nn, np.cross(Tn, Nn)), cap0="flat",
                  cap1="flat"))
    sy = shank_top - ez * 0.12 * s
    Ps = np.linspace(sy - ex * 0.25 * s, sy + ex * 0.25 * s, 4)
    Ps = np.vstack([Ps[0] - ex * 0.03 * s, Ps, Ps[-1] + ex * 0.03 * s])
    rs = np.array([0.0, 0.034, 0.022, 0.022, 0.034, 0.0]) * s
    mb.merge(tube(Ps, rs, sides=6, cap0="tip", cap1="tip"))
    ang = np.linspace(-np.pi * 0.42, np.pi * 0.42, 9)
    R = 0.36 * s
    centre = crown + ez * R
    Pa = centre + np.outer(np.sin(ang), ex) * R - np.outer(np.cos(ang), ez) * R
    ra = 0.036 * s * (1 - 0.3 * np.abs(ang) / (np.pi * 0.42))
    Ta = nrm(np.gradient(Pa, axis=0))
    Na = np.repeat(-ey[None], 9, 0)
    mb.merge(tube(Pa, np.stack([ra * 0.85, ra], 1), sides=6, frames=(Ta, Na, np.cross(Ta, Na)), cap0="flat",
                  cap1="flat"))
    for side in (-1, 1):
        tip_end = Pa[0] if side < 0 else Pa[-1]
        tdir = nrm(Pa[0] - Pa[1]) if side < 0 else nrm(Pa[-1] - Pa[-2])
        outw = nrm(tip_end - centre)
        w = 0.09 * s
        base_c = tip_end - tdir * 0.10 * s
        apex = tip_end + tdir * 0.08 * s
        l_ = base_c + outw * w
        r_ = base_c - outw * w * 0.30
        th = ey * 0.028 * s
        vs = np.array([apex - th, l_ - th, r_ - th, apex + th, l_ + th, r_ + th])
        faces = [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)]
        cen = vs.mean(0)
        fixed = []
        for fc in faces:
            p = vs[list(fc)]
            n_ = np.cross(p[1] - p[0], p[2] - p[0])
            fixed.append(fc if np.dot(n_, p.mean(0) - cen) >= 0 else fc[::-1])
        uvf = np.stack([vs @ ex, vs @ ez], 1)
        mb.add(vs, fixed, uvf, uvf / s)
    return mb.part(name, "iron_anchor", uv_priority=1.6)


def hook(top, facing, size=1.0, name="Hook"):
    f = nrm(facing)
    ez = np.array([0, 0, 1.0])
    ex = nrm(np.cross(ez, f))
    top = np.asarray(top, dtype=np.float64)
    s = size
    mb = MeshBuilder()
    er = 0.05 * s
    ec = top - ez * er
    mb.merge(planar_loop(ec, ez, f, er, er, 0.017 * s, major=6, minor=4, squash=1.0))
    start = ec - ez * er
    straight = np.linspace(start, start - ez * 0.20 * s, 3)
    R = 0.095 * s
    c = straight[-1] + ex * R
    ang = np.linspace(np.pi, np.pi * 2.15, 8)[1:]
    bend = c + np.outer(np.cos(ang), ex) * R + np.outer(np.sin(ang), ez) * R
    P = np.vstack([straight, bend])
    t = np.linspace(0, 1, len(P))
    rr = 0.026 * s * (1 - t ** 3) + 0.004 * s
    mb.merge(tube(P, rr, sides=6, cap0="flat", cap1="tip"))
    return mb.part(name, "iron", uv_priority=1.2)


def shell(origin, normal, up, size, name="Shell", seg=10):
    """Scallop: domed fan opening downward (local -Y) from the hinge at `origin`."""
    ang = np.linspace(-1.15, 1.15, seg + 1)
    radii = [0.42, 0.75, 1.0]
    top_v, top_uv = [], []
    for r in radii:
        for a in ang:
            x, y = np.sin(a) * r, -np.cos(a) * r
            z = 0.26 * (1 - r ** 2) + 0.05 + 0.045 * np.cos(a * 9.0) * r
            top_v.append((x, y, z))
            top_uv.append((x, y))
    top_v = np.asarray(top_v)
    bottom_v = top_v.copy()
    bottom_v[:, 2] = -0.14 * (top_v[:, 2] - 0.05)
    verts = np.vstack([[[0.0, 0.0, 0.30]], top_v, [[0.0, 0.0, -0.02]], bottom_v])
    top_uv = np.asarray(top_uv)
    uv = np.vstack([[[0.0, 0.0]], top_uv, [[0.0, 0.0]], top_uv * np.array([-1, 1]) + np.array([2.4, 0])])
    W = seg + 1
    hb = 1 + 3 * W
    F = []
    for k in range(seg):
        F.append((0, 1 + k, 1 + k + 1))                      # top fan (+Z)
        F.append((hb, hb + 1 + k + 1, hb + 1 + k))           # bottom fan (-Z)
    for ri in range(2):
        for k in range(seg):
            a = 1 + ri * W + k
            F.append((a, a + W, a + W + 1, a + 1))           # top rings
            b = hb + 1 + ri * W + k
            F.append((b, b + 1, b + W + 1, b + W))           # bottom rings
    lt, lb = 1 + 2 * W, hb + 1 + 2 * W
    for k in range(seg):
        F.append((lt + k, lb + k, lb + k + 1, lt + k + 1))   # outer rim
    for side_k in (0, seg):
        col_t = [0] + [1 + ri * W + side_k for ri in range(3)]
        col_b = [hb] + [hb + 1 + ri * W + side_k for ri in range(3)]
        for q in range(3):
            fc = (col_t[q], col_t[q + 1], col_b[q + 1], col_b[q])
            F.append(fc if side_k == seg else fc[::-1])
    verts = verts * size * 0.5
    M = frame_matrix(origin, y_axis=up, z_axis=normal)
    p = Part(name, verts, F, "shell", uv=uv * size * 0.5, local_uv=uv, matrix=M, uv_priority=1.6)
    p.attrs["t"] = np.r_[0.0, np.zeros(len(top_v)), 1.0, np.ones(len(top_v))]
    return p


def lantern_parts(center, size=1.0):
    """Egg-shaped glowing glass in a rusty cage. Returns ([glass, cage], top_point, bottom_point)."""
    c = np.asarray(center, dtype=np.float64)
    s = size
    H, R = 0.30 * s, 0.185 * s
    zs = np.linspace(1.0, -1.0, 8)
    rr = np.sqrt(np.clip(1 - zs ** 2, 0, 1)) * (1 + 0.12 * (-zs))
    glass = revolve(np.stack([rr * R, zs * H], 1), 10, "LanternGlass", "lantern_glass",
                    matrix=frame_matrix(c, x_axis=[1, 0, 0], y_axis=[0, 1, 0]), uv_priority=1.5)
    mb = MeshBuilder()
    for k in range(6):
        a = TAU * k / 6 + 0.3
        zz = np.linspace(0.93, -0.93, 6)
        r_ = np.sqrt(np.clip(1 - zz ** 2, 0, 1)) * (1 + 0.12 * (-zz)) * R + 0.02 * s
        Pb = np.stack([c[0] + r_ * np.cos(a), c[1] + r_ * np.sin(a), c[2] + zz * H], 1)
        mb.merge(tube(Pb, 0.012 * s, sides=4, cap0=None, cap1=None))  # ends hidden inside the caps
    ez = np.array([0, 0, 1.0])
    for zf, rf in ((0.05, 1.0), (0.70, 0.78)):
        zc = c + ez * zf * H
        rad = np.sqrt(max(1 - zf ** 2, 0)) * (1 + 0.12 * (-zf)) * R + 0.026 * s
        mb.merge(planar_loop(zc, [1, 0, 0], [0, 1, 0], rad, rad, 0.014 * s, major=12, minor=3, squash=1.0))
    capt = np.array([[0.0, 1.10], [0.55, 0.95], [0.62, 0.86], [0.0, 0.84]])
    for flip in (1, -1):
        pr = capt * np.array([1, flip])
        if flip < 0:
            pr = pr[::-1]
        pc = revolve(np.stack([pr[:, 0] * R, pr[:, 1] * H], 1), 8, "cap", "iron",
                     matrix=frame_matrix(c, x_axis=[1, 0, 0], y_axis=[0, 1, 0]))
        mb.add_part_world(pc)
    rt = c + ez * (1.10 * H + 0.045 * s)
    mb.merge(planar_loop(rt, ez, [0, 1, 0], 0.045 * s, 0.045 * s, 0.013 * s, major=8, minor=3, squash=1.0))
    rb = c - ez * (1.10 * H + 0.04 * s)
    mb.merge(planar_loop(rb, ez, [1, 0, 0], 0.04 * s, 0.04 * s, 0.013 * s, major=8, minor=3, squash=1.0))
    cage = mb.part("LanternCage", "iron", uv_priority=1.2)
    return [glass, cage], rt + ez * 0.045 * s, rb - ez * 0.04 * s


def compass(origin, normal, up, size, name="Compass", sides=16):
    """Brass compass: planar dial island + bezel/side strip + planar back + bail ring."""
    R = size / 2
    th = np.linspace(0, TAU, sides + 1)
    ring = lambda r, z: np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z)], 1)  # noqa: E731
    mb = MeshBuilder()
    # dial (glass) - planar UV, LocalUV in [-1, 1]
    dv = np.vstack([[[0, 0, 0.10 * R]], ring(0.80 * R, 0.07 * R)[:-1]])
    duv = dv[:, :2] / R
    mb.add(dv, [(0, 1 + j, 1 + (j + 1) % sides) for j in range(sides)], duv * R, duv)
    # bezel + side strip
    prof = [(0.80, 0.07), (0.88, 0.12), (1.0, 0.08), (1.0, -0.04)]
    rings = [ring(r * R, z * R) for r, z in prof]
    W = sides + 1
    V = np.concatenate(rings)
    lens = np.concatenate([[0], np.cumsum([np.hypot(prof[i + 1][0] - prof[i][0], prof[i + 1][1] - prof[i][1])
                                           for i in range(len(prof) - 1)])]) * R
    uv = np.stack([np.tile(th / TAU * TAU * R, len(prof)), np.repeat(-lens, W)], 1)
    F = []
    for i in range(len(prof) - 1):
        for j in range(sides):
            a, b = i * W + j, i * W + j + 1
            F.append((a, (i + 1) * W + j, (i + 1) * W + j + 1, b))
    mb.add(V, F, uv, np.full((len(V), 2), 1.6))
    # back
    bv = np.vstack([[[0, 0, -0.06 * R]], ring(R, -0.04 * R)[:-1]])
    buv = bv[:, :2] / R
    mb.add(bv, [(0, 1 + (j + 1) % sides, 1 + j) for j in range(sides)], buv * R, np.full((len(bv), 2), 1.6))
    # bail ring at the top edge
    mb.merge(planar_loop(np.array([0, R * 1.06, 0.0]), [0, 1, 0], [0, 0, 1], 0.05 * size, 0.05 * size,
                         0.016 * size, major=6, minor=3, squash=1.0))
    p = mb.part(name, "compass", matrix=frame_matrix(origin, z_axis=normal, y_axis=up), uv_priority=2.6)
    return p


def starfish(origin, normal, up, size, rng, name="Starfish"):
    k5 = np.arange(5)
    a_tip = TAU * k5 / 5 + np.pi / 2 + rng.uniform(-0.2, 0.2)
    a_val = a_tip + np.pi / 5
    rt = rng.uniform(0.85, 1.1, 5)
    V = [(0, 0, 0.20)]
    V += [(0.55 * np.cos(a), 0.55 * np.sin(a), 0.13) for a in a_tip]
    V += [(rt[k] * np.cos(a_tip[k]), rt[k] * np.sin(a_tip[k]), 0.03) for k in range(5)]
    V += [(0.36 * np.cos(a), 0.36 * np.sin(a), 0.06) for a in a_val]
    V.append((0, 0, -0.03))
    V = np.asarray(V)
    C, R0, T0, V0, CB = 0, 1, 6, 11, 16
    F = []
    for k in range(5):
        vp, vn = V0 + (k - 1) % 5, V0 + k
        F += [(C, vp, R0 + k), (C, R0 + k, vn), (vp, T0 + k, R0 + k), (R0 + k, T0 + k, vn)]
        F += [(CB, T0 + k, vp), (CB, vn, T0 + k)]
    uv = V[:, :2].copy()
    p = Part(name, V * (size / 2), F, "starfish", uv=uv * size / 2, local_uv=uv,
             matrix=frame_matrix(origin, z_axis=normal, y_axis=up), uv_priority=1.4)
    p.attrs["t"] = np.zeros(len(V))
    return p


def ribbon_on_surface(points, normals, widths, lift=0.008, name="Seaweed", material="seaweed", priority=0.9):
    """Single-sided leaf ribbon lying on a surface: 3 verts across (edge, raised midrib, edge)."""
    P = np.asarray(points)
    Nn = nrm(np.asarray(normals))
    T = nrm(np.gradient(P, axis=0))
    side = nrm(np.cross(Nn, T))
    n = len(P)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    V, uv = [], []
    for i in range(n):
        w = widths[i]
        V += [P[i] + side[i] * w / 2 + Nn[i] * lift * 0.3, P[i] + Nn[i] * lift, P[i] - side[i] * w / 2]
        uv += [(0.0, s[i]), (0.5, s[i]), (1.0, s[i])]
    V = np.asarray(V)
    uv = np.asarray(uv)
    F = []
    for i in range(n - 1):
        a = 3 * i
        for f in ((a, a + 3, a + 4, a + 1), (a + 1, a + 4, a + 5, a + 2)):
            p = V[list(f)]
            fn = np.cross(p[1] - p[0], p[2] - p[0])
            F.append(f if np.dot(fn, Nn[i]) >= 0 else f[::-1])
    real_uv = np.stack([uv[:, 0] * float(np.mean(widths)), uv[:, 1]], 1)
    p = Part(name, V, F, material, uv=real_uv, local_uv=uv, weight_mode="cloth", uv_priority=priority)
    p.attrs["t"] = np.repeat(s / max(s[-1], 1e-9), 3)
    return p
