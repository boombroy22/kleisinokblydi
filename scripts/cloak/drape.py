"""Drape surface of the cloak.

Coordinates are Blender (Z up), 1 unit = 1 stud, origin = centre of the R6 Torso.
The character faces -Y; +X is the character's LEFT; +Y is the back.

The cloth is parametrised by pattern coordinates (u, v):
  u : arc length along the shoulder "rim" (a rounded rectangle just outside the
      shoulders+arms block), running from the front-left end, around the left
      side, across the back, the right side, to the front-right end.
  v : drape distance.  v < 0 is the flat part lying on top of the shoulders
      (towards the neck), v = 0 is the shoulder edge, v > 0 hangs downwards.
"""
import numpy as np

from .noise import fbm1, fbm3, smoothstep

XR, YR, RC = 2.06, 0.60, 0.30   # rim half-extents and corner radius
Z_TOP = 1.035                    # cloth mid-surface height on the shoulders
NECK_AX, NECK_AY = 0.72, 0.60    # neckline ellipse
X_FRONT = 0.72                   # |x| where the front edges of the cloak begin
RB = 0.15                        # bend radius over the shoulder edge (rounded shoulders)
LB = np.pi * RB / 2.0
H_REF = 3.6                      # reference hanging length for flare profiles


def _interp_table(x, table):
    xs = np.array([p[0] for p in table])
    ys = np.array([p[1] for p in table])
    return np.interp(x, xs, ys)


class Drape:
    def __init__(self, seed=7):
        self.seed = seed
        segs = []

        def line(a, b, k):
            t = np.linspace(0.0, 1.0, k, endpoint=False)[:, None]
            return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t

        def arc(c, r, a0, a1, k):
            t = np.linspace(a0, a1, k, endpoint=False)
            return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)

        segs = [
            line((X_FRONT, -YR), (XR - RC, -YR), 500),
            arc((XR - RC, -YR + RC), RC, -np.pi / 2, 0.0, 300),
            line((XR, -YR + RC), (XR, YR - RC), 300),
            arc((XR - RC, YR - RC), RC, 0.0, np.pi / 2, 300),
            line((XR - RC, YR), (-XR + RC, YR), 1600),
            arc((-XR + RC, YR - RC), RC, np.pi / 2, np.pi, 300),
            line((-XR, YR - RC), (-XR, -YR + RC), 300),
            arc((-XR + RC, -YR + RC), RC, np.pi, 1.5 * np.pi, 300),
            line((-XR + RC, -YR), (-X_FRONT, -YR), 500),
        ]
        P = np.concatenate(segs + [np.array([[-X_FRONT, -YR]])])
        d = np.linalg.norm(np.diff(P, axis=0), axis=1)
        self.us = np.concatenate([[0.0], np.cumsum(d)])
        self.P = P
        self.U = float(self.us[-1])
        T = np.gradient(P, axis=0)
        T /= np.linalg.norm(T, axis=1)[:, None]
        self.T = T
        self.N = np.stack([T[:, 1], -T[:, 0]], 1)  # outward normal (path is CCW from above)

        # Landmarks measured from the back centre (w = |u - U/2|)
        front_len = XR - RC - X_FRONT
        arc_len = np.pi * RC / 2
        side_len = 2 * (YR - RC)
        back_half = XR - RC
        self.W_BACK_CORNER = back_half
        self.W_SIDE = back_half + arc_len
        self.W_FRONT_CORNER = self.W_SIDE + side_len
        self.W_FRONT = self.W_FRONT_CORNER + arc_len
        self.W_END = self.W_FRONT + front_len
        assert abs(self.W_END - self.U / 2) < 1e-3, (self.W_END, self.U / 2)

        wb, ws, wf, wF = self.W_BACK_CORNER, self.W_SIDE, self.W_FRONT_CORNER, self.W_FRONT
        # outward flare at the bottom (along rim normal)
        self.flare_tab = [(0.0, 0.52), (wb - 0.4, 0.58), (wb + 0.25, 0.66), (ws, 0.46), (wf, 0.22), (wF, 0.08), (self.W_END, 0.05)]
        # extra backward sweep (+Y) at the bottom
        self.sweep_tab = [(0.0, 0.16), (wb, 0.30), (ws, 0.42), (wf, 0.20), (wF, 0.03), (self.W_END, 0.0)]
        # fold amplitude at the bottom
        self.fold_tab = [(0.0, 0.13), (wb, 0.15), (ws, 0.08), (wf, 0.05), (wF, 0.035), (self.W_END, 0.03)]

    # ------------------------------------------------------------------ rim
    def w_of(self, u):
        return np.abs(np.asarray(u, dtype=np.float64) - self.U / 2.0)

    def rim(self, u):
        u = np.clip(np.asarray(u, dtype=np.float64), 0.0, self.U)
        R = np.stack([np.interp(u, self.us, self.P[:, 0]), np.interp(u, self.us, self.P[:, 1])], -1)
        n = np.stack([np.interp(u, self.us, self.N[:, 0]), np.interp(u, self.us, self.N[:, 1])], -1)
        n /= np.linalg.norm(n, axis=-1)[..., None]
        return R, n

    def neck(self, R):
        phi = np.arctan2(R[..., 1], R[..., 0])
        r = 1.0 / np.sqrt(np.cos(phi) ** 2 / NECK_AX ** 2 + np.sin(phi) ** 2 / NECK_AY ** 2)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], -1)

    def top_len(self, u):
        R, _ = self.rim(u)
        Nk = self.neck(R)
        L = np.linalg.norm(R - Nk, axis=-1)
        inside = np.linalg.norm(R, axis=-1) < np.linalg.norm(Nk, axis=-1)
        return np.where(inside, 0.0, L)

    # -------------------------------------------------------------- surface
    def _base(self, u, v, details=True):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        R, n = self.rim(u)
        Nk = self.neck(R)
        Lt = self.top_len(u)
        out = np.zeros(u.shape + (3,))

        # flat part on the shoulders (radial from the neckline to the rim)
        frac = np.where(Lt > 1e-6, 1.0 + v / np.maximum(Lt, 1e-6), 1.0)
        frac = np.clip(frac, 0.0, 1.0)
        Ptop = Nk + (R - Nk) * frac[..., None]

        # bend over the shoulder edge, then hang
        a = np.clip(v, 0.0, LB) / RB
        out_b = RB * np.sin(a)
        drop_b = RB * (1.0 - np.cos(a))
        hh = np.maximum(v - LB, 0.0)
        w = self.w_of(u)
        F = _interp_table(w, self.flare_tab)
        S = _interp_table(w, self.sweep_tab)
        A = _interp_table(w, self.fold_tab)
        g = (hh / H_REF) ** 1.5
        outward = out_b + F * g
        if details:
            phase = 2 * np.pi * u / 0.86 + 2.6 * fbm1(u * 0.5 + 3.1, self.seed, 3)
            env = smoothstep(0.45, 2.5, hh) ** 0.85  # upper back stays smooth for the lantern/crest
            wave = np.sin(phase) + 0.3 * np.sin(2.0 * phase + 0.7 + 1.5 * fbm1(u * 1.1, self.seed + 2, 2))
            # folds die out towards the free front edges so the edges hang flat
            edge_fade = smoothstep(0.0, 0.35, np.minimum(u, self.U - u))
            fold = A * env * wave * (0.35 + 0.65 * edge_fade)
            outward = outward + fold
        # dome the shoulder tops slightly (higher at the neck)
        ztop = Z_TOP + 0.025 * np.clip(-v, 0.0, 1.2) / 1.2
        Ph = R + n * outward[..., None]
        zh = Z_TOP - drop_b - hh
        yh = Ph[..., 1] + S * g

        top = v < 0.0
        out[..., 0] = np.where(top, Ptop[..., 0], Ph[..., 0])
        out[..., 1] = np.where(top, Ptop[..., 1], yh)
        out[..., 2] = np.where(top, ztop, zh)

        if details:
            # organic low-frequency wobble, growing towards the hem
            amp = 0.045 * smoothstep(0.4, 3.4, hh)
            nz = fbm3(out * np.array([0.9, 0.9, 0.6]) + 11.0, self.seed + 5, 3)
            out[..., 0] += n[..., 0] * amp * nz
            out[..., 1] += n[..., 1] * amp * nz
            # slight sag/lift so the hem line is not perfectly level
            out[..., 2] += 0.05 * smoothstep(1.0, 3.5, hh) * fbm1(u * 1.3 + 7.0, self.seed + 9, 2)
        return out

    def normal(self, u, v, eps=2e-3):
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        u0 = np.clip(u - eps, 0.0, self.U)
        u1 = np.clip(u + eps, 0.0, self.U)
        dPu = (self._base(u1, v) - self._base(u0, v)) / np.maximum(u1 - u0, 1e-9)[..., None]
        dPv = (self._base(u, v + eps) - self._base(u, v - eps)) / (2 * eps)
        nrm = np.cross(dPv, dPu)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1), 1e-12)[..., None]
        return nrm

    def point(self, u, v, offset=0.0):
        P = self._base(u, v)
        off = np.broadcast_to(np.asarray(offset, dtype=np.float64), np.shape(u))
        if np.any(off != 0.0):
            P = P + self.normal(u, v) * off[..., None]
        return P

    # ------------------------------------------------------------- inverse
    def build_lookup(self, du=0.02, dv=0.02, vmin=-1.6, vmax=4.2):
        uu, vv = np.meshgrid(np.arange(0.0, self.U + du, du), np.arange(vmin, vmax, dv))
        uu, vv = uu.ravel(), vv.ravel()
        Lt = self.top_len(uu)
        keep = vv >= -Lt
        uu, vv = uu[keep], vv[keep]
        P = self._base(uu, vv)
        from scipy.spatial import cKDTree

        self._lk_uv = np.stack([uu, vv], 1)
        self._lk_tree = cKDTree(P)

    def project(self, pts):
        """Nearest pattern coordinates (u, v) for arbitrary 3D points."""
        if not hasattr(self, "_lk_tree"):
            self.build_lookup()
        d, idx = self._lk_tree.query(np.asarray(pts, dtype=np.float64))
        return self._lk_uv[idx], d
