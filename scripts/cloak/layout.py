"""Places every decoration on the R6 mannequin / drape surface and returns Parts."""
import numpy as np

from . import parts as pt
from .cloth import THICK
from .drape import Drape
from .meshdata import Part
from .noise import fbm1

OUT = THICK / 2  # outer surface of the navy layer relative to the drape mid-surface


def surf(d, u, v, off=OUT):
    u = np.atleast_1d(np.asarray(u, dtype=np.float64))
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    P = d.point(u, v, offset=off)
    N = d.normal(u, v)
    Pd = d.point(u, v + 0.01, offset=off)
    return P[0], N[0], pt.nrm(Pd - P)[0]


def hug(d, pts, min_off, max_dist=0.45):
    """Push points that sink below `min_off` above the drape mid-surface back out along its normal."""
    pts = np.asarray(pts, dtype=np.float64)
    uv, dist = d.project(pts)
    S = d.point(uv[:, 0], uv[:, 1])
    N = d.normal(uv[:, 0], uv[:, 1])
    h = np.einsum("ij,ij->i", pts - S, N)
    push = np.where(dist < max_dist, np.maximum(0.0, min_off - h), 0.0)
    return pts + N * push[:, None]


class Layout:
    def __init__(self, d: Drape, seed=3):
        self.d = d
        self.rng = np.random.default_rng(seed)
        self.parts = []
        self.points = {}

    def add(self, p, mode=None, anchor=None):
        if mode:
            p.weight_mode = mode
        if anchor is not None:
            p.anchor = np.asarray(anchor, dtype=np.float64)
        self.parts.append(p)
        return p

    def lift_out(self, p, clear=0.006, iters=4):
        """Rigidly move a part out of the cloth if any vertex sinks below the navy outer surface."""
        d = self.d
        for _ in range(iters):
            W = p.world_verts()
            uv, dist = d.project(W)
            S = d.point(uv[:, 0], uv[:, 1])
            N = d.normal(uv[:, 0], uv[:, 1])
            h = np.einsum("ij,ij->i", W - S, N)
            need = (OUT + clear) - h
            mask = (dist < 0.3) & (need > 0) & (h > -0.08)
            if not mask.any():
                break
            shift = pt.nrm(N[mask].mean(0)) * (need[mask].max() + 0.001)
            p.matrix = p.matrix.copy()
            p.matrix[:3, 3] += shift
            if p.anchor is not None:
                p.anchor = np.asarray(p.anchor) + shift
        return p

    # ---------------------------------------------------------- shoulders
    def pauldron(self, side):
        """side=+1 left shoulder (x>0, u small), -1 right shoulder (mirrored in u)."""
        d, U, rng = self.d, self.d.U, self.rng
        S = "L" if side > 0 else "R"
        um = (lambda u: u) if side > 0 else (lambda u: U - u)
        made = []
        us = np.array([0.60, 0.64, 0.70, 0.78, 0.84])
        vs = np.array([-0.42, -0.14, 0.10, 0.44, 0.80])
        P = np.array([surf(d, um(u), v, OUT + 0.09)[0] for u, v in zip(us, vs)])
        made.append(pt.long_bone(None, None, None, 0.085, 0.13, 0.15, segs=8, seed=1 + side,
                                 name=f"Pauldron{S}_Bone", path=P))
        us = np.linspace(0.98, 2.60, 5)
        P = np.array([surf(d, um(u), -0.12, OUT + 0.085)[0] for u in us])
        P[:, 2] += 0.03 * np.sin(np.linspace(0, np.pi, 5))
        made.append(pt.long_bone(None, None, None, 0.085, 0.12, 0.125, segs=8, seed=5 + side,
                                 name=f"Pauldron{S}_Bar", path=P))
        rib = [(1.40 * side, 0.55, 1.26), (1.55 * side, 0.25, 1.34), (1.72 * side, -0.10, 1.32),
               (1.86 * side, -0.42, 1.22)]
        made.append(pt.long_bone(None, None, None, 0.072, 0.10, 0.105, segs=8, seed=9 + side,
                                 name=f"Pauldron{S}_Rib", path=rib))
        for k, (u, L, r0, up) in enumerate(((1.10, 1.02, 0.130, 0.32), (1.50, 1.14, 0.143, 0.50),
                                            (1.92, 1.10, 0.140, 0.52), (2.38, 0.97, 0.127, 0.40))):
            R, n = d.rim(np.array([um(u)]))
            out_h = np.array([n[0, 0], n[0, 1], 0.0])
            root = surf(d, um(u), -0.10, OUT + 0.10)[0] - out_h * 0.03
            out = pt.nrm(out_h + np.array([0, 0, up]))
            axis = pt.nrm(np.cross(out_h, [0, 0, -1.0]))
            made.append(pt.claw(root, out, L * rng.uniform(0.96, 1.04), r0, axis, curl=2.1 + 0.08 * k,
                                name=f"Pauldron{S}_Claw{k + 1}"))
        for k, (x, y, sz, face, lift) in enumerate(((1.22, -0.18, 0.27, (0.15, -1, 0.6), 0.0),
                                                    (1.20, 0.22, 0.25, (0.35, -0.2, 1.0), 0.0),
                                                    (1.50, 0.02, 0.28, (1.0, -0.45, 0.65), 0.07),
                                                    (1.58, -0.38, 0.24, (0.6, -1, 0.35), 0.0))):
            uv, _ = d.project([[x * side, y, 1.05]])
            base, nn, _ = surf(d, uv[0, 0], uv[0, 1], OUT)
            org = base + np.array([0, 0, sz * 0.45 + 0.05 + lift])
            f = pt.nrm(np.array([face[0] * side, face[1], face[2]]) + rng.normal(0, 0.1, 3))
            upv = pt.nrm(np.cross(f, np.cross([0, 0, 1.0], f)) + rng.normal(0, 0.12, 3))
            made.append(pt.skull(org, f, upv, sz, name=f"Pauldron{S}_Skull{k + 1}"))
        for p in made:
            self.add(self.lift_out(p), "base")
        return made

    def barnacles_on(self, hosts, count, prefix, min_up=-0.1, size=(0.055, 0.10)):
        rng = self.rng
        pool = []
        for h in hosts:
            W = h.world_verts()
            c = W.mean(0)
            for i in rng.choice(len(W), size=min(len(W), 50), replace=False):
                n = pt.nrm(W[i] - c)
                if n[2] >= min_up:
                    pool.append((W[i], n))
        idx = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
        out = []
        for k, i in enumerate(idx):
            p, n = pool[i]
            sz = rng.uniform(*size)
            b = pt.barnacle(p - n * sz * 0.2, n, sz, rng, name=f"{prefix}{k + 1}")
            out.append(self.add(self.lift_out(b), "base"))
        return out

    # -------------------------------------------------------------- front
    def collar(self):
        d = self.d
        ctrl = [(0.0, -0.605, 0.86), (0.30, -0.61, 0.93), (0.56, -0.57, 1.10), (0.80, -0.30, 1.155),
                (0.88, 0.08, 1.155), (0.77, 0.46, 1.145), (0.40, 0.675, 1.135), (0.0, 0.70, 1.135)]
        full = ctrl + [(-x, y, z) for (x, y, z) in ctrl[1:-1][::-1]]
        # small max_dist: only correct points resting on the shoulders, never shove the rope sideways
        P = hug(d, pt.catmull(full, 8, closed=True), OUT + 0.074, max_dist=0.16)
        self.add(pt.rope(P, 0.068, sides=8, seg_len=0.12, closed=True, name="CollarRope"), "base")
        ctrl2 = [(0.50, -0.62, 1.03), (0.42, -0.585, 0.80), (0.22, -0.548, 0.56), (0.0, -0.538, 0.47)]
        full2 = ctrl2 + [(-x, y, z) for (x, y, z) in ctrl2[:-1][::-1]]
        P2 = hug(d, pt.catmull(full2, 8), OUT + 0.052, max_dist=0.16)
        self.add(pt.rope(P2, 0.036, sides=6, seg_len=0.1, name="ChestRope"), "base")
        self.add(pt.anchor((0.0, -0.645, 0.84), size=1.0), "anchor")
        self.points["anchor_top"] = np.array([0.0, -0.645, 0.84])
        self.points["anchor_crown"] = np.array([0.0, -0.645, 0.84 - 0.93])

    # --------------------------------------------------------------- back
    def crest(self):
        made = [
            pt.long_bone(None, None, None, 0.09, 0.13, 0.14, segs=7, seed=11, name="Crest_Upper",
                         path=[(0, 0.86, 1.84), (0, 0.86, 1.48), (0, 0.92, 1.14), (0, 0.95, 0.97)]),
            pt.long_bone(None, None, None, 0.085, 0.14, 0.105, segs=7, seed=12, name="Crest_Lower",
                         path=[(0, 0.95, 1.06), (0, 0.99, 0.80), (0, 1.07, 0.60), (0, 1.18, 0.50)]),
        ]
        for side in (1, -1):
            s = "L" if side > 0 else "R"
            made.append(pt.long_bone(None, None, None, 0.078, 0.12, 0.135, segs=8, seed=13 + side,
                                     name=f"Crest_Arm{s}",
                                     path=[(0.06 * side, 0.95, 1.04), (0.50 * side, 0.92, 1.19),
                                           (1.00 * side, 0.84, 1.26), (1.46 * side, 0.72, 1.27)]))
            made.append(pt.claw((0.04 * side, 0.86, 1.80), (0.6 * side, 0.12, 1.0), 0.40, 0.062,
                                pt.nrm(np.cross([0.6 * side, 0.12, 1.0], [0, 0, -1.0])), curl=1.0,
                                name=f"Crest_Horn{s}"))
            path = np.array([(0.18 * side, 1.04, 0.66), (0.37 * side, 1.07, 0.36), (0.39 * side, 1.09, 0.02),
                             (0.27 * side, 1.11, -0.28), (0.06 * side, 1.13, -0.45)])
            P, _ = pt.resample_path(pt.catmull(path, 10), n=11)
            t = np.linspace(0, 1, 11)
            rr = 0.085 * (1 - t) ** 0.85 + 0.002
            mb = pt.tube(P, np.stack([rr, rr * 0.82], 1), sides=7, cap0="flat", cap1="tip")
            made.append(mb.part(f"Crescent{s}", "bone", uv_priority=1.3))
        for p in made:
            self.add(self.lift_out(p), "base")
        return made

    def lantern(self):
        (glass, cage), top, bottom = pt.lantern_parts((0.0, 1.22, 0.02), size=1.0)
        self.add(glass, "lantern")
        self.add(cage, "lantern")
        self.points["lantern_top"] = top
        self.points["lantern_bottom"] = bottom

    def back_chains(self):
        d = self.d
        paths = {
            "ChainBackL": [(0.12, 1.03, 1.30), (0.32, 1.05, 0.86), (0.64, 1.03, 0.54), (1.00, 0.97, 0.62),
                           (1.26, 0.89, 0.92), (1.40, 0.81, 1.15)],
            "ChainBackR": [(-0.12, 1.03, 1.26), (-0.34, 1.06, 0.78), (-0.68, 1.03, 0.40), (-1.04, 0.97, 0.54),
                           (-1.30, 0.89, 0.88), (-1.42, 0.81, 1.12)],
        }
        for name, ctrl in paths.items():
            P = hug(d, pt.catmull(ctrl, 12), OUT + 0.10)
            self.add(pt.chain(P, link_len=0.30, link_w=0.18, wire=0.036, name=name, twist0=0.3), "cloth")

    def surface_chain(self, name, u, v0, v1, link=0.20, minor=4):
        d = self.d
        vs = np.linspace(v0, v1, 24)
        us = np.full(24, u)
        lw = link * 0.62
        P = hug(d, d.point(us, vs, offset=OUT + lw / 2 + 0.006), OUT + lw / 2 + 0.006)
        return self.add(pt.chain(P, link_len=link, link_w=lw, wire=link * 0.13, name=name, minor=minor), "cloth")

    def pendants(self):
        d, U, rng = self.d, self.d.U, self.rng
        for name, u, v0, v1, size in (("HookL", 2.30, 0.85, 1.90, 1.05), ("HookR", U - 2.34, 1.20, 2.40, 1.1)):
            ch = self.surface_chain(name + "_Chain", u, v0, v1, link=0.26)
            end = ch.props["end"]
            uv, _ = d.project([end])
            p, n, down = surf(d, uv[0, 0], uv[0, 1], OUT)
            top = end + down * 0.03
            h = pt.hook(top, n, size=size, name=name)
            self.add(self.lift_out(h, clear=0.01), "rigid", anchor=top)
        for name, u, v0 in (("ShellFL", 0.30, 1.72), ("ShellFR", U - 0.28, 1.98), ("ShellBL", U / 2 - 1.22, 2.12),
                            ("ShellBR", U / 2 + 0.80, 2.70)):
            ch = self.surface_chain(name + "_Chain", u, v0, v0 + 0.32, link=0.16, minor=3)
            end = ch.props["end"]
            uv, _ = d.project([end])
            p, n, down = surf(d, uv[0, 0], uv[0, 1], OUT + 0.05)
            org = end + down * 0.02 + n * 0.035
            sh = pt.shell(org, n, -down, 0.29 * rng.uniform(0.92, 1.08), name=name)
            self.add(self.lift_out(sh), "rigid", anchor=org)
        p, n, down = surf(d, U / 2 - 0.70, 1.95, OUT + 0.035)
        comp = pt.compass(p, n, -down, 0.36)
        self.add(self.lift_out(comp), "rigid", anchor=p)
        bail = p - down * (0.18 * 1.06 + 0.03)
        lb = self.points["lantern_bottom"]
        ctrl = [lb, lb + np.array([0.16, -0.02, -0.15]), (lb + bail) / 2 + np.array([0, 0, -0.05]), bail]
        P = hug(d, pt.catmull(ctrl, 10), OUT + 0.05)
        cc = pt.chain(P, link_len=0.15, link_w=0.095, wire=0.02, name="CompassChain", minor=3)
        cc.props["blend"] = (lb, bail)
        self.add(cc, "lantern_blend", anchor=p)
        for name, u, v, sz in (("StarfishL", U / 2 - 0.56, 1.30, 0.32), ("StarfishR", U / 2 + 1.98, 1.58, 0.26)):
            p, n, down = surf(d, u, v, OUT + 0.012)
            self.add(self.lift_out(pt.starfish(p, n, -down, sz, rng, name=name)), "rigid", anchor=p)

    def seaweed(self):
        d, U, rng = self.d, self.d.U, self.rng
        specs = [(0.32, 0.10, 1.45, 0.17), (0.98, 0.05, 1.25, 0.16), (0.13, 1.22, 1.60, 0.16),
                 (U - 0.36, 0.12, 1.50, 0.17), (U - 1.02, 0.08, 1.20, 0.16), (U - 0.15, 1.16, 1.70, 0.16),
                 (1.84, 0.02, 1.30, 0.16), (U - 1.80, 0.02, 1.25, 0.16),
                 (U / 2 - 1.45, 0.05, 1.90, 0.18), (U / 2 + 1.35, 0.10, 1.75, 0.18),
                 (U / 2 - 0.66, 0.80, 1.30, 0.16), (U / 2 + 0.58, 0.85, 1.45, 0.16),
                 (U / 2 + 1.05, 1.90, 1.60, 0.17), (U / 2 - 2.05, 0.15, 1.40, 0.16),
                 (2.62, 0.10, 1.65, 0.16), (U - 2.66, 0.10, 1.55, 0.16),
                 # extra strands: the reference is draped in seaweed
                 (0.62, 0.15, 1.35, 0.15), (U - 0.64, 0.15, 1.40, 0.15), (1.55, 0.02, 1.15, 0.15),
                 (U - 1.55, 0.02, 1.20, 0.15), (U / 2 - 1.02, 0.10, 1.55, 0.17), (U / 2 + 0.98, 0.10, 1.60, 0.17),
                 (U / 2 - 1.78, 0.05, 1.30, 0.16), (U / 2 + 1.72, 0.05, 1.40, 0.16),
                 (U / 2 + 0.25, 1.25, 1.40, 0.15), (U / 2 - 0.32, 1.95, 1.30, 0.15)]
        for k, (u0, v0, L, w0) in enumerate(specs):
            n = max(4, int(L / 0.115))
            t = np.linspace(0, 1, n + 1)
            us = u0 + 0.07 * fbm1(t * 3 + k * 7.1, 40 + k, 2) + rng.uniform(-0.05, 0.05) * t
            vs = v0 + t * L
            P = hug(d, d.point(us, vs, offset=OUT + 0.016), OUT + 0.016)
            N = d.normal(us, vs)
            wid = w0 * (0.40 + 0.60 * np.clip(t / 0.18, 0, 1)) * (1 - 0.6 * t ** 2)
            wid *= 0.62 + 0.38 * np.abs(np.sin(t * L / 0.085 * np.pi + k))  # leafy, serrated outline
            self.add(pt.ribbon_on_surface(P, N, wid, name=f"Seaweed{k + 1:02d}"), "cloth")

    def net(self, name, u0, u1, v0, v1, cell=0.16, width=0.017, seed=0):
        import shapely
        from shapely.geometry import Polygon

        d, rng = self.d, np.random.default_rng(seed)
        ts = np.linspace(0, 1, 40)
        top = np.stack([u0 + (u1 - u0) * ts, v0 + 0.05 * fbm1(ts * 4, seed, 2)], 1)
        bot = np.stack([u1 - (u1 - u0) * ts, v1 + 0.25 * fbm1(ts * 5 + 3, seed + 1, 2) - 0.12 * np.sin(ts * np.pi)], 1)
        poly = Polygon(np.vstack([top, [[u1, (v0 + v1) / 2]], bot, [[u0, (v0 + v1) / 2]]])).buffer(0)
        segs = []
        diag = (u1 - u0) + (v1 - v0)
        for direction in (1, -1):
            for kk in range(-int(diag / cell) - 2, int(diag / cell) + 3):
                c = u0 + kk * cell
                vv = np.arange(v0 - 0.3, v1 + 0.3, cell / 2)
                uu = c + direction * (vv - v0)
                inside = shapely.contains_xy(poly, uu, vv)
                for i in range(len(vv) - 1):
                    if inside[i] and inside[i + 1]:
                        segs.append(((uu[i], vv[i]), (uu[i + 1], vv[i + 1])))
        V, F, UV = [], [], []
        for a, b in segs:
            uvp = np.array([a, b]) + rng.normal(0, 0.006, (2, 2))
            P = d.point(uvp[:, 0], uvp[:, 1], offset=OUT + 0.02)
            N = d.normal(uvp[:, 0], uvp[:, 1])
            T = pt.nrm(P[1] - P[0])
            base = len(V)
            for i in range(2):
                side = pt.nrm(np.cross(N[i], T))
                V += [P[i] + side * width / 2, P[i] - side * width / 2]
                UV += [(0.0, float(i)), (1.0, float(i))]
            q = (base, base + 2, base + 3, base + 1)
            pq = np.asarray(V)[list(q)]
            fn = np.cross(pq[1] - pq[0], pq[2] - pq[0])
            F.append(q if np.dot(fn, N[0]) >= 0 else q[::-1])
        V = hug(d, np.asarray(V), OUT + 0.012)
        UV = np.asarray(UV)
        p = Part(name, V, F, "net", uv=UV * np.array([width, cell * 0.7]), local_uv=UV, weight_mode="cloth",
                 uv_priority=0.5)
        p.attrs["t"] = np.zeros(len(V))
        p.merge = False
        return self.add(p, "cloth")

    def build_all(self):
        U = self.d.U
        self.collar()
        pl = self.pauldron(+1)
        pr = self.pauldron(-1)
        crest = self.crest()
        self.lantern()
        self.barnacles_on([p for p in pl if p.name.endswith(("Bone", "Bar", "Rib"))] + pl[-4:], 13, "BarnacleL")
        self.barnacles_on([p for p in pr if p.name.endswith(("Bone", "Bar", "Rib"))] + pr[-4:], 13, "BarnacleR")
        self.barnacles_on([p for p in crest if p.name.startswith("Crest_Arm")], 5, "BarnacleC")
        self.back_chains()
        self.pendants()
        self.seaweed()
        self.net("NetBackR", U / 2 + 0.95, U / 2 + 1.75, 0.95, 2.75, seed=5)
        self.net("NetSideL", U / 2 - 2.38, U / 2 - 1.78, 1.00, 2.40, seed=9)
        return self.parts
