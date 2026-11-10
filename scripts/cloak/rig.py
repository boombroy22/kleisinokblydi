"""Armature for the cloak: Root -> CloakBase -> 9 hanging cape chains (+ tips), Lantern, Anchor.

Weights are computed analytically from the drape pattern coordinates (u, v):
  * everything on the shoulders (v <= 0) and all rigid hardware -> CloakBase
  * hanging cloth blends between the two nearest chains horizontally and between the two
    nearest bones of each chain vertically (max 4 influences, normalised)
"""
import bpy
import numpy as np
from mathutils import Vector

from .noise import smoothstep

BONE_SEG = 0.72  # target bone length along the drape
N_BONES = 5      # every chain gets the same bone count and joint rows (regular 9 x 5 grid for physics)


def column_defs(U):
    return [("L4", 0.45), ("L3", 1.80), ("L2", U / 2 - 1.85), ("L1", U / 2 - 0.95), ("C", U / 2),
            ("R1", U / 2 + 0.95), ("R2", U / 2 + 1.85), ("R3", U - 1.80), ("R4", U - 0.45)]


class CapeRig:
    def __init__(self, drape, cloth_uv, lantern_top, lantern_bottom, anchor_top, anchor_crown):
        """cloth_uv: (N,2) pattern coords of every hanging cloth vertex (to size the chains)."""
        self.d = drape
        U = drape.U
        self.cols = []
        defs = column_defs(U)
        us = np.array([c[1] for c in defs])
        for i, (name, uc) in enumerate(defs):
            lo = us[i - 1] if i > 0 else -1.0
            hi = us[i + 1] if i < len(us) - 1 else U + 1.0
            # narrow band around the column so a neighbour's long panel does not stretch this chain
            half = 0.3 * min(uc - lo, hi - uc)
            sel = (cloth_uv[:, 0] > uc - half) & (cloth_uv[:, 0] < uc + half)
            ext = float(np.percentile(cloth_uv[sel, 1], 99.0)) if sel.any() else 1.2
            n_b = int(np.clip(round(ext / BONE_SEG), 2, 5))
            self.cols.append(dict(name=name, u=uc, ext=ext, n=n_b))
        # One shared set of joint rows for all chains: neighbouring chains then blend joints that sit at the
        # same drape height, which keeps weight gradients bounded (no hand-over where a short chain ends)
        # and lets physics code link chains row by row.
        v_max = max(c["ext"] for c in self.cols)
        self.v_max = v_max
        for c in self.cols:
            c["cloth_ext"] = c["ext"]
            c["ext"] = v_max
            c["n"] = N_BONES
            c["joints"] = np.linspace(0.0, v_max, N_BONES + 1)
        self.lantern = (np.asarray(lantern_top), np.asarray(lantern_bottom))
        self.anchor = (np.asarray(anchor_top), np.asarray(anchor_crown))

    # ------------------------------------------------------------ armature
    def build_armature(self, collection, name="CloakRig"):
        arm = bpy.data.armatures.new(name)
        ob = bpy.data.objects.new(name, arm)
        collection.objects.link(ob)
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        eb = arm.edit_bones
        root = eb.new("Root")
        root.head, root.tail = (0, 0, 0), (0, 0, 0.5)
        base = eb.new("CloakBase")
        base.head, base.tail = (0, 0, 1.0), (0, 0, 1.45)
        base.parent = root
        d = self.d
        for c in self.cols:
            P = d.point(np.full(c["n"] + 1, c["u"]), c["joints"], offset=-0.006)
            Nn = d.normal(np.full(c["n"] + 1, c["u"]), c["joints"])
            prev = base
            for k in range(c["n"]):
                b = eb.new(f"Cape_{c['name']}_{k + 1}")
                b.head, b.tail = Vector(P[k]), Vector(P[k + 1])
                b.parent = prev
                b.use_connect = k > 0
                b.align_roll(Vector((Nn[k] + Nn[k + 1]) / 2))  # local Z points away from the body
                prev = b
            tip = eb.new(f"Cape_{c['name']}_Tip")
            down = (P[-1] - P[-2]) / np.linalg.norm(P[-1] - P[-2])
            tip.head, tip.tail = Vector(P[-1]), Vector(P[-1] + down * 0.18)
            tip.parent = prev
            tip.use_connect = True
            tip.align_roll(Vector(Nn[-1]))
            tip.use_deform = False
        for nm, (top, bot) in (("Lantern", self.lantern), ("Anchor", self.anchor)):
            b = eb.new(nm)
            b.head, b.tail = Vector(top), Vector(bot)
            b.parent = base
            b.align_roll(Vector((0, 1, 0)) if nm == "Lantern" else Vector((0, -1, 0)))
            t = eb.new(nm + "_Tip")
            t.head, t.tail = Vector(bot), Vector(bot + (bot - top) / np.linalg.norm(bot - top) * 0.12)
            t.parent = b
            t.use_connect = True
            t.use_deform = False
        root.use_deform = False
        bpy.ops.object.mode_set(mode="OBJECT")
        self.arm_ob = ob
        return ob

    # -------------------------------------------------------------- weights
    def _vertical(self, c, v):
        """Weights for CloakBase and chain bones of column c at drape distance v (vectorised)."""
        n = c["n"]
        s = np.clip(v / (c["ext"] / n), 0.0, n)
        W = np.zeros((len(v), n + 1))  # col 0 = CloakBase
        top = s <= 0.5
        W[top, 0] = 1.0 - 2.0 * s[top]
        W[top, 1] = 2.0 * s[top]
        rest = ~top
        for k in range(1, n + 1):
            centre = k - 0.5
            w = np.clip(1.0 - np.abs(s[rest] - centre), 0.0, 1.0)
            if k == n:
                w = np.where(s[rest] >= centre, 1.0, w)
            if k == 1:
                w = np.where(s[rest] <= centre, 1.0, w)
            W[rest, k] += w
        W[v <= 0.0] = 0.0
        W[v <= 0.0, 0] = 1.0
        return W

    def weights_for(self, uv):
        """uv (N,2) -> list of dicts {bone: weight} (<= 4 entries, normalised)."""
        uv = np.asarray(uv, dtype=np.float64)
        u, v = uv[:, 0], uv[:, 1]
        cu = np.array([c["u"] for c in self.cols])
        ext = np.array([c["ext"] for c in self.cols])
        N = len(u)
        i1 = np.clip(np.searchsorted(cu, u), 1, len(cu) - 1)
        i0 = i1 - 1
        a = smoothstep(0.0, 1.0, (u - cu[i0]) / (cu[i1] - cu[i0]))
        w0, w1 = 1.0 - a, a.copy()
        cov0 = smoothstep(ext[i0] + 0.3, ext[i0] - 0.05, v)
        cov1 = smoothstep(ext[i1] + 0.3, ext[i1] - 0.05, v)
        w0 *= cov0
        w1 *= cov1
        none = (w0 + w1) < 1e-6
        w0 = np.where(none & (ext[i0] >= ext[i1]), 1.0, w0)
        w1 = np.where(none & (ext[i1] > ext[i0]), 1.0, w1)
        tot = w0 + w1
        w0, w1 = w0 / tot, w1 / tot
        out = [dict() for _ in range(N)]
        for ci in range(len(self.cols)):
            for idx, wc in ((i0, w0), (i1, w1)):
                m = (idx == ci) & (wc > 1e-6)
                if not m.any():
                    continue
                c = self.cols[ci]
                Wv = self._vertical(c, v[m])
                names = ["CloakBase"] + [f"Cape_{c['name']}_{k}" for k in range(1, c["n"] + 1)]
                rows = np.nonzero(m)[0]
                for j, r in enumerate(rows):
                    for b in np.nonzero(Wv[j] > 1e-6)[0]:
                        out[r][names[b]] = out[r].get(names[b], 0.0) + wc[r] * Wv[j, b]
        for dct in out:
            items = sorted(dct.items(), key=lambda kv: -kv[1])
            items = [(k, w) for k, w in items if w >= 0.01][:4]
            s = sum(w for _, w in items)
            dct.clear()
            dct.update({k: w / s for k, w in items})
        return out

    def assign(self, ob, weights):
        groups = {}
        for vi, dct in enumerate(weights):
            for bone, w in dct.items():
                g = groups.get(bone)
                if g is None:
                    g = ob.vertex_groups.get(bone) or ob.vertex_groups.new(name=bone)
                    groups[bone] = g
                g.add([vi], float(w), "REPLACE")

    def skin_object(self, ob, drape):
        mode = ob.get("weight_mode", "base")
        nv = len(ob.data.vertices)
        co = np.empty(nv * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        M = np.array(ob.matrix_world)
        Wp = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        if mode == "base":
            self.assign(ob, [{"CloakBase": 1.0}] * nv)
        elif mode in ("lantern", "anchor"):
            self.assign(ob, [{mode.capitalize(): 1.0}] * nv)
        elif mode == "rigid":
            uv, _ = drape.project([ob["anchor"]])
            w = self.weights_for(uv)[0]
            self.assign(ob, [w] * nv)
        elif mode == "lantern_blend":
            # chain hanging from the lantern to a pendant on the cloth: fade Lantern -> pendant weights
            a, b = np.array(ob["blend_start"]), np.array(ob["blend_end"])
            t = np.clip((Wp - a) @ (b - a) / max(float((b - a) @ (b - a)), 1e-9), 0.0, 1.0)
            uv, _ = drape.project([ob["anchor"]])
            wb = self.weights_for(uv)[0]
            out = []
            for ti in smoothstep(0.0, 1.0, t):
                dct = {k: w * ti for k, w in wb.items()}
                dct["Lantern"] = dct.get("Lantern", 0.0) + (1.0 - ti)
                items = [(k, w) for k, w in sorted(dct.items(), key=lambda kv: -kv[1])[:4] if w >= 0.01]
                s = sum(w for _, w in items)
                out.append({k: w / s for k, w in items})
            self.assign(ob, out)
        else:  # per-vertex cloth weights
            if "pat_u" in ob.data.attributes:
                pu = np.empty(nv, dtype=np.float32)
                pv = np.empty(nv, dtype=np.float32)
                ob.data.attributes["pat_u"].data.foreach_get("value", pu)
                ob.data.attributes["pat_v"].data.foreach_get("value", pv)
                uv = np.stack([pu, pv], 1)
            else:
                from scipy.spatial import cKDTree

                uv, _ = drape.project(Wp)
                # nearest-point projection can hop between fold ridges for parts floating off the cloth:
                # average it over a small neighbourhood so weights vary smoothly along the part
                tree = cKDTree(Wp)
                uv = np.array([uv[nb].mean(0) for nb in tree.query_ball_point(Wp, 0.12)])
            self.assign(ob, self.weights_for(uv))
