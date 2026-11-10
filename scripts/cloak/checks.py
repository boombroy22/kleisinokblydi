"""Geometry sanity checks: body penetration and decorations sinking into the cloth."""
import numpy as np
import shapely

from .cloth import THICK

# R6 mannequin boxes in the Blender torso-centred frame (x, y, z ranges)
BODY_BOXES = {
    "Torso": ((-1, 1), (-0.5, 0.5), (-1, 1)),
    "LeftArm": ((1, 2), (-0.5, 0.5), (-1, 1)),
    "RightArm": ((-2, -1), (-0.5, 0.5), (-1, 1)),
    "LeftLeg": ((0, 1), (-0.5, 0.5), (-3, -1)),
    "RightLeg": ((-1, 0), (-0.5, 0.5), (-3, -1)),
}
HEAD_PROFILE = [(0.90, 0.30), (0.92, 0.42), (0.99, 0.513), (1.08, 0.578), (1.20, 0.601), (1.80, 0.601),
                (1.92, 0.578), (2.01, 0.513), (2.08, 0.416), (2.10, 0.30)]


def inside_body(P, margin=0.004):
    P = np.asarray(P)
    hit = np.zeros(len(P), dtype=bool)
    for (x0, x1), (y0, y1), (z0, z1) in BODY_BOXES.values():
        hit |= ((P[:, 0] > x0 + margin) & (P[:, 0] < x1 - margin) & (P[:, 1] > y0 + margin) & (P[:, 1] < y1 - margin)
                & (P[:, 2] > z0 + margin) & (P[:, 2] < z1 - margin))
    zs = np.array([a for a, _ in HEAD_PROFILE])
    rs = np.array([b for _, b in HEAD_PROFILE])
    r = np.hypot(P[:, 0], P[:, 1])
    inz = (P[:, 2] > zs[0]) & (P[:, 2] < zs[-1])
    hit |= inz & (r < np.interp(P[:, 2], zs, rs) - margin)
    return hit


def body_report(parts):
    rows = []
    for p in parts:
        W = p.world_verts()
        hit = inside_body(W)
        if hit.any():
            rows.append((p.name, int(hit.sum()), len(W), W[hit][:2].round(3).tolist()))
    return rows


def cloth_report(parts, drape, layers, min_clear=0.0):
    """Vertices of decoration parts lying beneath the outer surface of an existing cloth layer."""
    polys = [(L.poly, L.offset) for L in layers]
    rows = []
    for p in parts:
        if p.weight_mode == "cloth" and p.material.startswith("cloth"):
            continue
        W = p.world_verts()
        uv, dist = drape.project(W)
        near = dist < 0.25
        if not near.any():
            continue
        S = drape.point(uv[:, 0], uv[:, 1])
        N = drape.normal(uv[:, 0], uv[:, 1])
        h = np.einsum("ij,ij->i", W - S, N)
        worst = 0.0
        count = 0
        for poly, off in polys:
            has = shapely.contains_xy(poly, uv[:, 0], uv[:, 1]) & near
            # inside the slab of this layer (between its inner and outer surfaces) = penetration
            pen = has & (h < off + THICK / 2 + min_clear) & (h > off - THICK / 2 - 0.06)
            if pen.any():
                depth = (off + THICK / 2 + min_clear - h[pen]).max()
                worst = max(worst, float(depth))
                count += int(pen.sum())
        if count:
            rows.append((p.name, count, len(W), round(worst, 4)))
    return rows
