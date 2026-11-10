"""R6 head surface (Blender frame, origin = head centre, 1 unit = 1 stud).

Front of the face = -Y, up = +Z, character's LEFT = +X (viewer's right when looking at the face).
The classic R6 head is a cylinder of radius 0.6 and height 1.2 whose top/bottom edges are rounded
with a 0.3 fillet (measured from the supplied mannequin OBJ).

Face coordinates (s, z): s = arc length along the cylinder (R * theta, theta = 0 at the front,
positive towards the character's left), z = height.
"""
import numpy as np

R = 0.6       # head radius
F = 0.3       # fillet radius of the top/bottom edges
ZS = 0.3      # half height of the straight cylinder part
HH = 0.6      # half height of the head


def head_rz(z):
    a = np.abs(np.asarray(z, dtype=np.float64)) - ZS
    return np.where(a <= 0.0, R, (R - F) + np.sqrt(np.clip(F * F - np.clip(a, 0, None) ** 2, 0.0, None)))


def _profile_normal(z):
    z = np.asarray(z, dtype=np.float64)
    a = np.abs(z) - ZS
    r = head_rz(z)
    inside = a <= 0.0
    nr = np.where(inside, 1.0, (r - (R - F)) / F)
    nz = np.where(inside, 0.0, np.sign(z) * np.clip(a, 0, None) / F)
    return nr, nz


def head_point(s, z, off=0.0):
    """Point on (or `off` studs along the normal from) the head surface."""
    s, z = np.broadcast_arrays(np.asarray(s, dtype=np.float64), np.asarray(z, dtype=np.float64))
    off = np.broadcast_to(np.asarray(off, dtype=np.float64), s.shape)
    th = s / R
    r = head_rz(z)
    nr, nz = _profile_normal(z)
    rr = r + off * nr
    return np.stack([rr * np.sin(th), -rr * np.cos(th), z + off * nz], -1)


def head_normal(s, z):
    s, z = np.broadcast_arrays(np.asarray(s, dtype=np.float64), np.asarray(z, dtype=np.float64))
    th = s / R
    nr, nz = _profile_normal(z)
    return np.stack([nr * np.sin(th), -nr * np.cos(th), nz], -1)


def to_sz(P):
    """Inverse of head_point for points near the cylinder band: returns (s, z, radial distance)."""
    P = np.asarray(P, dtype=np.float64)
    th = np.arctan2(P[..., 0], -P[..., 1])
    rho = np.hypot(P[..., 0], P[..., 1])
    return R * th, P[..., 2], rho
