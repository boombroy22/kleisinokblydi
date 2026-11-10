"""Small deterministic noise toolkit (numpy only) used by every generator."""
import numpy as np


def _hash(i, seed):
    """Integer lattice hash -> float in [-1, 1]."""
    i = np.asarray(i, dtype=np.int64)
    h = (i * 374761393 + (seed + 1) * 668265263) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h.astype(np.float64) / 4294967295.0) * 2.0 - 1.0


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def noise1(x, seed=0):
    """1D gradient noise, roughly in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64)
    i0 = np.floor(x).astype(np.int64)
    f = x - i0
    g0 = _hash(i0, seed)
    g1 = _hash(i0 + 1, seed)
    n0 = g0 * f
    n1 = g1 * (f - 1.0)
    return (n0 + _fade(f) * (n1 - n0)) * 2.0


def fbm1(x, seed=0, octaves=4, lac=2.03, gain=0.5):
    x = np.asarray(x, dtype=np.float64)
    total = np.zeros_like(x)
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * noise1(x * freq, seed + 131 * o)
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm


def _lattice3(ix, iy, iz, seed):
    k = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)
    return _hash(k, seed)


def noise3(p, seed=0):
    """3D value noise with quintic interpolation, roughly in [-1, 1]."""
    p = np.asarray(p, dtype=np.float64)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    ix, iy, iz = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
    fx, fy, fz = _fade(x - ix), _fade(y - iy), _fade(z - iz)
    out = 0.0
    for dx in (0, 1):
        wx = fx if dx else 1.0 - fx
        for dy in (0, 1):
            wy = fy if dy else 1.0 - fy
            for dz in (0, 1):
                wz = fz if dz else 1.0 - fz
                out = out + wx * wy * wz * _lattice3(ix + dx, iy + dy, iz + dz, seed)
    return out


def fbm3(p, seed=0, octaves=3, lac=2.1, gain=0.5):
    p = np.asarray(p, dtype=np.float64)
    total = np.zeros(p.shape[:-1])
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * noise3(p * freq, seed + 977 * o)
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
