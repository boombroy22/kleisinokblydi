"""Plain-numpy mesh container shared by the generators (no bpy dependency)."""
import numpy as np


class Part:
    """One generated mesh piece.

    verts       (N,3) local coordinates (object space)
    faces       list of index tuples (tris/quads), outward winding
    uv          (N,2) per-vertex atlas basis UV in "real size" units (studs) or None
    local_uv    (N,2) per-vertex procedural coordinates for the shader or None
    attrs       dict name -> (N,) float per-vertex attributes
    matrix      4x4 object -> world placement
    weight_mode 'base' | 'cloth' | 'rigid' | 'lantern'
    anchor      world point used by weight_mode == 'rigid'
    """

    def __init__(self, name, verts, faces, material, uv=None, local_uv=None, attrs=None,
                 matrix=None, weight_mode="base", anchor=None, bake=True, uv_priority=1.0,
                 smooth=True, merge=True):
        self.name = name
        self.verts = np.asarray(verts, dtype=np.float64)
        self.faces = [tuple(int(i) for i in f) for f in faces]
        self.material = material
        self.uv = None if uv is None else np.asarray(uv, dtype=np.float64)
        self.local_uv = None if local_uv is None else np.asarray(local_uv, dtype=np.float64)
        self.attrs = attrs or {}
        self.matrix = np.eye(4) if matrix is None else np.asarray(matrix, dtype=np.float64)
        self.weight_mode = weight_mode
        self.anchor = anchor
        self.bake = bake
        self.uv_priority = uv_priority
        self.smooth = smooth
        self.merge = merge
        self.props = {}

    def world_verts(self):
        M = self.matrix
        return self.verts @ M[:3, :3].T + M[:3, 3]

    def tri_count(self):
        return sum(len(f) - 2 for f in self.faces)


def frame_matrix(origin, x_axis=None, y_axis=None, z_axis=None, scale=1.0):
    """Build an orthonormal placement matrix from two of the three axes."""
    o = np.asarray(origin, dtype=np.float64)
    if z_axis is not None and y_axis is not None:
        z = _n(z_axis)
        x = _n(np.cross(y_axis, z))
        y = np.cross(z, x)
    elif z_axis is not None and x_axis is not None:
        z = _n(z_axis)
        y = _n(np.cross(z, x_axis))
        x = np.cross(y, z)
    elif y_axis is not None and x_axis is not None:
        y = _n(y_axis)
        z = _n(np.cross(x_axis, y))
        x = np.cross(y, z)
    else:
        raise ValueError("need two axes")
    M = np.eye(4)
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64), (3,))
    M[:3, 0] = x * s[0]
    M[:3, 1] = y * s[1]
    M[:3, 2] = z * s[2]
    M[:3, 3] = o
    return M


def _n(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)
