"""Face design: every curve, size, layer depth and colour of the (original) calm male anime face.

All shapes are defined for the character's LEFT eye/brow (s > 0) and mirrored with s -> -s.
Eye-local coordinate x runs from the inner corner (0) to the outer corner (1).
"""
import numpy as np
from scipy.interpolate import PchipInterpolator

# ----------------------------------------------------------------- layer depths (studs)
OFF_LASH_UP = 0.010     # raised upper lash strip
OFF_LASH_LO = 0.006     # raised lower lash strip
OFF_BROW = 0.008        # raised brows
OFF_MOUTH = 0.004       # mouth decal planes
OFF_BLUSH = 0.0035      # blush decal planes
OFF_LID_UP = -0.005     # upper eyelid plate (inside the socket)
OFF_LID_LO = -0.009     # lower eyelid plate
OFF_IRIS = -0.014       # flat iris disc
OFF_SCLERA = -0.020     # back wall of the socket (eye white)
OFF_STRIP_BOTTOM = -0.002
HIDE_SHIFT = 0.07       # hidden mouth/blush planes sit this far inside the head

# ----------------------------------------------------------------- eye
S_IN, S_OUT = 0.095, 0.312
Z_IN, Z_OUT = -0.080, -0.066
H_UP, H_LO = 0.104, 0.030
CREASE_GAP = 0.030      # opening above the resting upper lid (room to open the eye wider)
BOTTOM_GAP = 0.010      # opening below the resting lower lid

_f_up = PchipInterpolator([0, .06, .20, .42, .62, .82, 1.0], [0, .42, .80, 1.0, .97, .70, 0.0])
_f_lo = PchipInterpolator([0, .12, .35, .58, .80, .93, 1.0], [0, .45, .88, 1.0, .78, .40, 0.0])


def eye_s(x):
    return S_IN + (S_OUT - S_IN) * np.asarray(x, dtype=np.float64)


def eye_x(s):
    return (np.abs(np.asarray(s, dtype=np.float64)) - S_IN) / (S_OUT - S_IN)


def base_z(x):
    return Z_IN + (Z_OUT - Z_IN) * np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _bump(x, p):
    return np.sin(np.pi * np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)) ** p


def y_up(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return base_z(x) + H_UP * _f_up(x)


def y_lo(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return base_z(x) - H_LO * _f_lo(x)


def y_crease(x):
    return y_up(x) + CREASE_GAP * _bump(x, 0.6)


def y_bottom(x):
    return y_lo(x) - BOTTOM_GAP * _bump(x, 0.7)


D_CLOSE = float(np.max(y_up(np.linspace(0, 1, 2001)) - y_lo(np.linspace(0, 1, 2001))))


def lid_weight(x):
    """Eyelid bone weight: translating the lid bone by D_CLOSE moves y_up exactly onto y_lo."""
    x = np.asarray(x, dtype=np.float64)
    w = (y_up(x) - y_lo(x)) / D_CLOSE
    return np.where((x <= 0.0) | (x >= 1.0), 0.0, np.clip(w, 0.0, 1.0))


def opening_polygon(n=64):
    """Socket opening (max-open eye) as a CCW (s, z) polygon for the LEFT eye."""
    t = np.linspace(0.0, 1.0, n + 1)
    x = (1 - np.cos(np.pi * t)) / 2
    top = np.stack([eye_s(x), y_crease(x)], 1)            # inner -> outer along the crease
    bot = np.stack([eye_s(x[::-1]), y_bottom(x[::-1])], 1)  # outer -> inner along the bottom
    poly = np.vstack([top, bot[1:-1]])
    # CCW in (s, z): bottom edge left->right, top edge right->left
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    return poly if area > 0 else poly[::-1]


# iris (flat disc)
IRIS_X = 0.50
IRIS_A, IRIS_B = 0.050, 0.064
IRIS_S = float(eye_s(IRIS_X))
IRIS_Z = float((y_up(IRIS_X) + y_lo(IRIS_X)) / 2 + 0.003)
EYE_CENTER_Z = float((y_up(0.5) + y_lo(0.5)) / 2)

# upper lash strip: centre line follows y_up; thickness grows towards the outer corner, then a wing
WING = np.array([[0.016, 0.002], [0.032, 0.007], [0.046, 0.013]])  # offsets from the outer corner


def lash_up_thickness(x):
    x = np.asarray(x, dtype=np.float64)
    ss = lambda a, b, v: np.clip((v - a) / (b - a), 0, 1) ** 2 * (3 - 2 * np.clip((v - a) / (b - a), 0, 1))  # noqa: E731
    return 0.004 + 0.018 * ss(0.0, 0.35, x) + 0.006 * ss(0.5, 0.95, x)


LASH_UP_BELOW = 0.4     # fraction of the strip thickness below the lid line
LASH_SPIKES = [(0.80, 0.020, 0.012, 0.55), (0.93, 0.024, 0.013, 0.30)]  # x, length, base width, tilt

# lower lash strip (outer part only)
LASH_LO_X0 = 0.32


def lash_lo_thickness(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.clip((x - LASH_LO_X0) / 0.28, 0, 1)
    return 0.0075 * a * a * (3 - 2 * a) * (1 - 0.35 * np.clip((x - 0.8) / 0.2, 0, 1))


# brows: centre line and thickness (inner end blunt, outer end pointed)
BROW_S = np.array([0.072, 0.13, 0.20, 0.27, 0.336])
BROW_Z = np.array([0.064, 0.080, 0.087, 0.081, 0.062])
BROW_W = np.array([0.028, 0.026, 0.022, 0.014, 0.003])

# ----------------------------------------------------------------- mouth / nose / blush
MOUTH_S, MOUTH_Z = 0.0, -0.262
MOUTH_W, MOUTH_H = 0.30, 0.15
NOSE_S, NOSE_Z = 0.012, -0.168
BLUSH_S, BLUSH_Z = 0.235, -0.158
BLUSH_W, BLUSH_H = 0.16, 0.075

MOUTH_NAMES = ["Calm", "Neutral", "Smile", "Grin", "Laugh", "A", "E", "O",
               "U", "Frown", "Surprised", "Smirk", "Pout", "Worried", "Shout", "Grit"]

# ----------------------------------------------------------------- colours (sRGB hex)
COL = {
    "skin": "#f8dfcd", "skin_shade": "#ecc3ae", "skin_blush": "#f2aaa0", "crease": "#b07a66",
    "nose": "#d79e88", "nose_line": "#a8685a",
    "sclera": "#fbfbfd", "sclera_shadow": "#c4cadb", "wall": "#e6e5ec",
    "lash": "#1f1a1d", "lash_low": "#4a3432", "brow": "#3b2b24", "brow_dark": "#2a1e19",
    "iris_rim": "#0b2230", "iris_top": "#123447", "iris_mid": "#1f6577", "iris_low": "#5fc2c9",
    "pupil": "#07121a", "glint": "#ffffff",
    "mouth_line": "#5a2c2a", "mouth_in": "#6e2632", "tongue": "#d4686a", "teeth": "#ffffff",
    "blush": "#ef8e8e", "blush_line": "#de6f73",
}


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])
