"""Shared geometry helpers: analytic Equal Earth (WGS84, matches PROJ eqearth),
mapping from projected metres to frame pixels, and the frame constants."""
import json
import numpy as np

HERE = __import__("os").path.dirname(__file__)
W, H = 960, 540
A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
_a = 6378137.0
_f = 1 / 298.257223563
_e2 = _f * (2 - _f)
_e = np.sqrt(_e2)


def _q(sinphi):
    return (1 - _e2) * (sinphi / (1 - _e2 * sinphi ** 2)
                        - np.log((1 - _e * sinphi) / (1 + _e * sinphi)) / (2 * _e))


_qp = _q(1.0)
_Rq = _a * np.sqrt(_qp / 2)


def authalic_lat(lat_deg):
    return np.arcsin(np.clip(_q(np.sin(np.radians(lat_deg))) / _qp, -1, 1))


def equal_earth(beta, lon_deg, lon0_deg):
    """beta: authalic latitude (radians, precomputed); returns x, y in metres."""
    lam = np.radians(((lon_deg - lon0_deg + 180) % 360) - 180)
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(beta))
    t2 = th * th
    t6 = t2 * t2 * t2
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * A4 * t6 * t2 + 7 * A3 * t6 + 3 * A2 * t2 + A1))
    y = th * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
    return _Rq * x, _Rq * y


def load_ref_geom():
    with open(f"{HERE}/reference_geom.json") as f:
        return json.load(f)


def ee_to_px(x, y, g):
    """projected metres -> frame pixel coords (col, row), origin top-left."""
    sx = (g["ax_x1"] - g["ax_x0"]) / (g["xlim"][1] - g["xlim"][0])
    sy = (g["ax_y1"] - g["ax_y0"]) / (g["ylim"][1] - g["ylim"][0])
    col = g["ax_x0"] + (x - g["xlim"][0]) * sx
    row = g["fig_h"] - (g["ax_y0"] + (y - g["ylim"][0]) * sy)
    return col, row
