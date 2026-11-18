"""Minimal loader for Elastic projection h5 files.

Mirrors notebooks/elastic/core.py (load_elastic_projection / Section / project_points),
re-implemented here because core.py uses `X | Y` type hints that need Python >= 3.10
and the `rad` env runs an older Python.
"""
import numpy as np
import h5py
from scipy import interpolate
from matplotlib.path import Path


class Section:
    def __init__(self, name, lat_nodes, lon_nodes, xy_nodes, border):
        self.name = name
        self.xi = interpolate.RegularGridInterpolator((lat_nodes, lon_nodes), xy_nodes["x"])
        self.yi = interpolate.RegularGridInterpolator((lat_nodes, lon_nodes), xy_nodes["y"])
        self.border_latlon = np.stack([border["latitude"], border["longitude"]], -1)
        self.path = Path(self.border_latlon)
        v = self.border_latlon
        area = np.sum(np.roll(v[:, 1], 1) * v[:, 0] - np.roll(v[:, 0], 1) * v[:, 1])
        self.ccw = area > 0

    def project(self, lat, lon):
        pts = (lat, lon)
        return self.xi(pts), self.yi(pts)

    def contains(self, lat, lon):
        pts = np.stack([lat, lon], -1)
        if self.ccw:
            return self.path.contains_points(pts, radius=-1e-9)
        return ~self.path.contains_points(pts, radius=1e-9)


def load(path):
    with h5py.File(path, "r") as f:
        secs = []
        for h in range(f.attrs["number of sections"]):
            g = f[f"section {h}"]
            name = g.attrs.get("name", str(h))
            if isinstance(name, bytes):
                name = name.decode()
            secs.append(Section(name, g["latitude"][:], g["longitude"][:],
                                g["projected points"][:, :], g["boundary"][:]))
        bbox = f["bounding box"][:]
        boundary = f["projected boundary"][:]
    return secs, bbox, boundary


def project(lat, lon, secs):
    """Forward project like core.project_points: later sections overwrite earlier.
    Returns x, y (km) and section id (-1 if in no section)."""
    x = np.full(lat.shape, np.nan)
    y = np.full(lat.shape, np.nan)
    sid = np.full(lat.shape, -1, int)
    for i, s in enumerate(secs):
        m = s.contains(lat, lon)
        x[m], y[m] = s.project(lat[m], lon[m])
        sid[m] = i
    return x, y, sid
