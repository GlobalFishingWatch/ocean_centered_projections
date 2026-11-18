"""Shared ocean-bathymetry styling for the projection animations.

Source: ETOPO1 elevation averaged to 0.1 deg (etopo1_0p1deg.tif, north-up,
row 0 = 90N, col 0 = 180W, metres, negative = below sea level).

`ocean_rgb()` returns a float RGB image (0-1) of the sea floor coloured in the
GFW/pyseas dark palette: the abyss stays at the pyseas ocean colour (#0a1738)
and the water brightens toward the continental shelves, with a faint hillshade
for sea-floor relief. Land pixels are coloured as if they were depth 0; callers
are expected to composite their own land layer (#374a6d) on top.
"""
from pathlib import Path

import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy import ndimage

HERE = Path(__file__).parent
ETOPO_PATH = HERE / "etopo1_0p1deg.tif"

# (depth in metres, colour). Anchored on the pyseas dark ocean colour at -4500 m.
DEPTH_STOPS = [
    (-8000, "#050c22"),
    (-4500, "#0a1738"),  # pyseas dark "ocean"
    (-3000, "#0e2450"),
    (-1500, "#12356a"),
    (-200, "#1a5488"),   # shelf break
    (0, "#2478ad"),      # shoreline
]
HILLSHADE_STRENGTH = 0.10  # +/- fraction of brightness


def load_elevation():
    """Elevation in metres, shape (1800, 3600), row 0 = north."""
    with rasterio.open(ETOPO_PATH) as src:
        return src.read(1).astype("float32")


def depth_cmap():
    d0, d1 = DEPTH_STOPS[0][0], DEPTH_STOPS[-1][0]
    stops = [((d - d0) / (d1 - d0), to_rgb(c)) for d, c in DEPTH_STOPS]
    return LinearSegmentedColormap.from_list("gfw_depth", stops), d0, d1


def ocean_rgb(hillshade=True):
    elev = load_elevation()
    cmap, d0, d1 = depth_cmap()
    depth = np.clip(elev, d0, d1)
    rgb = cmap((depth - d0) / (d1 - d0))[..., :3].astype("float32")
    if hillshade:
        sea = np.minimum(elev, 0)
        sea = ndimage.gaussian_filter(sea, 1.0, mode=("nearest", "wrap"))
        # east-west gradient scaled by cos(lat) so relief strength is uniform
        lat = np.linspace(89.95, -89.95, elev.shape[0])[:, None]
        gx = (np.roll(sea, -1, 1) - np.roll(sea, 1, 1)) / np.maximum(np.cos(np.radians(lat)), 0.05)
        gy = np.gradient(sea, axis=0)
        shade = np.tanh((-gx + gy) / 600.0)  # light from the north-west
        rgb = np.clip(rgb * (1 + HILLSHADE_STRENGTH * shade[..., None]), 0, 1)
    return rgb


if __name__ == "__main__":
    import imageio
    rgb = ocean_rgb()
    imageio.imwrite(HERE / "ocean_bathy_0p1deg.png", (rgb * 255).round().astype("uint8"))
    # quick preview with a crude land overlay (elev >= 0) in pyseas land colour
    prev = rgb.copy()
    prev[load_elevation() >= 0] = to_rgb("#374a6d")
    imageio.imwrite(HERE / "preview_with_land.png", (prev[::2, ::2] * 255).round().astype("uint8"))
    print("wrote", HERE / "ocean_bathy_0p1deg.png")
