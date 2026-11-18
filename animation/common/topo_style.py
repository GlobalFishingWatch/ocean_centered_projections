"""Shared land + sea-floor styling using the cmocean `topo` colormap.

Source: ETOPO1 Ice Surface (NOAA NGDC), fetched from NOAA CoastWatch ERDDAP
dataset `etopo180` at every 3rd arc-minute node (0.05 deg):
  https://coastwatch.pfeg.noaa.gov/erddap/griddap/etopo180.nc?altitude[(-90):3:(90)][(-180):3:(180)]
Unlike etopo1_0p1deg.tif (bedrock), this includes the Antarctic and Greenland
ice sheets, so land looks like land.

Colours follow notebooks/elastic/worldbanktalks.ipynb (topography.png):
cmocean.cm.topo with Normalize(vmin=-5000, vmax=5000), i.e. sea level sits at
the colormap's pivot, as in the cmocean docs' topo example.

`topo_rgb()` returns float RGB (0-1), shape (3600, 7200, 3), row 0 = 90N,
col 0 = 180W, 0.05 deg cells. `load_elevation()` returns the matching metres.
"""
from pathlib import Path

import cmocean
import numpy as np
from matplotlib.colors import Normalize
from scipy.io import netcdf_file

HERE = Path(__file__).parent
ETOPO_ICE_PATH = HERE / "etopo1_ice_0p05deg.nc"
VMIN, VMAX = -5000.0, 5000.0
HILLSHADE_STRENGTH = 0.12  # +/- fraction of brightness; 0 disables


def load_elevation():
    """Elevation (m) on 0.05 deg cells, shape (3600, 7200), row 0 = north."""
    with netcdf_file(ETOPO_ICE_PATH, mmap=False) as nc:
        z = np.array(nc.variables["altitude"][:], dtype="float32")  # (3601, 7201) nodes
        lat = np.array(nc.variables["latitude"][:])
    if lat[0] < lat[-1]:
        z = z[::-1]  # make row 0 = north
    # node-registered -> cell-centred: average the 4 corner nodes of each cell
    return 0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:])


def topo_rgb(hillshade=True):
    elev = load_elevation()
    rgb = cmocean.cm.topo(Normalize(VMIN, VMAX, clip=True)(elev))[..., :3].astype("float32")
    if hillshade and HILLSHADE_STRENGTH:
        lat = np.linspace(90 - 0.025, -90 + 0.025, elev.shape[0])[:, None]
        gx = (np.roll(elev, -1, 1) - np.roll(elev, 1, 1)) / np.maximum(np.cos(np.radians(lat)), 0.05)
        gy = np.gradient(elev, axis=0)
        shade = np.tanh((-gx + gy) / 400.0)  # light from the north-west
        rgb = np.clip(rgb * (1 + HILLSHADE_STRENGTH * shade[..., None]), 0, 1)
    return rgb


if __name__ == "__main__":
    import imageio
    rgb = topo_rgb()
    imageio.imwrite(HERE / "topo_0p05deg.png", (rgb * 255).round().astype("uint8"))
    imageio.imwrite(HERE / "preview_topo.png", (rgb[::4, ::4] * 255).round().astype("uint8"))
    e = load_elevation()
    f = lambda la, lo: e[int((90 - la) / 0.05), int((lo + 180) / 0.05)]
    print(e.shape, e.min(), e.max(), "Greenland", f(72.6, -38.5), "Vostok", f(-78.5, 106.8),
          "Everest", f(27.99, 86.93), "Mariana", f(11.3, 142.2))
