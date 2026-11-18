"""HD textures for the globe animation (0.0625 deg plate carree, 5760x2880):
  texture_hd_flat.png   pyseas dark style (land #374a6d, borders #222d4b, ocean #0a1738)
  texture_hd_landmask.png  antialiased alpha of land + country/coast strokes
  texture_hd_bathy.png  same as flat but with the shared bathymetry ocean
                        (animation/common/bathy_style.py) under the land
Also writes reference_pyseas_1080p.png (pyseas default map at 1920x1080).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cv2
import pyseas.maps as psm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from bathy_style import ocean_rgb  # noqa: E402

DPI = 100
TEX_W, TEX_H = 5760, 2880
BORDER_LW = 2.0  # pt at DPI 100 in the texture (~0.9 px at 1080p EqualEarth)
OCEAN = np.array([10, 23, 56]) / 255


def pc_axes(fig):
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
    ax.axis("off")
    return ax


def grab(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.float32) / 255
    plt.close(fig)
    assert a.shape[:2] == (TEX_H, TEX_W), a.shape
    return a


# reference at 1080p: identical layout to reference_pyseas.png, dpi doubled
with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(9.6, 5.4), dpi=200)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)
    ax = psm.create_map(projection="global.default")
    psm.add_land(ax)
    psm.add_countries(ax)
    psm.add_figure_background(fig, color="#0a1738")
    fig.savefig(HERE / "reference_pyseas_1080p.png", dpi=200, facecolor="#0a1738")
    plt.close(fig)

with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(TEX_W / DPI, TEX_H / DPI), dpi=DPI, facecolor="#0a1738")
    ax = pc_axes(fig); ax.set_facecolor("#0a1738")
    psm.add_land(ax, linewidth=0)
    psm.add_countries(ax, linewidth=BORDER_LW)
    flat = grab(fig)

    fig = plt.figure(figsize=(TEX_W / DPI, TEX_H / DPI), dpi=DPI, facecolor="black")
    ax = pc_axes(fig); ax.set_facecolor("black")
    psm.add_land(ax, facecolor="white", linewidth=0)
    # facecolor="none": with the default, a few country polygons get filled
    # with the land colour, which would punch dark wedges into the mask
    psm.add_countries(ax, edgecolor="white", facecolor="none", linewidth=BORDER_LW)
    mask = grab(fig)[:, :, 0]

# bathymetry: 0.1 deg cell centres -> 0.0625 deg cell centres, bilinear
bathy = ocean_rgb()
bathy = cv2.resize(bathy, (TEX_W, TEX_H), interpolation=cv2.INTER_LINEAR)
# flat = a*landlayer + (1-a)*OCEAN  ->  swap OCEAN for the bathymetry colour
bathy_tex = np.clip(flat + (1 - mask)[..., None] * (bathy - OCEAN), 0, 1)

for name, img in [("flat", flat), ("bathy", bathy_tex)]:
    cv2.imwrite(str(HERE / f"texture_hd_{name}.png"), (img[:, :, ::-1] * 255).round().astype(np.uint8))
cv2.imwrite(str(HERE / "texture_hd_landmask.png"), (mask * 255).round().astype(np.uint8))
print("wrote HD textures")
