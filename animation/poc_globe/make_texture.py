"""Render (1) a pyseas reference map (EqualEarth, dark) and (2) a plate-carree
texture in the same style, used as the colour source for the point cloud."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyseas.maps as psm

W, H, DPI = 960, 540, 100
TEX_W, TEX_H = 2880, 1440  # 0.125 deg

# --- reference pyseas map at the animation frame size
with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.97)
    ax = psm.create_map(projection="global.default")
    psm.add_land(ax)
    psm.add_countries(ax)
    psm.add_figure_background(fig, color="#0a1738")
    fig.canvas.draw()
    # record where the map lands in the frame so frame 0 can match it exactly
    pos = ax.get_position()
    info = dict(
        ax_pos=[pos.x0, pos.y0, pos.x1, pos.y1],
        xlim=list(ax.get_xlim()),
        ylim=list(ax.get_ylim()),
        W=W, H=H,
    )
    fig.savefig("reference_pyseas.png", dpi=DPI, facecolor="#0a1738")
    plt.close(fig)
json.dump(info, open("reference_axes.json", "w"), indent=1)
print(info)

# --- plate carree texture, no margins
with psm.context(psm.styles.dark):
    fig = plt.figure(figsize=(TEX_W / DPI, TEX_H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
    ax.set_facecolor("#0a1738")
    ax.axis("off")
    # borders a bit thicker than the 0.4pt default because the texture is
    # ~3x finer than the final frame
    psm.add_land(ax, linewidth=0)
    psm.add_countries(ax, linewidth=1.0)
    fig.patch.set_facecolor("#0a1738")
    fig.savefig("texture_platecarree.png", dpi=DPI, facecolor="#0a1738")
    plt.close(fig)
img = plt.imread("texture_platecarree.png")
print("texture", img.shape)
