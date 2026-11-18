"""Step 1: render (a) the pyseas reference map and (b) a plate-carree texture
in pyseas dark style, used as the source image for the warp animation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pyseas.maps as psm
import imageio
import json

W, H = 960, 540
TEX_W, TEX_H = 3600, 1800  # 0.1 deg
OCEAN = "#0a1738"


def reference():
    with psm.context(psm.styles.dark):
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        psm.add_figure_background(fig, color=OCEAN)
        ax = psm.create_map(projection="global.default")
        psm.add_land(ax)
        psm.add_countries(ax)
        fig.savefig("reference_pyseas.png", dpi=100, facecolor=fig.get_facecolor())
        # record where the map sits in the frame so the animation can match it
        fig.canvas.draw()
        bb = ax.get_window_extent()
        geom = dict(
            fig_w=W, fig_h=H,
            ax_x0=bb.x0, ax_x1=bb.x1, ax_y0=bb.y0, ax_y1=bb.y1,  # px, origin bottom-left
            xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),  # projected metres
            proj4=ax.projection.proj4_init,
        )
        with open("reference_geom.json", "w") as f:
            json.dump(geom, f, indent=1)
        plt.close(fig)


def texture():
    # line width scaled so borders are about as thick (in output pixels) as in
    # the reference once the texture is downsampled (~4 texels per output pixel)
    lw = 0.4 * TEX_W / 900
    with psm.context(psm.styles.dark):
        fig = plt.figure(figsize=(TEX_W / 100, TEX_H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
        fig.patch.set_facecolor(OCEAN)
        try:  # cartopy < 0.21
            ax.background_patch.set_facecolor(OCEAN)
        except AttributeError:
            ax.set_facecolor(OCEAN)
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
        ax.axis("off")
        psm.add_land(ax, linewidth=lw)
        psm.add_countries(ax, linewidth=lw)
        try:
            ax.spines["geo"].set_visible(False)
        except Exception:
            pass
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
    assert img.shape == (TEX_H, TEX_W, 3), img.shape
    imageio.imwrite("texture_platecarree.png", img)


if __name__ == "__main__":
    reference()
    texture()
