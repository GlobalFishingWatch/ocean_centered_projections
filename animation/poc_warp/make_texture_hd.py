"""HD step 1: textures for the 1080p (or any resolution) warp animation.

Writes (suffix = f"_{W}x{H}_{TEX_W}"):
  reference_pyseas_{W}x{H}.png     pyseas default dark global map at W x H
  reference_geom_{W}x{H}.json      where the map sits in that frame
  tex_flat_{TEX_W}.png             plate-carree texture, pyseas dark style (flat ocean)
  tex_bathy_{TEX_W}.png            same, but ocean = shared bathymetry style
  tex_landalpha_{TEX_W}.png        land coverage (incl. the coastline stroke), 0..255

Usage: python make_texture_hd.py [--width 1920] [--tex-w 7200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cv2
import imageio.v2 as imageio
import pyseas.maps as psm

COMMON = Path(__file__).resolve().parent.parent / "common"
sys.path.insert(0, str(COMMON))
import bathy_style  # noqa: E402  (shared style, read-only)

OCEAN = "#0a1738"
OCEAN_RGB = np.array([10, 23, 56], np.float32)


def reference(W, H):
    dpi = 100 * W / 960  # same figure size in inches as the POC => same line widths in "points"
    with psm.context(psm.styles.dark):
        fig = plt.figure(figsize=(9.6, 5.4), dpi=dpi)
        psm.add_figure_background(fig, color=OCEAN)
        ax = psm.create_map(projection="global.default")
        psm.add_land(ax)
        psm.add_countries(ax)
        fig.savefig(f"reference_pyseas_{W}x{H}.png", dpi=dpi, facecolor=fig.get_facecolor())
        fig.canvas.draw()
        bb = ax.get_window_extent()
        geom = dict(fig_w=W, fig_h=H, ax_x0=bb.x0, ax_x1=bb.x1, ax_y0=bb.y0, ax_y1=bb.y1,
                    xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                    proj4=ax.projection.proj4_init)
        plt.close(fig)
    with open(f"reference_geom_{W}x{H}.json", "w") as f:
        json.dump(geom, f, indent=1)


def _pc_axes(TEX_W, TEX_H, face):
    fig = plt.figure(figsize=(TEX_W / 100, TEX_H / 100), dpi=100)
    fig.patch.set_facecolor(face)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    try:  # cartopy < 0.21
        ax.background_patch.set_facecolor(face)
        ax.outline_patch.set_visible(False)
    except AttributeError:
        ax.set_facecolor(face)
        ax.spines["geo"].set_visible(False)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.axis("off")
    return fig, ax


def _grab(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def textures(W, TEX_W):
    TEX_H = TEX_W // 2
    # ~4 texels per output pixel on the Equal Earth map; line width scaled so
    # borders come out as wide (in output px) as in the pyseas reference
    lw = 0.4 * TEX_W / 900  # texels/output px is ~constant when TEX_W scales with W
    with psm.context(psm.styles.dark):
        fig, ax = _pc_axes(TEX_W, TEX_H, OCEAN)
        psm.add_land(ax, linewidth=lw)
        psm.add_countries(ax, linewidth=lw)
        flat = _grab(fig)
        # land coverage: same land polygons + same coastline stroke, white on black
        fig, ax = _pc_axes(TEX_W, TEX_H, "black")
        psm.add_land(ax, linewidth=lw, facecolor="white", edgecolor="white")
        alpha = _grab(fig)[..., 0]
    assert flat.shape == (TEX_H, TEX_W, 3), flat.shape

    bathy = bathy_style.ocean_rgb() * 255.0  # (1800, 3600, 3), row 0 = 90N, col 0 = 180W
    if bathy.shape[:2] != (TEX_H, TEX_W):
        # bilinear, pixel-centre aligned (cv2 convention); wrap in longitude
        pad = np.concatenate([bathy[:, -1:], bathy, bathy[:, :1]], 1)
        sx = TEX_W / bathy.shape[1]
        big = cv2.resize(pad, (int(round(pad.shape[1] * sx)), TEX_H), interpolation=cv2.INTER_LINEAR)
        o = int(round(sx))
        bathy = big[:, o:o + TEX_W]
    a = alpha.astype(np.float32)[..., None] / 255.0
    # flat = a*land_stuff + (1-a)*OCEAN  =>  replace the ocean part by bathymetry
    bathy_tex = flat.astype(np.float32) + (1 - a) * (bathy - OCEAN_RGB)
    bathy_tex = np.clip(bathy_tex + 0.5, 0, 255).astype(np.uint8)

    imageio.imwrite(f"tex_flat_{TEX_W}.png", flat)
    imageio.imwrite(f"tex_bathy_{TEX_W}.png", bathy_tex)
    imageio.imwrite(f"tex_landalpha_{TEX_W}.png", alpha)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--tex-w", type=int, default=7200)
    a = ap.parse_args()
    H = a.width * 9 // 16
    reference(a.width, H)
    textures(a.width, a.tex_w)
