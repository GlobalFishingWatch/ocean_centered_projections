import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from math import sqrt

from core import (
    ΦΛPoint,
    load_elastic_projection,
    add_data_to_ax,
    rotate_points,
    project_points,
)


def hex_to_rgb(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) / 255.0 for i in range(0, lv, lv // 3))


def create_latlon_raster(raster):
    n_lat, n_lon = raster.shape
    source_lon = -180 + (0.5 + np.arange(n_lon) * 360) / n_lon
    source_lat = -90 + (0.5 + np.arange(n_lat) * 180) / n_lat

    latlon_raster = np.empty(raster.shape, dtype=ΦΛPoint)
    latlon_raster[:]["longitude"] = source_lon[newaxis, :]
    latlon_raster[:]["latitude"] = source_lat[:, newaxis]

    return latlon_raster


def turn_off_labels(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def add_labels_to_ax(ax, sections, dpi):
    scale = dpi / 300
    labels = [
        ("Europe", (23, 50.0), ("center", "baseline"), "small", (0, 0)),
        ("West\nAfrica", (7, 11), ("center", "baseline"), "small", (0, 0)),
        ("East\nAfrica", (34, -15), ("center", "baseline"), "small", (0, 0)),
        ("Asia", (103, 23.0), ("center", "baseline"), "small", (0, 0)),
        (
            "Eastern\nNorth America",
            (-95, 40.0),
            ("center", "baseline"),
            "small",
            (0, 0),
        ),
        ("Eastern\nSouth America", (-68, -30), ("center", "baseline"), "small", (0, 0)),
        (
            "Western\nSouth America",
            (-71, -30),
            ("center", "baseline"),
            "small",
            (-1300, 0),
        ),
        ("Western\nNorth America", (-109, 39), ("center", "baseline"), "small", (0, 0)),
        ("Suez Canal", (32.3, 30.7), ("left", "baseline"), "x-small", (100, 100)),
        ("Panama Canal", (-81, 9), ("left", "top"), "x-small", (300, 50)),
        # ("Bering\nStrait", (-168, 66), ('right', 'center'), 'x-small', (-50, 50)),
        ("Bering\nStrait", (-168, 60), ("left", "center"), "x-small", (750, 100)),
    ]

    for label, (lon, lat), (halign, valign), size, (dx, dy) in labels:
        line = np.zeros([1], dtype=ΦΛPoint)
        line["latitude"] = [lat]
        line["longitude"] = [lon]
        xy = project_points(line, sections)[0]
        plt.text(
            xy["x"] + dx,
            xy["y"] + dy,
            label,
            zorder=10,
            color="white",
            horizontalalignment=halign,
            verticalalignment=valign,
            size=size,
            fontstretch=1,
        )

    for (lon_1, lat_1), (dx1, dy1), (lon_2, lat_2), (dx2, dy2), con in [
        (
            (32.3, 30.7),
            (-200, 0),
            (34.3, 29.2),
            (50, 200),
            # "arc3,rad=0.3"
            f"arc,angleA=180,angleB=0,armA={scale * 200},armB={scale * 1000},rad={scale * 100}",
        ),
        (
            (-81, 9),
            (-100, 100),
            (-83, 9),
            (-100, -400),
            # "arc3,rad=-0.3"
            f"arc,angleA=180,angleB=0,armA={scale * 700},armB={scale * 800},rad={scale * 100}",
        ),
        (
            (-168, 66),
            (-50, 100),
            (-168, 60),
            (1450, 0),
            f"arc,angleA=-90,angleB=-90,armA={scale * 1100},armB={scale * 1360},rad={scale * 100}",
        ),
    ]:
        line = np.zeros([2], dtype=ΦΛPoint)
        line["latitude"] = [lat_1, lat_2]
        line["longitude"] = [lon_1, lon_2]
        xy = project_points(line, sections)
        x1, x2 = xy["x"]
        y1, y2 = xy["y"]
        ax.annotate(
            "",
            xy=(x1 + dx1, y1 + dy1),
            xycoords="data",
            xytext=(x2 + dx2, y2 + dy2),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",
                color="0.5",
                shrinkA=5,
                shrinkB=5,
                patchA=None,
                patchB=None,
                connectionstyle=con,
                linestyle=":",
            ),
        )


def plot_elastic(
    foreground_raster,
    mask=None,
    cmap=mpcm.plasma,
    background_color="#000000",
    resolution=2000,
    # vmin=1,
    # vmax=10000,
    norm=mpcolors.LogNorm(vmin=1, vmax=10000, clip=True),
    nominal_size=16,
    dpi=300,
    projection_name="Elastic-II",
    add_coastline=False,
    add_rivers=False,
    add_lakes=False,
    ax=None,
    alpha=1.0,
    zorder=1,
    nominal_pixel_size=72, # This should scale with raster size.
    rotation_deg=0,
    add_labels=True,
):
    projection_path = f"../../projection/{projection_name}.h5"
    sections, boundary, aspect_ratio = load_elastic_projection(projection_path)
    latlon_raster = create_latlon_raster(foreground_raster)
    if mask is not None:
        assert foreground_raster.shape == mask.shape
        latlon_raster = latlon_raster[mask]
        foreground_raster = foreground_raster[mask]
    xy_points = project_points(latlon_raster.flatten(), sections)

    if isinstance(background_color, str):
        background_color = hex_to_rgb(background_color)

    z = foreground_raster.flatten()

    if rotation_deg:
        xy_points = rotate_points(xy_points, rotation_deg)

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                nominal_size * sqrt(aspect_ratio),
                nominal_size / sqrt(aspect_ratio),
            ),
            facecolor="none",
            dpi=dpi,
        )
        turn_off_labels(ax)

    plt.scatter(
        x=xy_points["x"],
        y=xy_points["y"],
        c=z,
        edgecolors="none",
        # marker="s",
        s=nominal_pixel_size / dpi,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        zorder=zorder,
    )

    if background_color is not None:
        ax.patch.set_facecolor(background_color)

    # Alpha is really used for toning down colors, so just push toward black
    water = tuple(alpha * x / 255 for x in (10, 0, 215))
    if add_lakes:
        add_data_to_ax(
            ax,
            "ne_110m_lakes",
            dict(
                facecolor=water,
                edgecolor=water,
                linewidth=0.2,
            ),
            zorder=-1,
            sections=sections,
            rotation_deg=rotation_deg,
        )
    if add_rivers:
        add_data_to_ax(
            ax,
            "ne_50m_rivers_lake_centerlines_scale_rank",
            dict(
                color=water,
                linewidth=0,
            ),
            zorder=-1,
            sections=sections,
            rotation_deg=rotation_deg,
        )
    if add_coastline:
        add_data_to_ax(
            ax,
            "ne_50m_coastline",
            dict(
                color="#000000",
                linewidth=0.2,
            ),
            zorder=-1,
            sections=sections,
            rotation_deg=rotation_deg,
        )

    add_labels_to_ax(ax, sections, dpi)

    return ax
