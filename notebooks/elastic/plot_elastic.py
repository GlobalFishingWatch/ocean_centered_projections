import matplotlib.colors as mpcolors
import matplotlib.cm as mpcm
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from math import sqrt

from core import ΦΛPoint, load_elastic_projection, add_data_to_ax, rotate_points, project_points


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


def plot_elastic(
    foreground_raster,
    ocean_raster,
    cmap=mpcm.plasma,
    background_color="#000000",
    resolution=2000,
    # vmin=1,
    # vmax=10000,
    norm=mpcolors.LogNorm(vmin=1, vmax=10000, clip=True),
    nominal_size=6,
    dpi=300,
    projection_name="Elastic-II",
    add_context=True,
    ax = None,
    alpha=1.0,
    zorder=1,
    rotation_deg=-15,
):
    assert (
        foreground_raster.shape == ocean_raster.shape
    )  # not _really_ necessary, but simpler

    projection_path = f"../../projection/{projection_name}.h5"
    sections, boundary, aspect_ratio = load_elastic_projection(projection_path)
    latlon_raster = create_latlon_raster(foreground_raster)
    if True:
        mask = ocean_raster != 0
        latlon_raster = latlon_raster[mask]
        foreground_raster = foreground_raster[mask]
    xy_points = project_points(latlon_raster.flatten(), sections)

    if isinstance(background_color, str):
        background_color = hex_to_rgb(background_color)

    z = foreground_raster.flatten() #np.minimum(foreground_raster.flatten(), 0.99 * vmax)
    if rotation_deg:
        xy_points = rotate_points(xy_points, rotation_deg)

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(nominal_size * sqrt(aspect_ratio), nominal_size / sqrt(aspect_ratio)),
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
        s=36.0 / dpi,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        zorder=zorder,
    )


    if background_color is not None:
        ax.patch.set_facecolor(background_color)

    if add_context:
        # Alpha is really used for toning down colors, so just push toward black
        water = tuple(alpha * x / 255 for x in (10, 0, 215))
        add_data_to_ax(
            ax,
            "ne_110m_lakes",
            dict(
                facecolor=water,
                edgecolor=water,
                linewidth=0.2,
            ),
            zorder=5,
            sections=sections,
            rotation_deg=rotation_deg
        )
        add_data_to_ax(
            ax,
            "ne_50m_rivers_lake_centerlines_scale_rank",
            dict(
                color=water,
                linewidth=0,
            ),
            zorder=5,
            sections=sections,
            rotation_deg=rotation_deg
        )
        add_data_to_ax(
            ax,"ne_50m_coastline", dict(
                       color="#000000",
                       linewidth=0.2,
                   ),
            zorder=5,
            sections=sections,
            rotation_deg=rotation_deg
        )

    return ax


def plot_elastic_interp(
    foreground_raster,
    ocean_raster,
    cmap=mpcm.plasma,
    background_color="#000000",
    resolution=2000,
    # vmin=1,
    # vmax=10000,
    norm=mpcolors.LogNorm(vmin=1, vmax=10000, clip=True),
    nominal_size=6,
    dpi=300,
    projection_name="Elastic-II",
    add_context=True,
    ax = None,
    alpha=1.0,
    zorder=1,
    rotation_deg=0,
):
    assert (
        foreground_raster.shape == ocean_raster.shape
    )  # not _really_ necessary, but simpler

    projection_path = f"../../projection/{projection_name}.h5"
    sections, boundary, aspect_ratio = load_elastic_projection(projection_path)
    latlon_raster = create_latlon_raster(foreground_raster)
    if True:
        mask = ocean_raster != 0
        latlon_raster = latlon_raster[mask]
        foreground_raster = foreground_raster[mask]
    xy_points = project_points(latlon_raster.flatten(), sections)

    if isinstance(background_color, str):
        background_color = hex_to_rgb(background_color)

    z = foreground_raster.flatten() #np.minimum(foreground_raster.flatten(), 0.99 * vmax)
    if rotation_deg:
        xy_points = rotate_points(xy_points, rotation_deg)

    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(nominal_size * sqrt(aspect_ratio), nominal_size / sqrt(aspect_ratio)),
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
        s=36.0 / dpi,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        zorder=zorder,
    )


    if background_color is not None:
        ax.patch.set_facecolor(background_color)

    if add_context:
        # Alpha is really used for toning down colors, so just push toward black
        water = tuple(alpha * x / 255 for x in (10, 0, 215))
        add_data_to_ax(
            ax,
            "ne_110m_lakes",
            dict(
                facecolor=water,
                edgecolor=water,
                linewidth=0.2,
            ),
            zorder=5,
            sections=sections,
            rotation_deg=rotation_deg
        )
        add_data_to_ax(
            ax,
            "ne_50m_rivers_lake_centerlines_scale_rank",
            dict(
                color=water,
                linewidth=0,
            ),
            zorder=5,
            sections=sections,
            rotation_deg=rotation_deg
        )
        add_data_to_ax(
            ax,"ne_50m_coastline", dict(
                       color="#000000",
                       linewidth=0.2,
                   ),
            zorder=5,
            sections=sections,
            rotation_deg=rotation_deg
        )

    return ax