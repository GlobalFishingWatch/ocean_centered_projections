import numpy as np
from numpy.typing import NDArray
from typing import Any
from scipy import interpolate
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import h5py
import shapefile
from math import nan
from pathlib import Path as FilePath
import shutil
from typing import Optional
# TODO: import this stuff in create_example_maps

SNIPPING_LENGTH = 1000

Style = dict[str, Any]
XYPoint = np.dtype([("x", float), ("y", float)])
XYLine = NDArray[XYPoint]
XYFeature = tuple[int, float, list[XYLine]]
ΦΛPoint = np.dtype([("latitude", float), ("longitude", float)])
ΦΛLine = NDArray[ΦΛPoint]
ΦΛFeature = tuple[int, float, list[ΦΛLine]]


class Section:
    def __init__(
        self,
        ф_nodes: NDArray[float],
        λ_nodes: NDArray[float],
        xy_nodes: NDArray[XYPoint],
        border: ΦΛLine,
    ):
        """one lobe of an Elastic projection, containing a grid of latitudes and
        longitudes as well as the corresponding x and y coordinates
        :param ф_nodes: the node latitudes (deg)
        :param λ_nodes: the node longitudes (deg)
        :param xy_nodes: the grid of x- and y-values at each ф and λ (km)
        :param border: the path that encloses the region this section defines (d"eg)
        """
        self.x_projector = interpolate.RegularGridInterpolator(
            (ф_nodes, λ_nodes), xy_nodes["x"]
        )
        self.y_projector = interpolate.RegularGridInterpolator(
            (ф_nodes, λ_nodes), xy_nodes["y"]
        )
        self.border = Path(np.stack([border["latitude"], border["longitude"]], axis=-1))  # type: ignore
        self.border_is_counterclockwise = is_counterclockwise(self.border)

    def get_planar_coordinates(self, points: NDArray[ΦΛPoint]) -> NDArray[XYPoint]:
        """take a point on the sphere and smoothly interpolate it to x and y"""
        result = np.empty(points.size, dtype=XYPoint)
        result["x"] = self.x_projector((points["latitude"], points["longitude"]))
        result["y"] = self.y_projector((points["latitude"], points["longitude"]))
        return result

    def contains(self, points: NDArray[ΦΛPoint]) -> NDArray[bool]:
        """whether the given point is within this Section’s boundary"""
        points = np.stack([points["latitude"], points["longitude"]], axis=-1)
        # make sure you check the border orientation, because Matplotlib won't
        if (
            self.border_is_counterclockwise
        ):  # use the radius parameter to ensure points on the boundary are counted
            return self.border.contains_points(points, radius=-1e-9)  # type: ignore
        else:
            return ~self.border.contains_points(points, radius=1e-9)  # type: ignore


def is_counterclockwise(path: Path) -> bool:
    """determines whether the polygon is oriented in the normal direction"""
    area = 0
    for i in range(len(path)):
        area += (
            path.vertices[i - 1, 1] * path.vertices[i, 0]
            - path.vertices[i - 1, 0] * path.vertices[i, 1]
        )
    return area > 0


def load_elastic_projection(
    path: FilePath | str,
) -> tuple[list[Section], XYLine, float]:
    """load the hdf5 file that defines an elastic projection
    :param name: one of "elastic-I", "elastic-II", or "elastic-III"
    :return: the list of sections that comprise this projection, and the map’s full projected outer shape
    """
    path = FilePath(path)
    with h5py.File(path, "r") as file:
        sections = []
        for h in range(file.attrs["number of sections"]):
            sections.append(
                Section(
                    file[f"section {h}/latitude"][:],
                    file[f"section {h}/longitude"][:],
                    file[f"section {h}/projected points"][:, :],
                    file[f"section {h}/boundary"][:],
                )
            )
        boundary = file["projected boundary"][:]
        aspect_ratio = (file["bounding box"]["x"][1] - file["bounding box"]["x"][0]) / (
            file["bounding box"]["y"][1] - file["bounding box"]["y"][0]
        )
    return sections, boundary, aspect_ratio


def load_geographic_data(filename: str) -> tuple[list[ΦΛFeature], bool]:
    """load a bunch of polylines from a shapefile
    :param filename: the name of the shapefile zip file
    :return: a list of features, each comprising a "category" (the biome if available, the index
             otherwise), a "width" (only available from Natural Earth’s "rivers with scale
             ranks" dataset), and a list of series of geographic coordinates (degrees);
             and a bool indicating whether this is a closed polygon rather than an open polyline
    """
    encoding = "latin-1" if "wwf_" in filename else "utf-8"
    features: list[ΦΛFeature] = []
    closed = True
    # TODO: make take paths
    with shapefile.Reader(
        f"../../resources/shapefiles/{filename}.zip", encoding=encoding
    ) as f:
        for index, (record, shape) in enumerate(zip(f.records(), f.shapes())):
            closed = shape.shapeTypeName == "POLYGON"
            try:
                category = min(15, int(record["BIOME"])) - 1
            except IndexError:
                category = index
            try:
                width = 0.5 * record["strokeweig"]  # SIC
                print("using strokeweight of", width)
            except IndexError:
                width = 1
            lines: list[XYLine] = []
            for i in range(len(shape.parts)):
                start = shape.parts[i]
                end = (
                    shape.parts[i + 1]
                    if i + 1 < len(shape.parts)
                    else len(shape.points)
                )
                line = np.empty(end - start, dtype=ΦΛPoint)
                for j, (λ, ф) in enumerate(shape.points[start:end]):
                    line[j] = (max(-90, min(90, ф)), max(-180, min(180, λ)))
                lines.append(line)
            features.append((category, width, lines))
    return features, closed


def rotate_points(v, degrees):
    radians = np.radians(degrees)
    cos = np.cos(radians)
    sin = np.sin(radians)
    x = cos * v["x"] + sin * v["y"]
    y = cos * v["y"] - sin * v["x"]
    rotated = np.empty_like(v)
    rotated["x"] = x
    rotated["y"] = y
    return rotated


def project_points(points: list[ΦΛPoint], projection: list[Section]) -> list[XYPoint]:
    """apply the given Elastic projection to a list of lat/lon points"""
    projected_points: list[XYPoint] = np.empty(points.size, dtype=XYPoint)
    for section in projection:
        in_this_section = section.contains(points)
        projected_points[in_this_section] = section.get_planar_coordinates(
            points[in_this_section]
        )
    assert not np.any(np.isnan(projected_points["x"]))
    return projected_points


def project(features: list[ΦΛFeature], projection: list[Section]) -> list[XYFeature]:
    """apply the given Elastic projection, defined by a list of sections, to the given series of
    latitudes and longitudes.
    """
    projected_features: list[XYFeature] = []
    for j, (category, width, lines) in enumerate(features):
        # print(f"projecting feature {j: 3d}/{len(features): 3d} ({sum(len(line) for line in lines)} points)")
        projected_lines: list[XYLine] = []
        for line in lines:
            projected_line = np.empty(line.size, dtype=XYPoint)
            projected_line[:] = (nan, nan)
            # for each line, project it into whichever section that can accommodate it
            for section in projection:
                in_this_section = section.contains(line)
                projected_line[in_this_section] = section.get_planar_coordinates(
                    line[in_this_section]
                )
            # check that each point was projected by at least one section
            assert not np.any(np.isnan(projected_line["x"]))
            projected_lines.append(projected_line)
        projected_features.append((category, width, projected_lines))
    # print(f"projected {len(projected_features)} features!")
    return projected_features


def cut_lines_that_cross_interruptions(
    features: list[XYFeature], closed: bool
) -> list[XYFeature]:
    """if you naively project lines on a map projection that are not pre-cut at the interruptions,
    you’ll get a lot of extraneous lines crisscrossing the map.  this function deals with that
    problem by cutting any lines that seem suspiciusly long.
    :param features: the data to investigate and adjust
    :param closed: whether to worry about forming closed paths from the continuus regions of each line
    """
    new_features: list[XYFeature] = []
    for category, width, lines in features:
        new_lines: list[XYLine] = []
        new_lines_to_be_merged: list[XYLine] = []
        for line in lines:
            # start by finding line segments longer than 100 km
            j = np.arange(0 if closed else 1, line.size)
            lengths = np.hypot(
                line["x"][j] - line["x"][j - 1], line["y"][j] - line["y"][j - 1]
            )
            cuts = j[np.nonzero(lengths > SNIPPING_LENGTH)[0]]
            # don’t try to do anything if there are not cuts
            if cuts.size == 0:
                new_lines.append(line)
            else:
                if closed:
                    # cycle the points based on the discontinuities you found so it starts and ends with one
                    line = np.roll(line, -cuts[0])
                    cuts = np.concatenate([cuts - cuts[0], [line.size]])
                else:
                    cuts = np.concatenate([[0], cuts, [line.size]])
                sections: list[XYLine] = []
                # break the input up into separate continuus segments
                for k in range(1, cuts.size):
                    sections.append(line[cuts[k - 1] : cuts[k]])
                # remove any length 1 sections
                for k in range(len(sections) - 1, -1, -1):
                    if len(sections[k]) <= 1:
                        sections.pop(k)
                # don’t mark them as finalized yet if we need to close the paths
                if closed:
                    new_lines_to_be_merged += sections
                else:
                    new_lines += sections

        # if it’s closed, you have to reorder the lines so they go together
        new_lines_to_be_merged = sorted(
            new_lines_to_be_merged, key=lambda feature: len(feature[1])
        )
        # go thru all of these lines that are not finalized
        if len(new_lines_to_be_merged) > 0:
            merged_line: Optional[XYLine] = None
            while True:
                if merged_line is not None:
                    # take the endpoint of you current line
                    endpoint = merged_line[-1]
                    potential_next_startpoints = np.array(
                        [line[0] for line in new_lines_to_be_merged] + [merged_line[0]]
                    )
                    # and find the pending startpoint closest to it
                    next_index = np.argmin(
                        np.hypot(
                            potential_next_startpoints["x"] - endpoint["x"],
                            potential_next_startpoints["y"] - endpoint["y"],
                        )
                    )
                    # if that nearest startpoint is its own, close it and finish it
                    if next_index == len(new_lines_to_be_merged):
                        new_lines.append(merged_line)
                        merged_line = None
                    # if the nearest startpoint is a different segment, merge them
                    else:
                        next_line = new_lines_to_be_merged.pop(next_index)
                        merged_line = np.concatenate([merged_line, next_line])
                else:
                    if len(new_lines_to_be_merged) > 0:
                        # arbitrarily take the next pending line whenever we need to restart
                        merged_line = new_lines_to_be_merged.pop()
                    else:
                        # stop when we run out of lines to be merged
                        break
        new_features.append((category, width, new_lines))
    return new_features


def add_data_to_ax(ax, data_name, style, zorder, sections, rotation_deg=0):
    multiple_colors = "facecolor" in style and type(style["facecolor"]) is list
    multiple_widths = "linewidth" in style and style["linewidth"] == 0
    unprojected_data, closed = load_geographic_data(data_name)
    projected_data = project(unprojected_data, sections)
    projected_data = cut_lines_that_cross_interruptions(projected_data, closed)
    if closed:
        for category, width, lines in projected_data:
            feature_specific_style = {**style}
            if multiple_colors:
                color_index = category % len(style["facecolor"])
                feature_specific_style["facecolor"] = style["facecolor"][color_index]
            if style["edgecolor"] == "facecolor":
                feature_specific_style["edgecolor"] = feature_specific_style[
                    "facecolor"
                ]
            points: list[tuple[float, float]] = []
            codes: list[int] = []
            for line in lines:
                if rotation_deg:
                    line = rotate_points(line, rotation_deg)
                for k, point in enumerate(line):
                    points.append((point["x"], point["y"]))
                    codes.append(Path.MOVETO if k == 0 else Path.LINETO)
                points.append((nan, nan))
                codes.append(Path.CLOSEPOLY)
            path = Path(points, codes)  # type: ignore
            patch = PathPatch(path, zorder=zorder, **feature_specific_style)
            ax.add_patch(patch)
    else:
        for category, width, lines in projected_data:
            feature_specific_style = {**style}
            if multiple_widths:
                feature_specific_style["linewidth"] = width
            for line in lines:
                if rotation_deg:
                    line = rotate_points(line, rotation_deg)
                ax.plot(line["x"], line["y"], zorder=zorder, **feature_specific_style)


def copy_rotated_projection(input_path, output_path, rotation_deg):
    shutil.copy(input_path, output_path)
    with h5py.File(output_path, "r+") as data:
        projected_boundary = rotate_points(data["projected boundary"][:], rotation)
        data["projected boundary"][:] = projected_boundary
        data["bounding box"]["x"][0] = np.nanmin(projected_boundary["x"])
        data["bounding box"]["x"][1] = np.nanmax(projected_boundary["x"])
        data["bounding box"]["y"][0] = np.nanmin(projected_boundary["y"])
        data["bounding box"]["y"][1] = np.nanmax(projected_boundary["y"])
        for i in range(data.attrs["number of sections"]):
            data[f"section {i}/projected points"][:] = rotate_points(
                data[f"section {i}/projected points"][:], rotation
            )
