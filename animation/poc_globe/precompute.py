"""Precompute the point cloud for the 'via the globe' animation.

For every texture cell (0.125 deg) we store its colour and three positions in
frame-pixel units (origin at frame centre, x right, y up):
  ee  : flat EqualEarth position, matching reference_pyseas.png exactly
  sph : unit-sphere position, (x right, y up, z towards viewer) when the
        globe faces lon 0 / lat 0
  ex  : flat Elastic-X position, scaled/centred to fit the frame
Also fits the globe rotation Q (and scale) that best matches the Elastic-X
layout, and builds the cut-line point set.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# core.py uses PEP 604 annotations (X | Y) which this Python (3.9) can't evaluate,
# so load it unmodified but with postponed annotation evaluation.
import types
_core = types.ModuleType("core")
_core.__file__ = str(REPO / "notebooks/elastic/core.py")
exec(compile("from __future__ import annotations\n" + open(_core.__file__).read(), _core.__file__, "exec"), _core.__dict__)
ΦΛPoint, load_elastic_projection, project_points = _core.ΦΛPoint, _core.load_elastic_projection, _core.project_points

info = json.load(open(HERE / "reference_axes.json"))
W, H = info["W"], info["H"]


def sphere(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.sin(lo), np.sin(la), np.cos(la) * np.cos(lo)], -1)


def ee_px(lat, lon):
    """EqualEarth position in centred frame pixels, same mapping as the reference png"""
    xy = ccrs.EqualEarth().transform_points(ccrs.PlateCarree(), lon, lat)
    x0, y0, x1, y1 = info["ax_pos"]
    (ex0, ex1), (ey0, ey1) = info["xlim"], info["ylim"]
    px = x0 * W + (xy[:, 0] - ex0) / (ex1 - ex0) * (x1 - x0) * W
    py_up = y0 * H + (xy[:, 1] - ey0) / (ey1 - ey0) * (y1 - y0) * H
    return np.stack([px - W / 2, py_up - H / 2], -1)


def main():
    t0 = time.time()
    tex = (plt.imread(HERE / "texture_platecarree.png")[:, :, :3] * 255).astype(np.uint8)
    th, tw = tex.shape[:2]
    lat = 90 - (np.arange(th) + 0.5) * 180 / th
    lon = -180 + (np.arange(tw) + 0.5) * 360 / tw
    LON, LAT = np.meshgrid(lon, lat)
    lat_f, lon_f = LAT.ravel(), LON.ravel()
    col = tex.reshape(-1, 3)

    sections, boundary, _ = load_elastic_projection(REPO / "projection/elastic-X.h5")
    pts = np.empty(lat_f.size, dtype=ΦΛPoint)
    pts["latitude"], pts["longitude"] = lat_f, lon_f
    xy = project_points(pts, sections)
    exy = np.stack([xy["x"], xy["y"]], -1)
    # which section each point belongs to (for reference)
    sec_id = np.zeros(lat_f.size, np.int8)
    sec_id[sections[1].contains(pts)] = 1
    print("projected elastic", time.time() - t0)

    # Elastic-X km -> frame px: fit bounding box (+/-16384 x +/-11484 km) with margin
    # (the h5 bounding box doesn't enclose all points, e.g. the Bering Strait
    # spike, so fit the actual extent)
    lo, hi = exy.min(0), exy.max(0)
    ex_centre = (lo + hi) / 2
    ex_scale = min(0.96 * W / (hi - lo)[0], 0.96 * H / (hi - lo)[1])
    exy = exy - ex_centre
    ex = exy * ex_scale
    ee = ee_px(lat_f, lon_f)
    sph = sphere(lat_f, lon_f)
    print("ee/sph", time.time() - t0)

    # ---- fit globe orientation: minimise area-weighted squared displacement
    # between orthographic screen position lam*(Q p)_xy and the Elastic-X target
    rng = np.random.default_rng(0)
    w_all = np.cos(np.radians(lat_f))
    idx = rng.choice(lat_f.size, 40000, p=w_all / w_all.sum())
    P, T = sph[idx], ex[idx]

    def cost(rv, return_lam=False):
        q = Rotation.from_rotvec(rv).as_matrix()
        s = (P @ q.T)[:, :2]
        lam = (s * T).sum() / (s * s).sum()
        c = ((lam * s - T) ** 2).sum(1).mean()
        return (c, lam) if return_lam else c

    best = None
    for rv0 in Rotation.random(300, random_state=1).as_rotvec():
        r = minimize(cost, rv0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-3))
        if best is None or r.fun < best.fun:
            best = r
    c, lam = cost(best.x, True)
    Q = Rotation.from_rotvec(best.x)
    if lam < 0:  # negative scale == extra 180 deg rotation about the view axis
        Q = Rotation.from_euler("z", 180, degrees=True) * Q
        lam = -lam
    # describe: which lat/lon faces the viewer
    centre = Q.inv().apply([0, 0, 1])
    clat = np.degrees(np.arcsin(centre[1])); clon = np.degrees(np.arctan2(centre[0], centre[2]))
    up = Q.inv().apply([0, 1, 0])
    print(f"fit: rms={np.sqrt(c):.1f}px lam={lam:.1f}px centre lat={clat:.1f} lon={clon:.1f} "
          f"screen-up dir lat={np.degrees(np.arcsin(up[1])):.1f} lon={np.degrees(np.arctan2(up[0], up[2])):.1f}")
    # baseline cost with identity for comparison
    print("identity rms", np.sqrt(cost(np.zeros(3))))

    # ---- cut lines: boundary vertices where the two sections disagree
    import h5py
    with h5py.File(REPO / "projection/elastic-X.h5") as f:
        b = f["section 0/boundary"][:]
    # densify boundary to ~0.1 deg steps
    bl, bo = [], []
    for i in range(len(b) - 1):
        n = max(1, int(np.ceil(max(abs(b["latitude"][i + 1] - b["latitude"][i]),
                                   abs(b["longitude"][i + 1] - b["longitude"][i])) / 0.08)))
        s = np.arange(n) / n
        bl.append(b["latitude"][i] + s * (b["latitude"][i + 1] - b["latitude"][i]))
        bo.append(b["longitude"][i] + s * (b["longitude"][i + 1] - b["longitude"][i]))
    bl, bo = np.concatenate(bl), np.concatenate(bo)
    bp = np.empty(bl.size, dtype=ΦΛPoint); bp["latitude"], bp["longitude"] = bl, bo
    # For each side of the boundary take the Elastic-X position of the nearest
    # texture point belonging to that section (robust vs. extrapolated grid nodes),
    # then keep only the stretches where the two sides land far apart = real cuts.
    from scipy.spatial import cKDTree
    bs = sphere(bl, bo)
    side = []
    for h in range(2):
        m = np.nonzero(sec_id == h)[0]
        tree = cKDTree(sph[m])
        dist, j = tree.query(bs, k=1)
        side.append(exy[m[j]])
    d = np.hypot(*(side[0] - side[1]).T)
    is_cut = (d > 300) & (np.abs(bl) < 89.0)  # pole rows of the boundary are not cuts
    print("cut vertices", is_cut.sum(), "of", bl.size)
    cl, co = bl[is_cut], bo[is_cut]
    cut_ex = np.concatenate([side[0][is_cut], side[1][is_cut]])
    cut_lat = np.concatenate([cl, cl]); cut_lon = np.concatenate([co, co])
    cut_sph = sphere(cut_lat, cut_lon)
    cut_ee = ee_px(cut_lat, cut_lon)

    # EqualEarth map outline (pyseas draws it as a thin light line)
    n = 721
    L = 179.999
    elat = np.concatenate([np.linspace(-90, 90, n), np.full(n, 90.0), np.linspace(90, -90, n), np.full(n, -90.0)])
    elon = np.concatenate([np.full(n, L), np.linspace(L, -L, n), np.full(n, -L), np.linspace(-L, L, n)])
    edge_ee = ee_px(elat, elon)
    edge_sph = sphere(elat, elon)

    np.savez_compressed(
        HERE / "pointcloud.npz",
        edge_ee=edge_ee.astype(np.float32), edge_sph=edge_sph.astype(np.float32),
        col=col, ee=ee.astype(np.float32), sph=sph.astype(np.float32), ex=ex.astype(np.float32),
        lat=lat_f.astype(np.float32), lon=lon_f.astype(np.float32), sec=sec_id,
        cut_ee=cut_ee.astype(np.float32), cut_sph=cut_sph.astype(np.float32),
        cut_ex=(cut_ex * ex_scale).astype(np.float32), cut_lat=cut_lat, cut_lon=cut_lon,
        Q=Q.as_quat(), lam=lam, ex_scale=ex_scale,
    )
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
