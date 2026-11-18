"""HD step 2: point cloud for the HD warp (one point per texture cell).

Same as precompute.py (POC) but: any texture resolution, two colours per point
(flat pyseas ocean and bathymetry), target positions stored in *normalised*
frame units (fraction of frame height, so any output resolution works), and
cached as .npy files in cache_{TEX_W}/ for fast memory-mapped loading.

Usage: python precompute_hd.py [--tex-w 7200] [--pan-lon 72.5]
"""
import argparse
import os
import time
import numpy as np
import imageio.v2 as imageio
import elastic_io as E

MARGIN = 0.04  # fraction of frame height left free around the final map
ASPECT = 16 / 9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex-w", type=int, default=7200)
    ap.add_argument("--pan-lon", type=float, default=72.5,
                    help="pan longitude; 72.5 was chosen in the POC by minimising mean displacement")
    a = ap.parse_args()
    t0 = time.time()
    out = f"cache_{a.tex_w}"
    os.makedirs(out, exist_ok=True)

    flat = imageio.imread(f"tex_flat_{a.tex_w}.png")[..., :3]
    bathy = imageio.imread(f"tex_bathy_{a.tex_w}.png")[..., :3]
    h, w = flat.shape[:2]
    lat = 90 - (np.arange(h) + 0.5) * 180 / h
    lon = -180 + (np.arange(w) + 0.5) * 360 / w
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    LAT, LON = LAT.ravel(), LON.ravel()

    secs, bbox, boundary = E.load("../../projection/elastic-X.h5")
    x1, y1, sid = E.project(LAT, LON, secs)
    bad = ~np.isfinite(x1) | ~np.isfinite(y1) | (sid < 0)
    print(f"projected {LAT.size} points in {time.time() - t0:.1f}s; NaN/unassigned: {bad.sum()}")
    assert not bad.any(), "unexpected NaN points"

    # fit Elastic-X into a 16:9 frame; units: frame height = 1
    xmin, xmax, ymin, ymax = x1.min(), x1.max(), y1.min(), y1.max()
    s = min((1 - 2 * MARGIN) * ASPECT / (xmax - xmin), (1 - 2 * MARGIN) / (ymax - ymin))
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    p1u = ASPECT / 2 + (x1 - cx) * s   # x in units of frame height
    p1v = 0.5 - (y1 - cy) * s          # y down, units of frame height

    # cut (tear) lines, as in precompute.py
    b = secs[0].border_latlon
    dense, artefact = [], []
    for k in range(len(b) - 1):
        n = max(2, int(np.ceil(np.hypot(*(b[k + 1] - b[k])) / 0.05)))
        tt = np.linspace(0, 1, n, endpoint=False)[:, None]
        dense.append(b[k] + tt * (b[k + 1] - b[k]))
        on_frame = (np.all(np.abs(b[k:k + 2, 1]) > 179.99) or np.all(np.abs(b[k:k + 2, 0]) > 89.99)
                    or (np.abs(b[k, 1]) > 179.99 and np.abs(b[k + 1, 0]) > 89.99)
                    or (np.abs(b[k, 0]) > 89.99 and np.abs(b[k + 1, 1]) > 179.99))
        artefact.append(np.full(n, on_frame))
    dense = np.concatenate(dense)
    artefact = np.concatenate(artefact)
    lat_c, lon_c = dense[:, 0], dense[:, 1]
    xy = [sec.project(lat_c, lon_c) for sec in secs]
    gap = np.hypot(xy[0][0] - xy[1][0], xy[0][1] - xy[1][1])
    tear = np.isfinite(gap) & (gap > 50) & ~artefact

    save = dict(
        lat=LAT.astype(np.float32), lon=LON.astype(np.float32),
        col_flat=flat.reshape(-1, 3), col_bathy=bathy.reshape(-1, 3),
        sid=sid.astype(np.int8),
        p1u=p1u.astype(np.float32), p1v=p1v.astype(np.float32),
        cut_lat=lat_c, cut_lon=lon_c, cut_tear=tear,
        cut_p1u=np.stack([ASPECT / 2 + (xy[i][0] - cx) * s for i in range(2)]),
        cut_p1v=np.stack([0.5 - (xy[i][1] - cy) * s for i in range(2)]),
        meta=np.array([a.pan_lon, h, w], np.float64),
    )
    for k, v in save.items():
        np.save(f"{out}/{k}.npy", v)
    print(f"done in {time.time() - t0:.1f}s -> {out}/")


if __name__ == "__main__":
    main()
