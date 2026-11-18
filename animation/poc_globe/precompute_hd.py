"""HD point cloud for the globe animation (run make_texture_hd.py first).

Texture is 0.0625 deg; rows are thinned towards the poles by averaging blocks
of k longitude cells (k = largest divisor of the row length <= 1/cos(lat)) so
the point density is roughly uniform per unit area (~10.6M points).

All positions are stored in 'reference units' = pixels of the 960x540 POC
frame (origin at frame centre, y up); render_hd.py scales them to any size.
The globe orientation is the POC's best fit (globe_fit.json, from precompute.py).
"""
import json, time
from pathlib import Path
import numpy as np
import cv2
import h5py
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from precompute import ee_px, sphere, ΦΛPoint, load_elastic_projection, project_points, REPO, W, H  # noqa

HERE = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    flat = cv2.imread(str(HERE / "texture_hd_flat.png"))[:, :, ::-1]
    bathy = cv2.imread(str(HERE / "texture_hd_bathy.png"))[:, :, ::-1]
    th, tw = flat.shape[:2]
    divisors = np.array([d for d in range(1, tw + 1) if tw % d == 0])
    lats, lons, cf, cb = [], [], [], []
    lon_c = -180 + (np.arange(tw) + 0.5) * 360 / tw
    for i in range(th):
        lat = 90 - (i + 0.5) * 180 / th
        k = divisors[divisors <= 1 / max(np.cos(np.radians(lat)), 1e-9)].max()
        n = tw // k
        lats.append(np.full(n, lat))
        lons.append(lon_c.reshape(n, k).mean(1))
        cf.append(flat[i].reshape(n, k, 3).mean(1))
        cb.append(bathy[i].reshape(n, k, 3).mean(1))
    lat = np.concatenate(lats); lon = np.concatenate(lons)
    col_flat = np.rint(np.concatenate(cf)).astype(np.uint8)
    col_bathy = np.rint(np.concatenate(cb)).astype(np.uint8)
    print(f"{lat.size/1e6:.2f}M points", time.time() - t0)

    sections, _, _ = load_elastic_projection(REPO / "projection/elastic-X.h5")
    pts = np.empty(lat.size, dtype=ΦΛPoint)
    pts["latitude"], pts["longitude"] = lat, lon
    xy = project_points(pts, sections)
    exy = np.stack([xy["x"], xy["y"]], -1)
    sec_id = sections[1].contains(pts).astype(np.int8)
    lo, hi = exy.min(0), exy.max(0)
    exy -= (lo + hi) / 2
    ex_scale = min(0.96 * W / (hi - lo)[0], 0.96 * H / (hi - lo)[1])
    sph = sphere(lat, lon)
    ee = ee_px(lat, lon)
    print("projected", time.time() - t0)

    # cut lines (same rule as the POC), kept as ordered polylines per side
    with h5py.File(REPO / "projection/elastic-X.h5") as f:
        b = f["section 0/boundary"][:]
    bl, bo = [], []
    for i in range(len(b) - 1):
        n = max(1, int(np.ceil(max(abs(b["latitude"][i + 1] - b["latitude"][i]),
                                   abs(b["longitude"][i + 1] - b["longitude"][i])) / 0.05)))
        s = np.arange(n) / n
        bl.append(b["latitude"][i] + s * (b["latitude"][i + 1] - b["latitude"][i]))
        bo.append(b["longitude"][i] + s * (b["longitude"][i + 1] - b["longitude"][i]))
    bl, bo = np.concatenate(bl), np.concatenate(bo)
    bs = sphere(bl, bo)
    side = []
    for h in range(2):
        m = np.nonzero(sec_id == h)[0]
        _, j = cKDTree(sph[m]).query(bs, k=1)
        side.append(exy[m[j]])
    d = np.hypot(*(side[0] - side[1]).T)
    is_cut = (d > 300) & (np.abs(bl) < 89.0)
    seq = np.nonzero(is_cut)[0]          # consecutive seq numbers = connected
    print("cut vertices", is_cut.sum())

    # EqualEarth outline
    n = 1441; L = 179.999
    elat = np.concatenate([np.linspace(-90, 90, n), np.full(n, 90.0), np.linspace(90, -90, n), np.full(n, -90.0)])
    elon = np.concatenate([np.full(n, L), np.linspace(L, -L, n), np.full(n, -L), np.linspace(-L, L, n)])

    fit = json.load(open(HERE / "globe_fit.json"))
    f32 = lambda a: np.asarray(a, np.float32)
    np.savez(
        HERE / "pointcloud_hd.npz",
        col_flat=col_flat, col_bathy=col_bathy, lat=f32(lat), lon=f32(lon),
        ee=f32(ee), sph=f32(sph), ex=f32(exy * ex_scale),
        cut_lat=f32(bl[seq]), cut_lon=f32(bo[seq]), cut_seq=seq,
        cut_sph=f32(sphere(bl[seq], bo[seq])), cut_ee=f32(ee_px(bl[seq], bo[seq])),
        cut_ex0=f32(side[0][seq] * ex_scale), cut_ex1=f32(side[1][seq] * ex_scale),
        edge_ee=f32(ee_px(elat, elon)), edge_sph=f32(sphere(elat, elon)),
        Q=np.array(fit["Q"]), R=fit["lam"], ex_scale=ex_scale,
    )
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
