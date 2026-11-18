"""Step 2: build the point cloud (one point per texture cell) with
  - colour (from the pyseas-style plate-carree texture)
  - authalic latitude / longitude (for Equal Earth at any central longitude)
  - Elastic-X target position in frame pixels (P1) and section id
choose the pan longitude, extract the cut (tear) lines, and cache to points.npz."""
import time
import numpy as np
import imageio.v2 as imageio
import elastic_io as E
import geo

MARGIN = 0.04  # fraction of frame height left free around the final map
STEP = 1       # texture subsampling (1 = full 0.1 deg)


def main():
    t0 = time.time()
    tex = imageio.imread("texture_platecarree.png")[::STEP, ::STEP, :3]
    h, w = tex.shape[:2]
    lat = 90 - (np.arange(h) + 0.5) * 180 / h
    lon = -180 + (np.arange(w) + 0.5) * 360 / w
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    LAT, LON = LAT.ravel(), LON.ravel()
    col = tex.reshape(-1, 3)

    secs, bbox, boundary = E.load("../../projection/elastic-X.h5")
    x1, y1, sid = E.project(LAT, LON, secs)
    bad = ~np.isfinite(x1) | ~np.isfinite(y1) | (sid < 0)
    print(f"projected {LAT.size} points in {time.time()-t0:.1f}s; NaN/unassigned: {bad.sum()}")
    if bad.any():
        # fall back: try the other section's interpolator (cells right on a boundary)
        for i, s in enumerate(secs):
            m = bad.copy()
            xx, yy = s.project(LAT[m], LON[m])
            ok = np.isfinite(xx) & np.isfinite(yy)
            idx = np.where(m)[0][ok]
            x1[idx], y1[idx], sid[idx] = xx[ok], yy[ok], i
            bad = ~np.isfinite(x1) | ~np.isfinite(y1)
        print("still bad after fallback:", bad.sum())
        keep = ~bad
        LAT, LON, col, x1, y1, sid = LAT[keep], LON[keep], col[keep], x1[keep], y1[keep], sid[keep]

    # --- fit Elastic-X (km, y up) into the frame, centred, with margin ---------
    xmin, xmax, ymin, ymax = x1.min(), x1.max(), y1.min(), y1.max()
    s = min((1 - 2 * MARGIN) * geo.W / (xmax - xmin), (1 - 2 * MARGIN) * geo.H / (ymax - ymin))
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    p1c = geo.W / 2 + (x1 - cx) * s
    p1r = geo.H / 2 - (y1 - cy) * s
    fit = dict(s=s, cx=cx, cy=cy)

    beta = geo.authalic_lat(LAT)
    g = geo.load_ref_geom()

    # --- choose pan longitude: area-weighted mean displacement EE(L) -> P1 ----
    rng = np.random.default_rng(0)
    sub = rng.choice(LAT.size, 200000, replace=False)
    wgt = np.cos(np.radians(LAT[sub]))
    scores = []
    for L in np.arange(-180, 180, 2.5):
        x, y = geo.equal_earth(beta[sub], LON[sub], L)
        c, r = geo.ee_to_px(x, y, g)
        d = np.hypot(c - p1c[sub], r - p1r[sub])
        scores.append((np.sum(d * wgt) / wgt.sum(), L))
    scores.sort()
    pan_lon = float(scores[0][1])
    print("best pan longitudes (mean px displacement, lon):", [(round(a, 1), b) for a, b in scores[:5]])
    print("displacement at lon0=0:", [round(a, 1) for a, b in scores if b == 0])

    # --- cut (tear) lines: parts of the section boundary where the two sections
    #     disagree about where the point goes (where they agree, it's a seam) ---
    b = secs[0].border_latlon
    # densify
    dense, artefact = [], []
    for k in range(len(b) - 1):
        n = max(2, int(np.ceil(np.hypot(*(b[k + 1] - b[k])) / 0.1)))
        tt = np.linspace(0, 1, n, endpoint=False)[:, None]
        dense.append(b[k] + tt * (b[k + 1] - b[k]))
        # segments lying on the lat/lon frame (antimeridian or pole rows) are
        # artefacts of describing the polygon in lat/lon, not real cuts
        on_frame = (np.all(np.abs(b[k:k + 2, 1]) > 179.99) or np.all(np.abs(b[k:k + 2, 0]) > 89.99)
                    or (np.abs(b[k, 1]) > 179.99 and np.abs(b[k + 1, 0]) > 89.99)
                    or (np.abs(b[k, 0]) > 89.99 and np.abs(b[k + 1, 1]) > 179.99))
        artefact.append(np.full(n, on_frame))
    dense = np.concatenate(dense)
    artefact = np.concatenate(artefact)
    lat_c, lon_c = dense[:, 0], dense[:, 1]
    xy = [sec.project(lat_c, lon_c) for sec in secs]
    gap = np.hypot(xy[0][0] - xy[1][0], xy[0][1] - xy[1][1])
    tear = np.isfinite(gap) & (gap > 50) & ~artefact  # km
    print(f"cut line: {dense.shape[0]} pts, {tear.sum()} on tear")
    cut = dict(
        lat=lat_c, lon=lon_c, tear=tear,
        p1c=np.stack([geo.W / 2 + (xy[i][0] - cx) * s for i in range(2)]),
        p1r=np.stack([geo.H / 2 - (xy[i][1] - cy) * s for i in range(2)]),
    )

    np.savez(
        "points.npz",
        beta=beta.astype(np.float32), lon=LON.astype(np.float32), lat=LAT.astype(np.float32),
        col=col.astype(np.uint8), sid=sid.astype(np.int8),
        p1c=p1c.astype(np.float32), p1r=p1r.astype(np.float32),
        pan_lon=pan_lon, fit_s=s, fit_cx=cx, fit_cy=cy,
        cut_lat=cut["lat"], cut_lon=cut["lon"], cut_tear=cut["tear"],
        cut_p1c=cut["p1c"], cut_p1r=cut["p1r"],
        pan_scores=np.array(scores),
    )
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
