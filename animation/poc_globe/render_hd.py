"""HD renderer for the 'via the globe' animation, with bathymetry.

pyseas EqualEarth (flat colours) -> bathymetry fades in -> wrap onto globe ->
rotate -> red cut lines -> unpeel into Elastic-X -> hold.

Forward point splatting of the ~12.7M-point cloud (precompute_hd.py), processed
in chunks: pass 1 builds a z-buffer, pass 2 averages the colours of all points
within a small depth tolerance of the front surface (gives antialiasing).
Then hole fill, 1-px silhouette feathering, globe rim glow, and antialiased
polylines for the cut lines and the EqualEarth outline.

usage: python render_hd.py [--width 1920 --height 1080 --fps 30]
                           [--frames 0,100] [--times 0,7.5] [--out DIR] [--procs 10]
"""
import argparse, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
REF_H = 540.0                                   # positions are stored in 960x540 units

BG = np.array([10, 23, 56], np.float32)         # pyseas dark background/ocean #0a1738
BG_DARK = np.array([2, 5, 14], np.float32)      # backdrop once the bathymetry is in
GLOW = np.array([70, 140, 210], np.float32)
RED = np.array([235, 40, 40], np.float32)
EDGE = np.array([160, 167, 182], np.float32)    # pyseas map outline (light grey)

# ---- storyboard (seconds)
T_BATHY = (1.0, 2.0)    # flat ocean -> bathymetry (static EqualEarth)
T_WRAP = (2.0, 5.0)     # flat -> globe
T_ROT = (4.4, 6.6)      # rotate to best-fit orientation (overlaps the end of the wrap)
T_CUT = (6.6, 7.8)      # red cut lines fade in
T_PEEL = (7.8, 12.8)    # cut + unpeel globe -> Elastic-X
T_CUTOUT = (13.3, 14.3) # red lines fade out
T_END = 15.5


def apply_style_timing(style):
    """topo: no flat-pyseas opening / cross-fade (texture weight 1 from frame 0),
    so everything from the wrap onward starts 0.5 s earlier (total 15.0 s).
    bathy: timeline above, unchanged."""
    global T_BATHY, T_WRAP, T_ROT, T_CUT, T_PEEL, T_CUTOUT, T_END
    if style != "topo":
        return
    sh = lambda r: (r[0] - 0.5, r[1] - 0.5)
    T_BATHY = (-2.0, -1.0)  # fade finished before t=0: full texture + dark backdrop at frame 0
    T_WRAP, T_ROT, T_CUT, T_PEEL, T_CUTOUT = map(sh, (T_WRAP, T_ROT, T_CUT, T_PEEL, T_CUTOUT))
    T_END = 15.0
PEEL_DELAY = 0.3
# EqualEarth px per radian of longitude at the map centre, reference units
# (map half-width 0.47*960 px spans pi radians of longitude at the equator)
R0_EE = 0.47 * 960 / np.pi
CHUNK = 2_000_000

D = {}
CFG = dict(W=1920, H=1080, FPS=30, STYLE="bathy")
# per-style look: rim-glow colour/strength, dark casing under the red cut lines
STYLE = {
    "bathy": dict(glow=GLOW, glow_a=0.5, casing=0.0),
    # topo: brighter land/shelves -> more neutral, subtler rim; dark casing so
    # the red stays legible on green/tan land and cream shelves
    "topo": dict(glow=np.array([150, 165, 185], np.float32), glow_a=0.35, casing=0.6),
}


def load():
    if D:
        return D
    z = np.load(HERE / "pointcloud_hd.npz")
    D.update({k: z[k] for k in z.files})
    if CFG["STYLE"] == "topo":
        # the 'after fade' colours become cmocean topo (land + sea floor)
        D["col_bathy"] = np.load(HERE / "pointcloud_hd_topo.npz")["col_topo"]
    D["rot"] = Rotation.from_quat(D["Q"])
    D["R"] = float(D["R"])
    D["slerp"] = Slerp([0, 1], Rotation.concatenate([Rotation.identity(), D["rot"]]))
    return D


def smoother(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)


def smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def phase(t, rng):
    return float(np.clip((t - rng[0]) / (rng[1] - rng[0]), 0, 1))


def positions(t, lat, lon, ee, sph, ex):
    """positions in reference units (x, y, z towards viewer), normal-z, sphericity"""
    d = D
    R = d["R"]
    tw, tr, tp = phase(t, T_WRAP), phase(t, T_ROT), phase(t, T_PEEL)
    if tp > 0:
        rs = d["rot"].apply(sph)
        # far-side points start later and move with a gentle smoothstep
        delay = PEEL_DELAY * (1 - rs[:, 2]) / 2
        e = smoothstep((tp - delay) / (1 - PEEL_DELAY))[:, None]
        flat = np.concatenate([ex, np.zeros((len(ex), 1), np.float32)], 1)
        return ((1 - e) * R * rs + e * flat).astype(np.float32), rs[:, 2], 1 - smoother(tp / 0.3)
    e = smoother(tw)
    if e == 0:
        pos = np.concatenate([ee, np.full((len(ee), 1), R, np.float32)], 1)
        return pos, sph[:, 2], 0.0
    # wrap: bend the map onto spheres of shrinking radius rho = R/k (arc
    # lengths preserved, so both longitude and latitude curl together), plus a
    # correction term that makes t=0 exactly the EqualEarth layout.
    # The sheet's scale also grows from EqualEarth's central scale (R0) to the
    # globe radius, so the map doesn't balloon sideways mid-wrap.
    k = e
    Rk = R0_EE + (R - R0_EE) * e
    phi, lam = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
    rho = Rk / k
    cp = np.cos(k * phi)
    fam = np.stack([rho * cp * np.sin(k * lam), rho * np.sin(k * phi), rho * cp * np.cos(k * lam) - rho + Rk], 1)
    rot = d["slerp"]([smoother(tr)])[0] if tr > 0 else None
    if rot is not None:
        fam = rot.apply(fam)
    fam[:, 0] += (1 - e) * (ee[:, 0] - Rk * lam)
    fam[:, 1] += (1 - e) * (ee[:, 1] - Rk * phi)
    nz = rot.apply(sph)[:, 2] if rot is not None else sph[:, 2]
    return fam.astype(np.float32), nz, smoother((tw - 0.5) / 0.5)


def to_px(pos):
    s = CFG["H"] / REF_H
    W, H = CFG["W"], CFG["H"]
    x = pos[:, 0] * s + W / 2 - 0.5
    y = H / 2 - pos[:, 1] * s - 0.5
    return x, y, pos[:, 2] * s


def render_frame(t):
    d = load()
    W, H = CFG["W"], CFG["H"]
    s = H / REF_H
    n = len(d["lat"])
    b = smoother(phase(t, T_BATHY))
    tw, tp = phase(t, T_WRAP), phase(t, T_PEEL)
    need_depth = (tw > 0) and (tp < 1)
    tol = 1.5 * s
    # ---- pass 1: positions, pixel ids, z-buffer
    chunks = []
    zbuf = np.full(W * H, -np.inf, np.float32)
    g = 0.0
    for a0 in range(0, n, CHUNK):
        sl = slice(a0, min(n, a0 + CHUNK))
        pos, nz, g = positions(t, d["lat"][sl], d["lon"][sl], d["ee"][sl], d["sph"][sl], d["ex"][sl])
        x, y, z = to_px(pos)
        ix, iy = np.rint(x).astype(np.int32), np.rint(y).astype(np.int32)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        pix = (iy * W + ix)[ok]; z = z[ok]
        if need_depth:
            o = np.argsort(z)
            zl = np.full(W * H, -np.inf, np.float32)
            zl[pix[o]] = z[o]
            np.maximum(zbuf, zl, out=zbuf)
        chunks.append((sl, ok, pix, z, nz[ok]))
    # ---- pass 2: depth-tolerant colour average
    acc = np.zeros((3, W * H)); cnt = np.zeros(W * H)
    for sl, ok, pix, z, nz in chunks:
        keep = z >= zbuf[pix] - tol if need_depth else slice(None)
        cf = d["col_flat"][sl][ok][keep].astype(np.float32)
        if b > 0:
            cf = cf + b * (d["col_bathy"][sl][ok][keep].astype(np.float32) - cf)
        if g > 0:
            cf *= (1 - 0.3 * g * np.clip(1 - nz[keep], 0, 1) ** 1.5)[:, None]
        p = pix[keep]
        for c in range(3):
            acc[c] += np.bincount(p, weights=cf[:, c], minlength=W * H)
        cnt += np.bincount(p, minlength=W * H)
    img = (acc / np.maximum(cnt, 1)).T.reshape(H, W, 3).astype(np.float32)
    cov = (cnt > 0).reshape(H, W).astype(np.float32)
    k3 = np.ones((3, 3), np.float32)
    for _ in range(2):  # hole fill
        sm = cv2.filter2D(img * cov[..., None], -1, k3, borderType=cv2.BORDER_CONSTANT)
        nn = cv2.filter2D(cov, -1, k3, borderType=cv2.BORDER_CONSTANT)
        fill = (cov == 0) & (nn >= 4)
        img[fill] = sm[fill] / nn[fill, None]
        cov[fill] = 1
    # 1-px feathered silhouette: alpha = local coverage, colour from neighbours
    sm = cv2.filter2D(img * cov[..., None], -1, k3, borderType=cv2.BORDER_CONSTANT)
    nn = cv2.filter2D(cov, -1, k3, borderType=cv2.BORDER_CONSTANT)
    col = np.where(cov[..., None] > 0, img, sm / np.maximum(nn, 1)[..., None])
    alpha = np.where(cov > 0, 0.5 + 0.5 * nn / 9, 0.5 * nn / 9)[..., None]
    bgk = smoother(phase(t, T_BATHY))
    bg = (1 - bgk) * BG + bgk * BG_DARK
    out = alpha * col + (1 - alpha) * bg
    if g > 0:  # atmosphere rim outside the globe
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(xx + 0.5 - W / 2, H / 2 - yy - 0.5)
        Rp = d["R"] * s
        st = STYLE[CFG["STYLE"]]
        gl = g * st["glow_a"] * np.exp(-np.clip(r - Rp, 0, None) / (7.0 * s)) * (r > Rp - 1)
        gl = (gl * (1 - alpha[..., 0]))[..., None]
        out = out * (1 - gl) + st["glow"] * gl
    # EqualEarth outline, fades out early in the wrap
    ea = 1 - smoother(phase(t, (T_WRAP[0], T_WRAP[0] + 0.6)))
    if ea > 0:
        lat_e, lon_e = edge_latlon()
        pos, _, _ = positions(t, lat_e, lon_e, d["edge_ee"], d["edge_sph"], d["edge_ee"])
        x, y, _ = to_px(pos)
        out = draw_lines(out, [np.stack([x, y], 1)], EDGE, 0.8 * ea, max(1, int(round(0.6 * s))), closed=True)
    # red cut lines with depth test at each vertex
    ca = smoother(phase(t, T_CUT)) * (1 - smoother(phase(t, T_CUTOUT)))
    if ca > 0:
        lines = []
        for ex in (d["cut_ex0"], d["cut_ex1"]):
            pos, _, _ = positions(t, d["cut_lat"], d["cut_lon"], d["cut_ee"], d["cut_sph"], ex)
            x, y, z = to_px(pos)
            ix, iy = np.clip(np.rint(x).astype(int), 0, W - 1), np.clip(np.rint(y).astype(int), 0, H - 1)
            vis = (z >= zbuf[iy * W + ix] - 3 * s) if need_depth else np.ones(len(x), bool)
            brk = np.nonzero((np.diff(d["cut_seq"]) != 1) | ~vis[1:] | ~vis[:-1]
                             | (np.hypot(np.diff(x), np.diff(y)) > 25 * s))[0] + 1
            for seg_idx in np.split(np.arange(len(x)), brk):
                seg_idx = seg_idx[vis[seg_idx]]
                if len(seg_idx) > 1:
                    lines.append(np.stack([x[seg_idx], y[seg_idx]], 1))
        thick = max(1, int(round(1.1 * s)))
        casing = STYLE[CFG["STYLE"]]["casing"]
        if casing > 0:
            out = draw_lines(out, lines, np.array([20, 8, 8], np.float32), ca * casing, thick + 2)
        out = draw_lines(out, lines, RED, ca, thick)
    return np.clip(out, 0, 255).astype(np.uint8)


_EDGE = {}


def edge_latlon():
    if not _EDGE:
        n = len(D["edge_ee"]) // 4
        L = 179.999
        _EDGE["lat"] = np.concatenate([np.linspace(-90, 90, n), np.full(n, 90.0), np.linspace(90, -90, n), np.full(n, -90.0)]).astype(np.float32)
        _EDGE["lon"] = np.concatenate([np.full(n, L), np.linspace(L, -L, n), np.full(n, -L), np.linspace(-L, L, n)]).astype(np.float32)
    return _EDGE["lat"], _EDGE["lon"]


def draw_lines(out, lines, color, a, thick, closed=False):
    H, W = out.shape[:2]
    layer = np.zeros((H, W), np.uint8)
    pts = [np.rint((l + 0.5) * 16).astype(np.int32) for l in lines]  # +0.5: pixel centres
    if pts:
        cv2.polylines(layer, pts, closed, 255, thick, cv2.LINE_AA, shift=4)
    al = (layer.astype(np.float32) / 255 * a)[..., None]
    return out * (1 - al) + color * al


def _init_worker(cfg):
    CFG.update(cfg)
    apply_style_timing(CFG["STYLE"])
    load()


def save(i, outdir):
    img = render_frame(i / CFG["FPS"])
    cv2.imwrite(str(outdir / f"f{i:04d}.png"), img[:, :, ::-1])
    return i


def save_t(t, outdir):
    img = render_frame(t)
    cv2.imwrite(str(outdir / f"t{t:06.3f}.png"), img[:, :, ::-1])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--times", default=None, help="comma-separated times in s (test renders)")
    ap.add_argument("--out", default=None, help="default frames_1080p[_topo]/")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--style", choices=["bathy", "topo"], default="bathy",
                    help="colours after the 1-2 s fade: bathy (default, globe_1080p.mp4) or topo")
    a = ap.parse_args()
    CFG.update(W=a.width, H=a.height, FPS=a.fps, STYLE=a.style)
    apply_style_timing(a.style)
    outdir = Path(a.out or HERE / ("frames_1080p" + ("_topo" if a.style == "topo" else "")))
    outdir.mkdir(exist_ok=True)
    load()
    t0 = time.time()
    from functools import partial
    if a.times:
        jobs, fn = [float(x) for x in a.times.split(",")], partial(save_t, outdir=outdir)
    else:
        nf = int(round(T_END * a.fps))
        jobs = [int(x) for x in a.frames.split(",")] if a.frames else list(range(nf))
        fn = partial(save, outdir=outdir)
    # macOS uses 'spawn': workers re-import this module, so hand them the config
    with Pool(min(a.procs, len(jobs)), initializer=_init_worker, initargs=(dict(CFG),)) as p:
        for _ in p.imap_unordered(fn, jobs):
            pass
    print(f"rendered {len(jobs)} frames at {a.width}x{a.height} in {time.time()-t0:.1f}s")
