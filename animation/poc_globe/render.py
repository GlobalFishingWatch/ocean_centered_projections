"""Render the 'via the globe' POC animation: pyseas EqualEarth -> globe -> Elastic-X.

Point-cloud forward splatting with a z-buffer (+ depth-tolerant colour
averaging for antialiasing) and a small hole fill.  Run precompute.py first.

usage: python render.py [--frames 0,60,120] [--out frames_dir]
"""
import argparse, json, os, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import cv2
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
info = json.load(open(HERE / "reference_axes.json"))
W, H = info["W"], info["H"]
FPS = 24
BG = np.array([10, 23, 56], np.float32)        # pyseas dark background/ocean #0a1738
GLOW = np.array([90, 130, 200], np.float32)    # faint atmosphere rim
BG_DARK = np.array([3, 8, 22], np.float32)    # darker backdrop so the ocean's shape reads
EDGE = np.array([160, 167, 182], np.float32)  # pyseas map outline (light grey)
RED = np.array([235, 40, 40], np.float32)

# ---- storyboard (seconds)
T_HOLD0 = (0.0, 1.5)    # pyseas EqualEarth
T_WRAP = (1.5, 4.5)     # flat -> globe
T_ROT = (4.5, 7.0)      # rotate globe to best-fit Elastic-X orientation
T_CUT = (7.0, 8.5)      # fade in red cut lines
T_PEEL = (8.5, 13.0)    # cut + unpeel globe -> Elastic-X
T_CUTOUT = (13.5, 14.5) # fade red lines out
T_END = 15.5
PEEL_DELAY = 0.35
NFRAMES = int(round(T_END * FPS))

D = None  # data, loaded once per process


def load():
    global D
    if D is not None:
        return D
    z = np.load(HERE / "pointcloud.npz")
    D = {k: z[k] for k in z.files}
    q = Rotation.from_quat(D["Q"])
    lam = float(D["lam"])
    D["rot"] = q
    D["R"] = lam
    D["colf"] = D["col"].astype(np.float32)
    return D


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)


def phase(t, rng):
    return np.clip((t - rng[0]) / (rng[1] - rng[0]), 0, 1)


def positions(t, d, ee, sph, ex):
    """3D positions (x, y right/up in px from frame centre, z towards viewer) at time t.
    Returns pos, normal-z (for shading), sphericity weight."""
    R = d["R"]
    tw, tr, tp = phase(t, T_WRAP), phase(t, T_ROT), phase(t, T_PEEL)
    if tp > 0:
        rs = d["rot"].apply(sph).astype(np.float32)
        e = peel_ease(tp, rs)
        flat = np.concatenate([ex, np.zeros((len(ex), 1), np.float32)], 1)
        pos = (1 - e)[:, None] * (R * rs) + e[:, None] * flat
        g = 1 - smooth(tp / 0.3)
        return pos, rs[:, 2], g
    if tw < 1:
        e = smooth(tw)
        flat = np.concatenate([ee, np.zeros((len(ee), 1), np.float32)], 1)
        pos = (1 - e) * flat + e * (R * sph)
        return pos, sph[:, 2], smooth((tw - 0.5) / 0.5)
    rot = Slerp([0, 1], Rotation.concatenate([Rotation.identity(), d["rot"]]))([smooth(tr)])[0]
    rs = rot.apply(sph).astype(np.float32)
    return R * rs, rs[:, 2], 1.0


def peel_ease(tp, rs):
    """per-point progress for the unpeel.  Points on the visible hemisphere
    start first; points on the far side are delayed (by up to PEEL_DELAY of
    the phase) so they swing out from behind after the front has opened.
    The delay is a continuous function on the sphere, so the surface never
    tears except along the cuts."""
    delay = PEEL_DELAY * (1 - rs[:, 2]) / 2
    return smooth((tp - delay) / (1 - PEEL_DELAY)).astype(np.float32)


def splat(pos, col, weight=None, tol=2.0):
    """z-buffered, depth-tolerant average splat. returns (sum_rgb, count)"""
    ix = np.rint(pos[:, 0] + W / 2 - 0.5).astype(np.int32)
    iy = np.rint(H / 2 - pos[:, 1] - 0.5).astype(np.int32)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    pix = (iy * W + ix)[ok]
    z = pos[ok, 2]
    zbuf = np.full(W * H, -np.inf, np.float32)
    if np.ptp(z) > 1e-3:
        o = np.argsort(z, kind="stable")
        zbuf[pix[o]] = z[o]          # last (= nearest) wins
        keep = z >= zbuf[pix] - tol
    else:
        keep = np.ones(len(z), bool)
    pix = pix[keep]
    w = np.ones(len(pix), np.float32) if weight is None else weight[ok][keep]
    c = col[ok][keep]
    acc = np.stack([np.bincount(pix, weights=c[:, k] * w, minlength=W * H) for k in range(3)], -1)
    cnt = np.bincount(pix, weights=w, minlength=W * H)
    return acc, cnt, zbuf


def render_frame(i):
    d = load()
    t = i / FPS
    pos, nz, g = positions(t, d, d["ee"], d["sph"], d["ex"])
    col = d["colf"]
    if g > 0:  # gentle limb shading on the globe
        shade = 1 - 0.45 * g * np.clip(1 - nz, 0, 1) ** 1.5
        col = col * shade[:, None]
    # red cut lines: extra heavily weighted points, nudged towards viewer
    a = smooth(phase(t, T_CUT)) * (1 - smooth(phase(t, T_CUTOUT)))
    if a > 0:
        cpos, _, _ = positions(t, d, d["cut_ee"], d["cut_sph"], d["cut_ex"])
        cpos = cpos.copy(); cpos[:, 2] += 1.5
        offs = np.array([[0, 0], [0.7, 0], [0, 0.7], [0.7, 0.7]], np.float32) - 0.35
        cp = np.concatenate([cpos + np.r_[o, 0] for o in offs])
        allpos = np.concatenate([pos, cp])
        allcol = np.concatenate([col, np.tile(RED, (len(cp), 1))])
        wts = np.concatenate([np.ones(len(pos), np.float32), np.full(len(cp), 30 * a, np.float32)])
        acc, cnt, _ = splat(allpos, allcol, wts, tol=2.5)
    else:
        acc, cnt, _ = splat(pos, col)
    acc = acc.reshape(H, W, 3).astype(np.float32); cnt = cnt.reshape(H, W).astype(np.float32)
    cov = (cnt > 0).astype(np.float32)
    img = np.where(cov[..., None] > 0, acc / np.maximum(cnt, 1e-6)[..., None], 0)
    # hole fill: average of covered neighbours where most neighbours are covered
    for _ in range(2):
        k = np.ones((3, 3), np.float32)
        s = cv2.filter2D(img * cov[..., None], -1, k, borderType=cv2.BORDER_CONSTANT)
        n = cv2.filter2D(cov, -1, k, borderType=cv2.BORDER_CONSTANT)
        fill = (cov == 0) & (n >= 4)
        img[fill] = s[fill] / n[fill, None]
        cov = cov + fill
    # after the opening hold, darken the backdrop slightly so the ocean (which is
    # the same colour as the pyseas background) becomes visible as a shape
    bgk = smooth(phase(t, (T_WRAP[0], T_WRAP[0] + 1.0)))
    bg = (1 - bgk) * BG + bgk * BG_DARK
    out = np.where(cov[..., None] > 0, img, bg)
    # atmosphere glow outside the globe silhouette
    if g > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(xx + 0.5 - W / 2, H / 2 - yy - 0.5)
        glow = g * 0.55 * np.exp(-np.clip(r - d["R"], 0, None) / 7.0) * (r > d["R"] - 0.5)
        glow *= (cov == 0)
        out = out * (1 - glow[..., None]) + GLOW * glow[..., None]
    # thin EqualEarth outline, as pyseas draws it; fades out early in the wrap
    ea = 1 - smooth(phase(t, (T_WRAP[0], T_WRAP[0] + 0.6)))
    if ea > 0:
        epos, _, _ = positions(t, d, d["edge_ee"], d["edge_sph"], d["edge_ee"])
        pts = np.stack([epos[:, 0] + W / 2, H / 2 - epos[:, 1]], -1)
        layer = np.zeros((H, W), np.uint8)
        cv2.polylines(layer, [np.rint(pts * 16).astype(np.int32)], True, 255, 1, cv2.LINE_AA, shift=4)
        al = (layer.astype(np.float32) / 255 * 0.75 * ea)[..., None]
        out = out * (1 - al) + EDGE * al
    return np.clip(out, 0, 255).astype(np.uint8)


def save(i, outdir):
    img = render_frame(i)
    cv2.imwrite(str(outdir / f"f{i:04d}.png"), img[:, :, ::-1])
    return i


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=None)
    ap.add_argument("--out", default=str(HERE / "frames"))
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(exist_ok=True)
    frames = [int(x) for x in a.frames.split(",")] if a.frames else list(range(NFRAMES))
    load()
    t0 = time.time()
    with Pool(min(a.procs, len(frames))) as p:
        for k, i in enumerate(p.imap_unordered(lambda_save := __import__("functools").partial(save, outdir=outdir), frames)):
            pass
    print(f"rendered {len(frames)} frames in {time.time()-t0:.1f}s")
