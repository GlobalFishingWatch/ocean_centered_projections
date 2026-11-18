"""Step 3: render the flat-2D-warp animation frames, encode mp4/gif, contact sheet.

Usage:
  python render.py              # full render + encode
  python render.py --frames 0 100 200   # just write those frames to frames_preview/
"""
import os
import sys
import time
import numpy as np
import imageio.v2 as imageio
import cv2
from scipy import ndimage
import geo

FPS = 24
W, H = geo.W, geo.H
SS = 2  # supersampling factor of the splat buffer
RIGID = float(os.environ.get("RIGID", "1.0"))
A_END = float(os.environ.get("A_END", "0.75"))
B_START = float(os.environ.get("B_START", "0.25"))  # fraction of per-piece rotation/scale done rigidly

OCEAN_BG = np.array([10, 23, 56], float)  # pyseas dark background/ocean (#0a1738)
FINAL_BG = np.array([0, 0, 0], float)     # outside-the-map colour once the map is torn
CUT_RGB = (230, 30, 30)
OUTLINE_RGB = (140, 146, 165)  # thin map outline, as in the pyseas reference

# ---------------- storyboard (seconds) ----------------
T_HOLD0 = 1.5    # hold pyseas-look Equal Earth @ 0E
T_PAN = 2.5      # pan central longitude 0 -> pan_lon; background darkens
T_CUTS = 1.5     # fade in cut lines
T_WARP = 5.0     # tear + warp to Elastic-X
T_HOLD1 = 2.5    # hold final
SEGS = np.cumsum([0, T_HOLD0, T_PAN, T_CUTS, T_WARP, T_HOLD1])
DURATION = SEGS[-1]
NFRAMES = int(round(DURATION * FPS))


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)  # smoothstep


def ease2(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)  # smootherstep


def frame_state(i):
    tsec = i / FPS
    st = {}
    u_pan = ease((tsec - SEGS[1]) / T_PAN)
    st["pan_u"] = u_pan
    st["bg_u"] = ease((tsec - SEGS[1]) / (T_PAN + T_CUTS))
    st["cut_alpha"] = ease((tsec - SEGS[2]) / T_CUTS) * (1 - ease((tsec - (SEGS[4] - 0.6)) / 1.0))
    st["warp"] = np.clip((tsec - SEGS[3]) / T_WARP, 0, 1)
    st["t"] = tsec
    return st


# ---------------- pieces & per-piece similarity motion ----------------
def weighted_similarity(P, Q, w):
    """Umeyama: find s, R(theta), translation mapping P -> Q (weighted LSQ)."""
    w = w / w.sum()
    mp = (w[:, None] * P).sum(0)
    mq = (w[:, None] * Q).sum(0)
    Pc, Qc = P - mp, Q - mq
    a = (w * (Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])).sum()
    b = (w * (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])).sum()
    theta = np.arctan2(b, a)
    var = (w * (Pc ** 2).sum(1)).sum()
    s = np.hypot(a, b) / var
    return s, theta, mp, mq


class Model:
    def __init__(self, path="points.npz"):
        d = np.load(path)
        self.beta = d["beta"]
        self.lon = d["lon"]
        self.lat = d["lat"]
        self.col = d["col"].astype(np.float32)
        self.sid = d["sid"]
        self.p1 = np.stack([d["p1c"], d["p1r"]], 1).astype(np.float64)
        self.pan_lon = float(d["pan_lon"])
        self.cut_lat, self.cut_lon = d["cut_lat"], d["cut_lon"]
        self.cut_tear = d["cut_tear"]
        self.cut_p1 = np.stack([d["cut_p1c"], d["cut_p1r"]], -1)  # (2 sides, n, 2)
        self.g = geo.load_ref_geom()
        self.cut_beta = geo.authalic_lat(self.cut_lat)
        self.n = self.lat.size
        self._build_pieces()

    def ee_px(self, beta, lon, lon0):
        x, y = geo.equal_earth(beta, lon, lon0)
        c, r = geo.ee_to_px(x, y, self.g)
        return np.stack([c, r], 1)

    def _build_pieces(self):
        """Pieces = connected components of each Elastic section in the
        Equal Earth map centred on pan_lon (so a section crossing the map's
        edge meridian becomes two pieces that move independently)."""
        L = self.pan_lon
        h = int(round(np.sqrt(self.n / 2)))
        w = self.n // h
        assert h * w == self.n
        sid = self.sid.reshape(h, w)
        lon_axis = self.lon.reshape(h, w)[0]
        rl = ((lon_axis - L + 180) % 360) - 180
        order = np.argsort(rl)  # columns reordered so EE left edge is column 0
        sid_r = sid[:, order]
        labels = np.zeros_like(sid_r, dtype=np.int32)
        nlab = 0
        for s in np.unique(sid_r):
            lab, n = ndimage.label(sid_r == s)
            labels[lab > 0] = lab[lab > 0] + nlab
            nlab += n
        inv = np.argsort(order)
        piece = labels[:, inv].ravel() - 1
        # merge tiny components (single stray cells) into their section's biggest piece
        cnt = np.bincount(piece, minlength=nlab)
        self.piece_grid = labels[:, inv] - 1  # for cut-line lookup
        self.grid_shape = (h, w)
        self.P0 = self.ee_px(self.beta, self.lon, L)
        wgt = np.cos(np.radians(self.lat)).astype(np.float64) + 1e-3
        self.params = {}
        print("pieces:")
        for k in range(nlab):
            m = piece == k
            if cnt[k] < 50:
                continue
            s, th, mp, mq = weighted_similarity(self.P0[m], self.p1[m], wgt[m])
            s, th = s ** RIGID, th * RIGID
            self.params[k] = (s, th, mp, mq)
            print(f"  piece {k}: section {self.sid[m][0]}, {cnt[k]} cells, "
                  f"scale {s:.2f}, rot {np.degrees(th):.0f} deg")
        # tiny pieces: use section's largest piece
        for k in range(nlab):
            if k not in self.params:
                m = piece == k
                sec = self.sid[m][0]
                cand = [j for j in self.params if self.sid[piece == j][0] == sec]
                big = max(cand, key=lambda j: cnt[j])
                piece[m] = big
        self.piece = piece
        # residual after the rigid(similarity) part
        self.resid = np.empty_like(self.p1)
        for k, prm in self.params.items():
            m = piece == k
            self.resid[m] = self.p1[m] - self.apply_sim(self.P0[m], prm, 1.0)
        # cut-line pieces per side: nearest grid cell belonging to that section
        self.cut_piece = []
        r = np.clip(((90 - self.cut_lat) / 180 * h).astype(int), 0, h - 1)
        c = np.clip(((self.cut_lon + 180) / 360 * w).astype(int), 0, w - 1)
        for side in range(2):
            notmine = sid != side
            _, (ii, jj) = ndimage.distance_transform_edt(notmine, return_indices=True)
            pg = self.piece.reshape(h, w)
            self.cut_piece.append(pg[ii[r, c], jj[r, c]])
        self.cut_P0 = self.ee_px(self.cut_beta, self.cut_lon, L)
        self.cut_resid = []
        for side in range(2):
            res = np.zeros_like(self.cut_P0)
            for k, prm in self.params.items():
                m = self.cut_piece[side] == k
                res[m] = self.cut_p1[side][m] - self.apply_sim(self.cut_P0[m], prm, 1.0)
            self.cut_resid.append(res)

    @staticmethod
    def apply_sim(P, prm, a):
        s, th, mp, mq = prm
        sa = s ** a
        ang = th * a
        cs, sn = np.cos(ang), np.sin(ang)
        Pc = P - mp
        out = np.empty_like(P)
        out[:, 0] = sa * (cs * Pc[:, 0] - sn * Pc[:, 1])
        out[:, 1] = sa * (sn * Pc[:, 0] + cs * Pc[:, 1])
        out += mp + (mq - mp) * a
        return out

    def warp_positions(self, P0, piece, resid, u, camera=True):
        """rigid part runs over u in [0, A_END], local deformation over [B_START, 1]"""
        a = ease2(u / A_END)
        b = ease2((u - B_START) / (1 - B_START))
        out = np.empty_like(P0)
        for k, prm in self.params.items():
            m = piece == k
            out[m] = self.apply_sim(P0[m], prm, a)
        out += b * resid
        if camera:
            k, cx, cy = self.camera(u)
            out[:, 0] = W / 2 + k * (out[:, 0] - cx)
            out[:, 1] = H / 2 + k * (out[:, 1] - cy)
        return out

    def camera(self, u):
        """virtual camera that zooms out / recentres if pieces swing out of frame
        (eases back to identity at both ends of the warp)."""
        if not hasattr(self, "_cam"):
            us = np.linspace(0, 1, 121)
            sub = np.arange(0, self.n, 97)
            ks, cxs, cys = [], [], []
            for uu in us:
                P = self.warp_positions(self.P0[sub], self.piece[sub], self.resid[sub], uu, camera=False)
                x0, x1 = np.percentile(P[:, 0], [0.05, 99.95])
                y0, y1 = np.percentile(P[:, 1], [0.05, 99.95])
                k = min(1.0, (W - 30) / (x1 - x0), (H - 30) / (y1 - y0))
                ks.append(k)
                # recentre only as much as needed to keep the bbox inside
                # recentre only as much as needed to keep the bbox inside the frame
                hw, hh = (W / 2 - 15) / k, (H / 2 - 15) / k
                cxs.append(np.clip(W / 2, x1 - hw, x0 + hw) if x1 - x0 < 2 * hw else (x0 + x1) / 2)
                cys.append(np.clip(H / 2, y1 - hh, y0 + hh) if y1 - y0 < 2 * hh else (y0 + y1) / 2)
            ks, cxs, cys = map(np.array, (ks, cxs, cys))
            # make the zoom monotone-ish and smooth: running min then gaussian smooth
            ks = ndimage.minimum_filter1d(ks, 15, mode="nearest")
            ks = ndimage.gaussian_filter1d(ks, 5, mode="nearest")
            cxs = ndimage.gaussian_filter1d(cxs, 5, mode="nearest")
            cys = ndimage.gaussian_filter1d(cys, 5, mode="nearest")
            ks[0], ks[-1] = 1, 1
            self._cam = (us, ks, cxs, cys)
        us, ks, cxs, cys = self._cam
        return np.interp(u, us, ks), np.interp(u, us, cxs), np.interp(u, us, cys)

    def positions(self, st):
        if st["warp"] <= 0:
            lon0 = st["pan_u"] * self.pan_lon
            return self.ee_px(self.beta, self.lon, lon0)
        return self.warp_positions(self.P0, self.piece, self.resid, st["warp"])

    def cut_positions(self, st):
        """list of polylines (each an (n,2) px array) for the tear lines"""
        if st["warp"] <= 0:
            lon0 = st["pan_u"] * self.pan_lon
            sides = [self.ee_px(self.cut_beta, self.cut_lon, lon0)]
        else:
            sides = [self.warp_positions(self.cut_P0, self.cut_piece[s], self.cut_resid[s], st["warp"])
                     for s in range(2)]
        lines = []
        for P in sides:
            # break where the line leaves the tear part or jumps (EE edge wrap)
            jump = np.hypot(*np.diff(P, axis=0).T) > 20
            brk = jump | ~self.cut_tear[1:] | ~self.cut_tear[:-1]
            starts = np.r_[0, np.where(brk)[0] + 1]
            ends = np.r_[np.where(brk)[0] + 1, len(P)]
            for a, b in zip(starts, ends):
                if b - a >= 2 and self.cut_tear[a]:
                    lines.append(P[a:b])
        return lines


# ---------------- renderer ----------------
def splat(P, col, bg):
    """nearest splat into SS-times buffer, hole fill, box-downsample with coverage."""
    Ws, Hs = W * SS, H * SS
    c = np.floor(P[:, 0] * SS).astype(np.int64)
    r = np.floor(P[:, 1] * SS).astype(np.int64)
    ok = (c >= 0) & (c < Ws) & (r >= 0) & (r < Hs)
    idx = r[ok] * Ws + c[ok]
    cnt = np.bincount(idx, minlength=Ws * Hs).astype(np.float32)
    acc = np.stack([np.bincount(idx, weights=col[ok, k], minlength=Ws * Hs) for k in range(3)], -1).astype(np.float32)
    cnt = cnt.reshape(Hs, Ws)
    acc = acc.reshape(Hs, Ws, 3)
    # hole fill: normalised 3x3 convolution, applied only to empty cells that
    # are mostly surrounded by covered cells (two passes)
    for _ in range(2):
        empty = cnt == 0
        cov = (cnt > 0).astype(np.float32)
        nb = ndimage.uniform_filter(cov, 3)
        fill = empty & (nb >= 4 / 9 - 1e-6)
        if not fill.any():
            break
        cs = ndimage.uniform_filter(cnt, 3)
        accs = np.stack([ndimage.uniform_filter(acc[..., k], 3) for k in range(3)], -1)
        acc[fill] = accs[fill] / np.maximum(cs[fill], 1e-6)[:, None]
        cnt[fill] = 1.0
        # renormalise acc for filled cells so colour = acc/cnt
    colr = np.where(cnt[..., None] > 0, acc / np.maximum(cnt, 1e-6)[..., None], 0)
    covered = (cnt > 0).astype(np.float32)
    # downsample
    colr = colr.reshape(H, SS, W, SS, 3)
    covd = covered.reshape(H, SS, W, SS)
    csum = (colr * covd[..., None]).sum((1, 3))
    cov = covd.sum((1, 3))
    alpha = cov / (SS * SS)
    mean = csum / np.maximum(cov, 1e-6)[..., None]
    img = mean * alpha[..., None] + bg[None, None, :] * (1 - alpha[..., None])
    return img


def draw_lines(img, lines, alpha, rgb=CUT_RGB, thickness=2):
    if alpha <= 0.001 or not lines:
        return img
    over = img.copy()
    shift = 3
    for P in lines:
        pts = np.round(P * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(over, [pts], False, rgb, thickness=thickness, lineType=cv2.LINE_AA, shift=shift)
    return img * (1 - alpha) + over * alpha


def outline_line(model):
    """Equal Earth map outline (pyseas draws it as a thin light line)."""
    lat = np.linspace(-90, 90, 361)
    beta = geo.authalic_lat(lat)
    right = model.ee_px(beta, np.full_like(lat, 179.9999), 0.0)
    left = model.ee_px(beta[::-1], np.full_like(lat, -179.9999), 0.0)
    return [np.concatenate([right, left, right[:1]])]


def render_frame(model, i):
    st = frame_state(i)
    bg = OCEAN_BG * (1 - st["bg_u"]) + FINAL_BG * st["bg_u"]
    P = model.positions(st)
    img = splat(P, model.col, bg)
    img = draw_lines(img, outline_line(model), 1 - ease(st["warp"] / 0.08), OUTLINE_RGB, 1)
    img = draw_lines(img, model.cut_positions(st), st["cut_alpha"])
    return np.clip(img + 0.5, 0, 255).astype(np.uint8), st


def contact_sheet(frames_idx, frames, path):
    cols = 4
    rows = int(np.ceil(len(frames) / cols))
    tw, th = W // 2, H // 2
    sheet = np.full((rows * (th + 24), cols * tw, 3), 255, np.uint8)
    for n, (i, f) in enumerate(zip(frames_idx, frames)):
        small = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        y0 = (n // cols) * (th + 24)
        x0 = (n % cols) * tw
        sheet[y0 + 24:y0 + 24 + th, x0:x0 + tw] = small
        cv2.putText(sheet, f"t={i / FPS:5.2f}s  (frame {i})", (x0 + 6, y0 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    imageio.imwrite(path, sheet)


def main():
    t0 = time.time()
    model = Model()
    print(f"model ready {time.time() - t0:.1f}s; pan_lon={model.pan_lon}; {NFRAMES} frames")
    if "--sheet" in sys.argv:
        idx = [int(a) for a in sys.argv[sys.argv.index("--sheet") + 1:]]
        frames = [render_frame(model, i)[0] for i in idx]
        contact_sheet(idx, frames, os.environ.get("QA", "qa_sheet.png"))
        print(f"done {time.time() - t0:.1f}s")
        return
    if "--frames" in sys.argv:
        os.makedirs("frames_preview", exist_ok=True)
        for a in sys.argv[sys.argv.index("--frames") + 1:]:
            i = int(a)
            img, st = render_frame(model, i)
            imageio.imwrite(f"frames_preview/frame_{i:04d}.png", img)
        print(f"done {time.time() - t0:.1f}s")
        return

    import imageio_ffmpeg
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    sheet_idx = np.linspace(0, NFRAMES - 1, 12).round().astype(int)
    sheet_frames = []
    gif_frames = []
    writer = imageio.get_writer("poc_warp.mp4", fps=FPS, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=2)
    for i in range(NFRAMES):
        img, st = render_frame(model, i)
        writer.append_data(img)
        if i in sheet_idx:
            sheet_frames.append(img)
            imageio.imwrite(f"keyframe_{i:04d}.png", img) if i in (0, NFRAMES - 1) else None
        if i % 2 == 0:
            gif_frames.append(cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA))
        if i % 24 == 0:
            print(f"frame {i}/{NFRAMES}  {time.time() - t0:.0f}s", flush=True)
    writer.close()
    imageio.mimsave("poc_warp.gif", gif_frames, duration=2 / FPS)
    contact_sheet(sheet_idx, sheet_frames, "contact_sheet.png")
    print(f"all done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
