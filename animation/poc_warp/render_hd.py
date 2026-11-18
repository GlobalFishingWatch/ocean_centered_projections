"""HD step 3: render the flat-2D-warp animation (default 1920x1080 @ 30 fps).

Same storyboard / motion design as the POC (render.py), plus:
  * bathymetry: ocean cross-fades from flat pyseas #0a1738 to the shared
    bathymetry style (animation/common/bathy_style.py) after a 1 s hold;
  * minor pieces (parts of an Elastic section that sit on the far side of the
    Equal Earth edge meridian, e.g. the SE Pacific / Antarctic sliver) no longer
    move on their own: a copy attached to their section (Equal Earth extended
    past the edge meridian) fades in while the original fades out, so there are
    no wedges between it and its section;
  * deterministic draw order (one Elastic section drawn on top of the other)
    instead of colour averaging where pieces overlap;
  * short cut-line fragments removed;
  * multiprocessing.

Usage:
  python render_hd.py                          # 1920x1080@30 -> warp_1080p.mp4 + sheet + key frames
  python render_hd.py --width 960 --fps 24 --tex-w 3600 --out warp_540p
  python render_hd.py --frames 0 60 250        # QA: write individual frames to frames_hd/
"""
import argparse
import json
import os
import time
import multiprocessing as mp

import numpy as np
import cv2
import imageio.v2 as imageio
from scipy import ndimage

import geo  # analytic Equal Earth (POC helper)

# ---------------- colours ----------------
OCEAN_BG = np.array([10, 23, 56], np.float32)   # pyseas dark background/ocean (#0a1738)
FINAL_BG = np.array([0, 0, 0], np.float32)
CUT_RGB = (230, 30, 30)
CASING_RGB = (20, 10, 10)
OUTLINE_RGB = (140, 146, 165)

# ---------------- storyboard (seconds) ----------------
T_HOLD0 = 1.0    # hold pyseas default look
T_BATHY = 1.0    # cross-fade flat ocean -> bathymetry
T_PAN = 2.5      # pan central longitude 0 -> pan_lon; background darkens
T_CUTS = 1.5     # fade in cut lines
T_WARP = 5.0     # tear + warp to Elastic-X
T_HOLD1 = 2.5    # hold final
SEGS = np.cumsum([0, T_HOLD0, T_BATHY, T_PAN, T_CUTS, T_WARP, T_HOLD1])
DURATION = SEGS[-1]


def set_storyboard(style):
    """bathy: 1 s flat pyseas hold + 1 s cross-fade (as above).
    topo: textured from frame 0 (no flat opening, no cross-fade), 1.5 s hold."""
    global T_HOLD0, T_BATHY, SEGS, DURATION
    T_HOLD0, T_BATHY = (1.5, 0.0) if style == "topo" else (1.0, 1.0)
    SEGS = np.cumsum([0, T_HOLD0, T_BATHY, T_PAN, T_CUTS, T_WARP, T_HOLD1])
    DURATION = SEGS[-1]

A_END, B_START = 0.75, 0.25   # rigid part over [0, A_END], local deformation over [B_START, 1]
SWAP_END = 0.15               # minor-piece crossfade (fraction of warp)
TOP_SECTION = 1               # Elastic section drawn on top where pieces overlap
SEAM_DEG = 20.0               # width of the band around untorn seams where the two sections' motions blend
MIN_CUT_PTS = 40              # drop cut-line fragments shorter than this (~2 deg)


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def ease2(t):
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (6 * t - 15) + 10)


def frame_state(i, fps):
    t = i / fps
    return dict(
        t=t,
        bathy=ease((t - SEGS[1]) / T_BATHY) if T_BATHY > 0 else 1.0,
        pan_u=ease((t - SEGS[2]) / T_PAN),
        bg_u=ease((t - SEGS[2]) / (T_PAN + T_CUTS)),
        cut_alpha=ease((t - SEGS[3]) / T_CUTS) * (1 - ease((t - (SEGS[5] - 0.6)) / 1.0)),
        warp=float(np.clip((t - SEGS[4]) / T_WARP, 0, 1)),
    )


def weighted_similarity(P, Q, w):
    w = w / w.sum()
    mp_ = (w[:, None] * P).sum(0)
    mq = (w[:, None] * Q).sum(0)
    Pc, Qc = P - mp_, Q - mq
    a = (w * (Pc[:, 0] * Qc[:, 0] + Pc[:, 1] * Qc[:, 1])).sum()
    b = (w * (Pc[:, 0] * Qc[:, 1] - Pc[:, 1] * Qc[:, 0])).sum()
    theta = np.arctan2(b, a)
    var = (w * (Pc ** 2).sum(1)).sum()
    return np.hypot(a, b) / var, theta, mp_, mq


def apply_sim(P, prm, a):
    s, th, mp_, mq = prm
    sa, ang = s ** a, th * a
    cs, sn = np.cos(ang), np.sin(ang)
    x = P[:, 0] - mp_[0]
    y = P[:, 1] - mp_[1]
    out = np.empty_like(P)
    out[:, 0] = sa * (cs * x - sn * y) + mp_[0] + (mq[0] - mp_[0]) * a
    out[:, 1] = sa * (sn * x + cs * y) + mp_[1] + (mq[1] - mp_[1]) * a
    return out


class Model:
    def __init__(self, W, H, tex_w, style="bathy"):
        self.W, self.H = W, H
        cdir = f"cache_{tex_w}"
        ld = lambda k: np.load(f"{cdir}/{k}.npy", mmap_mode="r")
        self.lat = np.array(ld("lat"))
        self.lon = np.array(ld("lon"))
        self.col_flat = np.array(ld("col_flat"))
        if style == "bathy":
            # "after fade" colours: pyseas land/borders over shared bathymetry
            self.col_bathy = np.array(ld("col_bathy"))
        elif style == "topo":
            # "after fade" colours: shared cmocean topo texture (land + sea floor,
            # no borders), animation/common/topo_style.py -> topo_0p05deg.png
            topo = imageio.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "common", "topo_0p05deg.png"))[..., :3]
            assert topo.shape[:2] == (int(ld("meta")[1]), int(ld("meta")[2])), topo.shape
            self.col_bathy = topo.reshape(-1, 3)
        else:
            raise ValueError(style)
        self.sid = np.array(ld("sid"))
        self.p1 = np.stack([ld("p1u") * H, ld("p1v") * H], 1).astype(np.float32)
        meta = ld("meta")
        self.pan_lon, self.th, self.tw = float(meta[0]), int(meta[1]), int(meta[2])
        self.cut_lat, self.cut_lon = np.array(ld("cut_lat")), np.array(ld("cut_lon"))
        self.cut_tear = np.array(ld("cut_tear"))
        self.cut_p1 = np.stack([ld("cut_p1u") * H, ld("cut_p1v") * H], -1)
        with open(f"reference_geom_{W}x{H}.json") as f:
            self.g = json.load(f)
        self.beta = geo.authalic_lat(self.lat).astype(np.float32)
        self.cut_beta = geo.authalic_lat(self.cut_lat)
        self.n = self.lat.size
        self._build()

    # Equal Earth in frame pixels; `unwrap` adds k*360 deg to the relative longitude
    def ee_px(self, beta, lon, lon0, unwrap=None):
        rl = ((lon - lon0 + 180) % 360) - 180
        if unwrap is not None:
            rl = rl + 360 * unwrap
        x, y = geo.equal_earth(beta, rl, 0.0) if unwrap is None else self._ee_raw(beta, rl)
        c, r = geo.ee_to_px(x, y, self.g)
        return np.stack([c, r], 1).astype(np.float32)

    @staticmethod
    def _ee_raw(beta, rl_deg):
        # geo.equal_earth wraps longitude to +-180; this version does not
        lam = np.radians(rl_deg)
        th = np.arcsin(np.sqrt(3) / 2 * np.sin(beta))
        t2 = th * th
        t6 = t2 * t2 * t2
        A1, A2, A3, A4 = geo.A1, geo.A2, geo.A3, geo.A4
        x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * A4 * t6 * t2 + 7 * A3 * t6 + 3 * A2 * t2 + A1))
        y = th * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
        return geo._Rq * x, geo._Rq * y

    def _build(self):
        L = self.pan_lon
        h, w = self.th, self.tw
        sid = self.sid.reshape(h, w)
        lon_axis = self.lon.reshape(h, w)[0]
        rl_axis = ((lon_axis - L + 180) % 360) - 180
        order = np.argsort(rl_axis)
        sid_r = sid[:, order]
        labels = np.zeros(sid_r.shape, np.int32)
        nlab = 0
        for s in np.unique(sid_r):
            lab, n = ndimage.label(sid_r == s)
            labels[lab > 0] = lab[lab > 0] + nlab
            nlab += n
        piece = labels[:, np.argsort(order)].ravel() - 1
        cnt = np.bincount(piece, minlength=nlab)
        psec = np.array([self.sid[np.argmax(piece == k)] for k in range(nlab)])
        main = {s: max([k for k in range(nlab) if psec[k] == s], key=lambda k: cnt[k]) for s in (0, 1)}
        # minor = not the main piece of its section -> attach to section via unwrapped EE
        self.minor = np.zeros(self.n, bool)
        for k in range(nlab):
            if k != main[psec[k]]:
                self.minor |= piece == k
                print(f"  minor piece {k}: section {psec[k]}, {cnt[k]} cells")
        self.P0 = self.ee_px(self.beta, self.lon, L)
        rl = ((self.lon - L + 180) % 360) - 180
        self.wrap = np.where(rl < 0, 1, -1).astype(np.int8)  # direction that re-attaches
        P0e = self.P0.copy()
        m = self.minor
        P0e[m] = self.ee_px(self.beta[m], self.lon[m], L, unwrap=self.wrap[m])
        self.P0e = P0e
        self.alpha = self._seam_alpha(self.lat, self.lon)
        self.alpha[self.minor] = 1.0
        # start position of each point expressed in the *other* section's frame
        # (needed for blending near seams): section 1's frame is the plain
        # panned Equal Earth; section 0's frame has everything west of the 72W
        # seam unwrapped past the edge meridian (where its re-attached piece is)
        self.rl72 = ((-72.1 - L + 180) % 360) - 180
        self.Qo = self.P0.copy()
        m1 = (self.sid == 1) & (rl < self.rl72) & (self.alpha < 1)
        self.Qo[m1] = self.ee_px(self.beta[m1], self.lon[m1], L, unwrap=np.ones(m1.sum(), np.int8))
        wgt = np.cos(np.radians(self.lat)) + 1e-3
        self.params = {}
        sub = np.zeros(self.n, bool)
        sub[::7] = True
        for s in (0, 1):
            ms = (self.sid == s) & sub
            prm = weighted_similarity(P0e[ms].astype(np.float64), self.p1[ms].astype(np.float64), wgt[ms])
            self.params[s] = prm
            print(f"  section {s}: scale {prm[0]:.2f}, rot {np.degrees(prm[1]):.0f} deg")
        self.resid = np.empty_like(self.p1)
        for s in (0, 1):
            ms = np.where(self.sid == s)[0]
            self.resid[ms] = self.p1[ms] - self._rigid(P0e[ms], self.alpha[ms], s, 1.0, self.Qo[ms])
        self.sec_idx = {s: np.where((self.sid == s) & ~self.minor)[0] for s in (0, 1)}
        self.minor_idx = {s: np.where((self.sid == s) & self.minor)[0] for s in (0, 1)}

        # cut lines: per side, nearest texture cell of that section -> minor or not
        r = np.clip(((90 - self.cut_lat) / 180 * h).astype(int), 0, h - 1)
        c = np.clip(((self.cut_lon + 180) / 360 * w).astype(int), 0, w - 1)
        minor_g = self.minor.reshape(h, w)
        wrap_g = self.wrap.reshape(h, w)
        self.cut_P0 = self.ee_px(self.cut_beta, self.cut_lon, L)
        self.cut_P0e, self.cut_resid, self.cut_Qo, self.cut_minor = [], [], [], []
        self.cut_alpha_w = self._seam_alpha(self.cut_lat, self.cut_lon)
        cut_rl = ((self.cut_lon - L + 180) % 360) - 180
        for side in (0, 1):
            _, (ii, jj) = ndimage.distance_transform_edt(sid != side, return_indices=True)
            mi = minor_g[ii[r, c], jj[r, c]]
            self.cut_minor.append(mi)
            P = self.cut_P0.copy()
            if mi.any():
                P[mi] = self.ee_px(self.cut_beta[mi], self.cut_lon[mi], L, unwrap=wrap_g[ii[r, c], jj[r, c]][mi])
            self.cut_P0e.append(P)
            Q = self.cut_P0.copy()
            if side == 1:
                mw = cut_rl < self.rl72
                Q[mw] = self.ee_px(self.cut_beta[mw], self.cut_lon[mw], L, unwrap=np.ones(mw.sum(), np.int8))
            self.cut_Qo.append(Q)
            self.cut_resid.append(self.cut_p1[side] - self._rigid(P.astype(np.float64), self.cut_alpha_w, side, 1.0, Q))
        self._camera_table()

    def _seam_alpha(self, lat, lon):
        """weight of a point's own section's rigid motion: 0.5 on the seams
        (section boundary where the map is NOT torn), rising to 1 at SEAM_DEG
        away. Points on a seam then move identically on both sides, so the
        seam never opens mid-warp."""
        from scipy.spatial import cKDTree
        if not hasattr(self, "_seam_tree"):
            # Only the seam south of Africa (30E). The other seam (72W, Drake
            # Passage -> pole) borders the re-attached SE Pacific piece: its two
            # sides start ~a map-width apart (one is unwrapped), so blending there
            # smears geometry across the frame. That seam is left to open/close.
            ok = (~self.cut_tear) & (self.cut_lat < 85) & (np.abs(self.cut_lon - 30) < 1.0)
            sl, so = np.radians(self.cut_lat[ok][::4]), np.radians(self.cut_lon[ok][::4])
            self._seam_tree = cKDTree(np.c_[np.cos(sl) * np.cos(so), np.cos(sl) * np.sin(so), np.sin(sl)])
        la, lo = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
        xyz = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
        dmax = 2 * np.sin(np.radians(SEAM_DEG) / 2)
        d, _ = self._seam_tree.query(xyz, distance_upper_bound=dmax, workers=-1)
        ang = np.degrees(2 * np.arcsin(np.clip(d / 2, 0, 1)))
        ang[~np.isfinite(d)] = SEAM_DEG
        return (0.5 + 0.5 * ease(ang / SEAM_DEG)).astype(np.float32)

    # ---------- motion ----------
    def _rigid(self, P0e, alpha, s, a, Qo):
        out = apply_sim(P0e, self.params[s], a)
        m = alpha < 1
        if m.any():
            other = apply_sim(Qo[m], self.params[1 - s], a)
            al = alpha[m][:, None]
            out[m] = al * out[m] + (1 - al) * other
        return out

    def _warp(self, P0e, resid, s, u, alpha, Qo):
        a = ease2(u / A_END)
        b = ease2((u - B_START) / (1 - B_START))
        return self._rigid(P0e, alpha, s, a, Qo) + b * resid

    def _camera_table(self):
        W, H = self.W, self.H
        mg = 15 * H / 540
        us = np.linspace(0, 1, 121)
        sub = np.arange(0, self.n, 397)
        ks, cxs, cys = [], [], []
        for uu in us:
            P = np.concatenate([self._warp(self.P0e[q], self.resid[q], s, uu, self.alpha[q], self.Qo[q])
                                for s in (0, 1) for q in [sub[self.sid[sub] == s]]])
            x0, x1 = np.percentile(P[:, 0], [0.05, 99.95])
            y0, y1 = np.percentile(P[:, 1], [0.05, 99.95])
            k = min(1.0, (W - 2 * mg) / (x1 - x0), (H - 2 * mg) / (y1 - y0))
            hw, hh = (W / 2 - mg) / k, (H / 2 - mg) / k
            ks.append(k)
            cxs.append(np.clip(W / 2, x1 - hw, x0 + hw) if x1 - x0 < 2 * hw else (x0 + x1) / 2)
            cys.append(np.clip(H / 2, y1 - hh, y0 + hh) if y1 - y0 < 2 * hh else (y0 + y1) / 2)
        ks = ndimage.minimum_filter1d(np.array(ks), 15, mode="nearest")
        ks = ndimage.gaussian_filter1d(ks, 5, mode="nearest")
        cxs = ndimage.gaussian_filter1d(np.array(cxs), 5, mode="nearest")
        cys = ndimage.gaussian_filter1d(np.array(cys), 5, mode="nearest")
        ks[0] = ks[-1] = 1
        cxs[0] = cxs[-1] = W / 2
        cys[0] = cys[-1] = H / 2
        self._cam = (us, ks, cxs, cys)

    def camera(self, P, u):
        us, ks, cxs, cys = self._cam
        k, cx, cy = np.interp(u, us, ks), np.interp(u, us, cxs), np.interp(u, us, cys)
        P[:, 0] = self.W / 2 + k * (P[:, 0] - cx)
        P[:, 1] = self.H / 2 + k * (P[:, 1] - cy)
        return P

    def layers(self, st):
        """list of (point indices or None=all, positions, alpha), bottom to top"""
        u = st["warp"]
        if u <= 0:
            return [(None, self.ee_px(self.beta, self.lon, st["pan_u"] * self.pan_lon), 1.0)]
        out = []
        f = ease(u / SWAP_END)
        # ghost of the minor pieces at their original Equal Earth position, fading out
        if f < 1:
            gi = np.concatenate([self.minor_idx[0], self.minor_idx[1]])
            out.append((gi, self.camera(self.P0[gi].copy(), u), 1 - f))
        for s in sorted((0, 1), key=lambda s: s == TOP_SECTION):
            mi = self.minor_idx[s]
            if len(mi) and f > 0:
                out.append((mi, self.camera(self._warp(self.P0e[mi], self.resid[mi], s, u, self.alpha[mi], self.Qo[mi]), u), f))
            si = self.sec_idx[s]
            out.append((si, self.camera(self._warp(self.P0e[si], self.resid[si], s, u, self.alpha[si], self.Qo[si]), u), 1.0))
        return out

    def cut_lines(self, st, minor=False):
        """tear lines; minor=True returns only the parts riding on re-attached
        (minor) pieces, which are drawn fading in together with those pieces"""
        u = st["warp"]
        if u <= 0:
            if minor:
                return []
            sides = [self.ee_px(self.cut_beta, self.cut_lon, st["pan_u"] * self.pan_lon)]
            keeps = [np.ones(len(self.cut_lat), bool)]
        else:
            sides = [self.camera(self._warp(self.cut_P0e[s].astype(np.float64), self.cut_resid[s], s, u, self.cut_alpha_w, self.cut_Qo[s]), u)
                     for s in (0, 1)]
            keeps = [self.cut_minor[s] if minor else ~self.cut_minor[s] for s in (0, 1)]
        lines = []
        for P, keep in zip(sides, keeps):
            ok = self.cut_tear & keep
            jump = np.hypot(*np.diff(P, axis=0).T) > 20 * self.H / 540
            brk = jump | ~ok[1:] | ~ok[:-1]
            starts = np.r_[0, np.where(brk)[0] + 1]
            ends = np.r_[np.where(brk)[0] + 1, len(P)]
            for a, b in zip(starts, ends):
                if b - a >= MIN_CUT_PTS and ok[a]:
                    lines.append(P[a:b])
        return lines

    def outline(self):
        lat = np.linspace(-90, 90, 721)
        beta = geo.authalic_lat(lat)
        right = self.ee_px(beta, np.full_like(lat, 179.99999), 0.0)
        left = self.ee_px(beta[::-1], np.full_like(lat, -179.99999), 0.0)
        return [np.concatenate([right, left, right[:1]])]


# ---------------- renderer ----------------
def splat_layer(P, col, W, H, ss):
    """nearest splat into an ss-times buffer + small-hole fill.
    returns colour (Hs, Ws, 3) and coverage (Hs, Ws) in {0, 1}"""
    Ws, Hs = W * ss, H * ss
    c = (P[:, 0] * ss).astype(np.int64)
    r = (P[:, 1] * ss).astype(np.int64)
    ok = (P[:, 0] >= 0) & (c < Ws) & (P[:, 1] >= 0) & (r < Hs)
    idx = r[ok] * Ws + c[ok]
    cnt = np.bincount(idx, minlength=Ws * Hs).astype(np.float32).reshape(Hs, Ws)
    acc = np.stack([np.bincount(idx, weights=col[ok, k], minlength=Ws * Hs) for k in range(3)], -1)
    acc = acc.astype(np.float32).reshape(Hs, Ws, 3)
    for _ in range(2):
        empty = cnt == 0
        nb = ndimage.uniform_filter((~empty).astype(np.float32), 3)
        fill = empty & (nb >= 4 / 9 - 1e-6)
        if not fill.any():
            break
        cs = ndimage.uniform_filter(cnt, 3)
        accs = np.stack([ndimage.uniform_filter(acc[..., k], 3) for k in range(3)], -1)
        acc[fill] = accs[fill] / np.maximum(cs[fill], 1e-6)[:, None]
        cnt[fill] = 1.0
    cov = (cnt > 0).astype(np.float32)
    colr = acc / np.maximum(cnt, 1e-6)[..., None]
    return colr, cov


def draw_lines(img, lines, alpha, rgb, thickness):
    if alpha <= 0.001 or not lines:
        return img
    over = img.copy()
    shift = 4
    for P in lines:
        pts = np.round(P * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(over, [pts], False, rgb, thickness=thickness, lineType=cv2.LINE_AA, shift=shift)
    return img * (1 - alpha) + over * alpha


MODEL = None
OPTS = None


def render_frame(i):
    m, o = MODEL, OPTS
    W, H, ss = m.W, m.H, o.ss
    st = frame_state(i, o.fps)
    bg = OCEAN_BG * (1 - st["bg_u"]) + FINAL_BG * st["bg_u"]
    fb = st["bathy"]
    canvas = np.empty((H * ss, W * ss, 3), np.float32)
    canvas[:] = bg
    for idx, P, alpha in m.layers(st):
        if idx is None:
            cf, cb = m.col_flat, m.col_bathy
        else:
            cf, cb = m.col_flat[idx], m.col_bathy[idx]
        if fb <= 0:
            col = cf
        elif fb >= 1:
            col = cb
        else:
            col = cf.astype(np.float32) * (1 - fb) + cb.astype(np.float32) * fb
        colr, cov = splat_layer(P, col, W, H, ss)
        a = (cov * alpha)[..., None]
        canvas = canvas * (1 - a) + colr * a
    img = canvas.reshape(H, ss, W, ss, 3).mean((1, 3))
    scale = H / 540
    if st["warp"] < 0.08:
        img = draw_lines(img, m.outline(), 1 - ease(st["warp"] / 0.08), OUTLINE_RGB, max(1, int(round(scale))))
    lw = max(2, int(round(1.5 * scale)))
    a_main = st["cut_alpha"]
    a_minor = st["cut_alpha"] * ease(st["warp"] / SWAP_END)
    main_l, minor_l = m.cut_lines(st), m.cut_lines(st, minor=True)
    if o.style == "topo":  # thin dark casing so red reads over green/tan/cream
        img = draw_lines(img, main_l, a_main * 0.8, CASING_RGB, lw + 2)
        img = draw_lines(img, minor_l, a_minor * 0.8, CASING_RGB, lw + 2)
    img = draw_lines(img, main_l, a_main, CUT_RGB, lw)
    img = draw_lines(img, minor_l, a_minor, CUT_RGB, lw)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def _init(model, opts):
    global MODEL, OPTS
    MODEL, OPTS = model, opts


def contact_sheet(idx, frames, path, fps, tw=960, cols=3):
    th = int(round(tw * frames[0].shape[0] / frames[0].shape[1]))
    rows = int(np.ceil(len(frames) / cols))
    lab = 34
    sheet = np.full((rows * (th + lab), cols * tw, 3), 255, np.uint8)
    for n, (i, f) in enumerate(zip(idx, frames)):
        small = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        y0, x0 = (n // cols) * (th + lab), (n % cols) * tw
        sheet[y0 + lab:y0 + lab + th, x0:x0 + tw] = small
        cv2.putText(sheet, f"t={i / fps:5.2f}s  (frame {i})", (x0 + 8, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    imageio.imwrite(path, sheet)


def main():
    global MODEL, OPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor")
    ap.add_argument("--tex-w", type=int, default=7200)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--crf", type=int, default=17)
    ap.add_argument("--style", choices=["bathy", "topo"], default="bathy",
                    help="texture after the opening cross-fade")
    ap.add_argument("--out", default=None, help="output prefix (default warp_{H}p)")
    ap.add_argument("--frames", type=int, nargs="*")
    ap.add_argument("--sheet", type=int, nargs="*")
    o = ap.parse_args()
    W = o.width
    H = W * 9 // 16
    set_storyboard(o.style)  # before forking workers, so they inherit it
    out = o.out or (f"warp_{H}p" if o.style == "bathy" else f"warp_{H}p_{o.style}")
    nframes = int(round(DURATION * o.fps))
    t0 = time.time()
    model = Model(W, H, o.tex_w, o.style)
    MODEL, OPTS = model, o
    print(f"model ready {time.time() - t0:.1f}s; {nframes} frames at {W}x{H}@{o.fps}", flush=True)

    if o.frames is not None or o.sheet is not None:
        idx = o.frames if o.frames is not None else o.sheet
        with mp.get_context("fork").Pool(min(o.workers, len(idx)), _init, (model, o)) as pool:
            frames = pool.map(render_frame, idx)
        if o.frames is not None:
            os.makedirs("frames_hd", exist_ok=True)
            for i, f in zip(idx, frames):
                imageio.imwrite(f"frames_hd/{o.style}_frame_{i:04d}.png", f)
        else:
            contact_sheet(idx, frames, f"qa_sheet_hd_{o.style}.png", o.fps)
        print(f"done {time.time() - t0:.1f}s")
        return

    import imageio_ffmpeg
    os.environ.setdefault("IMAGEIO_FFMPEG_EXE", imageio_ffmpeg.get_ffmpeg_exe())
    sheet_idx = set(np.linspace(0, nframes - 1, 12).round().astype(int).tolist())
    key_idx = {0, int(round(SEGS[2] * o.fps)),
               int(round((SEGS[4] + 0.35 * T_WARP) * o.fps)),
               int(round((SEGS[4] + 0.6 * T_WARP) * o.fps)), nframes - 1}
    sheet = {}
    writer = imageio.get_writer(f"{out}.mp4", fps=o.fps, codec="libx264", quality=None,
                                pixelformat="yuv420p", macro_block_size=2,
                                ffmpeg_params=["-crf", str(o.crf), "-preset", "slow"])
    with mp.get_context("fork").Pool(o.workers, _init, (model, o)) as pool:
        for i, img in enumerate(pool.imap(render_frame, range(nframes), chunksize=2)):
            writer.append_data(img)
            if i in sheet_idx:
                sheet[i] = img
            if i in key_idx:
                imageio.imwrite(f"{out}_key_{i:04d}.png", img)
            if i % o.fps == 0:
                print(f"frame {i}/{nframes}  {time.time() - t0:.0f}s", flush=True)
    writer.close()
    ks = sorted(sheet)
    contact_sheet(ks, [sheet[k] for k in ks], f"{out}_contact_sheet.png", o.fps)
    print(f"all done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
