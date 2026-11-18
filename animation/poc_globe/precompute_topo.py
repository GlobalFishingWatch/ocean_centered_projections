"""Per-point colours for the 'topo' style (cmocean topo land + sea floor).

Resamples animation/common/topo_0p05deg.png (output of topo_style.topo_rgb())
to the HD texture grid (0.0625 deg) and applies exactly the same row thinning
as precompute_hd.py, so the colours line up 1:1 with pointcloud_hd.npz.
Writes texture_hd_topo.png and pointcloud_hd_topo.npz (col_topo only).
"""
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
TOPO = HERE.parent / "common" / "topo_0p05deg.png"

flat = cv2.imread(str(HERE / "texture_hd_flat.png"))
th, tw = flat.shape[:2]
topo = cv2.imread(str(TOPO))[:, :, ::-1]        # 3600x7200, row 0 = 90N, col 0 = 180W
assert topo.shape[:2] == (3600, 7200), topo.shape
# 0.05 deg cells -> 0.0625 deg cells (area average; both grids are cell-centred over the globe)
topo = cv2.resize(topo, (tw, th), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(HERE / "texture_hd_topo.png"), topo[:, :, ::-1])

divisors = np.array([d for d in range(1, tw + 1) if tw % d == 0])
cols = []
for i in range(th):
    lat = 90 - (i + 0.5) * 180 / th
    k = divisors[divisors <= 1 / max(np.cos(np.radians(lat)), 1e-9)].max()
    cols.append(topo[i].reshape(tw // k, k, 3).mean(1))
col = np.rint(np.concatenate(cols)).astype(np.uint8)
n_hd = len(np.load(HERE / "pointcloud_hd.npz")["col_bathy"])
assert len(col) == n_hd, (len(col), n_hd)
np.savez(HERE / "pointcloud_hd_topo.npz", col_topo=col)
print("wrote", len(col), "topo colours")
