"""Encode frames_1080p/ into globe_1080p.mp4, globe_1080p_contact_sheet.png and
full-res key frames globe_1080p_key_XXXX.png.  usage: python make_video_hd.py [--fps 30]"""
import argparse, shutil, subprocess
from pathlib import Path
import numpy as np
import cv2
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--frames", default=str(HERE / "frames_1080p"))
ap.add_argument("--name", default="globe_1080p")
ap.add_argument("--key-times", default="0,2.0,6.6,10.3,end",
                help="times (s) of the full-res key frames; 'end' = last frame")
a = ap.parse_args()
FR, FPS = Path(a.frames), a.fps
files = sorted(FR.glob("f*.png"))
ff = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FR / "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow",
                "-movflags", "+faststart", str(HERE / f"{a.name}.mp4")], check=True)

idx = np.linspace(0, len(files) - 1, 12).round().astype(int)
tiles = []
for i in idx:
    im = cv2.resize(cv2.imread(str(files[i])), (960, 540), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f"t={i / FPS:5.2f}s  (frame {i})", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 1, cv2.LINE_AA)
    tiles.append(im)
cv2.imwrite(str(HERE / f"{a.name}_contact_sheet.png"), np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)]))

# key frames: first, bathymetry in, globe (best-fit orientation), mid-unpeel, final
for k in a.key_times.split(","):
    t = len(files) / FPS if k == "end" else float(k)
    i = min(int(round(t * FPS)), len(files) - 1)
    shutil.copy(files[i], HERE / f"{a.name}_key_{i:04d}.png")
print("done:", len(files), "frames")
