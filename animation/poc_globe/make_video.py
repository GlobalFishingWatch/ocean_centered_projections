"""Encode frames/ into poc_globe.mp4 (H.264 yuv420p), poc_globe.gif, contact_sheet.png"""
import subprocess
from pathlib import Path
import numpy as np
import cv2
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
FR = HERE / "frames"
FPS = 24
files = sorted(FR.glob("f*.png"))
ff = imageio_ffmpeg.get_ffmpeg_exe()

subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FR / "f%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                "-movflags", "+faststart", str(HERE / "poc_globe.mp4")], check=True)
# small gif: 480px wide, 12 fps, shared palette
subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(FR / "f%04d.png"),
                "-vf", "fps=12,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                str(HERE / "poc_globe.gif")], check=True)

# contact sheet: 12 evenly spaced frames, 4x3, labelled with time
idx = np.linspace(0, len(files) - 1, 12).round().astype(int)
tiles = []
for i in idx:
    im = cv2.imread(str(files[i]))
    im = cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA)
    cv2.putText(im, f"t={i / FPS:5.2f}s  (frame {i})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    tiles.append(im)
sheet = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)])
cv2.imwrite(str(HERE / "contact_sheet.png"), sheet)
print("wrote mp4, gif, contact sheet;", len(files), "frames")
