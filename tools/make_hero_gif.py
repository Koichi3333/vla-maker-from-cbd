#!/usr/bin/env python3
"""Build the side-by-side GIF at the top of the README (docs/media/hero_video_to_vla.gif). CPU, 0 CU.

Left: the CBD sample (a human video with the tracked skeleton overlaid; Common Behavior Data's
    examples/human-capture/sample_output/01_mediapipe_overlay/mediapipe_overlay.mp4)
Right: one trial of the VLA's closed-loop evaluation (drawn as in tools/make_reach_gif.py: the glass is magenta and moves with the hand once reached)

    gh api repos/Koichi3333/common-behavior-data/contents/examples/human-capture/sample_output/01_mediapipe_overlay/mediapipe_overlay.mp4 \
        -H 'Accept: application/vnd.github.raw' > work/cbd/mediapipe_overlay.mp4
    MUJOCO_GL=osmesa python3 tools/make_hero_gif.py work/cbd/mediapipe_overlay.mp4 work/v10b/evalV_box/h2/traj_2237.npz

The right side defaults to scene 2237 (pick up a glass, drink, put it back), second "shifted" trial (trial 5). Of this scene's 12 trials,
it is the only one that brings the glass to the mouth (closest to the head 0.227 m; the recorded human clip of the same scene gets 0.203 m). Both sides play in real time at 10 fps.
The right side is smoothed with vla/smooth_motion.py by default (the hand jitters at chunk seams; hand jitter while lifting drops from 1.29 to 0.79 cm on average).
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_reach_gif import DEFAULT_COLOR, ROOT, load_scene, render_trial, save_gif

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read_video(path):
    import imageio_ffmpeg
    r = imageio_ffmpeg.read_frames(str(path))
    meta = next(r)
    w, h = meta["size"]
    return np.stack([np.frombuffer(f, np.uint8).reshape(h, w, 3) for f in r]), meta["fps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="CBD の mediapipe_overlay.mp4")
    ap.add_argument("traj", help="traj_<clip>.npz")
    ap.add_argument("--trial", type=int, default=5)
    ap.add_argument("--end", type=int, default=170, help="右をこのフレームで切る（評価のフレーム、20fps）")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--labels", nargs=2, default=["Human video → CBD", "VLA on CBD → MuJoCo"])
    ap.add_argument("--color", default=DEFAULT_COLOR)
    ap.add_argument("--raw", action="store_true", help="右を均さずに描く（既定は vla/smooth_motion.py で均す）")
    ap.add_argument("--out", default=str(ROOT / "docs/media/hero_video_to_vla.gif"))
    a = ap.parse_args()
    from PIL import Image, ImageDraw, ImageFont

    H = a.height
    vid, vfps = read_video(a.video)
    W = round(H * vid.shape[2] / vid.shape[1])        # Left width (keeps portrait). The right side is cropped to the same width

    z, label, scene = load_scene(a.traj, a.color, H)
    sim = render_trial(z, label, scene, a.trial, a.end, 20 // a.fps, smooth=not a.raw)
    bright = sim.max(axis=3) > 150                    # body and object (brighter than the floor and background)
    xs = np.nonzero(bright.any(axis=(0, 1)))[0]
    cx = int(np.clip((xs.min() + xs.max()) // 2, W // 2, H - W // 2))
    sim = sim[:, :, cx - W // 2: cx - W // 2 + W]

    n = max(len(sim), int(len(vid) / vfps * a.fps))
    bar = 32
    font = ImageFont.truetype(FONT, 15)
    frames = []
    for i in range(n):
        left = np.asarray(Image.fromarray(vid[min(round(i * vfps / a.fps), len(vid) - 1)]).resize((W, H), Image.LANCZOS))
        right = sim[min(i, len(sim) - 1)]
        img = Image.new("RGB", (2 * W + 4, H + bar), (17, 17, 17))
        img.paste(Image.fromarray(left), (0, bar))
        img.paste(Image.fromarray(right), (W + 4, bar))
        d = ImageDraw.Draw(img)
        for k, text in enumerate(a.labels):
            d.text((k * (W + 4) + 10, bar // 2), text, font=font, fill=(235, 235, 235), anchor="lm")
        frames.append(np.asarray(img))
    save_gif(np.stack(frames), a.out, a.color, 1000 // a.fps)


if __name__ == "__main__":
    main()
