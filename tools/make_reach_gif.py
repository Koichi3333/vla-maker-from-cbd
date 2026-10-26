#!/usr/bin/env python3
"""Redraw the README GIF (docs/media/reach_three_positions.gif) from closed-loop evaluation trajectories. CPU, 0 CU.

    MUJOCO_GL=osmesa python3 tools/make_reach_gif.py work/v10b/evalV_box/h2/traj_2243.npz --smooth --grip 0.10 0 0.04
    MUJOCO_GL=osmesa python3 tools/make_reach_gif.py work/v10b/evalV_box/h2/traj_2237.npz --trials 5 --size 360 --end 170 \
        --out work/media/glass_to_mouth.gif      # a single trial (bringing the glass to the mouth)

Trajectories are the `traj_<clip>.npz` files written by `groot_eval.py --attach` (`traj`, `lo`, `conds`, `obj`, `attach`, `label`).
The README's three-position GIF uses `evalV_box/h2/traj_2243.npz` in `VLA_1_0/v10b/v10b_results.tgz` on Drive (VLM point + box push, 2026-10-11).
The first trial of each condition (original, shifted, mirrored) is placed side by side. Only the object's color is changed (shape and size stay as in
PROXY in vla/render_with_objects.py), and its position comes from `obj` in the npz, so it moves with the hand from the frame it is reached. No goal marker is drawn.
The side-by-side GIF at the top is tools/make_hero_gif.py (which uses the functions here).
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_COLOR = "1 0.1 0.75 1"                        # Magenta: distinct from the body's blue-grey and the feet's orange, and survives color reduction


def load_scene(traj, color=DEFAULT_COLOR, size=240):
    """Load the npz and build a Scene with only the object's color changed. -> (z, label, scene)"""
    sys.path.insert(0, str(ROOT / "vla"))
    import render_with_objects as rwo
    z = np.load(traj, allow_pickle=False)
    label = str(z["label"])
    rwo.PROXY[label] = (rwo.PROXY[label][0], color)
    from groot_eval import Scene
    return z, label, Scene(str(ROOT / "vla/humanoid.xml"), size=size, objects=True)


def in_hand(scene, traj, lo, side, grip):
    """Object position per frame in the hand's own frame: `grip` = (along the fingers, across, toward the palm) in m from the wrist.
    The finger bones run along the hand's x axis and curl toward its -z (the palm side), so (0.10, 0, 0.04) puts a glass of radius 0.035 in the curl of the fingers."""
    import mujoco
    q = scene.qpos(traj)
    q[:, 2] -= lo
    bid = scene.m.body(f"{side}_hand").id
    sign = 1.0 if side == "left" else -1.0                 # the right hand's fingers run along -x
    off = np.array([sign * grip[0], grip[1], -grip[2]])
    out = np.zeros((len(q), 3))
    for t in range(len(q)):
        scene.d.qpos[:] = q[t]
        mujoco.mj_kinematics(scene.m, scene.d)
        out[t] = scene.d.xpos[bid] + scene.d.xmat[bid].reshape(3, 3) @ off
    return out


def render_trial(z, label, scene, b, end=None, step=2, smooth=False, grip=None):
    """Render trial b. The object is placed at `obj` from the npz (the hand position once reached). -> (T, size, size, 3)
    With smooth, apply smooth_states from vla/smooth_motion.py (a forward-backward 1-euro filter) to the trajectory and re-attach the object to the smoothed hand once reached.
    With grip (see in_hand), the reached object is drawn in the curl of the fingers instead of at the wrist-to-finger midpoint (drawing only; the evaluation is unchanged)."""
    a0 = int(z["attach"][b])
    print(f"試行 {b}（{z['conds'][b]}）: 掴んだフレーム {a0}（-1 は掴まず）・手 {z['side'][b]}"
          + ("・均す" if smooth else ""), flush=True)
    traj, obj, lo = z["traj"][b], z["obj"][b], float(z["lo"][b])
    if smooth:
        from smooth_motion import smooth_states
        traj = smooth_states(traj)
        obj = obj.copy()
        if a0 >= 0:
            obj[a0:] = scene.hands(traj, lo)[str(z["side"][b])][a0:]
            if grip is not None:
                obj[a0:] = in_hand(scene, traj, lo, str(z["side"][b]), grip)[a0:]
    return scene.render(traj[:end], lo, (label, obj[:end]))[::step]


def save_gif(frames, out, color=DEFAULT_COLOR, duration=100):
    """Write (T, H, W, 3) as a GIF. GIFs have 256 colors and the object is small on screen, so a palette built from all frames drops the object's color.
    Add the object's pixels to weight it in."""
    from PIL import Image
    w = frames.shape[2]
    rgb = np.array([float(v) for v in color.split()[:3]]) * 255
    flat = frames.reshape(-1, 3)
    obj_px = flat[np.linalg.norm(flat.astype(float) - rgb, axis=1) < 120]
    parts = [frames[::5].reshape(-1, w, 3)]
    if len(obj_px):
        parts.append(np.resize(obj_px, (40 * w, 3)).reshape(40, w, 3))
    pal = Image.fromarray(np.concatenate(parts).astype(np.uint8)).quantize(
        colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=duration, loop=0, optimize=True)
    left = sum(np.any(np.linalg.norm(np.asarray(im.convert("RGB")).reshape(-1, 3).astype(float) - rgb, axis=1) < 120) for im in imgs)
    print(f"{out}  {len(imgs)} コマ・{out.stat().st_size / 1e6:.2f} MB・物の色が残ったコマ {left}/{len(imgs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", help="traj_<clip>.npz")
    ap.add_argument("--out", default=str(ROOT / "docs/media/reach_three_positions.gif"))
    ap.add_argument("--color", default=DEFAULT_COLOR, help="物の rgba")
    ap.add_argument("--size", type=int, default=240, help="1 場面の幅・高さ（px）")
    ap.add_argument("--step", type=int, default=2, help="何フレームおきに使うか（評価は 20fps。既定で 10fps）")
    ap.add_argument("--trials", type=int, nargs="+", default=None,
                    help="並べる試行の番号（npz の 0 始まり）。既定は元・ずらし・反対側の各 1 回目")
    ap.add_argument("--end", type=int, default=None, help="このフレームで切る（評価のフレーム、20fps）")
    ap.add_argument("--grip", type=float, nargs=3, default=None, metavar=("ALONG", "ACROSS", "PALM"),
                    help="draw the reached object in the curl of the fingers: offset from the wrist in the hand's frame (m), e.g. 0.10 0 0.04")
    ap.add_argument("--smooth", action="store_true", help="軌道を vla/smooth_motion.py で均してから描く（チャンクの継ぎ目のカクつきを落とす）")
    a = ap.parse_args()

    z, label, scene = load_scene(a.traj, a.color, a.size)
    conds = [str(c) for c in z["conds"]]
    trials = a.trials or [conds.index(c) for c in ("orig", "shift", "mirror")]   # Default: the first trial of each condition (same as mp4_object)
    frames = np.concatenate([render_trial(z, label, scene, b, a.end, a.step, a.smooth, a.grip) for b in trials], axis=2)
    save_gif(frames, a.out, a.color, 1000 * a.step // 20)


if __name__ == "__main__":
    main()
