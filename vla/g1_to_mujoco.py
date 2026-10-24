#!/usr/bin/env python3
"""Play back the joint angles produced by CBD's GR00T G1 adapter on MuJoCo's G1.

`adapters/isaac_sim_groot_g1/` writes USD for Isaac Sim, but the retargeting result
itself (43 joint angles and the pelvis pose in `motion.npz`) drops straight into
MuJoCo Menagerie's `unitree_g1/g1_with_hands.xml`. All 43 joint names
match. **Choosing the G1 and using Isaac Sim are independent.**

Playback is kinematic. qpos is assigned every frame and mj_forward is called; gravity and
contact play no part. The adapter itself says in adapter_report.json that
"this is not a dynamically feasible trajectory".
"""
import argparse
import pathlib

import numpy as np


def build_qpos(model, npz):
    import mujoco
    names = [str(n) for n in npz["joint_names"]]
    ang = npz["joint_angles_rad"].astype(np.float64)
    pos = npz["pelvis_position_robot"].astype(np.float64)
    quat = npz["pelvis_orientation_wxyz"].astype(np.float64)
    # Joint name -> qpos index. Every G1 joint is a 1-DoF hinge, so one element each.
    adr = {}
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if n:
            adr[n] = model.jnt_qposadr[j]
    missing = [n for n in names if n not in adr]
    if missing:
        raise SystemExit(f"MuJoCo 側に無い関節: {missing[:5]}")
    q = np.zeros((len(ang), model.nq))
    q[:, 0:3] = pos
    q[:, 3:7] = quat
    for k, n in enumerate(names):
        q[:, adr[n]] = ang[:, k]
    return q


def ground(model, data, q):
    """Shift the whole clip up or down by a constant so the lowest point of the feet sits on the floor."""
    import mujoco
    g = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
    rad = np.array([model.geom_size[i].max() for i in g])
    lo = np.inf
    for t in range(len(q)):
        data.qpos[:] = q[t]
        mujoco.mj_forward(model, data)
        lo = min(lo, float((data.geom_xpos[g, 2] - rad).min()))
    q[:, 2] -= lo
    return lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", nargs="+", help="アダプタが出した motion.npz")
    ap.add_argument("--xml", required=True,
                    help="CBD リポジトリの G1 の scene（床と光源を含む scene_with_hands.xml）")
    ap.add_argument("--out", default="work/g1_preview")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--size", default="640x480")
    a = ap.parse_args()

    import mujoco
    model = mujoco.MjModel.from_xml_path(a.xml)
    data = mujoco.MjData(model)
    w, h = (int(v) for v in a.size.split("x"))
    # Reuse one Renderer. Creating one per clip exhausts OSMesa's GL contexts,
    # and from the second clip on it keeps returning the same image (the video becomes a still).
    renderer = mujoco.Renderer(model, h, w) if a.render else None

    for src in a.motion:
        src = pathlib.Path(src)
        npz = np.load(src, allow_pickle=True)
        q = build_qpos(model, npz)
        off = ground(model, data, q)
        fps = float(npz["fps"][0])
        name = src.parent.parent.parent.name
        out = pathlib.Path(a.out) / name
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "motion.npz", qpos=q.astype(np.float32),
                 fps=np.array([fps], np.float32),
                 timestamp_sec=npz["timestamp_sec"])
        note = ""
        if a.render:
            import imageio
            r = renderer
            frames = []
            for t in range(len(q)):
                data.qpos[:] = q[t]
                mujoco.mj_forward(model, data)
                # Create the camera anew every frame. update_scene rewrites the MjvCamera,
                # so reusing it mixes in shots looking up from under the floor.
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(cam)
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.trackbodyid = -1
                cam.fixedcamid = -1
                cam.distance, cam.elevation, cam.azimuth = 2.6, -10, 135
                cam.lookat[:] = [q[t, 0], q[t, 1], 0.6]
                r.update_scene(data, cam)
                frames.append(r.render().copy())
            imageio.mimwrite(out / "preview.mp4", frames, fps=fps, quality=8,
                             macro_block_size=1)
            note = f"  -> {out / 'preview.mp4'}"
        print(f"{name:18s} {len(q):4d}f  接地補正 {-off:+.3f}m{note}")


if __name__ == "__main__":
    main()
