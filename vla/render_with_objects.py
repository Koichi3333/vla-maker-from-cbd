#!/usr/bin/env python3
"""Render third-person views with objects (256x256, 20 fps) into a video LeRobot dataset (V-4).

Camera, skeleton and ground correction are the same as stage B (`humanml3d_to_video.py` / `cbd_to_mujoco.py`). The only difference is one added object proxy.

**The object is placed relative to the rendered hand** (from the states in `objects/episode_XXXXXX.json`):

| State | Placement |
| --- | --- |
| `held_left` / `held_right` | that hand (midpoint of the wrist and the base of the middle finger) |
| `held_both` | midpoint of both hands |
| `rest` | **where the hand was when it last let go** (if not grasped yet, where it will first be grasped) |
| `hidden` | off screen (flying away from the hand; a `tracked_2d` stretch with no 3D) |

CBD's `position_canonical` (3D with the body as the ruler) is not used. The rendered skeleton differs in size from the recorded person,
and canonical's absolute scale is also 25% short (F-5), so placed as is the object does not land in the hand. Placing it relative to the hand
guarantees on screen that "while held it is in the hand" and "after release it stays put". **No table is drawn** (the object stays in the air).

    MUJOCO_GL=osmesa python3 vla/render_with_objects.py --dataset work/lerobot_video --workers 2 --preview work/v4_preview
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
B = HERE
sys.path.insert(0, str(B))

# label -> proxy shape (MuJoCo geom; sizes in m, close to the real thing). Colors distinguishable from the body (pale blue-grey) and feet (orange)
PROXY = {
    "cup": ('type="cylinder" size="0.035 0.05"', "0.05 0.55 0.65 1"),
    "bottle": ('type="cylinder" size="0.033 0.10"', "0.15 0.3 0.9 1"),
    "bowl": ('type="cylinder" size="0.07 0.03"', "0.8 0.25 0.55 1"),
    "ball": ('type="sphere" size="0.034"', "0.85 0.95 0.2 1"),
    "book": ('type="box" size="0.10 0.075 0.015"', "0.7 0.35 0.15 1"),
    "phone": ('type="box" size="0.036 0.075 0.004"', "0.08 0.08 0.1 1"),
}
AWAY = np.array([0.0, 0.0, 100.0])              # where hidden objects go (outside the camera, at a height that does not affect the ground correction)


def scaled(geom, scale):
    """Scale the size in a PROXY geom string by scale (for V-7a's "drawn larger" control)."""
    if scale == 1.0:
        return geom
    import re
    return re.sub(r'size="([^"]*)"', lambda mt: 'size="' + " ".join(f"{float(v) * scale:g}" for v in mt.group(1).split()) + '"', geom)


def xml_with_objects(xml, scale=1.0):
    """Add a mocap body per label to humanoid.xml's worldbody (no joints, so the qpos size does not change)."""
    s = pathlib.Path(xml).read_text(encoding="utf-8")
    add = "".join(f'<body name="obj_{k}" mocap="true" pos="0 0 100"><geom name="obj_{k}" {scaled(g, scale)} rgba="{c}" '
                  f'contype="0" conaffinity="0"/></body>' for k, (g, c) in PROXY.items())
    return s.replace("</worldbody>", add + "</worldbody>", 1)


_W = {}


def _init(xml, size, scale=1.0):
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml_with_objects(xml, scale))
    mocap, hand = body_ids(m)
    _W.update(m=m, d=mujoco.MjData(m), r=mujoco.Renderer(m, size, size), size=size, mocap=mocap, hand=hand)


def body_ids(m):
    """Object mocap indices, and the body indices of both hands (wrist, base of the middle finger)."""
    import mujoco
    mocap = {k: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{k}")] for k in PROXY}
    hand = {s: (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand"),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_middle")) for s in ("left", "right")}
    return mocap, hand


def anchors(m, d, hand, q):
    """Positions of both hands per frame (midpoint of the wrist and the base of the middle finger, MuJoCo world coordinates)."""
    import mujoco
    out = {s: np.zeros((len(q), 3)) for s in hand}
    for t in range(len(q)):
        d.qpos[:] = q[t]
        mujoco.mj_forward(m, d)
        for s, (a, b) in hand.items():
            out[s][t] = (d.xpos[a] + d.xpos[b]) / 2
    return out


def object_track(mode, hands):
    """Sequence of states -> sequence of object positions ((T,3), AWAY when hidden)."""
    T = len(mode)
    held = [m.startswith("held_") for m in mode]
    pos = np.tile(AWAY, (T, 1))
    for t, m in enumerate(mode):
        if m.startswith("held_"):
            h = m[5:]
            pos[t] = (hands["left"][t] + hands["right"][t]) / 2 if h == "both" else hands[h][t]
    first = next((t for t in range(T) if held[t]), None)
    last = None
    for t, m in enumerate(mode):
        if held[t]:
            last = t
        elif m == "rest":
            ref = last if last is not None else first
            if ref is not None:
                pos[t] = pos[ref]
    return pos


def frames_with_object(m, d, r, mocap, q, pos, label):
    """Draw with the object placed, using the same camera as cbd_to_mujoco.frames. pos is (T,3); if label is None no object is shown."""
    import mujoco
    for t in range(len(q)):
        d.qpos[:] = q[t]
        for k, i in mocap.items():
            d.mocap_pos[i] = pos[t] if (k == label) else AWAY
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1
        cam.fixedcamid = -1
        cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 135
        cam.lookat[:] = [q[t, 0], q[t, 1], 0.9]
        r.update_scene(d, cam)
        yield r.render()


def _render_one(job):
    from cbd_to_mujoco import ground
    from humanml3d_to_video import load_state, state_to_qpos
    from lerobot_add_video import encode
    ep, parquet, objp, dst, fps, preview = job
    m, d = _W["m"], _W["d"]
    for i in _W["mocap"].values():                     # move objects away before the ground correction (mocap bodies have geoms too)
        d.mocap_pos[i] = AWAY
    q = state_to_qpos(load_state(parquet))
    lift = -ground(m, d, q)
    obj = json.loads(pathlib.Path(objp).read_text()) if objp else None
    pos = object_track(obj["mode"], anchors(m, d, _W["hand"], q)) if obj else np.tile(AWAY, (len(q), 1))
    label = obj["label"] if obj else None
    frames = [f.copy() for f in frames_with_object(m, d, _W["r"], _W["mocap"], q, pos, label)]
    encode(pathlib.Path(dst), iter(frames), _W["size"], fps)
    out = dict(ep=ep, T=len(q), lift=lift, label=label,
               hidden=int(sum(1 for x in (obj["mode"] if obj else []) if x == "hidden")),
               same=int(sum(np.array_equal(a, b) for a, b in zip(frames, frames[1:]))))
    if preview:                                        # one still each for viewing: held, after release, and the first frame
        import imageio
        pathlib.Path(preview).mkdir(parents=True, exist_ok=True)
        mode = obj["mode"] if obj else []
        pick = sorted({0, len(q) // 2, len(q) - 1} | {next((t for t, x in enumerate(mode) if x.startswith("held")), 0)})
        imageio.imwrite(f"{preview}/episode_{ep:06d}.png", np.hstack([frames[t] for t in pick]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="work/lerobot_video")
    ap.add_argument("--xml", default=str(HERE / "humanoid.xml"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "osmesa"), choices=["egl", "osmesa", "glfw"])
    ap.add_argument("--preview", default=None, help="1本あたり数枚の静止画を書く（目視用）")
    ap.add_argument("--object-scale", type=float, default=1.0,
                    help="物の代理形状の寸法の倍率（V-7a で「見えていない」と出たときの大きく描いた版。既定 1 = 従来どおり）")
    ap.add_argument("--only-missing", action="store_true",
                    help="mp4 が既にある episode は描かない（作り直したデータセットで、変わった本だけ描く）")
    a = ap.parse_args()
    os.environ["MUJOCO_GL"] = a.gl
    from lerobot_add_video import VIDEO_KEY, VIDEO_PATH

    ds = pathlib.Path(a.dataset)
    info = json.loads((ds / "meta/info.json").read_text())
    eps = [json.loads(line) for line in (ds / "meta/episodes.jsonl").read_text().splitlines() if line]
    jobs = []
    for e in eps:
        i = e["episode_index"]
        objp = ds / "objects" / f"episode_{i:06d}.json"
        jobs.append((i, str(ds / info["data_path"].format(episode_chunk=0, episode_index=i)),
                     str(objp) if objp.exists() else None,
                     str(ds / VIDEO_PATH.format(episode_chunk=0, video_key=VIDEO_KEY, episode_index=i)),
                     info["fps"], a.preview))
    if a.only_missing:
        n_all = len(jobs)
        jobs = [j for j in jobs if not pathlib.Path(j[3]).exists()]
        print(f"mp4 が無い {len(jobs)} / {n_all} 本だけ描く")
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.xml, a.size, a.object_scale)) as pool:
        res = sorted(pool.imap_unordered(_render_one, jobs), key=lambda r: r["ep"])
    for r in res:
        print(f"  {r['ep']:3d}  {r['T']:4d}f  {r['label'] or '-':6s}  隠した {r['hidden']:3d}f  "
              f"接地 {r['lift']:+.3f} m  静止フレーム {r['same']}")
    still = [r["ep"] for r in res if r["same"] >= 0.5 * (r["T"] - 1)]
    assert not still, f"静止画になっている: {still}"

    info["total_videos"] = len(eps)
    info["video_path"] = VIDEO_PATH
    info["features"][VIDEO_KEY] = {
        "dtype": "video", "shape": [a.size, a.size, 3], "names": ["height", "width", "channels"],
        "info": {"video.height": a.size, "video.width": a.size, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                 "video.is_depth_map": False, "video.fps": info["fps"], "video.channels": 3, "has_audio": False}}
    info["cbd"]["video_pattern"] = "render_with_objects"
    info["cbd"]["object_scale"] = a.object_scale
    (ds / "meta/info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    mod = json.loads((ds / "meta/modality.json").read_text())
    mod["video"] = {"front_view": {"original_key": VIDEO_KEY}}
    (ds / "meta/modality.json").write_text(json.dumps(mod, indent=1))
    print(f"{len(res)} 本 -> {ds}/videos/")


if __name__ == "__main__":
    main()
