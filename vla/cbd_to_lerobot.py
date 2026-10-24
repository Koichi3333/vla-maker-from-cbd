#!/usr/bin/env python3
"""Bake CBD bone rotations into a LeRobot dataset (parquet). For training.

CBD's frames.jsonl is a layer for exchange and provenance, not something to put in a training loop.
Measured: loading 421 clips / 74,907 frames takes 5.4 s, 124 times slower than arrays.
That cost is paid every epoch, so for training the data is re-baked into arrays.

The action space is **CBD bone rotations**. It does not depend on the output embodiment,
so retargeting to a G1 or anything else later needs no new training data. Training on qpos or
G1 joint angles would give weights that cannot be carried to other targets.

Quaternions are not regressed directly. Their sign flips between adjacent frames in 0.38% of cases
(double cover), which makes the loss discontinuous. They are converted to the continuous 6D representation
that stacks the first two columns of the rotation matrix (Zhou et al. 2019).
"""
import argparse
import json
import pathlib

import numpy as np

from rotation import normalize, qinv, qmul, quat_to_6d, sixd_to_quat

# Layout of the 117 dims. modality.json names these ranges.
TORSO = ["spine", "chest", "neck", "head"]
LARM = ["left_shoulder", "left_upper_arm", "left_lower_arm", "left_hand"]
RARM = ["right_shoulder", "right_upper_arm", "right_lower_arm", "right_hand"]
LLEG = ["left_upper_leg", "left_lower_leg", "left_foot"]
RLEG = ["right_upper_leg", "right_lower_leg", "right_foot"]
BONES = ["hips"] + TORSO + LARM + RARM + LLEG + RLEG          # 19
GROUPS = [("root_translation", 0, 3), ("hips", 3, 9), ("torso", 9, 33),
          ("left_arm", 33, 57), ("right_arm", 57, 81),
          ("left_leg", 81, 99), ("right_leg", 99, 117)]
DIM = 3 + len(BONES) * 6


def mirror_spec():
    """Express left-right mirroring as (permutation, sign). At training time x[perm] * sign mirrors a frame.

    In CBD canonical, +X is the subject's left, so mirroring is the reflection M = diag(-1,1,1).
    A rotation becomes R -> M R M; in the 6D representation column 1 becomes (x,-y,-z) and column 2 (-x,y,z).
    Left and right bones are swapped. Mirrored data is not stored on disk; it is made at training time.
    """
    swap = {b: (b.replace("left", "right") if b.startswith("left")
                else b.replace("right", "left") if b.startswith("right") else b)
            for b in BONES}
    perm = np.arange(DIM)
    sign = np.ones(DIM, np.float32)
    sign[0] = -1.0                                          # root x
    for i, b in enumerate(BONES):
        s, d = 3 + i * 6, 3 + BONES.index(swap[b]) * 6
        perm[d:d + 6] = np.arange(s, s + 6)
        sign[d:d + 6] = [1, -1, -1, -1, 1, 1]
    return perm, sign


def load_npy(root, i):
    """Build the 117 dims directly from HumanML3D .npy (without going through frames.jsonl).

    There is no point exchanging every clip in human-readable form, so training uses this path.
    """
    from humanml3d_to_cbd import convert
    pos = np.load(root / "new_joints" / f"{i}.npy").astype(np.float64)
    loc, hips, _ = convert(pos)
    Q = np.stack([loc[b] for b in BONES], 1)
    state = np.concatenate(
        [hips.astype(np.float32),
         quat_to_6d(Q).reshape(len(Q), -1).astype(np.float32)], 1)
    return state, Q


def load_clip(d):
    q = {b: [] for b in BONES}
    root, caps = [], []
    for line in open(d / "timeline" / "frames.jsonl", encoding="utf-8"):
        f = json.loads(line)
        h = f["human"]
        for b in BONES:
            x, y, z, w = h["bone_rotations_xyzw"][b]
            q[b].append([w, x, y, z])
        root.append(h["hips_position"])
        caps.append((f["caption"] or {}).get("en") or "")
    # frames.jsonl is rounded to 5 digits, so the norm is off from 1 by about 6e-6.
    # A rotation matrix built from a non-unit quaternion is not orthogonal, so normalize on read.
    Q = normalize(np.stack([np.asarray(q[b], np.float64) for b in BONES], 1))
    v = [np.asarray(root, np.float32), quat_to_6d(Q).reshape(len(Q), -1).astype(np.float32)]
    return np.concatenate(v, 1), caps, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["cbd", "humanml3d"], default="cbd",
                    help="humanml3d は .npy から直接焼く（全件モード）")
    ap.add_argument("--cbd", default="work/cbd_humanml3d")
    ap.add_argument("--humanml3d-root", default="work/humanml3d")
    ap.add_argument("--out", default="work/lerobot_humanml3d")
    ap.add_argument("--robot-type", default="cbd_humanoid")
    ap.add_argument("--fps", type=float, default=20.0)
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = pathlib.Path(a.out)
    if a.source == "cbd":
        cbd = pathlib.Path(a.cbd)
        clips = sorted(p.parent.parent for p in cbd.glob("*/*/timeline/frames.jsonl"))
    else:
        hroot = pathlib.Path(a.humanml3d_root)
        idx = json.load(open(hroot / "standing.json", encoding="utf-8"))
        clips = idx["ids"]
        print(f"全件モード: {len(clips)} clips（frames.jsonl は経由しない）")
    (out / "meta").mkdir(parents=True, exist_ok=True)

    tasks, task_id = [], {}
    episodes, chunks = [], {}
    n_frames, worst = 0, 0.0
    # Holding every state for all clips gets close to 1 GB, so statistics are accumulated incrementally
    cnt = 0
    ssum = np.zeros(DIM, np.float64)
    ssq = np.zeros(DIM, np.float64)
    smin = np.full(DIM, np.inf)
    smax = np.full(DIM, -np.inf)
    for ep, d in enumerate(clips):
        if a.source == "cbd":
            state, caps, quats = load_clip(d)
            cap = caps[0] or d.name
            src = f"{d.parent.name}/{d.name}"
        else:
            state, quats = load_npy(hroot, d)
            cap = idx["meta"][d]["caption"] or d
            src = d
        T = len(state)
        if a.source == "humanml3d" and ep % 500 == 0:
            print(f"  {ep:6d}/{len(clips)}")
        # action is the next frame's pose (the last frame stays put)
        action = np.concatenate([state[1:], state[-1:]], 0)

        # Convert 6D back to quaternions and measure the angle error (round-trip check of the representation).
        # arccos(dot) is ill-conditioned when dot is near 1, so measure with arcsin
        # of the relative rotation's vector part (well-conditioned near 0).
        back = sixd_to_quat(state[:, 3:].reshape(T, len(BONES), 6).astype(np.float64))
        back *= np.sign((back * quats).sum(-1, keepdims=True))
        rel = qmul(qinv(quats), back)
        worst = max(worst, float(np.degrees(
            2 * np.arcsin(np.linalg.norm(rel[..., 1:], axis=-1).clip(0, 1))).max()))

        if cap not in task_id:
            task_id[cap] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": cap})
        ti = task_id[cap]

        ch = ep // 1000
        cd = out / "data" / f"chunk-{ch:03d}"
        cd.mkdir(parents=True, exist_ok=True)
        chunks[ch] = chunks.get(ch, 0) + 1
        pq.write_table(pa.table({
            "observation.state": list(state),
            "action": list(action),
            "timestamp": np.arange(T) / a.fps,
            "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, ep, np.int64),
            "index": np.arange(n_frames, n_frames + T, dtype=np.int64),
            "task_index": np.full(T, ti, np.int64),
            "annotation.human.task_description": np.full(T, ti, np.int64),
            "next.done": np.arange(T) == T - 1,
        }), cd / f"episode_{ep:06d}.parquet")
        episodes.append({"episode_index": ep, "tasks": [cap], "length": T,
                         "source_clip": src})
        n_frames += T
        x = state.astype(np.float64)
        cnt += T
        ssum += x.sum(0)
        ssq += (x * x).sum(0)
        smin = np.minimum(smin, x.min(0))
        smax = np.maximum(smax, x.max(0))

    mean = ssum / cnt
    std = np.sqrt(np.maximum(ssq / cnt - mean * mean, 0.0))
    stat = dict(mean=mean.tolist(), std=std.tolist(),
                max=smax.tolist(), min=smin.tolist())
    json.dump({"observation.state": stat, "action": stat},
              open(out / "meta" / "stats.json", "w"))

    # Record the dimensions that never move, so training can mask them.
    # The hands are unobserved (identity rotation), and elbows and knees are pure hinges, so the component along the axis is constant
    # (the latter also shows that the hinge constraint works).
    dead = np.where(std < 1e-6)[0]
    const = {}
    for i in dead:
        b = "root" if i < 3 else BONES[(i - 3) // 6]
        const.setdefault(b, []).append(int(i))

    feats = {
        "observation.state": {"dtype": "float32", "shape": [DIM],
                              "names": [f"{g}" for g, _, _ in GROUPS]},
        "action": {"dtype": "float32", "shape": [DIM],
                   "names": [f"{g}" for g, _, _ in GROUPS]},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "annotation.human.task_description": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
    }
    json.dump({
        "codebase_version": "v2.0", "robot_type": a.robot_type,
        "total_episodes": len(clips), "total_frames": n_frames,
        "total_tasks": len(tasks), "total_chunks": len(chunks),
        "chunks_size": 1000, "fps": a.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": feats,
        "cbd": {
            "action_space": "CBD bone rotations, 6D continuous representation",
            "layout": {g: [s, e] for g, s, e in GROUPS},
            "bones": BONES,
            "rest_pose": "T-pose",
            "coordinate_system": "y-up, right-handed, person faces +z",
            "constant_dims": const,
            "effective_dim": DIM - len(dead),
            "note": "回転の素性は元の CBD の runtime_info.json の twist_source を見ること。"
                    "前腕の回内は推定値、手は観測不能で恒等回転。"
                    "constant_dims は分散ゼロの次元（手は恒等回転、肘と膝は純ヒンジ）。",
        },
    }, open(out / "meta" / "info.json", "w"), indent=1, ensure_ascii=False)

    for name, rows in (("tasks", tasks), ("episodes", episodes)):
        with open(out / "meta" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mod = {k: {g: {"start": s, "end": e} for g, s, e in GROUPS}
           for k in ("state", "action")}
    mod["annotation"] = {"human.task_description": {"original_key":
                         "annotation.human.task_description"}}
    json.dump(mod, open(out / "meta" / "modality.json", "w"), indent=1)

    perm, sign = mirror_spec()
    json.dump({"note": "x_mirrored = x[perm] * sign。左右反転はディスクに持たず"
                       "学習時に作る（HumanML3D の M* と同義）。",
               "perm": perm.tolist(), "sign": sign.tolist()},
              open(out / "meta" / "mirror.json", "w"), ensure_ascii=False)

    print(f"{len(clips)} episodes / {n_frames} frames / {len(tasks)} tasks -> {out}")
    print(f"6D 表現の往復誤差: 最大 {worst:.2e} 度")


if __name__ == "__main__":
    main()
