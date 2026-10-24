#!/usr/bin/env python3
"""Turn a 117-d state (training data, GR00T output) back into CBD that the G1 adapter can read.

    # from GR00T output (groot_eval.py's rollouts.npz)
    python3 state_to_cbd.py --rollouts work/b2/results/eval/ckpt10000/rollouts.npz \\
        --text walks_forward --inits 0 1 2 --out work/g1_gen
    # check: from a training parquet (compare with the existing work/g1/<clip>)
    python3 state_to_cbd.py --parquet work/lerobot_humanml3d/data/chunk-000/episode_000276.parquet \\
        --out /tmp/check

The state holds local rotations relative to the T-pose (the default of `humanml3d_to_cbd.py`), so it lacks what the G1 adapter needs:
**the torso's neutral rest pose (--torso-rest neutral) and pose_canonical**. Rather than re-mapping the rotation convention
directly, **FK turns it back into 22 joint positions, which go through the same `convert()` as the real data**.
That way what reaches the G1 follows the same path as the real data (`ws/cbd_g1_src`).

Skeleton dimensions are measured from the source clip's new_joints (`measure_offsets`). HumanML3D shares bone lengths
across all clips, but the direction of offsets in the parent frame differs slightly per clip, and building with another clip's dimensions
puts the toes off by up to 20 mm (0.026 mm with its own). Generated output uses the dimensions of "the held-out clip
the initial pose was taken from".

Output: <out>/<name>/04_behavior_dataset/ (CBD). Then
    python3 work/cbd_ref/cbd_adapter_groot_g1.py --dataset <out>/<name> --output <out>/<name>/out
    MUJOCO_GL=osmesa python3 g1_to_mujoco.py --render <out>/*/out/06_groot_g1/motion.npz
"""
import argparse
import json
import os
import pathlib

import numpy as np

from cbd_to_lerobot import BONES as ORDER          # state layout
from humanml3d_to_cbd import (BONE_AT_JOINT, BONES, PARENT, convert, joint_angles,
                              measure_offsets, pose_canonical, write_cbd)
from rotation import qmul, qrot, sixd_to_quat

HERE = pathlib.Path(__file__).resolve().parent
WORK = pathlib.Path(os.environ.get("VLA_WORK", HERE.parent / "work"))   # where outputs go (not tracked by git)
NEW_JOINTS = WORK / "humanml3d/subset/new_joints"
DATASET = WORK / "lerobot_humanml3d"


def globals_from_local(loc):
    glob = {}
    for name, _, _, parent in BONES:               # order with parents first
        glob[name] = loc[name] if parent is None else qmul(glob[parent], loc[name])
    return glob


def source_npy(ep, dataset=DATASET):
    """Training data episode_index -> the source new_joints .npy."""
    for l in (pathlib.Path(dataset) / "meta/episodes.jsonl").read_text().splitlines():
        e = json.loads(l)
        if e["episode_index"] == ep:
            return NEW_JOINTS / (e["source_clip"].split("/")[-1] + ".npy")
    raise KeyError(ep)


def skeleton_offsets(ref):
    """Measure each joint's rest offset in its parent's frame (T-pose reference) from ref's new_joints."""
    pos = np.load(ref).astype(np.float64)
    loc, _, _ = convert(pos)
    off, resid = measure_offsets(pos, globals_from_local(loc))
    return off, resid


def state_to_pos(state, off):
    """(T,117) -> (T,22,3)。"""
    T = len(state)
    Q = sixd_to_quat(np.asarray(state[:, 3:], np.float64).reshape(T, len(ORDER), 6))
    glob = globals_from_local({b: Q[:, i] for i, b in enumerate(ORDER)})
    pos = np.zeros((T, 22, 3))
    pos[:, 0] = state[:, :3]
    for c in range(1, 22):
        pos[:, c] = pos[:, PARENT[c]] + qrot(glob[BONE_AT_JOINT[PARENT[c]]],
                                              np.broadcast_to(off[c], (T, 3)))
    return pos


def write_g1_cbd(out, name, category, caption, pos, fps=20.0):
    loc, hips, diag = convert(pos, "neutral", "neutral")
    write_cbd(out / name / "04_behavior_dataset", name, category, caption, loc, hips, diag,
              joint_angles(pos), fps, pose_canonical(pos), "neutral")
    return diag


def main():
    global NEW_JOINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", help="groot_eval.py の rollouts.npz")
    ap.add_argument("--text", help="rollouts.npz のキー（walks_forward など）")
    ap.add_argument("--inits", type=int, nargs="*", default=[0], help="何番目の初期姿勢を出すか")
    ap.add_argument("--parquet", help="学習データの episode parquet（検証用）")
    ap.add_argument("--dataset", default=str(DATASET),
                    help="episode_index → 元クリップを引くデータセット（全件なら work/lerobot_standing_all）")
    ap.add_argument("--new-joints", default=str(NEW_JOINTS),
                    help="元クリップの new_joints（全件なら work/humanml3d/new_joints）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="", help="出力名の接頭辞（ckpt10000 など）")
    a = ap.parse_args()

    NEW_JOINTS = pathlib.Path(a.new_joints)
    out = pathlib.Path(a.out)
    jobs = []
    if a.parquet:
        import pyarrow.parquet as pq
        col = pq.read_table(a.parquet, columns=["observation.state"]).column(0).combine_chunks()
        st = col.values.to_numpy().reshape(len(col), -1)
        ep = int(pathlib.Path(a.parquet).stem.split("_")[-1])
        jobs.append((a.tag + pathlib.Path(a.parquet).stem, "check", "", st, source_npy(ep, a.dataset)))
    if a.rollouts:
        z = np.load(a.rollouts)
        eps = z["init_eps"]
        for k in a.inits:
            st = z[a.text][k]
            jobs.append((f"{a.tag}{a.text}_init{int(eps[k]):03d}", a.text,
                         "a person " + a.text.replace("_", " "), st,
                         source_npy(int(eps[k]), a.dataset)))
    for name, cat, cap, st, ref in jobs:
        off, _ = skeleton_offsets(ref)
        pos = state_to_pos(st, off)
        diag = write_g1_cbd(out, name, cat, cap, pos)
        msg = f"{name}: {len(st)}f  convert の FK 往復 {diag['fk_max_m'] * 1000:.4f} mm"
        if cat == "check":
            gt = np.load(ref).astype(np.float64)[:len(pos)]
            msg += f"  / 元の関節位置との差 最大 {np.abs(pos - gt).max() * 1000:.3f} mm"
        print(msg)
    (out / "index.json").write_text(json.dumps([j[0] for j in jobs], indent=1))


if __name__ == "__main__":
    main()
