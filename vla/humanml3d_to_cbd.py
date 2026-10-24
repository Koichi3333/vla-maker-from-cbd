#!/usr/bin/env python3
"""Convert HumanML3D joint positions (new_joints) to CBD bone_rotations.

HumanML3D distributes only the **positions** of 22 joints; no rotations are included.
The original `skeleton.py` inverse_kinematics_np uses qbetween(u, v), i.e. the minimal rotation
(zero twist), so even the rot_data in new_joint_vecs has no twist.

When recovering rotations from positions, this script **chooses each parent's twist so that the child joint
is an anatomical hinge**. Upper-arm twist (shoulder internal/external rotation) and thigh twist are uniquely
determined by the elbow and knee flexion planes, so they are geometric facts, not estimates.

The hands (beyond the wrist) do not exist in HumanML3D (its 22 joints end at the wrist). So
forearm pronation/supination cannot be observed. By default it is filled with "neutral pronation":
assuming the palm normal matches the elbow's flexion axis, the hand orientation is derived from the arm's configuration.
Values filled under this assumption are marked with twist_source in frames.jsonl to keep their provenance.
"""
import argparse
import csv
import datetime as dt
import json
import pathlib

import numpy as np

from rotation import (mat_to_quat, normalize as _n, qaxis, qbetween,
                      qinv, qmul, qrot)

# ---------------------------------------------------------------- Skeleton definition
# SMPL/t2m 22 joints. +X is the subject's left, +Y up, +Z forward (checked on the data).
J = dict(pelvis=0, l_hip=1, r_hip=2, spine1=3, l_knee=4, r_knee=5, spine2=6,
         l_ankle=7, r_ankle=8, spine3=9, l_foot=10, r_foot=11, neck=12,
         l_collar=13, r_collar=14, head=15, l_shoulder=16, r_shoulder=17,
         l_elbow=18, r_elbow=19, l_wrist=20, r_wrist=21)

PARENT = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8,
          12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17,
          20: 18, 21: 19}

# Direction of each joint from its parent in the T-pose (CBD requires a "T-pose rest" reference).
# HumanML3D's internal t2m_raw_offsets has the arms down, so it is not used.
REST_DIR = {
    1: (1, 0, 0),  2: (-1, 0, 0),  3: (0, 1, 0),
    4: (0, -1, 0), 5: (0, -1, 0),  6: (0, 1, 0),
    7: (0, -1, 0), 8: (0, -1, 0),  9: (0, 1, 0),
    10: (0, 0, 1), 11: (0, 0, 1), 12: (0, 1, 0),
    13: (1, 0, 0), 14: (-1, 0, 0), 15: (0, 1, 0),
    16: (1, 0, 0), 17: (-1, 0, 0),
    18: (1, 0, 0), 19: (-1, 0, 0),
    20: (1, 0, 0), 21: (-1, 0, 0),
}


# Torso directions in the "neutral pose" (pelvis frame, mean of 421 standing clips).
#
# The SMPL spine is anatomically curved: even standing upright, each segment leans 16-20 degrees
# from vertical (an S-curve, so they cancel and the net trunk lean stays at 9 degrees).
# Defining the T-pose as "the whole spine vertical" puts this curvature into the rotations
# as a constant offset. Arms and legs are solved by direction so no harm there, but **consumers that map rotations
# straight to joint angles** (such as the G1's waist) see an apparent 40-degree forward lean. Measured: the G1's
# waist_pitch sat at its limit in 197 of 199 frames.
#
# --torso-rest neutral uses this as the rest pose. Upright becomes the identity rotation,
# so it can go straight to a robot's waist. The default stays tpose (literally what the CBD spec says,
# and rigs with a vertical spine, such as MuJoCo's humanoid.xml, assume it).
# Pelvis frame correction. up is defined as pelvis->spine1, so the sacral tilt
# (16.8 degrees in neutral) becomes the "pelvis orientation" as is. Consumers that reproduce directions, like MuJoCo,
# cancel it in the torso so no harm there, but passing the root orientation straight to a robot's pelvis
# leans it forward all the time, and the waist joint cannot fully undo it. Rotate by a constant so the pelvis is upright in neutral.
NEUTRAL_HIPS = np.array([0.989264, 0.145869, -0.0, 0.008887])

NEUTRAL_REST = {
    3: (0.0078, 0.93621, -0.35135),
    6: (0.01377, 0.97817, 0.20736),
    9: (-0.05314, 0.95516, 0.29128),
    12: (-0.06524, 0.99496, -0.07608),
    15: (0.02284, 0.80166, 0.59734),
}

# CBD 19 bones: (name, joint carrying the rotation, child joint that sets the direction, parent bone)
BONES = [
    ("hips",             0,  3, None),
    ("spine",            3,  6, "hips"),
    ("chest",            6,  9, "spine"),
    ("neck",             9, 12, "chest"),
    ("head",            12, 15, "neck"),
    ("left_shoulder",   13, 16, "neck"),
    ("left_upper_arm",  16, 18, "left_shoulder"),
    ("left_lower_arm",  18, 20, "left_upper_arm"),
    ("left_hand",       20, None, "left_lower_arm"),
    ("right_shoulder",  14, 17, "neck"),
    ("right_upper_arm", 17, 19, "right_shoulder"),
    ("right_lower_arm", 19, 21, "right_upper_arm"),
    ("right_hand",      21, None, "right_lower_arm"),
    ("left_upper_leg",   1,  4, "hips"),
    ("left_lower_leg",   4,  7, "left_upper_leg"),
    ("left_foot",        7, 10, "left_lower_leg"),
    ("right_upper_leg",  2,  5, "hips"),
    ("right_lower_leg",  5,  8, "right_upper_leg"),
    ("right_foot",       8, 11, "right_lower_leg"),
]
BONE_AT_JOINT = {j: n for n, j, _, _ in BONES}

# Hinge axes in the T-pose (= the axis the child joint bends about).
# Left elbow: arm along +X, forearm bends toward +Z  -> cross(+X, +Z) = (0,-1,0)
# Knee      : thigh along -Y, shin bends toward -Z   -> cross(-Y, -Z) = (1,0,0)
HINGE_REST = {16: (0, -1, 0), 17: (0, 1, 0),      # shoulder (sets the upper-arm twist)
              18: (0, -1, 0), 19: (0, 1, 0),      # elbow (forearm pronation = the estimated part)
              1: (1, 0, 0),   2: (1, 0, 0),       # hip (sets the thigh twist)
              4: (1, 0, 0),   5: (1, 0, 0)}       # knee

# Of MediaPipe Pose's 33 landmarks, the 21 that CBD's GR00T G1 adapter
# reads to measure segment lengths. They fill almost directly from SMPL's 22 joints.
# Only fingertips (19/20) and heels (29/30) are missing in HumanML3D, so they are derived and their provenance recorded in the manifest.
MP_FROM_SMPL = {0: 15, 7: 15, 8: 15,            # nose / ear are stood in for by the head
                11: 16, 12: 17, 13: 18, 14: 19, 15: 20, 16: 21,
                23: 1, 24: 2, 25: 4, 26: 5, 27: 7, 28: 8, 31: 10, 32: 11}
MP_DERIVED = {19: "left_index", 20: "right_index", 29: "left_heel", 30: "right_heel"}
HAND_OVER_FOREARM = 0.70        # hand length / forearm length. A rough anthropometric ratio


def pose_canonical(pos):
    """(T,22,3) -> (T,33,3). Repack into MediaPipe's landmark layout."""
    T = len(pos)
    out = np.zeros((T, 33, 3))
    for mp, sm in MP_FROM_SMPL.items():
        out[:, mp] = pos[:, sm]
    for mp, (elb, wri) in {19: (18, 20), 20: (19, 21)}.items():
        d = pos[:, wri] - pos[:, elb]
        out[:, mp] = pos[:, wri] + d * HAND_OVER_FOREARM     # place the hand on the extension of the forearm
    for mp, (ank, toe) in {29: (7, 10), 30: (8, 11)}.items():
        h = pos[:, ank] - pos[:, toe]
        h[:, 1] = 0.0                                         # horizontal component only
        out[:, mp] = pos[:, ank] + h * 0.5
        out[:, mp, 1] = pos[:, toe, 1]                        # heel at the same height as the toe
    return out


# ------------------------------------------------------------------- Conversion
def resolve(v0, v, ref_rest, ref_world):
    """The swing v0 -> v, plus a twist that aligns the reference axis with ref_world.

    If ref_world is None, or the reference axis is parallel to the bone axis so its projection vanishes, returns the swing only
    (= zero twist).
    """
    sw = qbetween(np.broadcast_to(np.asarray(v0, float), v.shape), v)
    if ref_world is None:
        return sw, np.zeros(len(v), bool)
    a = qrot(sw, np.broadcast_to(np.asarray(ref_rest, float), v.shape))
    axis = _n(v)
    p = a - axis * (a * axis).sum(-1, keepdims=True)
    q = ref_world - axis * (ref_world * axis).sum(-1, keepdims=True)
    ok = (np.linalg.norm(p, axis=-1) > 1e-3) & (np.linalg.norm(q, axis=-1) > 1e-3)
    p, q = _n(p), _n(q)
    theta = np.arctan2((np.cross(p, q) * axis).sum(-1), (p * q).sum(-1))
    theta = np.where(ok, theta, 0.0)
    return qmul(qaxis(axis, theta), sw), ok


def measure_offsets(pos, glob):
    """Measure each joint's rest offset in its parent's frame.

    REST_DIR is only a convention for "the child that sets the bone's direction". For joints with several children,
    such as the pelvis (3 children) or the top of the spine (3 children), the relative placement of the siblings does not follow from the convention,
    so it is measured within the chosen frame. The skeleton is rigid, so it does not vary between frames.
    """
    off, resid = {}, 0.0
    for c in range(1, 22):
        p = PARENT[c]
        local = qrot(qinv(glob[BONE_AT_JOINT[p]]), pos[:, c] - pos[:, p])
        off[c] = local.mean(0)
        resid = max(resid, float(np.linalg.norm(local - off[c], axis=-1).max()))
    return off, resid


def convert(pos, pronation="neutral", torso_rest="tpose"):
    """(T,22,3) joint positions -> CBD bone rotations.

    Returns: dict(bone -> (T,4) wxyz local rotation), root position, diagnostics
    """
    T = len(pos)
    rest = dict(REST_DIR)
    if torso_rest == "neutral":
        rest.update(NEUTRAL_REST)

    def d(a, b):
        return _n(pos[:, b] - pos[:, a])

    # --- Root (hips): build an orthonormal frame from only the 3 points rigid to the pelvis.
    # spine3 moves when the spine bends, so it cannot be used. spine1 and both hips are direct, rigid children of the pelvis.
    y = _n(pos[:, J["spine1"]] - pos[:, J["pelvis"]])
    x = pos[:, J["l_hip"]] - pos[:, J["r_hip"]]
    x = _n(x - y * (x * y).sum(-1, keepdims=True))
    z = np.cross(x, y)
    root = mat_to_quat(np.stack([x, y, z], -1))
    if torso_rest == "neutral":
        root = qmul(root, np.broadcast_to(NEUTRAL_HIPS, root.shape))

    # --- Twist reference axes
    across_hip = _n(pos[:, J["l_hip"]] - pos[:, J["r_hip"]])
    # The shoulders' left-right axis comes from the clavicle joints (l_shoulder/r_shoulder are children of the clavicles,
    # so they are not rigid relative to the spine3 frame)
    across_sh = _n(pos[:, J["l_collar"]] - pos[:, J["r_collar"]])
    # Normals of the elbow and knee flexion planes (= the actual hinge axes). These set the upper-arm / thigh twist.
    hinge = {}
    for j, (a, b, c) in {16: (16, 18, 20), 17: (17, 19, 21),
                         1: (1, 4, 7), 2: (2, 5, 8)}.items():
        hinge[j] = _n(np.cross(d(a, b), d(b, c)))
    # the forearm inherits the elbow's hinge axis as is (neutral pronation)
    hinge[18], hinge[19] = hinge[16], hinge[17]
    hinge[4], hinge[5] = hinge[1], hinge[2]

    def mix(t):
        return _n((1 - t) * across_hip + t * across_sh)

    REF = {3: (( 1, 0, 0), mix(1 / 3)), 6: ((1, 0, 0), mix(2 / 3)),
           9: (( 1, 0, 0), across_sh),  12: ((1, 0, 0), across_sh)}
    for j, ax in HINGE_REST.items():
        if pronation == "zero" and j in (18, 19):
            continue                       # leave out only the forearm pronation
        REF[j] = (ax, hinge[j])
    if pronation == "none":
        REF = {}                           # no twist at all (same as the original HumanML3D)

    glob, loc, est = {}, {}, {}
    glob["hips"] = root
    loc["hips"] = root
    for name, jj, child, parent in BONES:
        if name == "hips":
            continue
        if child is None:                  # hands: no child, so they cannot be observed
            g = glob[parent]
            loc[name] = np.tile(np.array([1.0, 0, 0, 0]), (T, 1))
            glob[name] = g
            est[name] = "unobservable_set_to_forearm_frame"
            continue
        rr, rw = REF.get(jj, (None, None))
        g, ok = resolve(rest[child], d(jj, child), rr, rw)
        glob[name] = g
        loc[name] = qmul(qinv(glob[parent]), g)
        if jj not in REF:
            est[name] = "measured_swing_only"        # twist not determined
        elif jj in (18, 19):
            # Forearm pronation cannot be observed. Filled by aligning the palm with the elbow's flexion plane.
            est[name] = f"estimated_neutral_pronation ({100 * ok.mean():.0f}%_resolved)"
        elif jj in HINGE_REST:
            # the child is a hinge joint, so its flexion plane fixes the twist geometrically
            est[name] = "measured_from_hinge_plane"
        elif jj == 9:
            est[name] = "measured_from_clavicle_line"
        elif jj == 12:
            est[name] = "assumed_no_twist_relative_to_shoulders"
        else:
            est[name] = "interpolated_torso_twist"   # the middle of the spine cannot be observed, so it is distributed proportionally

    # --- Reconstruct positions by FK and measure the error
    off, resid = measure_offsets(pos, glob)
    rec = np.zeros_like(pos)
    rec[:, 0] = pos[:, 0]
    for c in range(1, 22):
        gb = glob[BONE_AT_JOINT[PARENT[c]]]
        rec[:, c] = rec[:, PARENT[c]] + qrot(gb, np.broadcast_to(off[c], (T, 3)))
    err = np.linalg.norm(rec - pos, axis=-1)
    return loc, pos[:, 0], dict(fk_rmse_m=float(np.sqrt((err ** 2).mean())),
                                fk_max_m=float(err.max()),
                                rest_offset_resid_m=resid,
                                fk_per_joint=err.mean(0).round(4).tolist(),
                                twist_source=est)


def joint_angles(pos):
    out = {}
    for side, (s, e, w, h, k, a) in {
            "left": (16, 18, 20, 1, 4, 7), "right": (17, 19, 21, 2, 5, 8)}.items():
        v1, v2 = _n(pos[:, e] - pos[:, s]), _n(pos[:, w] - pos[:, e])
        out[f"{side}_elbow_flexion"] = np.degrees(
            np.arccos(np.clip((v1 * v2).sum(-1), -1, 1)))
        v3, v4 = _n(pos[:, k] - pos[:, h]), _n(pos[:, a] - pos[:, k])
        out[f"{side}_knee_flexion"] = np.degrees(
            np.arccos(np.clip((v3 * v4).sum(-1), -1, 1)))
        down = np.array([0.0, -1.0, 0.0])
        out[f"{side}_shoulder_abduction"] = np.degrees(
            np.arccos(np.clip((v1 * down).sum(-1), -1, 1)))
    return out


# -------------------------------------------------------------- Output
def write_cbd(out, clip_id, category, caption, loc, hips, diag, ang, fps,
              pose=None, torso_rest="tpose"):
    out.mkdir(parents=True, exist_ok=True)
    (out / "timeline").mkdir(exist_ok=True)
    (out / "human").mkdir(exist_ok=True)
    (out / "quality").mkdir(exist_ok=True)
    T = len(hips)
    names = [n for n, _, _, _ in BONES]

    with open(out / "timeline" / "frames.jsonl", "w") as f:
        for t in range(T):
            f.write(json.dumps({
                "frame": t,
                "source_frame_index": t,
                "timestamp_ms": round(t * 1000.0 / fps, 2),
                "timestamp_sec": round(t / fps, 4),
                "frame_image": None,
                "human": {
                    # CBD uses xyzw order. Internally it is wxyz, so reorder.
                    "bone_rotations_xyzw": {
                        n: [round(float(x), 5) for x in
                            (loc[n][t][1], loc[n][t][2], loc[n][t][3], loc[n][t][0])]
                        for n in names},
                    "hips_position": [round(float(x), 5) for x in hips[t]],
                    # HumanML3D has no fingers. Put a null per hand to state explicitly that "both hands are missing"
                    # (a bare null leaves the consumer unable to look up a hand).
                    "finger_curls_rad": {"left": None, "right": None},
                    "joint_angles_deg": {k: round(float(v[t]), 2)
                                         for k, v in ang.items()},
                    **({"pose_canonical": [[round(float(c), 5) for c in p]
                                           for p in pose[t]]} if pose is not None
                       else {}),
                    "face_blendshapes": None,
                    "gestures": None,
                },
                "objects": [],
                "interactions": [],
                "phase": {"action": category, "phase": None, "hand": None},
                "caption": {"window_index": 0, "en": caption, "ja": None,
                            "source": "humanml3d_annotation"},
            }) + "\n")

    with open(out / "human" / "bone_rotations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_ms", "bone",
                    "rot_x", "rot_y", "rot_z", "rot_w"])
        for t in range(T):
            for n in names:
                q = loc[n][t]
                w.writerow([t, round(t * 1000.0 / fps, 2), n,
                            round(float(q[1]), 5), round(float(q[2]), 5),
                            round(float(q[3]), 5), round(float(q[0]), 5)])

    with open(out / "human" / "joint_angles.csv", "w", newline="") as f:
        keys = sorted(ang)
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_ms"] + keys)
        for t in range(T):
            w.writerow([t, round(t * 1000.0 / fps, 2)] +
                       [round(float(ang[k][t]), 2) for k in keys])

    json.dump({
        "dataset_version": "0.2.0",
        "demo_name": f"humanml3d_{clip_id}",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_video": None,
        "source_dataset": "HumanML3D new_joints (AMASS derived)",
        "source_clip_id": clip_id,
        "source_fps": fps, "fps": fps,
        "frame_count": T, "duration_sec": round(T / fps, 3),
        "person_count": 1, "face_landmark_count": 0,
        "coordinate_system": {
            "raw": "humanml3d (x=person left, y=up, z=person forward, floor at y=0)",
            "canonical": "body_capture (y-up, right-handed, person faces +z)",
            "adapters": {"mujoco": "z-up, person faces -y",
                         "unity_vrm": "gltf y-up"}},
        "rest_pose": "T-pose" if torso_rest == "tpose"
                     else "T-pose（胴体のみ実測の中立姿勢）",
        "bone_count": len(names),
        **({"pose_canonical": {
            "layout": "mediapipe pose, 33 landmarks",
            "measured": sorted(MP_FROM_SMPL),
            "derived": MP_DERIVED,
            "unused": [i for i in range(33)
                       if i not in MP_FROM_SMPL and i not in MP_DERIVED],
            "note": "SMPL 22 関節からの詰め替え。指先(19/20)は前腕の延長に手長比 "
                    f"{HAND_OVER_FOREARM} で置いた導出値、踵(29/30)は足首と爪先から"
                    "導出。顔・手の細部は HumanML3D に無いのでゼロのまま。",
        }} if pose is not None else {}),
    }, open(out / "manifest.json", "w"), indent=1)

    json.dump({"action": category, "caption_en": caption,
               "primary_hand": None, "events": []},
              open(out / "behavior_summary.json", "w"), indent=1)

    json.dump({
        "converter": "tools/humanml3d_to_cbd.py",
        "method": "position-only IK; twist locked to the anatomical hinge plane",
        "fk_roundtrip_rmse_m": round(diag["fk_rmse_m"], 6),
        "fk_roundtrip_max_m": round(diag["fk_max_m"], 6),
        "twist_source": diag["twist_source"],
    }, open(out / "runtime_info.json", "w"), indent=1)

    json.dump({
        "overall_quality": 1.0, "pose_coverage": 1.0,
        "left_hand_coverage": 0.0, "right_hand_coverage": 0.0,
        "face_coverage": 0.0, "object_coverage": 0.0,
        "missing_frame_ratio": 0.0,
        "notes": [
            "HumanML3D は 22 関節（手首で終端）。手・指・顔のデータは存在しない。",
            "left_hand / right_hand の回転は観測不能。前腕フレームに揃えた恒等回転。",
            "前腕の回内・回外は観測不能。肘の屈曲平面に手のひらを合わせる中立回内で埋めた。",
            "上腕・大腿の捻りは肘・膝の屈曲平面から幾何的に確定しており、推定ではない。",
            "脊椎途中(spine/chest)の捻りは観測不能。骨盤と鎖骨の左右軸の間で按分した。",
            "ボーンごとの素性は runtime_info.json の twist_source を見ること。",
        ],
    }, open(out / "quality" / "quality.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="work/humanml3d/subset")
    ap.add_argument("--out", default="work/cbd_humanml3d")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--pronation", choices=["neutral", "zero", "none"],
                    default="neutral")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--torso-rest", choices=["tpose", "neutral"], default="tpose",
                    help="neutral は胴体の静止姿勢を実測の中立姿勢にする"
                         "（ロボットの腰へ回転をそのまま渡す消費側向け）")
    ap.add_argument("--pose-canonical", action="store_true",
                    help="pose_canonical を書く。GR00T G1 アダプタが要求する")
    a = ap.parse_args()

    sub, out = pathlib.Path(a.subset), pathlib.Path(a.out)
    man = json.load(open(sub / "manifest.json"))
    cat_of = {i: c for c, v in man["categories"].items() for i in v}
    ids = sorted(cat_of)
    if a.limit:
        ids = ids[:a.limit]

    rmse, worst = [], (0.0, None)
    for i in ids:
        pos = np.load(sub / "new_joints" / f"{i}.npy").astype(np.float64)
        loc, hips, diag = convert(pos, a.pronation, a.torso_rest)
        ang = joint_angles(pos)
        cap = open(sub / "texts" / f"{i}.txt", encoding="utf-8",
                   errors="replace").readline().split("#")[0].strip()
        write_cbd(out / cat_of[i] / i, i, cat_of[i], cap, loc, hips, diag,
                  ang, a.fps, pose_canonical(pos) if a.pose_canonical else None,
                  a.torso_rest)
        rmse.append(diag["fk_rmse_m"])
        worst = max(worst, (diag["fk_max_m"], i))

    json.dump({"clips": len(ids), "fps": a.fps, "pronation": a.pronation,
               "categories": {c: [i for i in ids if cat_of[i] == c]
                              for c in sorted(set(cat_of[i] for i in ids))}},
              open(out / "index.json", "w"), indent=1)
    print(f"変換 {len(ids)} clips -> {out}")
    print(f"FK 逆算誤差  RMSE 平均 {np.mean(rmse)*1000:.4f} mm / "
          f"最悪 {worst[0]*1000:.4f} mm ({worst[1]})")


if __name__ == "__main__":
    main()
