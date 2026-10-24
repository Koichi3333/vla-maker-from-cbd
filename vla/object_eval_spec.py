#!/usr/bin/env python3
"""Build the evaluation spec for V-5's H-2 (with an object in the instruction, does the hand go toward it?). Codespace, 0 CU.

Of the scoring set (heldout in `meta/split.json`), only clips that **start with the object resting somewhere** are used
(throwing clips that hold the ball from the start have no reaching motion, so they are left out). For each clip:

- `init_state`: the 117 dims of the first frame (the closed-loop initial pose)
- `obj_raw`: **where the object was actually first grasped** (the same place V-4's renderer puts the object before the grasp). Coordinates with the ground correction undone
  (evaluation redoes the ground correction from the initial pose, so `obj_raw - (0, 0, lift)` is the world position in evaluation)
- `shift_dir`: the direction the control moves the object. Along the pelvis left-right axis, **toward the side opposite the object** (unit vector, horizontal)
- `obj_mirror_raw`: control (V-5b). The object on the other side of the body (mirrored about the plane through the pelvis perpendicular to the left-right axis)
- `h4_offsets` (with `--h4`; V-7's H-4): offsets of the object from its original position (world m, 8 of them). In the body orientation of the first frame:
  left-right +-12 cm, forward-back +-8 cm, up-down +-7 cm, and two diagonals (within reach of the same hand; separate from shift's 25 cm and mirror). Directions where the object would overlap a leg
  are flipped, and dropped if they still overlap. `h4_hand` is the hand that grasped in the original (H-4 is measured on that hand)

    python3 vla/object_eval_spec.py --dataset work/lerobot_video --out work/v5/object_eval.json
    python3 vla/object_eval_spec.py --dataset work/lerobot_video --out work/v7/object_eval.json --h4   # V-7

V-12 stage 3 (turning guided motion into training data): `--split train` picks training-side clips, and `--random-h4 N` puts N random
offsets into `h4_offsets` (within `--reach` of the grasping hand's shoulder, not overlapping a leg, at least 3 cm from the fixed evaluation offsets).
`--add-mirror` adds separate items with the object on the other side of the body and the grasping hand swapped (their h4 offsets are drawn around the mirrored position).

    python3 vla/object_eval_spec.py --split train --random-h4 40 --add-mirror --out work/v12/train_spec.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))


H4_PATTERN = [(12, 0, 0), (-12, 0, 0), (0, 8, 0), (0, -8, 0), (0, 0, 7), (0, 0, -7), (8, 6, 5), (-8, -6, -5)]   # cm (left, forward, up)


def h4_offsets(m, d, q0, p, hip, label):
    """H-4 offsets (world m). Directions that overlap a leg are flipped, and dropped if they still overlap."""
    import mujoco
    from reach_synth import LEGS, OBJ_EXTENT
    up = np.array([0.0, 0, 1])
    basis = np.stack([hip, np.cross(hip, up), up])                 # left, forward, up (same as reach_synth.body_frame)
    ext = OBJ_EXTENT.get(label, 0.05)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)

    def clear(x):
        for a_, b_, r in LEGS:
            a, b = d.xpos[m.body(a_).id], d.xpos[m.body(b_).id]
            u = np.clip((x - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            if np.linalg.norm(x - (a + u * (b - a))) < r + ext:
                return False
        return True
    out = []
    for c in H4_PATTERN:
        for sgn in (1, -1):
            o = sgn * np.array(c, np.float64) / 100 @ basis
            if clear(p + o):
                out.append(o.round(4).tolist())
                break
    return out


def leg_clear(m, d, q0, label):
    """Does the object center x stay clear of the legs (thighs, shins)? Judged in the q0 pose."""
    import mujoco
    from reach_synth import LEGS, OBJ_EXTENT
    ext = OBJ_EXTENT.get(label, 0.05)
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    segs = [(d.xpos[m.body(a_).id].copy(), d.xpos[m.body(b_).id].copy(), r) for a_, b_, r in LEGS]

    def clear(x):
        for a, b, r in segs:
            u = np.clip((x - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
            if np.linalg.norm(x - (a + u * (b - a))) < r + ext:
                return False
        return True
    return clear


RANDOM_BOX = np.array([0.30, 0.25, 0.20])        # range of the random offsets (left, forward, up; m, +-)


def random_offsets(rng, n, m, d, q0, p, hip, label, hand, reach, avoid):
    """V-12 stage 3: n random offsets of the object from p (world m). Within reach of the shoulder (`<hand>_upper_arm`), not overlapping a leg,
    and at least 3 cm from every offset in avoid (those used in evaluation). The body orientation is the pelvis left-right axis of the first frame."""
    import mujoco
    up = np.array([0.0, 0, 1])
    basis = np.stack([hip, np.cross(hip, up), up])
    clear = leg_clear(m, d, q0, label)
    sh = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{hand}_upper_arm")].copy()
    out, tries = [], 0
    while len(out) < n and tries < 20000:
        tries += 1
        o = rng.uniform(-1, 1, 3) * RANDOM_BOX @ basis
        x = p + o
        if np.linalg.norm(o) < 0.04 or np.linalg.norm(x - sh) > reach or x[2] < 0.03 or not clear(x):
            continue
        if any(np.linalg.norm(o - np.asarray(v)) < 0.03 for v in avoid):
            continue
        out.append(o.round(4).tolist())
    if len(out) < n:
        print(f"    乱数のずらしが {len(out)}/{n} 個しか取れなかった（{tries} 回）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="work/lerobot_video")
    ap.add_argument("--xml", default=str(HERE / "humanoid.xml"))
    ap.add_argument("--shift-m", type=float, default=0.25)
    ap.add_argument("--out", default="work/v5/object_eval.json")
    ap.add_argument("--h4", action="store_true", help="V-7 の H-4 のずらし（h4_offsets）を足す")
    ap.add_argument("--split", default="heldout", choices=["heldout", "train"], help="V-12 段 3: train で学習側の clip を選ぶ")
    ap.add_argument("--random-h4", type=int, default=0, help="V-12 段 3: h4_offsets を乱数のずらし N 個にする")
    ap.add_argument("--add-mirror", action="store_true", help="V-12 段 3: 物を反対側・掴む手を逆にした項目を足す")
    ap.add_argument("--reach", type=float, default=0.52, help="乱数のずらしの上限（掴む手の肩から、m）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import mujoco
    from cbd_to_mujoco import ground
    from humanml3d_to_video import load_state, state_to_qpos
    from render_with_objects import AWAY, anchors, body_ids, object_track, xml_with_objects

    ds = pathlib.Path(a.dataset)
    m = mujoco.MjModel.from_xml_string(xml_with_objects(a.xml))
    d = mujoco.MjData(m)
    mocap, hand = body_ids(m)
    for i in mocap.values():
        d.mocap_pos[i] = AWAY
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("pelvis", "left_upper_leg", "right_upper_leg")}
    split = json.loads((ds / "meta/split.json").read_text())
    eps = {e["episode_index"]: e for e in map(json.loads, (ds / "meta/episodes.jsonl").read_text().splitlines())}
    rng = np.random.default_rng(a.seed)
    flip = {"left": "right", "right": "left"}
    items = []
    for i in split[a.split]:
        op = ds / "objects" / f"episode_{i:06d}.json"
        if not op.exists():
            continue
        mode = json.loads(op.read_text())["mode"]
        if mode[0] != "rest":
            print(f"  外す: {eps[i]['source_clip']}（最初から {mode[0]}）")
            continue
        st = load_state(ds / f"data/chunk-000/episode_{i:06d}.parquet")
        q = state_to_qpos(st)
        lift = ground(m, d, q)                      # q[:, 2] -= lift (same as the training renders)
        pos = object_track(mode, anchors(m, d, hand, q))
        g = next(t for t, x in enumerate(mode) if x.startswith("held"))
        d.qpos[:] = q[0]
        mujoco.mj_forward(m, d)
        hip = d.xpos[bid["left_upper_leg"]] - d.xpos[bid["right_upper_leg"]]
        hip[2] = 0
        hip /= np.linalg.norm(hip)
        rel = pos[g] - d.xpos[bid["pelvis"]]
        side = np.sign(rel @ hip) or 1.0
        raw = pos[g] + np.array([0, 0, lift])
        # V-5b: the other side of the body (mirrored about the plane through the pelvis perpendicular to the left-right axis). A control matched to the spread of placements of a model trained with mirrored copies
        mir = pos[g] - 2 * (rel @ hip) * hip + np.array([0, 0, lift])
        label = json.loads(op.read_text())["label"]
        hand0 = mode[g][5:] if mode[g] != "held_both" else "left"
        h4 = h4_offsets(m, d, q[0], pos[g], hip, label) if a.h4 or a.random_h4 else None
        variants = [(raw, mir, side, hand0, pos[g], "")]
        if a.add_mirror:                # other side: object at the mirrored position, grasping hand swapped, shift direction reversed
            variants.append((mir, raw, -side, flip[hand0], mir - np.array([0, 0, lift]), "_mirror"))
        for raw_v, mir_v, side_v, hand_v, p_v, suf in variants:
            extra = {}
            if h4 is not None:
                offs = h4
                if a.random_h4:
                    avoid = h4 + [(-side_v * hip * a.shift_m).tolist(), (mir_v - raw_v).tolist()]
                    offs = random_offsets(rng, a.random_h4, m, d, q[0], p_v, hip, label, hand_v, a.reach, avoid)
                extra = dict(h4_offsets=offs, h4_hand=hand_v)
            items.append(dict(episode_index=i, clip=eps[i]["source_clip"] + suf, text=eps[i]["tasks"][0],
                              label=label, init_state=st[0].tolist(),
                              obj_raw=np.asarray(raw_v).tolist(), obj_mirror_raw=np.asarray(mir_v).tolist(),
                              shift_dir=(-side_v * hip).tolist(), shift_m=a.shift_m,
                              first_grasp_frame=g, lift_train=float(lift),
                              obj_from_pelvis_m=float(np.linalg.norm(p_v - d.xpos[bid["pelvis"]])),
                              obj_side="left" if side_v > 0 else "right", **extra,
                              **({"slug": f"ep{i:02d}{suf[1:2]}"} if a.split == "train" else {})))   # traj_<slug>.npz (m for mirrored)
            print(f"  {items[-1]['clip'][-44:]:44s} {label:5s} 掴むのは t={g / 20:.1f}s  "
                  f"腰から {items[-1]['obj_from_pelvis_m']:.2f} m（{items[-1]['obj_side']}）"
                  + (f"  ずらし {len(extra['h4_offsets'])} 個・{hand_v}" if extra else ""))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(dataset=str(ds), note=f"v5_object_eval_spec.py で作った（split={a.split}）", items=items),
                              indent=1, ensure_ascii=False))
    print(f"{len(items)} 本 -> {out}")


if __name__ == "__main__":
    main()
