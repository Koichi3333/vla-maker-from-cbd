#!/usr/bin/env python3
"""Video CBD -> a LeRobot dataset in the same shape as stage B (V-4). Codespace, 0 CU.

- **state / action stay 117-d** (same layout and same 6D representation as `cbd_to_lerobot.py`). root is fixed at 0 (F-3)
- **Resampled to 20 fps.** Video CBD is 15 fps (decimated by 2) or 30 fps. Bone rotations use slerp, object states the nearest frame
- **Objects are not put in the state** (decided in stage B). Only so the renderer can place them on screen, per-frame states are kept in
  `objects/episode_XXXXXX.json` (GR00T's loader reads only data/ and meta/):
  `held_left` / `held_right` / `held_both` (phase is Grasp / Carry and the object is visible), `hidden` (flying away from the hand = `tracked_2d`),
  `rest` (everything else, including `hidden` shorter than 0.25 s). **No positions are stored.** The object is placed relative to the rendered hand (`render_with_objects.py`)
- **Clips that `fail` the quality gate are left out.** The external example (`sample`) is left out too (the subject's consent is unknown)
- **Tuning and scoring sets are separate** (V-5's H-3 is measured on the scoring set). `train` / `heldout` in `meta/split.json`
- Captions are **English sentences made from the file names the recorder gave** (`CAPTIONS`), not generated text

    python3 vla/video_to_lerobot.py --cbd-dir work/cbd_video --out work/lerobot_video
    python3 vla/video_to_lerobot.py --cbd-dir work/cbd_video --out work/lerobot_video_mirror --mirror   # V-5b
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cbd_to_lerobot import BONES, DIM, GROUPS, mirror_spec   # noqa: E402
from rotation import normalize, quat_to_6d                    # noqa: E402

# Recorder's file name ("recording results" in `SHOOTING_LIST.md`) -> English sentence, in HumanML3D's style ("a person ...")
CAPTIONS = {
    "bodyweight_squat_x2": "a person does two bodyweight squats.",
    "jump_and_catch_ball": "a person tosses a ball up, jumps, and catches it.",
    "march_in_place_knee_raise": "a person marches in place, raising the knees.",
    "pick_and_place_glass_real": "a seated person picks up a glass from the table and puts it back down.",
    "seated_drink_from_glass": "a seated person picks up a glass from the table, drinks from it, and puts it back.",
    "seated_drink_from_water_bottle": "a seated person picks up a plastic bottle from the table, drinks from it, "
                                      "and puts it back.",
    "seated_lean_reach_glass_far_and_back": "a seated person leans forward to move a glass to the far side of "
                                            "the table and back.",
    "seated_lift_bowl_overhead": "a seated person lifts a bowl from the table over the head and puts it back.",
    "seated_lift_glass_reseat_and_hold_out_to_side": "a seated person lifts a glass, shifts in the seat, holds "
                                                     "the glass out to the side, and puts it back.",
    "seated_move_glass_far_and_back": "a seated person moves a glass to the far side of the table and back.",
    "seated_pick_up_and_look_at_smartphone": "a seated person picks up a smartphone from the table, looks at it, "
                                             "and puts it back.",
    "seated_pick_up_and_read_book": "a seated person picks up a book from the table, reads it, and puts it back.",
    "squat_jump": "a person squats down and jumps up, swinging the arms.",
    "squat_then_toss_ball_up_and_catch": "a person squats, tosses a ball up from below, and catches it.",
    "standing_ball_overhead_and_toss_between_hands": "a person raises a ball over the head and tosses it "
                                                     "between the hands.",
    "standing_ball_overhead_one_hand_then_both_hands": "a person picks up a ball from the table, raises it over "
                                                       "the head with one hand and then both hands, and puts it back.",
    "standing_pick_up_ball_raise_overhead_and_return": "a person picks up a ball from the table, raises it over "
                                                       "the head, and puts it back.",
    "standing_small_vertical_jump": "a person does a small jump in place.",
    "standing_toss_ball_up_and_catch": "a person tosses a ball straight up and catches it.",
}
# Tuning set (alternating in recording order; `SHOOTING_LIST.md`). The 2 existing real clips were also used for tuning, so they are train
TUNE = {"2236", "2238", "2240", "2242", "2244", "2246", "2248", "2250", "2252", "2254", "2256", "2258", "2261"}
EXCLUDE = {"sample"}


def caption(name):
    key = name.rsplit("_", 1)[0] if name[-4:].isdigit() else name
    return CAPTIONS[key]


def slerp(a, b, w):
    """Spherical linear interpolation of (..., 4) wxyz. w is a (...,) weight."""
    d = (a * b).sum(-1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d).clip(-1, 1)
    th = np.arccos(d)
    s = np.sin(th)
    w = w[..., None]
    small = s < 1e-6
    k0 = np.where(small, 1 - w, np.sin((1 - w) * th) / np.where(small, 1, s))
    k1 = np.where(small, w, np.sin(w * th) / np.where(small, 1, s))
    return normalize(k0 * a + k1 * b)


def object_modes(frames, fps, gap_sec=0.5):
    """Per-frame object states. **Held only when phase is Grasp / Carry and the object is visible (`held_by` is set)**.

    The rule carries "held" over while the object is lost, so CBD keeps stretches where phase is Grasp but `held_by` is None
    (2240: the last 4 s after the phone was laid flat on the desk and lost. Left as is, the phone was drawn stuck to the hand on the lap. 2026-09-26).
    Losses shorter than `gap_sec` (a moment cupped in both hands, 4 frames in 2250) are filled from the held frames on either side.
    """
    T = len(frames)
    ph = [f.get("phase") or {} for f in frames]
    held = [p.get("phase") in ("Grasp", "Carry") and f["objects"][0]["held_by"] is not None
            for p, f in zip(ph, frames)]
    phase_held = [p.get("phase") in ("Grasp", "Carry") for p in ph]
    n, i = max(1, int(round(gap_sec * fps))), 0
    while i < T:                                   # fill short losses sandwiched between held frames
        if not held[i] and phase_held[i] and i > 0 and held[i - 1]:
            j = i
            while j < T and not held[j] and phase_held[j]:
                j += 1
            if j < T and held[j] and j - i < n:
                held[i:j] = [True] * (j - i)
            i = j
        else:
            i += 1
    out, last = [], None
    for f, p, h in zip(frames, ph, held):
        if h:
            last = p.get("hand") or last or "right"
            out.append(f"held_{last}")
        elif f["objects"][0].get("position_source") == "tracked_2d":
            out.append("hidden")
        else:
            out.append("rest")
    return out


def drop_short_hidden(mode, fps, min_sec=0.25):
    """Turn `hidden` stretches shorter than `min_sec` back into `rest`.

    When the object moves 1-3 frames in a different direction from the hand at the moment of placing or sliding, it becomes `tracked_2d` and briefly vanishes in the render
    (2-4 frames for the plastic bottle, phone and cup; 2026-09-26). Throws in flight last 7-17 frames, so they are kept.
    """
    out, n, i = list(mode), max(1, int(round(min_sec * fps))), 0
    while i < len(out):
        if out[i] == "hidden":
            j = i
            while j < len(out) and out[j] == "hidden":
                j += 1
            if j - i < n:
                out[i:j] = ["rest"] * (j - i)
            i = j
        else:
            i += 1
    return out


def load(d, fps_out):
    frames = [json.loads(line) for line in (d / "timeline" / "frames.jsonl").read_text().splitlines()]
    fps_in = json.loads((d / "manifest.json").read_text())["fps"]
    Q = []
    for f in frames:
        r = f["human"]["bone_rotations_xyzw"]
        Q.append([[r[b][3], r[b][0], r[b][1], r[b][2]] for b in BONES])
    Q = normalize(np.asarray(Q, np.float64))                     # (T, 19, 4)
    T_in = len(Q)
    n = int(np.floor((T_in - 1) / fps_in * fps_out)) + 1
    t = np.arange(n) / fps_out * fps_in                           # source frame number (fractional)
    i0 = np.floor(t).astype(int).clip(0, T_in - 1)
    i1 = (i0 + 1).clip(0, T_in - 1)
    w = np.broadcast_to((t - i0)[:, None], (n, len(BONES)))
    Qo = slerp(Q[i0], Q[i1], w)
    obj = None
    if frames[0].get("objects"):
        modes = object_modes(frames, fps_in)
        near = np.rint(t).astype(int).clip(0, T_in - 1)
        obj = dict(label=frames[0]["objects"][0]["label"], mode=drop_short_hidden([modes[k] for k in near], fps_out))
    return Qo, obj, fps_in, T_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbd-dir", default="work/cbd_video")
    ap.add_argument("--out", default="work/lerobot_video")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--robot-type", default="cbd_humanoid")
    ap.add_argument("--mirror", action="store_true",
                    help="train に左右反転の複製を足す（V-5b。物の置き場所に幅を持たせる）。heldout は反転しない")
    a = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = pathlib.Path(a.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "objects").mkdir(exist_ok=True)
    clips, skipped = [], []
    for d in sorted(pathlib.Path(a.cbd_dir).iterdir()):
        q = json.loads((d / "quality" / "quality.json").read_text())
        if d.name in EXCLUDE or q["overall_quality"] == "fail":
            skipped.append((d.name, "外部の見本" if d.name in EXCLUDE else "品質ゲートで fail"))
            continue
        clips.append((d, q["overall_quality"]))
    # train first, heldout after (so they can be split by number)
    part = {d.name: ("train" if d.name[-4:] in TUNE or not d.name[-4:].isdigit() else "heldout") for d, _ in clips}
    clips.sort(key=lambda c: (part[c[0].name] != "train", c[0].name))

    # mirrored copies go after train (before heldout), so train / heldout stay separable by number
    jobs = [(d, q, False) for d, q in clips if part[d.name] == "train"]
    if a.mirror:
        jobs += [(d, q, True) for d, q in clips if part[d.name] == "train"]
    jobs += [(d, q, False) for d, q in clips if part[d.name] == "heldout"]
    perm, sign = mirror_spec()
    SWAP = {"held_left": "held_right", "held_right": "held_left"}

    tasks, task_id, episodes, n_frames = [], {}, [], 0
    split = {"train": [], "heldout": []}
    allx = []
    for ep, (d, qual, mirrored) in enumerate(jobs):
        Q, obj, fps_in, T_in = load(d, a.fps)
        T = len(Q)
        state = np.concatenate([np.zeros((T, 3), np.float32),              # F-3: root fixed at 0
                                quat_to_6d(Q).reshape(T, -1).astype(np.float32)], 1)
        if mirrored:
            # mirroring is x[perm] * sign from mirror_spec() (same as cbd_to_lerobot.py). The holding hand is swapped too
            state = (state[:, perm] * sign).astype(np.float32)
            if obj:
                obj = dict(obj, mode=[SWAP.get(m, m) for m in obj["mode"]], mirrored=True)
        assert state.shape[1] == DIM
        action = np.concatenate([state[1:], state[-1:]], 0)
        cap = caption(json.loads((d / "manifest.json").read_text())["demo_name"])
        if cap not in task_id:
            task_id[cap] = len(tasks)
            tasks.append({"task_index": len(tasks), "task": cap})
        ti = task_id[cap]
        cd = out / "data" / "chunk-000"
        cd.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "observation.state": list(state), "action": list(action),
            "timestamp": np.arange(T) / a.fps, "frame_index": np.arange(T, dtype=np.int64),
            "episode_index": np.full(T, ep, np.int64), "index": np.arange(n_frames, n_frames + T, dtype=np.int64),
            "task_index": np.full(T, ti, np.int64), "annotation.human.task_description": np.full(T, ti, np.int64),
            "next.done": np.arange(T) == T - 1,
        }), cd / f"episode_{ep:06d}.parquet")
        if obj:
            (out / "objects" / f"episode_{ep:06d}.json").write_text(json.dumps(obj))
        episodes.append({"episode_index": ep, "tasks": [cap], "length": T, "source_clip": d.name,
                         "part": part[d.name], "quality": qual, "object": obj["label"] if obj else None,
                         "source_fps": fps_in, "source_frames": T_in, "mirrored": mirrored})
        split[part[d.name]].append(ep)
        n_frames += T
        allx.append(state.astype(np.float64))
        print(f"  {ep:3d} {part[d.name]:7s}{'（反転）' if mirrored else ''} {d.name[-44:]:44s} {T_in:4d}f@{fps_in:5.2f} -> {T:4d}f  "
              f"{obj['label'] if obj else '-':6s} {qual}")

    x = np.concatenate(allx)
    stat = dict(mean=x.mean(0).tolist(), std=x.std(0).tolist(), max=x.max(0).tolist(), min=x.min(0).tolist())
    json.dump({"observation.state": stat, "action": stat}, open(out / "meta" / "stats.json", "w"))
    feats = {k: {"dtype": "float32", "shape": [DIM], "names": [g for g, _, _ in GROUPS]}
             for k in ("observation.state", "action")}
    feats.update({k: {"dtype": t, "shape": [1]} for k, t in (
        ("timestamp", "float64"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"),
        ("task_index", "int64"), ("annotation.human.task_description", "int64"), ("next.done", "bool"))})
    json.dump({
        "codebase_version": "v2.0", "robot_type": a.robot_type,
        "total_episodes": len(jobs), "total_frames": n_frames, "total_tasks": len(tasks),
        "total_chunks": 1, "chunks_size": 1000, "fps": a.fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": feats,
        "cbd": {
            "action_space": "CBD bone rotations, 6D continuous representation",
            "layout": {g: [s, e] for g, s, e in GROUPS}, "bones": BONES, "rest_pose": "T-pose",
            "coordinate_system": "y-up, right-handed, person faces +z",
            "source": "自前の動画（iPhone、固定カメラ）→ vla/video_to_cbd.py の CBD",
            "root_translation": "0 固定（F-3。動画からは世界での移動を復元できない）",
            "objects": "state / action には入れない。objects/episode_XXXXXX.json にフレームごとの状態だけ"
                       "（位置はレンダの手に合わせる。render_with_objects.py）",
            "captions": "撮影者が付けたファイル名から作った英文（video_to_lerobot.py の CAPTIONS）",
            "smpl_used": True,
            "license_note": "胴は SMPL（ROMP）で解いた値。SMPL の利用条件（非商用・再配布不可）が及ぶ",
            "split": "meta/split.json（train = 調整用 + 既存の実写、heldout = 採点用）",
            "mirrored_train": bool(a.mirror),
            "skipped": [dict(clip=c, why=w) for c, w in skipped],
        },
    }, open(out / "meta" / "info.json", "w"), indent=1, ensure_ascii=False)
    for name, rows in (("tasks", tasks), ("episodes", episodes)):
        (out / "meta" / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    mod = {k: {g: {"start": s, "end": e} for g, s, e in GROUPS} for k in ("state", "action")}
    mod["annotation"] = {"human.task_description": {"original_key": "annotation.human.task_description"}}
    json.dump(mod, open(out / "meta" / "modality.json", "w"), indent=1)
    json.dump({"note": "x_mirrored = x[perm] * sign。cbd_to_lerobot.py と同じ",
               "perm": perm.tolist(), "sign": sign.tolist()}, open(out / "meta" / "mirror.json", "w"))
    json.dump(split, open(out / "meta" / "split.json", "w"), indent=1)
    print(f"{len(jobs)} episodes（train {len(split['train'])} / heldout {len(split['heldout'])}）/ "
          f"{n_frames} frames / {len(tasks)} tasks -> {out}")
    for c, w in skipped:
        print(f"  入れなかった: {c}（{w}）")


if __name__ == "__main__":
    main()
