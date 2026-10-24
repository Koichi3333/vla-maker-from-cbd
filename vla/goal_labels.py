#!/usr/bin/env python3
"""V-8 stage 1: add a 4-d "goal the hand is heading to next" to the state of a LeRobot dataset (117 -> 121).

Goal representation (`goal`, state dims 117-120):
- 0-2: goal position in m, **seen from the pelvis of the current frame** (subject's left, forward, up). Measured with the pelvis position and the horizontal heading given by the two hip joints
- 3: goal present (1 = yes, 0 = no). GR00T maps each state key to [-1, 1] by its min and max, so "no goal" cannot be expressed as 0.
  Frames without a goal get position 0 and are told apart by this flag

All hand positions come from **forward kinematics of the MuJoCo skeleton (humanoid.xml)** (the same skeleton as the observation renders and the evaluation's hand position; hand = midpoint of the wrist and the base of the middle finger).
GR00T pads the state with zeros up to 132 dims before its input layer (`max_state_dim = 132`), so going to 121 dims does not change the input layer's shape.

How goals are assigned:
- **Clips with an object** (the first state in `objects/episode_*.json` is `rest`): until the first grasp (0..g), the **object's position** (= the hand at the moment of grasping; the renderer places the object there).
  After the grasp, the "hindsight goal" below
- **Everything else (HumanML3D etc.) and object clips after the grasp**: a **hindsight goal**. Detect the apexes of the hand's reaches and use the hand position at each apex
  as the goal from `WINDOW` seconds before the apex up to the apex. Apex conditions (on the hand trajectory seen from the pelvis, per side):
  hand speed at a local minimum and below `V_APEX` / the slow state lasts at least `HOLD` frames (a brief stop) / net distance from the position `WINDOW` seconds earlier is at least `NET` /
  speed reached at least `PEAK` in between (it moved there deliberately). Apexes within 0.5 s of each other, across both sides, are merged into the first.
  Each frame takes as its goal the nearest upcoming apex within `WINDOW` seconds
- The conditions were set on the 14 real object clips (2026-09-27): without using object information, the actual grasp moment was found within +-0.5 s and 10 cm in 12 clips.
  The first rule, "hand at least 0.40 m from the shoulder (arm extended)", found only 8: a relaxed hanging arm (0.44 m) is farther than a bent arm grasping an object (0.35-0.39 m).
  HumanML3D gives about 4 apexes per 10 s, video about 9 because of pose-estimation jitter (smoothing dropped grasp detection to 10 clips, so it is not smoothed)

    MUJOCO_GL=osmesa python3 vla/goal_labels.py --src work/lerobot_video_warp --out work/lerobot_video_warp_goal
    MUJOCO_GL=osmesa python3 vla/goal_labels.py --src work/lerobot_video --check-objects   # compare hindsight goals with object positions
    MUJOCO_GL=osmesa python3 vla/goal_labels.py --src <heldout> --out <heldout_nogoal> --no-goal   # control with no goal
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE))
FPS = 20.0
WINDOW = 1.5          # how many seconds before the apex the goal starts
V_APEX = 0.30         # m/s. Hand speed at the apex (seen from the pelvis)
HOLD = 3              # frames. How long the slow state must last (0.15 s)
NET = 0.15            # m. Net distance from the position WINDOW seconds earlier
PEAK = 0.50           # m/s. Peak speed reached in between
GOAL_DIM = 4
XML_DEFAULT = HERE / "humanoid.xml"

_W = {}


def _init(xml):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    _W.update(m=m, d=mujoco.MjData(m),
              ids={n: m.body(n).id for n in ("pelvis", "left_upper_leg", "right_upper_leg", "left_upper_arm", "right_upper_arm",
                                                 "left_hand", "left_middle", "right_hand", "right_middle")})


def kinematics(state):
    """(T,117) -> pelvis position (T,3), body orientation (T,3,3) (rows: left, forward, up), hands {side: (T,3)}, shoulders {side: (T,3)}. MuJoCo world coordinates."""
    import mujoco
    from humanml3d_to_video import state_to_qpos
    m, d, ids = _W["m"], _W["d"], _W["ids"]
    q = state_to_qpos(state[:, :117])
    T = len(q)
    pel = np.zeros((T, 3))
    basis = np.zeros((T, 3, 3))
    hand = {s: np.zeros((T, 3)) for s in ("left", "right")}
    sho = {s: np.zeros((T, 3)) for s in ("left", "right")}
    up = np.array([0.0, 0, 1])
    for t in range(T):
        d.qpos[:] = q[t]
        mujoco.mj_kinematics(m, d)
        x = d.xpos
        left = x[ids["left_upper_leg"]] - x[ids["right_upper_leg"]]
        left[2] = 0
        left /= np.linalg.norm(left) + 1e-9
        basis[t] = [left, np.cross(left, up), up]
        pel[t] = x[ids["pelvis"]]
        for s in ("left", "right"):
            hand[s][t] = (x[ids[f"{s}_hand"]] + x[ids[f"{s}_middle"]]) / 2
            sho[s][t] = x[ids[f"{s}_upper_arm"]]
    return pel, basis, hand, sho


def to_body(p, pel, basis):
    """World point p ((T,3) or (3,)) to (left, forward, up) as seen from each frame's pelvis."""
    return np.einsum("tij,tj->ti", basis, np.broadcast_to(p, pel.shape) - pel)


def apexes(pel, basis, hand, sho):
    """Apexes for hindsight goals [(t, world position, side)]. Judged on the hand trajectory seen from the pelvis (conditions in the module docstring)."""
    out = []
    W = int(round(WINDOW * FPS))
    for s in ("left", "right"):
        rel = to_body(hand[s], pel, basis)                      # hand seen from the pelvis
        v = np.linalg.norm(np.gradient(rel, axis=0), axis=1) * FPS
        slow = v < V_APEX
        for t in range(1, len(rel) - 1):
            if not (v[t] <= v[t - 1] and v[t] <= v[t + 1] and slow[t]):
                continue
            a, b = t, t
            while a > 0 and slow[a - 1]:
                a -= 1
            while b < len(v) - 1 and slow[b + 1]:
                b += 1
            t0 = max(0, t - W)
            if b - a + 1 < HOLD or np.linalg.norm(rel[t] - rel[t0]) < NET or v[t0:t + 1].max() < PEAK:
                continue
            out.append((t, hand[s][t].copy(), s))
    out.sort(key=lambda x: x[0])
    merged = []
    for a in out:
        if merged and a[0] - merged[-1][0] < 0.5 * FPS:
            continue
        merged.append(a)
    return merged


def label(state, obj=None):
    """(T,117) and object states -> (T,4) goals, and a breakdown."""
    pel, basis, hand, sho = kinematics(state)
    T = len(state)
    goal = np.zeros((T, GOAL_DIM), np.float32)
    src = np.zeros(T, np.int8)                                   # 0 none / 1 object / 2 hindsight
    W = int(round(WINDOW * FPS))
    start = 0
    if obj is not None and obj["mode"][0] == "rest":
        g = next(t for t, x in enumerate(obj["mode"]) if x.startswith("held"))
        m = obj["mode"][g]
        p = (hand["left"][g] + hand["right"][g]) / 2 if m == "held_both" else hand[m[5:]][g]
        goal[:g + 1, :3] = to_body(p, pel[:g + 1], basis[:g + 1])
        goal[:g + 1, 3] = 1
        src[:g + 1] = 1
        start = g + 1
    aps = [a for a in apexes(pel, basis, hand, sho) if a[0] >= start]
    for t in range(start, T):
        nxt = next((a for a in aps if t <= a[0] <= t + W), None)
        if nxt is not None:
            goal[t, :3] = to_body(nxt[1], pel[t:t + 1], basis[t:t + 1])[0]
            goal[t, 3] = 1
            src[t] = 2
    return goal, src, aps


def _job(args):
    src, dst, ep, objp, no_goal = args
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(src)
    col = t.column("observation.state").combine_chunks()
    st = col.values.to_numpy().reshape(len(col), -1).astype(np.float32)
    assert st.shape[1] in (117, 117 + GOAL_DIM), st.shape
    st = st[:, :117]                                             # re-assign even for data that already has goals (121)
    obj = json.loads(pathlib.Path(objp).read_text()) if objp else None
    goal, srcs, aps = label(st, obj)
    if no_goal:                                                  # control: same shape with only the goal removed (flag 0)
        goal[:] = 0
        srcs[:] = 0
    new = np.concatenate([st, goal], 1)
    cols = {c: t.column(c) for c in t.column_names}
    cols["observation.state"] = pa.array(list(new), type=pa.list_(pa.float32()))
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst)
    return dict(ep=ep, T=len(st), n_obj=int((srcs == 1).sum()), n_hind=int((srcs == 2).sum()), n_apex=len(aps),
                sum=new.sum(0).tolist(), sq=(new.astype(np.float64) ** 2).sum(0).tolist(), mn=new.min(0).tolist(), mx=new.max(0).tolist())


def write(src, out, xml, workers, sample=1, no_goal=False):
    src, out = pathlib.Path(src), pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    info = json.loads((src / "meta/info.json").read_text())
    eps = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l][::sample]
    jobs = []
    for e in eps:
        i = e["episode_index"]
        rel = info["data_path"].format(episode_chunk=i // info.get("chunks_size", 1000), episode_index=i)
        objp = src / "objects" / f"episode_{i:06d}.json"
        jobs.append((str(src / rel), str(out / rel), i, str(objp) if objp.exists() else None, no_goal))
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(xml),)) as pool:
        res = sorted(pool.imap_unordered(_job, jobs, chunksize=8), key=lambda r: r["ep"])
    # Videos, object states and meta stay as they are (videos are linked). Only the state shape and ranges are updated
    for name in ("videos", "objects"):
        if (src / name).exists():
            os.symlink((src / name).resolve(), out / name)
    for f in (src / "meta").iterdir():
        if f.name not in ("info.json", "modality.json", "stats.json", "relative_stats.json", "episodes.jsonl"):
            shutil.copy(f, out / "meta" / f.name)
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps))
    info["total_episodes"] = len(res)
    info["total_frames"] = sum(r["T"] for r in res)
    info["features"]["observation.state"]["shape"] = [117 + GOAL_DIM]
    names = info["features"]["observation.state"]["names"]
    info["features"]["observation.state"]["names"] = names if "goal" in names else names + ["goal"]
    info.setdefault("cbd", {})["goal"] = dict(
        note="state の 117〜120 に目標（今の腰から見た 左・前・上 の m と、有無 0/1）。vla/goal_labels.py",
        no_goal=no_goal, window_sec=WINDOW, v_apex=V_APEX, hold_frames=HOLD, net_m=NET, peak_mps=PEAK, source_dataset=str(src))
    (out / "meta/info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    mod = json.loads((src / "meta/modality.json").read_text())
    mod["state"]["goal"] = {"start": 117, "end": 117 + GOAL_DIM}
    (out / "meta/modality.json").write_text(json.dumps(mod, indent=1))
    n = sum(r["T"] for r in res)
    s, sq = np.sum([r["sum"] for r in res], 0), np.sum([r["sq"] for r in res], 0)
    mean = s / n
    stat = dict(mean=mean.tolist(), std=np.sqrt(np.maximum(sq / n - mean ** 2, 0)).tolist(),
                min=np.min([r["mn"] for r in res], 0).tolist(), max=np.max([r["mx"] for r in res], 0).tolist())
    old = json.loads((src / "meta/stats.json").read_text()) if (src / "meta/stats.json").exists() else {}
    (out / "meta/stats.json").write_text(json.dumps({"observation.state": stat, "action": old.get("action", {})}))
    n_obj, n_hind = sum(r["n_obj"] for r in res), sum(r["n_hind"] for r in res)
    with_goal = sum(1 for r in res if r["n_obj"] + r["n_hind"] > 0)
    print(f"{len(res)} 本 / {n} フレーム -> {out}")
    print(f"  目標あり: 物 {n_obj} フレーム（{n_obj / n:.1%}）・事後 {n_hind} フレーム（{n_hind / n:.1%}）・"
          f"目標が 1 フレームでもある本 {with_goal}（{with_goal / len(res):.1%}）・頂点 {sum(r['n_apex'] for r in res)} 個")
    return res


def check_objects(src, xml):
    """For clips with an object, can hindsight goals (which use no object information) find the actual grasp moment and position?"""
    from humanml3d_to_video import load_state
    _init(xml)
    src = pathlib.Path(src)
    info = json.loads((src / "meta/info.json").read_text())
    rows = []
    for op in sorted((src / "objects").glob("episode_*.json")):
        obj = json.loads(op.read_text())
        if obj["mode"][0] != "rest":
            continue
        i = int(op.stem.split("_")[1])
        st = load_state(src / info["data_path"].format(episode_chunk=i // info.get("chunks_size", 1000), episode_index=i))
        pel, basis, hand, sho = kinematics(st)
        g = next(t for t, x in enumerate(obj["mode"]) if x.startswith("held"))
        m = obj["mode"][g]
        p = (hand["left"][g] + hand["right"][g]) / 2 if m == "held_both" else hand[m[5:]][g]
        aps = apexes(pel, basis, hand, sho)
        near = min(aps, key=lambda a: abs(a[0] - g), default=None)
        dt = (near[0] - g) / FPS if near else np.nan
        dp = np.linalg.norm(near[1] - p) if near else np.nan
        rows.append((i, obj["label"], g, len(aps), dt, dp))
        print(f"  ep {i:2d} {obj['label']:6s} 掴む t={g / FPS:4.1f}s  頂点 {len(aps)} 個  一番近い頂点: "
              f"{dt:+5.2f} s・{dp * 100:5.1f} cm" if near else f"  ep {i:2d} 頂点なし")
    ok = [r for r in rows if abs(r[4]) <= 0.5 and r[5] <= 0.10]
    print(f"掴んだ瞬間を ±0.5 秒・10 cm 以内で拾えた: {len(ok)} / {len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--xml", default=str(XML_DEFAULT))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check-objects", action="store_true")
    ap.add_argument("--no-goal", action="store_true",
                    help="同じ 121 次元の形で目標をすべて「なし」にする（H-1 / H-3 を目標なしでも測る対照用）")
    ap.add_argument("--sample", type=int, default=1, help="k 本おきに間引く（試し用。episodes.jsonl も間引いた分だけ書く）")
    a = ap.parse_args()
    if a.check_objects:
        check_objects(a.src, a.xml)
        return
    write(a.src, a.out, a.xml, a.workers, a.sample, a.no_goal)


if __name__ == "__main__":
    main()
