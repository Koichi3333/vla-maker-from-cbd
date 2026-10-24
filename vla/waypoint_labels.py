#!/usr/bin/env python3
"""V-11: add a 7-d "change of both hands up to 15 frames ahead" (waypoint) to the state of a LeRobot dataset (117 -> 124).

`waypoint` (state dims 117-123):
- 0-2 / 3-5: left / right hand, **"hand seen from the pelvis at that time" K frames later minus "hand seen from the current pelvis" now** (left, forward, up in m). Hindsight (the recorded future itself).
  Whole-body movement (walking) is excluded, leaving only hand motion relative to the body. Measured in the current pelvis frame, walking clips move almost 1 m in 0.75 s, and the normalization range gets
  dragged by walking (q01-q99 width 1.25-1.9 m -> 0.88-1.37 m with this definition). Its effect on actions is also larger with this definition (421 HumanML3D clips, ridge including interactions:
  +0.169 -> +0.229, 2026-10-04). The planner at inference assumes the body stays put for a while and draws toward the object seen from the current pelvis (the two agree if the body does not move)
- 6: present (1 = yes). Frames without a waypoint have 0-5 set to 0 as well

V-11's preliminary check (`v11_future_goal.py`) found that only the hand position at the end of the chunk (K = 15) affects actions. At inference the upper layer (planner)
builds a waypoint of the same shape from the path "current hand -> object" and passes it (`--goal object` in `groot_eval.py`).
Hand positions come from the same MuJoCo skeleton forward kinematics as `goal_labels.py` (hand = midpoint of the wrist and the base of the middle finger). The last K frames use the change up to the final frame.

**Waypoint dropout:** so the model can also move without a waypoint (evaluation with off, after reaching the object), each frame is set to "none" with probability `--drop` (default 0.25).
The random seed comes from the episode number (the same on every rebuild). GR00T's state dropout would erase the waypoint too, so it is set to 0 in training (same as arm F).

    MUJOCO_GL=osmesa python3 vla/waypoint_labels.py --src work/lerobot_video_warp --out work/lerobot_video_warp_wp
    MUJOCO_GL=egl python3 waypoint_labels.py --src /content/b3/heldout --out /content/v11/hml_heldout_nowp --no-wp   # control (all none)
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
from goal_labels import XML_DEFAULT, _init, kinematics, to_body          # noqa: E402

K = 15
WP_DIM = 7
DROP = 0.25
V_PLAN = 0.5          # m/s. Average hand speed the planner uses to estimate duration (same as V-11's check)
MIN_DUR = 10          # frames
FPS = 20.0


def quintic(p0, v0, p1, D, t):
    """Position t frames into a quintic polynomial (minimum jerk) that goes from p0 (velocity v0, acceleration 0) to p1 (velocity 0, acceleration 0) in D frames (p1 if t >= D).
    v0 is per frame."""
    if t >= D:
        return np.array(p1, np.float64)
    s = t / D
    v = np.asarray(v0, np.float64) * D                        # velocity in normalized time
    d = np.asarray(p1, np.float64) - p0
    a3, a4, a5 = 10 * d - 6 * v, -15 * d + 8 * v, 6 * d - 3 * v
    return p0 + v * s + a3 * s ** 3 + a4 * s ** 4 + a5 * s ** 5


def plan(now, prev, tgt, t=K):
    """Upper layer (planner): from the current hand now, the hand one frame earlier prev and the destination tgt (all seen from the current pelvis), the hand's change up to t frames ahead.
    Duration is distance / V_PLAN (at least MIN_DUR). Called every chunk at inference (re-estimating the duration from the remaining distance)."""
    dist = np.linalg.norm(np.asarray(tgt) - now)
    D = max(MIN_DUR, dist / V_PLAN * FPS)
    return quintic(np.asarray(now, np.float64), np.asarray(now) - prev, tgt, D, t) - now


def label(state, ep, drop, no_wp=False):
    """(T,117) → (T,7)。"""
    pel, basis, hand, _ = kinematics(state)
    T = len(state)
    wp = np.zeros((T, WP_DIM), np.float32)
    if no_wp:
        return wp
    fut = np.minimum(np.arange(T) + K, T - 1)
    for j, s in enumerate(("left", "right")):
        loc = to_body(hand[s], pel, basis)                       # hand seen from each frame's pelvis
        wp[:, 3 * j:3 * j + 3] = loc[fut] - loc
    wp[:, 6] = 1.0
    keep = np.random.default_rng(ep).random(T) >= drop
    wp[~keep] = 0.0
    return wp


def _job(args):
    src, dst, ep, drop, no_wp = args
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(src)
    col = t.column("observation.state").combine_chunks()
    st = col.values.to_numpy().reshape(len(col), -1).astype(np.float32)[:, :117]   # even data with goals (121) is cut back to 117 before adding
    wp = label(st, ep, drop, no_wp)
    new = np.concatenate([st, wp], 1)
    cols = {c: t.column(c) for c in t.column_names}
    cols["observation.state"] = pa.array(list(new), type=pa.list_(pa.float32()))
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst)
    on = wp[:, 6] > 0.5
    return dict(ep=ep, T=len(st), n_on=int(on.sum()), mag=float(np.linalg.norm(wp[on, :3], axis=1).sum() + np.linalg.norm(wp[on, 3:6], axis=1).sum()),
                sum=new.sum(0).tolist(), sq=(new.astype(np.float64) ** 2).sum(0).tolist(), mn=new.min(0).tolist(), mx=new.max(0).tolist())


def write(src, out, xml, workers, drop=DROP, no_wp=False):
    src, out = pathlib.Path(src), pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    info = json.loads((src / "meta/info.json").read_text())
    eps = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l]
    jobs = []
    for e in eps:
        i = e["episode_index"]
        rel = info["data_path"].format(episode_chunk=i // info.get("chunks_size", 1000), episode_index=i)
        jobs.append((str(src / rel), str(out / rel), i, drop, no_wp))
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(xml),)) as pool:
        res = sorted(pool.imap_unordered(_job, jobs, chunksize=8), key=lambda r: r["ep"])
    for name in ("videos", "objects"):
        if (src / name).exists():
            os.symlink((src / name).resolve(), out / name)
    for f in (src / "meta").iterdir():
        if f.name not in ("info.json", "modality.json", "stats.json", "relative_stats.json", "episodes.jsonl"):
            shutil.copy(f, out / "meta" / f.name)
    (out / "meta/episodes.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in eps))
    info["total_episodes"] = len(res)
    info["total_frames"] = sum(r["T"] for r in res)
    info["features"]["observation.state"]["shape"] = [117 + WP_DIM]
    names = [n for n in info["features"]["observation.state"]["names"] if n != "goal"]
    info["features"]["observation.state"]["names"] = names + ["waypoint"]
    info.setdefault("cbd", {}).pop("goal", None)
    info["cbd"]["waypoint"] = dict(
        note="state の 117〜123 に経由点（左手・右手の、K フレーム後のその時の腰から見た手 − 今の腰から見た手。左・前・上 の m と、有無 0/1）。vla/waypoint_labels.py",
        k=K, drop=drop, no_wp=no_wp, source_dataset=str(src))
    (out / "meta/info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    mod = json.loads((src / "meta/modality.json").read_text())
    mod["state"].pop("goal", None)
    mod["state"]["waypoint"] = {"start": 117, "end": 117 + WP_DIM}
    (out / "meta/modality.json").write_text(json.dumps(mod, indent=1))
    n = sum(r["T"] for r in res)
    s, sq = np.sum([r["sum"] for r in res], 0), np.sum([r["sq"] for r in res], 0)
    mean = s / n
    stat = dict(mean=mean.tolist(), std=np.sqrt(np.maximum(sq / n - mean ** 2, 0)).tolist(),
                min=np.min([r["mn"] for r in res], 0).tolist(), max=np.max([r["mx"] for r in res], 0).tolist())
    old = json.loads((src / "meta/stats.json").read_text()) if (src / "meta/stats.json").exists() else {}
    (out / "meta/stats.json").write_text(json.dumps({"observation.state": stat, "action": old.get("action", {})}))
    n_on = sum(r["n_on"] for r in res)
    print(f"{len(res)} 本 / {n} フレーム -> {out}")
    print(f"  経由点あり {n_on} フレーム（{n_on / n:.1%}）・手の変化の平均 {sum(r['mag'] for r in res) / max(2 * n_on, 1) * 100:.1f} cm")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xml", default=str(XML_DEFAULT))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--drop", type=float, default=DROP)
    ap.add_argument("--no-wp", action="store_true", help="同じ 124 次元の形で経由点をすべて「なし」にする（H-1 / H-3 の対照）")
    a = ap.parse_args()
    write(a.src, a.out, a.xml, a.workers, a.drop, a.no_wp)


if __name__ == "__main__":
    main()
