#!/usr/bin/env python3
"""Render third-person tracking mp4s from the observation.state of a LeRobot dataset.

For GR00T N1.7's video input (observation.images.front_view). Settings fixed in B-1:
256x256 / 20 fps / azimuth 135, elevation -12, distance 3.2, lookat [x, y, 0.9].

cbd_to_mujoco.py reads frames.jsonl, but jsonl exists for only 421 clips (writing all of them would
take 5.1 GB, so it was not done). This script builds qpos directly from the 117 dims in the parquet written by
cbd_to_lerobot.py (root translation + 6D for 19 bones). Both the 421 clips (--source cbd) and
all clips (--source humanml3d) render through the same path, and **the observation image is a rendering of the
training state itself**, so frames stay aligned.

    MUJOCO_GL=egl python3 humanml3d_to_video.py --dataset work/lerobot_humanml3d \\
        --out /tmp/b1 --sample 10 --workers 8 --check

Output is <out>/videos/chunk-XXX/observation.images.front_view/episode_XXXXXX.mp4
(same layout as lerobot_add_video.py). meta is not touched. To update meta as well and have GR00T
read the result, use lerobot_add_video.py --pattern render.

Parallelism is per process. One mujoco.Renderer is reused within a process (creating one per clip
exhausts GL contexts and gives stills). The parent process does not import mujoco
(MUJOCO_GL is read at import time, so this makes sure it takes effect in the children).
"""
import argparse
import json
import multiprocessing as mp
import os
import pathlib
import subprocess
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TOTAL_FRAMES = 1_510_434      # all standing clips (10,427). Used to estimate the full run

_W = {}


def state_to_qpos(state):
    """(T,117) -> (T,81). 6D is turned back into quaternions by Gram-Schmidt (round-trip error below 1e-5 degrees)."""
    from cbd_to_lerobot import BONES
    from cbd_to_mujoco import qpos_from_rotations
    from rotation import sixd_to_quat
    T = len(state)
    Q = sixd_to_quat(np.asarray(state[:, 3:], np.float64).reshape(T, len(BONES), 6))
    return qpos_from_rotations({b: Q[:, i] for i, b in enumerate(BONES)}, state[:, :3])


def load_state(parquet):
    import pyarrow.parquet as pq
    col = pq.read_table(parquet, columns=["observation.state"]).column(0).combine_chunks()
    return col.values.to_numpy().reshape(len(col), -1)


def _init(xml, size):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml))
    _W.update(m=m, d=mujoco.MjData(m), r=mujoco.Renderer(m, size, size), size=size)


def decode(path):
    """Decode an mp4 back to (n, H, W, 3) uint8. Checks the encoded picture, as GR00T's loader sees it."""
    from lerobot_add_video import ffmpeg_exe
    p = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(path),
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True)
    s = _W["size"]
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, s, s, 3)


def _count_same(it, box):
    """Pass frames through while counting how many are **byte-identical** to the previous one.

    The still-image bug (the Renderer keeps returning the same image) is checked on raw frames before encoding.
    Differences in a decoded mp4 cannot tell them apart: encoding the same image repeatedly still gives an adjacent difference of 0.76
    from x264 P-frame noise, while the smallest in the real data was 0.96 (an almost motionless standing clip) (measured in B-1).
    """
    prev = None
    for f in it:
        if prev is not None and np.array_equal(f, prev):
            box[0] += 1
        prev = f.copy()
        yield f


def _render_one(job):
    from cbd_to_mujoco import frames, ground
    from lerobot_add_video import encode
    ep, parquet, dst, fps, check = job
    m, d, r = _W["m"], _W["d"], _W["r"]
    t0 = time.perf_counter()
    q = state_to_qpos(load_state(parquet))
    lift = -ground(m, d, q)
    t1 = time.perf_counter()
    same = [0]
    encode(pathlib.Path(dst), _count_same(frames(m, d, q, r), same), _W["size"], fps)
    t2 = time.perf_counter()
    out = dict(ep=ep, T=len(q), bytes=os.path.getsize(dst), lift=lift,
               same_frac=same[0] / max(1, len(q) - 1),
               t_prep=t1 - t0, t_render=t2 - t1, pid=os.getpid())
    if check:
        v = decode(dst).astype(np.int16)
        out.update(n_decoded=len(v),
                   # For reference: maximum difference from the first frame (mean over pixels, 0-255).
                   # Minimum over 621 real clips 2.72; the same image encoded repeatedly 0.78
                   diff_from_first=float(np.abs(v - v[0]).mean(axis=(1, 2, 3)).max()),
                   t_check=time.perf_counter() - t2)
    return out


def render_jobs(jobs, xml, size, workers, log=print, on_result=None):
    """jobs: list of (ep, parquet, dst_mp4, fps, check). Returns results in completion order.

    on_result is called with the result dict each time a clip finishes (for chunk completion markers).
    """
    res = []
    ctx = mp.get_context("spawn")      # do not carry the parent's state (pyarrow threads etc.) into the children
    t0 = time.perf_counter()
    with ctx.Pool(workers, initializer=_init, initargs=(xml, size)) as pool:
        for i, r in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=1)):
            res.append(r)
            if on_result:
                on_result(r)
            if log and (i + 1) % max(1, len(jobs) // 20) == 0:
                f = sum(x["T"] for x in res)
                log(f"  {i + 1:6d}/{len(jobs)}  {f / (time.perf_counter() - t0):7.1f} fps")
    return res, time.perf_counter() - t0


def still_eps(res):
    """With the still-image bug, every frame matches the previous one. Real data should be at 0%, so the threshold is half."""
    return [r["ep"] for r in res if r["same_frac"] >= 0.5]


def jobs_for(dataset, out, episodes, check=False):
    from lerobot_add_video import VIDEO_KEY, VIDEO_PATH
    info = json.loads((dataset / "meta/info.json").read_text())
    jobs = []
    for e in episodes:
        i = e["episode_index"]
        ch = i // info["chunks_size"]
        jobs.append((i, str(dataset / info["data_path"].format(episode_chunk=ch, episode_index=i)),
                     str(out / VIDEO_PATH.format(episode_chunk=ch, video_key=VIDEO_KEY, episode_index=i)),
                     info["fps"], check))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="cbd_to_lerobot.py の出力")
    ap.add_argument("--out", required=True, help="videos/ をここに書く")
    ap.add_argument("--sample", type=int, default=None,
                    help="全エピソードから等間隔に N 本（既定: 全部）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--xml", default=str(HERE / "humanoid.xml"))
    ap.add_argument("--gl", default=os.environ.get("MUJOCO_GL", "egl"),
                    choices=["egl", "osmesa", "glfw"])
    ap.add_argument("--check", action="store_true",
                    help="焼いた mp4 を読み戻してフレーム数を検査する（静止画の検査は常に行う）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="既にある mp4 は焼かない（途中で落ちたときの再開用）")
    ap.add_argument("--report", default=None, help="結果を JSON で書く")
    a = ap.parse_args()
    os.environ["MUJOCO_GL"] = a.gl        # children start with spawn, so setting it here takes effect

    dataset, out = pathlib.Path(a.dataset), pathlib.Path(a.out)
    eps = [json.loads(l) for l in (dataset / "meta/episodes.jsonl").read_text().splitlines() if l]
    if a.sample:
        eps = [eps[k] for k in np.unique(np.linspace(0, len(eps) - 1, a.sample).round().astype(int))]
    jobs = jobs_for(dataset, out, eps, a.check)
    if a.skip_existing:
        jobs = [j for j in jobs if not pathlib.Path(j[2]).exists()]
    print(f"{len(jobs)} episodes / {sum(e['length'] for e in eps)} frames / "
          f"{a.size}px / workers={a.workers} / MUJOCO_GL={a.gl}")

    res, wall = render_jobs(jobs, a.xml, a.size, a.workers)
    res.sort(key=lambda r: r["ep"])
    F = sum(r["T"] for r in res)
    B = sum(r["bytes"] for r in res)
    busy = sum(r["t_prep"] + r["t_render"] for r in res)
    summary = dict(
        episodes=len(res), frames=F, workers=a.workers, gl=a.gl, size=a.size, wall_sec=wall,
        fps_aggregate=F / wall,                      # effective fps including parallelism (the full-run estimate uses this)
        fps_per_worker=F / busy,                     # per process
        prep_share=sum(r["t_prep"] for r in res) / busy,
        bytes_per_frame=B / F, mb_total=B / 1e6,
        est_total_hours=TOTAL_FRAMES / (F / wall) / 3600,
        est_total_gb=TOTAL_FRAMES * B / F / 1e9,
    )
    still = still_eps(res)
    summary.update(still_videos=still, same_frac_max=max(r["same_frac"] for r in res))
    if a.check:
        bad_n = [r["ep"] for r in res if r["n_decoded"] != r["T"]]
        summary.update(frame_count_mismatch=bad_n,
                       diff_from_first_min=min(r["diff_from_first"] for r in res))
    print(json.dumps(summary, indent=1))
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(dict(summary=summary, clips=res), indent=1))
    assert not still, f"静止画になっている（生フレームが直前と一致）: {still}"
    if a.check:
        assert not bad_n, f"フレーム数が parquet と合わない: {bad_n}"
        print("CHECK OK: フレーム数一致・静止画なし")


if __name__ == "__main__":
    main()
