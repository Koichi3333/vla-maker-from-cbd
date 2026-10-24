#!/usr/bin/env python3
"""Assemble V-5's training data (runs on the VM). Add the video train set, duplicated `--dup` times, to HumanML3D's train set.

- **The HumanML3D side is not copied.** `data/chunk-XXX` and `videos/chunk-XXX` are symlinked (they are several GB)
- The video side gets `episode_index` renumbered from 20000 (the `index` column from 10 million) and goes into `chunk-020` and later (HumanML3D's full numbering is 0-10426.
  GR00T's loader just reads episodes.jsonl in order, so gaps in the numbering are fine; confirmed with groot_split.py during development)
- **Duplicates symlink the mp4s too** (it only shows the same picture several times). Parquets get `episode_index` / `index` /
  `task_index` rewritten and become separate files
- **Normalization statistics stay those of HumanML3D's train set** (`stats.json` / `relative_stats.json` are copied).
  Training starts from checkpoint-47000, so both are compared with the same statistics
- The scoring set (heldout in `split.json`) is written to `--heldout-out` as a separate dataset (for H-3's open-loop)

    python3 mix_datasets.py --hml /content/b3/train --video /content/v5/lerobot_video --dup 40 \\
        --out /content/v5/mix --heldout-out /content/v5/video_heldout
"""
import argparse
import json
import os
import pathlib
import shutil

import numpy as np

VIDEO = "observation.images.front_view"


def read_jsonl(p):
    return [json.loads(line) for line in pathlib.Path(p).read_text().splitlines() if line]


def write_jsonl(p, rows):
    pathlib.Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def rewrite(src, dst, ep, index0, task_map):
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(src)
    n = t.num_rows
    cols = {c: t.column(c) for c in t.column_names}
    cols["episode_index"] = pa.array(np.full(n, ep, np.int64))
    cols["index"] = pa.array(np.arange(index0, index0 + n, dtype=np.int64))
    ti = np.vectorize(task_map.get)(t.column("task_index").to_numpy()).astype(np.int64)
    cols["task_index"] = pa.array(ti)
    cols["annotation.human.task_description"] = pa.array(ti)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), dst)
    return n


def build(hml, video, eps_video, out, dup, start_ep=20000):
    hinfo = json.loads((hml / "meta/info.json").read_text())
    vinfo = json.loads((video / "meta/info.json").read_text())
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    heps = read_jsonl(hml / "meta/episodes.jsonl") if dup is not None else []
    htasks = read_jsonl(hml / "meta/tasks.jsonl") if dup is not None else []
    if dup is not None:                                # the HumanML3D side is symlinked
        for kind in ("data", "videos"):
            (out / kind).mkdir()
            for ch in sorted((hml / kind).iterdir()):
                os.symlink(ch.resolve(), out / kind / ch.name)
    vtasks = {r["task_index"]: r["task"] for r in read_jsonl(video / "meta/tasks.jsonl")}
    tasks = list(htasks)
    tid = {r["task"]: r["task_index"] for r in tasks}
    task_map = {}
    for k, text in vtasks.items():
        if text not in tid:
            tid[text] = len(tasks)
            tasks.append({"task_index": tid[text], "task": text})
        task_map[k] = tid[text]
    # The index column of HumanML3D's train set keeps the running number over all clips (up to about 1.51 million), so start from 10 million to avoid overlap
    index0 = 10_000_000 if heps else 0
    new_eps, ep = [], start_ep
    for r in range(dup or 1):
        for e in eps_video:
            i = e["episode_index"]
            src = video / vinfo["data_path"].format(episode_chunk=0, episode_index=i)
            ch = ep // hinfo["chunks_size"]
            dst = out / hinfo["data_path"].format(episode_chunk=ch, episode_index=ep)
            n = rewrite(src, dst, ep, index0, task_map)
            vsrc = video / vinfo["video_path"].format(episode_chunk=0, video_key=VIDEO, episode_index=i)
            vdst = out / hinfo["video_path"].format(episode_chunk=ch, video_key=VIDEO, episode_index=ep)
            vdst.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(vsrc.resolve(), vdst)
            new_eps.append(dict(e, episode_index=ep, copy=r, source_episode=i))
            index0 += n
            ep += 1
    eps = heps + new_eps
    info = dict(hinfo)
    info.update(total_episodes=len(eps), total_frames=int(sum(e["length"] for e in eps)), total_tasks=len(tasks),
                total_videos=len(eps), total_chunks=max(e["episode_index"] for e in eps) // hinfo["chunks_size"] + 1)
    info.setdefault("cbd", {})["v5_mix"] = dict(
        humanml3d_episodes=len(heps), video_episodes=len(eps_video), dup=dup,
        video_frames=int(sum(e["length"] for e in eps_video)),
        video_share_of_frames=float(sum(e["length"] for e in new_eps) / max(1, info["total_frames"])))
    (out / "meta/info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    write_jsonl(out / "meta/episodes.jsonl", eps)
    write_jsonl(out / "meta/tasks.jsonl", tasks)
    for f in ("modality.json", "stats.json", "relative_stats.json", "mirror.json"):
        if (hml / "meta" / f).exists():
            shutil.copy(hml / "meta" / f, out / "meta" / f)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hml", required=True, help="HumanML3D の train（/content/b3/train）")
    ap.add_argument("--video", required=True, help="video_to_lerobot.py の出力（レンダ済み）")
    ap.add_argument("--dup", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-out", required=True)
    a = ap.parse_args()
    hml, video = pathlib.Path(a.hml), pathlib.Path(a.video)
    split = json.loads((video / "meta/split.json").read_text())
    veps = {e["episode_index"]: e for e in read_jsonl(video / "meta/episodes.jsonl")}
    info = build(hml, video, [veps[i] for i in split["train"]], pathlib.Path(a.out), a.dup)
    mix = info["cbd"]["v5_mix"]
    print(f"[mix] HumanML3D {mix['humanml3d_episodes']} 本 + 動画 {mix['video_episodes']} 本 × {a.dup} "
          f"= {info['total_episodes']} 本 / {info['total_frames']} フレーム  動画のフレームの割合 "
          f"{mix['video_share_of_frames']:.4f} -> {a.out}")
    # Scoring set: no HumanML3D, only the video heldout (renumbered from 20000; statistics from HumanML3D's train set)
    hinfo = build(hml, video, [veps[i] for i in split["heldout"]], pathlib.Path(a.heldout_out), None)
    print(f"[heldout] 動画の採点用 {hinfo['total_episodes']} 本 / {hinfo['total_frames']} フレーム -> {a.heldout_out}")


if __name__ == "__main__":
    main()
