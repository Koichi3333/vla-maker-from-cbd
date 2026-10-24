#!/usr/bin/env python3
"""Add synthetic videos to a LeRobot dataset so that GR00T's loader can read it.

The output of cbd_to_lerobot.py has no video. GR00T N1.7 requires video, so to be able to run before rendering,
this writes synthetic mp4s exactly as long as each episode and registers the video in
meta/info.json and meta/modality.json.

    --pattern moving  for checking the scaffolding in B-0. The picture changes every frame (passes the still-image check)
    --pattern black   for the visual ablation in B-2b. The same black image in every frame
    --pattern render  the real thing from B-2 on. Renders the state with MuJoCo (humanml3d_to_video.py)

parquet, episodes and tasks are not touched (the first --episodes clips, or one side of --split, are copied as is).
stats.json has no q01/q99, so it is deleted. Rebuild it with GR00T's gr00t/data/stats.py.

render can resume if it dies on the full run (B-3). mp4s are written to a temporary file and renamed, so
"if it exists it is complete", and --skip-existing renders only the rest. When all mp4s of a chunk are present,
it writes <dst>/render_done/chunk-XXX.json (a completion marker).

Runs on the VM with only numpy and ffmpeg (no pandas). render also needs mujoco and pyarrow.
"""
import argparse
import json
import pathlib
import shutil
import subprocess

import numpy as np

VIDEO_KEY = "observation.images.front_view"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def frames(pattern, n, size):
    """Yield (n, size, size, 3) uint8 one frame at a time."""
    yy, xx = np.mgrid[0:size, 0:size]
    for t in range(n):
        if pattern == "black":
            yield np.zeros((size, size, 3), np.uint8)
            continue
        img = np.empty((size, size, 3), np.uint8)
        img[..., 0] = (xx + 3 * t) % 256
        img[..., 1] = (yy + 5 * t) % 256
        img[..., 2] = 128
        s = (7 * t) % (size - 32)
        img[s:s + 32, s:s + 32] = 255
        yield img


def ffmpeg_exe():
    """The VM has ffmpeg. The Codespace does not, so use the one bundled with imageio_ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encode(path, it, size, fps):
    """Write a sequence of (size, size, 3) uint8 frames as an h264 mp4.

    It writes to a temporary file and then renames, so if path exists it is complete
    (if it dies midway, just rebuild it on resume).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.mp4")
    cmd = [ffmpeg_exe(), "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-g", str(int(fps)),
           str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in it:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {path}")
    tmp.replace(path)


def render(args, src, dst, info, episodes):
    """Render the state into mp4s, and write a completion marker each time a chunk is complete."""
    import collections
    import os
    import time
    import humanml3d_to_video as hv
    os.environ.setdefault("MUJOCO_GL", "egl")    # children use spawn, so this takes effect before import
    chunk_of = lambda i: i // info["chunks_size"]
    length = {e["episode_index"]: e["length"] for e in episodes}
    jobs = hv.jobs_for(src, dst, episodes)
    todo = [j for j in jobs if not (args.skip_existing and pathlib.Path(j[2]).exists())]
    left = collections.Counter(chunk_of(j[0]) for j in todo)
    members = collections.defaultdict(list)
    for e in episodes:
        members[chunk_of(e["episode_index"])].append(e["episode_index"])
    marks = dst / "render_done"
    marks.mkdir(exist_ok=True)

    def mark(ch):
        eps = members[ch]
        (marks / f"chunk-{ch:03d}.json").write_text(json.dumps(dict(
            chunk=ch, episodes=len(eps), frames=sum(length[i] for i in eps),
            first=min(eps), last=max(eps), at=time.strftime("%Y-%m-%dT%H:%M:%S"))))

    for ch in members:
        if left[ch] == 0:                       # on resume: everything is already there
            mark(ch)

    def done(r):
        ch = chunk_of(r["ep"])
        left[ch] -= 1
        if left[ch] == 0:
            mark(ch)

    print(f"render: {len(todo)} / {len(jobs)} 本を焼く（残りは既存）")
    res, wall = hv.render_jobs(todo, args.xml or hv.HERE / "humanoid.xml",
                               args.size, args.workers, on_result=done)
    still = hv.still_eps(res)
    if args.report:
        F = sum(r["T"] for r in res)
        B = sum(r["bytes"] for r in res)
        summary = dict(episodes=len(jobs), rendered=len(res), skipped=len(jobs) - len(res),
                       frames=F, wall_sec=wall, fps_aggregate=F / wall if wall else None,
                       workers=args.workers, bytes=B, bytes_per_frame=B / F if F else None,
                       still_videos=still,
                       same_frac_max=max((r["same_frac"] for r in res), default=None))
        print(json.dumps(summary, indent=1))
        pathlib.Path(args.report).write_text(json.dumps(
            dict(summary=summary, clips=sorted(res, key=lambda r: r["ep"])), indent=1))
    assert not still, f"静止画になっている: {still}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="cbd_to_lerobot.py の出力")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--episodes", type=int, default=None, help="先頭から何本使うか（既定: 全部）")
    ap.add_argument("--split", default=None, help="分け方の JSON（vla/full_split.json の形式）。--part の側だけを使う")
    ap.add_argument("--part", choices=["train", "heldout", "excluded"], default="train",
                    help="excluded は full_split.json の「held-out と同じ試行」のクリップ（学習に使わない）")
    ap.add_argument("--pattern", choices=["moving", "black", "render"], default="moving")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8, help="render のプロセス数")
    ap.add_argument("--xml", default=None, help="render の humanoid.xml（既定: work/cbd_ref/humanoid.xml）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="render で既にある mp4 は焼かない（途中で落ちたときの再開用）")
    ap.add_argument("--report", default=None, help="render の結果（1本ごとの静止画判定・容量・所要）を JSON で書く")
    args = ap.parse_args()

    src, dst = pathlib.Path(args.src), pathlib.Path(args.dst)
    info = json.loads((src / "meta/info.json").read_text())
    episodes = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l]
    if args.split:
        keep = set(json.loads(pathlib.Path(args.split).read_text())[args.part])
        episodes = [e for e in episodes if e["episode_index"] in keep]
        assert len(episodes) == len(keep), "split の ID がデータセットに無い"
    if args.episodes is not None:
        episodes = episodes[:args.episodes]
    fps = info["fps"]

    (dst / "meta").mkdir(parents=True, exist_ok=True)
    for name in ["tasks.jsonl", "mirror.json"]:
        if (src / "meta" / name).exists():
            shutil.copy(src / "meta" / name, dst / "meta" / name)
    (dst / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))

    for e in episodes:
        i = e["episode_index"]
        chunk = i // info["chunks_size"]
        rel = info["data_path"].format(episode_chunk=chunk, episode_index=i)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / rel, dst / rel)
        if args.pattern != "render":
            encode(dst / VIDEO_PATH.format(episode_chunk=chunk, video_key=VIDEO_KEY, episode_index=i),
                   frames(args.pattern, e["length"], args.size), args.size, fps)
    if args.pattern == "render":
        render(args, src, dst, info, episodes)

    info["total_episodes"] = len(episodes)
    info["total_frames"] = sum(e["length"] for e in episodes)
    info["total_videos"] = len(episodes)
    info["total_chunks"] = max(e["episode_index"] for e in episodes) // info["chunks_size"] + 1
    info["video_path"] = VIDEO_PATH
    info["features"][VIDEO_KEY] = {
        "dtype": "video",
        "shape": [args.size, args.size, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.height": args.size, "video.width": args.size, "video.codec": "h264",
                 "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps,
                 "video.channels": 3, "has_audio": False},
    }
    info.setdefault("cbd", {})["video_pattern"] = args.pattern
    (dst / "meta/info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))

    modality = json.loads((src / "meta/modality.json").read_text())
    modality["video"] = {"front_view": {"original_key": VIDEO_KEY}}
    (dst / "meta/modality.json").write_text(json.dumps(modality, indent=1))

    print(f"{len(episodes)} episodes / {info['total_frames']} frames / pattern={args.pattern} -> {dst}")


if __name__ == "__main__":
    main()
