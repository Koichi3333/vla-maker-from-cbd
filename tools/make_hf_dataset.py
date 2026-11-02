#!/usr/bin/env python3
"""Cut the synthetic-only subset of data/video_reach into a self-contained LeRobot dataset for Hugging Face. Codespace, 0 CU.

The 24 real episodes (13 train + 11 held-out) come from the author's own videos and stay out. What is left is the 647
rollouts of an earlier model (arm B + inference-time guidance), cut to the reaching segment. Episodes are renumbered from 0.

    python3 tools/make_hf_dataset.py                       # writes work/hf_dataset/
    python3 tools/make_hf_dataset.py --check               # also verifies the result (shapes, ids, no real episode left)

Statistics (meta/stats.json) are recomputed on the subset. The same code run on the full dataset reproduces the shipped
meta/stats.json (checked by --check), so the numbers are comparable.
"""
import argparse
import json
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "video_reach"
CARD = ROOT / "hf_dataset" / "README.md"   # not under hf/: tools/hf_update_meta.py uploads all of hf/ to the model repo


def read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p, rows):
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def stats_of(arrays):
    x = np.concatenate(arrays).astype(np.float64)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "max": x.max(0).tolist(), "min": x.min(0).tolist()}


def episode_arrays(src_dir, eps):
    for e in eps:
        t = pq.read_table(src_dir / "data/chunk-000" / f"episode_{e['episode_index']:06d}.parquet")
        yield e, t


def build(src, out, keep):
    """Write the episodes where keep(e) is true into out. Returns (episodes, stats)."""
    eps = [e for e in read_jsonl(src / "meta/episodes.jsonl") if keep(e)]
    old_tasks = {t["task_index"]: t["task"] for t in read_jsonl(src / "meta/tasks.jsonl")}
    used = sorted({t for e in eps for t in e["tasks"]})
    task_id = {t: i for i, t in enumerate(used)}
    assert all(t in old_tasks.values() for t in used)

    if out.exists():
        shutil.rmtree(out)
    for d in ("data/chunk-000", "meta", "objects"):
        (out / d).mkdir(parents=True)

    new_eps, S, n_total = [], [], 0
    for new, (e, t) in enumerate(episode_arrays(src, eps)):
        n = t.num_rows
        assert n == e["length"], (e["episode_index"], n, e["length"])
        ti = task_id[e["tasks"][0]]
        cols = {name: t.column(name) for name in ("observation.state", "action", "timestamp", "frame_index", "next.done")}
        cols["episode_index"] = pa.array([new] * n, pa.int64())
        cols["index"] = pa.array(np.arange(n_total, n_total + n), pa.int64())
        cols["task_index"] = pa.array([ti] * n, pa.int64())
        cols["annotation.human.task_description"] = pa.array([ti] * n, pa.int64())
        pq.write_table(pa.table({k: cols[k] for k in t.column_names}), out / "data/chunk-000" / f"episode_{new:06d}.parquet")
        obj = src / "objects" / f"episode_{e['episode_index']:06d}.json"
        if obj.exists():
            shutil.copy(obj, out / "objects" / f"episode_{new:06d}.json")
        S.append(np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)))
        new_eps.append(dict(e, episode_index=new, source_episode_in_video_reach=e["episode_index"]))
        n_total += n

    write_jsonl(out / "meta/episodes.jsonl", new_eps)
    write_jsonl(out / "meta/tasks.jsonl", [{"task_index": i, "task": t} for t, i in task_id.items()])
    stats = stats_of(S)
    stats = {"observation.state": stats, "action": stats}     # same as video_to_lerobot.py: both use the state statistics
    json.dump(stats, open(out / "meta/stats.json", "w"))
    for name in ("modality.json", "mirror.json"):
        shutil.copy(src / "meta" / name, out / "meta" / name)

    info = json.load(open(src / "meta/info.json"))
    info.update(total_episodes=len(new_eps), total_frames=n_total, total_tasks=len(task_id))
    info["cbd"] = dict(info["cbd"], skipped=[], split="meta/split.json (everything is train; there is no held-out set here)",
                       source="Rollouts of an earlier model (arm B + inference-time guidance). No recording of a person is included.")
    info["cbd"].pop("synthetic_source", None)
    json.dump(info, open(out / "meta/info.json", "w"), indent=1, ensure_ascii=False)
    json.dump({"train": list(range(len(new_eps))), "heldout": []}, open(out / "meta/split.json", "w"))
    synth = json.load(open(src / "meta/synth_report.json"))
    json.dump(synth, open(out / "meta/synth_report.json", "w"), indent=1, ensure_ascii=False)
    return new_eps, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "work" / "hf_dataset"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    eps, stats = build(SRC, out, lambda e: "synthetic" in e)
    if CARD.exists():
        shutil.copy(CARD, out / "README.md")
    print(f"{len(eps)} episodes, {sum(e['length'] for e in eps)} frames -> {out}")

    if a.check:
        # 1. the stats code reproduces the shipped numbers on the full dataset
        tmp = out.parent / "hf_dataset_fullcheck"
        _, full = build(SRC, tmp, lambda e: True)
        shipped = json.load(open(SRC / "meta/stats.json"))
        for k in ("observation.state", "action"):
            for f in ("mean", "std", "max", "min"):
                d = np.abs(np.array(full[k][f]) - np.array(shipped[k][f])).max()
                assert d < 1e-4, (k, f, d)
        shutil.rmtree(tmp)
        # 2. nothing real is left, ids are dense, every episode has a task, object files line up
        rows = read_jsonl(out / "meta/episodes.jsonl")
        assert all("synthetic" in r for r in rows)
        assert [r["episode_index"] for r in rows] == list(range(len(rows)))
        ntask = len(read_jsonl(out / "meta/tasks.jsonl"))
        seen = 0
        for r in rows:
            t = pq.read_table(out / "data/chunk-000" / f"episode_{r['episode_index']:06d}.parquet")
            assert t.num_rows == r["length"]
            assert set(t.column("episode_index").to_pylist()) == {r["episode_index"]}
            assert t.column("index")[0].as_py() == seen
            assert 0 <= t.column("task_index")[0].as_py() < ntask
            seen += t.num_rows
        print("check ok: stats match the shipped file on the full set; subset has", len(rows), "synthetic episodes only")


if __name__ == "__main__":
    main()
