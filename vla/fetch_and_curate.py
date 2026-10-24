#!/usr/bin/env python3
"""Assemble HumanML3D and build a subset with only the standing actions.

**This script does not download the 3D joint positions (new_joints).** HumanML3D is derived from AMASS, and AMASS's terms
forbid giving the data (including modified data) to third parties. HumanML3D's authors do not distribute the data either,
only scripts that rebuild it from AMASS. Each user registers with AMASS, agrees to its terms, downloads it, and passes
the `HumanML3D/new_joints/` built with HumanML3D's official steps
(raw_pose_processing.ipynb -> motion_representation.ipynb in https://github.com/EricGuo5513/HumanML3D)
(a folder, or a zip of it) with `--joints`. The steps are in docs/data.md.

Text annotations and lists (texts.zip, all.txt, etc.) are taken from what HumanML3D's authors distribute on GitHub (MIT).
The 263-d training features (new_joint_vecs) and the videos are not needed for CBD conversion, so they are not used.

Standing is **not judged from the wording alone**. Many standing clips do not say "stand" in the caption,
and some clips that mention sit/lie are standing (512 clips measured).
Standing is judged geometrically from joint positions, and the caption is then used to classify the action.
"""
import argparse
import collections
import io
import json
import os
import pathlib
import re
import shutil
import urllib.request
import zipfile

import numpy as np

GH = "https://raw.githubusercontent.com/EricGuo5513/HumanML3D/main"
SMALL = ["HumanML3D/all.txt", "HumanML3D/train.txt", "HumanML3D/val.txt",
         "HumanML3D/test.txt", "HumanML3D/Mean.npy", "HumanML3D/Std.npy",
         "index.csv"]

PELVIS, HEAD, LANK, RANK = 0, 15, 7, 8

# Geometric conditions for standing: head and pelvis height, trunk verticality, ankles on the ground.
STAND = dict(head_min=1.35, pelvis_min=0.75, verticality=0.88, foot_max=0.30)

CATEGORIES = {
    "walk":      (r"\bwalks?\b|\bwalking\b", r"\bstairs?\b|\bcircle\b"),
    "turn":      (r"\bturns?\b|\bturning\b|\brotates?\b|\bpivots?\b", r"\bpage\b"),
    "wave":      (r"\bwaves?\b|\bwaving\b", r"\bwave of\b"),
    "raise_arm": (r"\braises? (his|her|their|both|the )?(right |left |)"
                  r"(arm|arms|hand|hands)\b|\blifts? (his|her|their|both)? ?(arm|arms)\b", r""),
    "punch":     (r"\bpunch(es|ing)?\b|\bjabs?\b|\bboxing\b", r""),
    "kick":      (r"\bkicks?\b|\bkicking\b", r""),
    "squat":     (r"\bsquats?\b|\bsquatting\b|\bcrouch(es|ing)?\b", r""),
    "jump":      (r"\bjumps?\b|\bjumping\b|\bhops?\b|\bleaps?\b", r""),
    "clap":      (r"\bclaps?\b|\bclapping\b|\bapplaud", r""),
    "pick_up":   (r"\bpicks? (something |it |an object |a .* )?up\b|"
                  r"\bbends? (down|over) .*\b(pick|grab)", r""),
    "throw":     (r"\bthrows?\b|\bthrowing\b|\btosses?\b", r""),
    "stretch":   (r"\bstretch(es|ing)?\b", r""),
    "bow":       (r"\bbows?\b|\bbowing\b", r"\bbow and arrow\b|\bbow tie\b"),
    "drink":     (r"\bdrinks?\b|\bdrinking\b|\bsips?\b", r""),
    "sidestep":  (r"\bsteps? (to (his|her|their) )?(the )?(left|right|side)\b|"
                  r"\bside ?steps?\b|\bshuffles?\b", r""),
}


def get(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  取得 {url.rsplit('/', 1)[-1]}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def fetch(root, joints=None):
    """Fetch text annotations and lists from the authors' GitHub, and place new_joints from `joints` (built by the user)."""
    for p in SMALL:
        get(f"{GH}/{p}", root / p.rsplit("/", 1)[-1])
    tz = get(f"{GH}/HumanML3D/texts.zip", root / "texts.zip")
    if not (root / "texts").exists():
        zipfile.ZipFile(tz).extractall(root)
    if (root / "new_joints").exists():
        return
    if joints is None:
        raise SystemExit(
            "new_joints が無い。AMASS に登録・同意してダウンロードし、HumanML3D の公式手順で作った\n"
            "HumanML3D/new_joints/（フォルダか zip）を --joints で渡すこと（docs/data.md）。")
    src = pathlib.Path(joints)
    (root / "new_joints").mkdir(parents=True)
    if src.is_dir():
        files = [f for f in src.glob("*.npy") if not f.name.startswith("M")]   # Mirrored clips (M*) are made at training time
        for f in files:
            shutil.copy(f, root / "new_joints" / f.name)
    else:
        with zipfile.ZipFile(src) as z:
            files = [n for n in z.namelist() if n.endswith(".npy") and not pathlib.PurePath(n).name.startswith("M")]
            for n in files:
                (root / "new_joints" / pathlib.PurePath(n).name).write_bytes(z.read(n))
    print(f"  new_joints: {len(files)} 本を {src} から置いた")
    assert files, f"{src} に new_joints の .npy が無い"


def standing(a):
    """Return the standing mask and head height."""
    p, h = a[:, PELVIS], a[:, HEAD]
    up = h - p
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-8)
    foot = np.minimum(a[:, LANK, 1], a[:, RANK, 1])
    m = ((h[:, 1] > STAND["head_min"]) & (p[:, 1] > STAND["pelvis_min"]) &
         (up[:, 1] > STAND["verticality"]) & (foot < STAND["foot_max"]))
    return m, h[:, 1]


def scan_standing(root, lo, hi):
    """Collect every clip that starts and ends standing, and write out only ids and captions.

    Used in the all-clips mode for training. No files are copied (no point duplicating 10,000 clips).
    """
    ids, meta = [], {}
    for f in sorted(os.listdir(root / "new_joints")):
        a = np.load(root / "new_joints" / f)
        if a.ndim != 3 or not (lo <= a.shape[0] <= hi):
            continue
        m, hy = standing(a)
        if m[:10].all() and m[-10:].all() and hy.min() > 0.60:
            i = f[:-4]
            cap = open(root / "texts" / f"{i}.txt", encoding="utf-8",
                       errors="replace").readline().split("#")[0].strip()
            ids.append(i)
            meta[i] = dict(T=int(a.shape[0]), stand_frac=float(m.mean()),
                           caption=cap)
    return ids, meta


def curate(root, per_cat, lo, hi):
    """Pick clips that start and end standing by geometry, and classify them by caption."""
    meta = {}
    for f in sorted(os.listdir(root / "new_joints")):
        a = np.load(root / "new_joints" / f)
        if a.ndim != 3 or not (lo <= a.shape[0] <= hi):
            continue
        m, hy = standing(a)
        if m[:10].all() and m[-10:].all() and hy.min() > 0.60:
            meta[f[:-4]] = dict(T=int(a.shape[0]), stand_frac=float(m.mean()))
    print(f"  立位で開始・終了し {lo}〜{hi} フレーム: {len(meta)} 本")

    caps = {i: [l.split("#")[0].strip().lower()
                for l in open(root / "texts" / f"{i}.txt", encoding="utf-8",
                              errors="replace") if l.strip()] for i in meta}
    out = {}
    for name, (pos, neg) in CATEGORIES.items():
        pos, neg = re.compile(pos), (re.compile(neg) if neg else None)
        hits = []
        for i, cs in caps.items():
            k = sum(1 for c in cs
                    if pos.search(c) and not (neg and neg.search(c)))
            # Keep only clips where most captions agree. A clip caught by only one caption
            # is often not mainly about that action.
            if k / len(cs) >= 0.5:
                hits.append((k / len(cs), meta[i]["T"], i))
        out[name] = [i for _, _, i in
                     sorted(hits, key=lambda x: (-x[0], -x[1]))[:per_cat]]
        print(f"    {name:10s} 候補 {len(hits):5d} -> {len(out[name])}")
    return out, meta, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work/humanml3d")
    ap.add_argument("--out", default="work/humanml3d/subset")
    ap.add_argument("--per-category", type=int, default=30)
    ap.add_argument("--min-frames", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--all", action="store_true",
                    help="カテゴリで絞らず立位クリップを全部拾い standing.json に書く")
    ap.add_argument("--joints", default=None,
                    help="AMASS から HumanML3D の公式手順で作った new_joints（フォルダか zip）。docs/data.md")
    a = ap.parse_args()

    root, out = pathlib.Path(a.root), pathlib.Path(a.out)
    print("1. 取得")
    fetch(root, a.joints)
    print("2. 選別")
    if a.all:
        ids, meta = scan_standing(root, a.min_frames, a.max_frames)
        sec = sum(v["T"] for v in meta.values()) / 20
        json.dump({
            "source": "HumanML3D new_joints (SMPL 22 joints, 20fps, "
                      "Y-up right-handed, faces +Z, floor at y=0)",
            "filter": f"立位で開始/終了 + {a.min_frames}〜{a.max_frames} フレーム",
            "stand_criteria": STAND,
            "note": "ミラー(M*)は含めない。学習時に mirror.json で反転する。",
            "ids": ids, "meta": meta,
        }, open(root / "standing.json", "w"), indent=1, ensure_ascii=False)
        print(f"3. 完了  {len(ids)} clips / {sec/3600:.2f} 時間 -> "
              f"{root / 'standing.json'}（ファイルはコピーしない）")
        return
    cats, meta, caps = curate(root, a.per_category, a.min_frames, a.max_frames)

    ids = sorted({i for v in cats.values() for i in v})
    shutil.rmtree(out, ignore_errors=True)
    (out / "new_joints").mkdir(parents=True)
    (out / "texts").mkdir()
    for i in ids:
        shutil.copy(root / "new_joints" / f"{i}.npy", out / "new_joints" / f"{i}.npy")
        shutil.copy(root / "texts" / f"{i}.txt", out / "texts" / f"{i}.txt")
    for f in ("Mean.npy", "Std.npy"):
        shutil.copy(root / f, out / f)
    json.dump({
        "source": "HumanML3D new_joints (SMPL 22 joints, 20fps, "
                  "Y-up right-handed, faces +Z, floor at y=0)",
        "filter": f"立位で開始/終了 + キャプション過半数一致 + "
                  f"{a.min_frames}〜{a.max_frames} フレーム",
        "stand_criteria": STAND,
        "categories": cats,
        "meta": {i: dict(meta[i], caption=caps[i][0]) for i in ids},
    }, open(out / "manifest.json", "w"), indent=1, ensure_ascii=False)

    mb = sum(os.path.getsize(out / "new_joints" / f"{i}.npy") for i in ids) / 1e6
    sec = sum(meta[i]["T"] for i in ids) / 20
    print(f"3. 完了  {len(ids)} clips / {mb:.1f} MB / {sec/60:.1f} 分 -> {out}")


if __name__ == "__main__":
    main()
