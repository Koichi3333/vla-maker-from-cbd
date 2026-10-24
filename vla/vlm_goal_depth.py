#!/usr/bin/env python3
"""Remaining task a of V-10: move the 3D goal built from the VLM's point from the near surface of the object toward its center (Codespace + Gemini API, 0 CU).

In redo (A) (2026-10-11) the goal stuck to the object's near surface, 7-12 cm short of the center for the bowl and the book, so it did not count as "held" (within 8 cm of the center).
Here, before using an A100, only the goal-to-center distance is measured locally to choose a method. **The object's true position is used only for scoring** (never to compute the goal).

Methods (all on the 21 cases with no distractor and the instruction prompt):
  surf      current method: of the rays within +-3 px of the VLM's point, the surface hit nearest the camera (same as v10_vlm_target.py goals)
  push<d>   move d cm from surf along the ray (d = 3, 5, 7, 10). No extra Gemini calls
  box       also ask Gemini for the object's bounding box, estimate the object's width (m) from the box width (px) and the distance to the surface, and move half of it along the ray
  boxc      the same push as box, but with the ray cast from the box center instead of the VLM's point

    python3 vla/vlm_goal_depth.py ask --limit 100     # ask for boxes (21 scenes x 3 = 63 calls, about $0.2). Spend log and cap shared with v10_vlm_target.py
    MUJOCO_GL=osmesa python3 vla/vlm_goal_depth.py eval          # goal-to-center distance per method
    MUJOCO_GL=osmesa python3 vla/vlm_goal_depth.py goals --method box   # write work/v10/object_eval_vlm_<method>.json with the chosen method
"""
import argparse
import datetime
import json
import pathlib
import sys
import time
import urllib.error

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import vlm_target as V                       # noqa: E402  images, prompts, spend log and rays are shared with V-10

OUT = V.OUT
PUSH = (3, 5, 7, 10)                             # cm


# ---------------------------------------------------------------- Ask for boxes
def box_prompt(q):
    return (f'This image shows a person in a simulator. The person is about to do this: "{q["text"]}"'
            " Give the 2D bounding box of the object the person should grasp first."
            ' Answer in JSON: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "<object name>"}], normalized to 0-1000.'
            " Return exactly one box.")


def parse_box(text):
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(v, dict):
        v = [v]
    for x in v if isinstance(v, list) else []:
        b = x.get("box_2d") if isinstance(x, dict) else None
        if isinstance(b, list) and len(b) == 4:
            return [float(c) * 256 / 1000 for c in b]                       # pixels (ymin, xmin, ymax, xmax)
    return None


def cases():
    """The 21 cases with no distractor and the instruction prompt (from the prompts of v10_vlm_target.py render)."""
    return [q for q in json.loads((OUT / "questions.json").read_text()) if q["dis"] == "none" and q["ask"] == "instr"]


def cmd_ask(a):
    from colab_secrets import load_codespaces_secrets
    if datetime.date.today() > V.PRICE_UNTIL:
        sys.exit(f"[ask] 単価の確認日（{V.PRICE_UNTIL} まで有効）を過ぎた。料金ページで確かめて PRICES を直すこと")
    cap = min(a.cap, V.CAP_USD)
    key = load_codespaces_secrets().get("GEMINI_API_KEY")
    if not key:
        sys.exit("[ask] GEMINI_API_KEY が無い")
    ans_p = OUT / f"answers_box_{a.model}.jsonl"
    have = set()
    if ans_p.exists():
        have = {(json.loads(l)["id"], json.loads(l)["rep"]) for l in ans_p.read_text().splitlines() if l.strip()}
    todo = [(q, rep) for rep in range(a.reps) for q in cases() if (q["id"], rep) not in have]
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}"
    n = 0
    for q, rep in todo[:a.limit]:
        body = V.body_of(q, a.model)
        body["contents"][0]["parts"][1]["text"] = box_prompt(q)
        ct = V.post(f"{base}:countTokens", {"contents": body["contents"]}, key, 60)["totalTokens"]
        worst = V.cost(a.model, ct, V.MAX_OUT)
        so_far = V.spent()
        if so_far + worst > cap:
            print(f"[ask] 上限で止めた: これまで ${so_far:.4f} + この 1 回の最大 ${worst:.4f} > ${cap:.2f}")
            break
        t0 = time.time()
        try:
            d = V.post(f"{base}:generateContent", body, key)
        except urllib.error.HTTPError as e:
            with V.LEDGER.open("a") as fh:
                fh.write(json.dumps(dict(t=time.time(), model=a.model, id="box_" + q["id"], rep=rep, usd=worst, error=e.code)) + "\n")
            print(f"[ask] HTTP {e.code}（{q['id']}）: {e.read()[:300]!r}")
            break
        u = d.get("usageMetadata", {})
        usd = V.cost(a.model, u.get("promptTokenCount", ct), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0))
        with V.LEDGER.open("a") as fh:
            fh.write(json.dumps(dict(t=time.time(), model=a.model, id="box_" + q["id"], rep=rep, usd=usd, usage=u)) + "\n")
        try:
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError):
            text = ""
        with ans_p.open("a") as fh:
            fh.write(json.dumps(dict(id=q["id"], rep=rep, text=text, usage=u, sec=round(time.time() - t0, 1)), ensure_ascii=False) + "\n")
        n += 1
        print(f"[ask] {q['id']} rep{rep}  ${usd:.4f}  {time.time() - t0:.1f}s  {text[:90]}")
    print(f"[ask] {n} 回送った。残り {max(len(todo) - n, 0)} 問。合計 ${V.spent():.4f} / 上限 ${cap:.2f}")


# ---------------------------------------------------------------- Build goals
def medians(path, parse):
    out = {}
    if path.exists():
        for l in path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                v = parse(r["text"])
                if v is not None:
                    out.setdefault(r["id"], []).append(v)
    return {k: np.median(np.array(v), 0) for k, v in out.items()}


def focal_px(rr):
    """Horizontal focal length in pixels of the camera of the scene just drawn. From the same frustum as pixel_ray."""
    gl = rr.scene.camera[0]
    H, W = rr.height, rr.width
    hw = gl.frustum_width or (gl.frustum_top - gl.frustum_bottom) / 2 * W / H
    return gl.frustum_near * W / (2 * hw)


def goals_for(m, d, rr, q, pos, place, pt, box, win):
    """-> {method: 3D goal or None}. place is used for drawing (without drawing the object the ray hits nothing). It is not used to compute the goal."""
    V.render_case(m, d, rr, rr, q, {"t": (place[0], place[1])})
    cam = np.array(rr.scene.camera[0].pos)
    out = {}
    surf = V.nearest_surface(m, d, rr, pt[0], pt[1], win)[0] if pt is not None else None
    out["surf"] = surf
    ray = None if surf is None else (surf - cam) / np.linalg.norm(surf - cam)
    for c in PUSH:
        out[f"push{c}"] = None if surf is None else surf + ray * c / 100
    out["box"] = out["boxc"] = None
    if box is not None and surf is not None:
        f = focal_px(rr)
        width_m = (box[3] - box[1]) * np.linalg.norm(surf - cam) / f          # box width -> object width at the surface distance
        out["box"] = surf + ray * width_m / 2
        cy, cx = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        sc = V.nearest_surface(m, d, rr, cy, cx, win)[0]
        if sc is not None:
            rc = (sc - cam) / np.linalg.norm(sc - cam)
            out["boxc"] = sc + rc * ((box[3] - box[1]) * np.linalg.norm(sc - cam) / f) / 2
        out["_width_cm"] = width_m * 100
    return out


def build(a):
    import mujoco
    spec = json.loads(V.SPEC.read_text())
    pts = medians(OUT / f"answers_{a.model}.jsonl", V.parse_point)
    boxes = medians(OUT / f"answers_box_{a.model}.jsonl", parse_box)
    m, d = V.scene()
    rr = mujoco.Renderer(m, 256, 256)
    rows = []
    for it in spec["items"]:
        clip = it["clip"][-4:]
        q, place = V.case_scene(it, m, d)
        for pos in ("orig", "shift", "mirror"):
            qid = f"{clip}_{pos}_none_instr"
            g = goals_for(m, d, rr, q, pos, (it["label"], place[pos]), pts.get(qid), boxes.get(qid), a.win)
            rows.append(dict(it=it, clip=clip, pos=pos, label=it["label"], goals=g, true=place[pos]))
    return spec, rows


def cmd_eval(a):
    spec, rows = build(a)
    methods = ["surf"] + [f"push{c}" for c in PUSH] + ["box", "boxc"]
    print(f"{'':16}" + "".join(f"{k:>9}" for k in methods) + "   枠の幅")
    for r in rows:
        dist = [None if r["goals"][k] is None else np.linalg.norm(r["goals"][k] - r["true"]) * 100 for k in methods]
        w = r["goals"].get("_width_cm")
        print(f"{r['clip']} {r['pos']:6} {r['label']:5}" + "".join(f"{'-':>9}" if x is None else f"{x:9.1f}" for x in dist)
              + (f"   {w:.1f} cm" if w is not None else ""))
    print("---- 目標と物の中心の距離（cm）")
    groups = {"全 21 件": lambda r: True, "コップ・ボール": lambda r: r["label"] in ("cup", "ball"), "お椀・本": lambda r: r["label"] in ("bowl", "book")}
    for gname, sel in groups.items():
        sub = [r for r in rows if sel(r)]
        line = []
        for k in methods:
            v = np.array([np.linalg.norm(r["goals"][k] - r["true"]) * 100 for r in sub if r["goals"][k] is not None])
            line.append(f"{k} 中央値 {np.median(v):.1f}・最大 {v.max():.1f}・8cm 以内 {np.mean(v <= 8):.2f}" if len(v) else f"{k} -")
        print(f"[{gname}（{len(sub)} 件）]\n  " + "\n  ".join(line))


def cmd_goals(a):
    spec, rows = build(a)
    by = {(r["clip"], r["pos"]): r["goals"][a.method] for r in rows}
    for it in spec["items"]:
        clip = it["clip"][-4:]
        it["vlm_goal"] = {p: (None if by[(clip, p)] is None else by[(clip, p)].tolist()) for p in ("orig", "shift", "mirror")}
        it.pop("h4_offsets", None)
    spec["note"] = (spec.get("note", "") + f" / V-10 残タスク a: vlm_goal = {a.model} の点（紛らわしい物なし・指示文・3 回の中央値）"
                    f"から方法 {a.method} で作った目標（v10c_goal_depth.py）").strip(" /")
    dst = OUT / f"object_eval_vlm_{a.method}.json"
    dst.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    print(f"-> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ask", "eval", "goals"])
    ap.add_argument("--model", default="gemini-robotics-er-2-preview", choices=sorted(V.PRICES))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cap", type=float, default=V.CAP_USD)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--method", default="box")
    a = ap.parse_args()
    dict(ask=cmd_ask, eval=cmd_eval, goals=cmd_goals)[a.cmd](a)


if __name__ == "__main__":
    main()
