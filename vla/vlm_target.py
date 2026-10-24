#!/usr/bin/env python3
"""V-10 part 1: have a VLM (Gemini Robotics-ER) pick "the object to grasp" from the frame and compare with the answer (Codespace + Gemini API, 0 CU).

The **first frame** of each of H-2's 7 heldout clips (ws/v7/object_eval.json) is drawn with the same camera, object proxies and 256x256 as training and evaluation, and the VLM points at it.
Conditions (7 clips x 3 x 3 x 2 = 126 questions, x reps):
  object position : orig (where it was grasped in the recording) / shift (moved 25 cm) / mirror (other side of the body)
  distractor      : none / other (one object of a different kind) / same (one more of the same kind; the question adds "on the left/right of the screen" to tell them apart)
                    the distractor goes "where the VLA tends to go": the mirror position for orig questions, the orig position for shift / mirror questions
  prompt          : instr (give the original instruction and ask for "the object to grasp first") / name ("point at <object name>")
Answer: the centroid of the target object's pixels in a segmentation render. **The choice is correct** if the returned point lies on the target object's pixels,
or is closer to the target object (its pixel centroid) than to the distractor.

Spending cap: the total across all runs is logged in work/v10/spend.jsonl, and **a request is not sent if "total so far + this call's maximum" would exceed CAP_USD**.
A call's maximum is input tokens (counted with countTokens before sending) x input price + (maxOutputTokens) x output price (thinking counts as output,
and maxOutputTokens caps it). Prices were checked on the pricing page on 2026-10-10 (doubling from 2027-01-01; update PRICES after that).

    MUJOCO_GL=osmesa python3 vla/vlm_target.py render            # build images, answers and questions (sends nothing)
    python3 vla/vlm_target.py ask --limit 10                      # send up to 10 unanswered questions (answers are cached)
    python3 vla/vlm_target.py score                               # score
    python3 vla/vlm_target.py spend                               # total of the spend log

The key is read from a file with the function in tools/colab_secrets.py. **Never print or echo the key** (CLAUDE.md).
"""
import argparse
import base64
import datetime
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

import numpy as np

R = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(R / "vla"), str(R / "tools")]
OUT = pathlib.Path(os.environ.get("VLA_WORK", R / "work")) / "vlm"     # images, questions, answers, spend log (not tracked by git)
SPEC = pathlib.Path(os.environ.get("VLA_SPEC", R / "data/eval/object_eval_heldout.json"))   # scenes whose goal is asked
XML = R / "vla/humanoid.xml"
CAP_USD = 10.0                                   # the owner's cap (2026-10-10). Must not be raised
PRICES = {                                       # USD / 1M tokens (input, output; output includes thinking). Until 2026-12-31
    "gemini-robotics-er-2-preview": (1.00, 5.00),
    "gemini-3.8-flash": (0.75, 3.75),
}
PRICE_UNTIL = datetime.date(2026, 12, 31)
MAX_OUT = 4096                                   # maxOutputTokens (cap including thinking -> used to compute a call's maximum cost)
NOUN = {"cup": "glass", "bottle": "water bottle", "bowl": "bowl", "ball": "ball", "book": "book", "phone": "smartphone"}
VISIBLE_PX = 20                                  # below this the object is hidden behind the body or legs and barely visible (2245 shift has 0 px)
OTHER = {"cup": "ball", "bowl": "book", "ball": "cup", "book": "bowl", "bottle": "bowl", "phone": "cup"}


# ---------------------------------------------------------------- Rendering
def scene():
    """Add two object bodies per label to humanoid.xml, "target" and "distractor" (so two of the same kind can be placed)."""
    import mujoco
    from render_with_objects import PROXY
    s = XML.read_text(encoding="utf-8")
    add = "".join(f'<body name="v10_{role}_{k}" mocap="true" pos="0 0 100"><geom name="v10_{role}_{k}" {g} rgba="{c}" '
                  f'contype="0" conaffinity="0"/></body>' for role in ("t", "d") for k, (g, c) in PROXY.items())
    m = mujoco.MjModel.from_xml_string(s.replace("</worldbody>", add + "</worldbody>", 1))
    return m, mujoco.MjData(m)


def render_case(m, d, rr, rs, q, objs):
    """q: (qpos,), objs: {role: (label, position)}. -> (RGB image, {role: pixel mask}). Same camera as render_with_objects.frames_with_object."""
    import mujoco
    for i in range(m.nmocap):
        d.mocap_pos[i] = [0, 0, 100]
    gid = {}
    for role, (lab, p) in objs.items():
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"v10_{role}_{lab}")
        d.mocap_pos[m.body_mocapid[b]] = p
        gid[role] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"v10_{role}_{lab}")
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.trackbodyid = cam.fixedcamid = -1
    cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 135
    cam.lookat[:] = [q[0], q[1], 0.9]
    rr.update_scene(d, cam)
    img = rr.render().copy()
    rs.update_scene(d, cam)
    seg = rs.render()
    masks = {role: (seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) for role, g in gid.items()}
    return img, masks


def pixel_ray(m, d, rr, py, px):
    """In the scene just drawn, cast a ray toward pixel (py, px) and return the first hit (world coordinates) and its geom id. (None, -1) if nothing is hit.
    rr has had update_scene called in render_case. The frustum is MuJoCo's mjvGLCamera (near plane left-right center +- half width, top-bottom bottom..top;
    the half width frustum_width is stored as 0, in which case it follows from the aspect ratio)."""
    import mujoco
    gl = rr.scene.camera[0]
    fwd, up, pos = np.array(gl.forward), np.array(gl.up), np.array(gl.pos)
    right = np.cross(fwd, up)
    H, W = rr.height, rr.width
    hw = gl.frustum_width or (gl.frustum_top - gl.frustum_bottom) / 2 * W / H     # if 0, from the aspect ratio (MuJoCo's default)
    x = gl.frustum_center - hw + (px + 0.5) / W * 2 * hw
    y = gl.frustum_top - (py + 0.5) / H * (gl.frustum_top - gl.frustum_bottom)
    v = fwd * gl.frustum_near + right * x + up * y
    v /= np.linalg.norm(v)
    gid = np.array([-1], np.int32)
    dist = mujoco.mj_ray(m, d, pos, v, None, 1, -1, gid)
    return (pos + v * dist, int(gid[0])) if dist >= 0 else (None, -1)


def nearest_surface(m, d, rr, py, px, win):
    """Of the rays within +-win pixels of the pointed spot, the one that hits nearest (the same as taking the minimum depth of a small window on a depth camera).
    Objects are small, around 10 px, and a point just off the edge hits the floor 3 m away."""
    best = (None, -1, np.inf)
    cam = np.array(rr.scene.camera[0].pos)
    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            g, gid = pixel_ray(m, d, rr, py + dy, px + dx)
            if g is not None and np.linalg.norm(g - cam) < best[2]:
                best = (g, gid, np.linalg.norm(g - cam))
    return best[0], best[1]


def case_scene(it, m, d):
    """One spec item -> (q (ground-corrected), placements {orig, shift, mirror}). Same coordinates as rendering and evaluation (groot_eval.object_eval)."""
    from cbd_to_mujoco import ground
    from humanml3d_to_video import state_to_qpos
    q = state_to_qpos(np.asarray(it["init_state"], np.float64)[None])
    for i in range(m.nmocap):
        d.mocap_pos[i] = [0, 0, 100]
    lo = ground(m, d, q)                          # q is lowered inside as well (subtracting again would lift only the body by lo; fixed 2026-10-10)
    down = np.array([0, 0, lo])
    place = {"orig": np.asarray(it["obj_raw"]) - down}
    place["shift"] = place["orig"] + np.asarray(it["shift_dir"]) * it["shift_m"]
    place["mirror"] = np.asarray(it["obj_mirror_raw"]) - down
    return q[0], place


def cmd_goals(a):
    """VLM point -> 3D goal by ray -> groot_eval spec (vlm_goal). Uses the median point over reps of the answers to the no-distractor, instruction prompt.
    With --check, the true pixels (centroid of the target object's pixels) are used instead of the VLM to measure the error of lifting back to 3D."""
    import mujoco
    spec = json.loads(SPEC.read_text())
    qs = {q["id"]: q for q in json.loads((OUT / "questions.json").read_text())}
    pts = {}
    if not a.check:
        for l in (OUT / f"answers_{a.model}.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                p = parse_point(r["text"])
                if p is not None:
                    pts.setdefault(r["id"], []).append(p)
    m, d = scene()
    rr = mujoco.Renderer(m, 256, 256)
    rep = []
    for it in spec["items"]:
        clip = it["clip"][-4:]
        q, place = case_scene(it, m, d)
        it["vlm_goal"] = {}
        for pos in ("orig", "shift", "mirror"):
            qid = f"{clip}_{pos}_none_{a.ask}"
            if a.check:
                c = qs[qid]["center"]["t"]
                pt = c if c is not None else None
            else:
                v = pts.get(qid, [])
                pt = np.median(np.array(v), 0).tolist() if v else None
            g, err, hit = None, None, None
            if pt is not None:
                render_case(m, d, rr, rr, q, {"t": (it["label"], place[pos])})   # build the same scene for ray casting (rr is used twice)
                g, gid = nearest_surface(m, d, rr, pt[0], pt[1], a.win)
                hit = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) if gid >= 0 else None
                if g is not None:
                    err = float(np.linalg.norm(g - place[pos]))
            it["vlm_goal"][pos] = None if g is None else g.tolist()
            rep.append((clip, pos, pt is not None, hit, err))
            print(f"  {clip} {pos:6} 点 {'-' if pt is None else [round(x) for x in pt]}  当たった {hit}  "
                  f"本当の物との距離 {'-' if err is None else f'{err * 100:.1f} cm'}")
    errs = [e for *_, e in rep if e is not None]
    print(f"[goals] {len(rep)} 件  点なし {sum(not r[2] for r in rep)}  3D の誤差の中央値 {np.median(errs) * 100:.1f} cm  "
          f"10 cm 以内 {np.mean([e <= 0.10 for e in errs]):.2f}")
    if not a.check:
        for it in spec["items"]:
            it.pop("h4_offsets", None)                 # H-4 has nothing to do with the VLM's goal, so it is not run (groot_eval skips H-4 when there are no h4_offsets)
        spec["note"] = (spec.get("note", "") + f" / V-10: vlm_goal = {a.model} の点（紛らわしい物なし・{a.ask}・reps の中央値）を光線で 3D にしたもの").strip(" /")
        dst = OUT / "object_eval_vlm.json"
        dst.write_text(json.dumps(spec, indent=1, ensure_ascii=False))
        print(f"-> {dst}")


def cmd_render(a):
    import cv2
    import mujoco
    from cbd_to_mujoco import ground
    from humanml3d_to_video import state_to_qpos
    spec = json.loads(SPEC.read_text())
    m, d = scene()
    rr = mujoco.Renderer(m, 256, 256)
    rs = mujoco.Renderer(m, 256, 256)
    rs.enable_segmentation_rendering()
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    qs = []
    for it in spec["items"]:
        clip = it["clip"][-4:]
        q = state_to_qpos(np.asarray(it["init_state"], np.float64)[None])
        for i in range(m.nmocap):
            d.mocap_pos[i] = [0, 0, 100]
        lo = ground(m, d, q)                      # q is lowered inside as well (same as case_scene)
        down = np.array([0, 0, lo])
        place = {"orig": np.asarray(it["obj_raw"]) - down}
        place["shift"] = place["orig"] + np.asarray(it["shift_dir"]) * it["shift_m"]
        place["mirror"] = np.asarray(it["obj_mirror_raw"]) - down
        lab = it["label"]
        for pos in ("orig", "shift", "mirror"):
            alt = place["mirror"] if pos == "orig" else place["orig"]          # where the VLA tends to go
            for dis in ("none", "other", "same"):
                objs = {"t": (lab, place[pos])}
                if dis != "none":
                    objs["d"] = (OTHER[lab] if dis == "other" else lab, alt)
                img, masks = render_case(m, d, rr, rs, q[0], objs)
                cid = f"{clip}_{pos}_{dis}"
                cv2.imwrite(str(OUT / "img" / f"{cid}.png"), img[..., ::-1])
                cen = {r: (np.argwhere(mk).mean(0).tolist() if mk.any() else None) for r, mk in masks.items()}   # (y, x) pixels
                np.savez_compressed(OUT / "img" / f"{cid}_mask.npz", **{r: mk for r, mk in masks.items()})
                side = None
                if dis == "same":
                    if cen["t"] is None or cen["d"] is None:
                        side = None
                    else:
                        side = "left" if cen["t"][1] < cen["d"][1] else "right"
                for ask in ("instr", "name"):
                    qs.append(dict(id=f"{cid}_{ask}", case=cid, clip=clip, label=lab, pos=pos, dis=dis, ask=ask,
                                   text=it["text"], side=side, center=cen, pixels={r: int(mk.sum()) for r, mk in masks.items()}))
    (OUT / "questions.json").write_text(json.dumps(qs, indent=1, ensure_ascii=False))
    hidden = [q["case"] for q in qs if q["pixels"]["t"] == 0]
    print(f"[render] {len(qs) // 2} 枚・{len(qs)} 問 → {OUT}  狙いの物が写っていない: {sorted(set(hidden))}")


# ---------------------------------------------------------------- Questions
def prompt(q):
    noun = NOUN[q["label"]]
    which = f" There are two {noun}s; choose the one on the {q['side']} side of the image." if q["dis"] == "same" else ""
    fmt = (' Answer in JSON: [{"point": [y, x], "label": "<object name>"}], with the point normalized to 0-1000'
           " (y from the top, x from the left). Return exactly one point.")
    if q["ask"] == "instr":
        return (f'This image shows a person in a simulator. The person is about to do this: "{q["text"]}"'
                f" Point to the object the person should grasp first to do it.{which}{fmt}")
    return f"Point to the {noun}.{which}{fmt}"


def body_of(q, model):
    png = (OUT / "img" / f"{q['case']}.png").read_bytes()
    return {"contents": [{"role": "user", "parts": [
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}},
                {"text": prompt(q)}]}],
            "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": MAX_OUT}}


def post(url, body, key, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


# ---------------------------------------------------------------- Spend log
LEDGER = OUT / "spend.jsonl"


def spent():
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def cost(model, prompt_tok, out_tok):
    pi, po = PRICES[model]
    return prompt_tok * pi / 1e6 + out_tok * po / 1e6


def cmd_ask(a):
    from colab_secrets import load_codespaces_secrets
    if datetime.date.today() > PRICE_UNTIL:
        sys.exit(f"[ask] 単価の確認日（{PRICE_UNTIL} まで有効）を過ぎた。料金ページで確かめて PRICES を直すこと")
    cap = min(a.cap, CAP_USD)
    key = load_codespaces_secrets().get("GEMINI_API_KEY")
    if not key:
        sys.exit("[ask] GEMINI_API_KEY が無い")
    qs = json.loads((OUT / "questions.json").read_text())
    ans_p = OUT / f"answers_{a.model}.jsonl"
    have = {}
    if ans_p.exists():
        for l in ans_p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                have[(r["id"], r["rep"])] = r
    todo = [(q, rep) for rep in range(a.reps) for q in qs if (q["id"], rep) not in have
            and (not a.only or q["id"].startswith(tuple(a.only.split(","))))]
    base = f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}"
    n = 0
    for q, rep in todo[:a.limit]:
        body = body_of(q, a.model)
        ct = post(f"{base}:countTokens", {"contents": body["contents"]}, key, 60)["totalTokens"]
        worst = cost(a.model, ct, MAX_OUT)
        so_far = spent()
        if so_far + worst > cap:
            print(f"[ask] 上限で止めた: これまで ${so_far:.4f} + この 1 回の最大 ${worst:.4f} > ${cap:.2f}")
            break
        t0 = time.time()
        try:
            d = post(f"{base}:generateContent", body, key)
        except urllib.error.HTTPError as e:
            # failed calls are logged at their maximum too (we cannot confirm whether they were billed, so err high)
            with LEDGER.open("a") as fh:
                fh.write(json.dumps(dict(t=time.time(), model=a.model, id=q["id"], rep=rep, usd=worst, error=e.code)) + "\n")
            print(f"[ask] HTTP {e.code}（{q['id']}）: {e.read()[:300]!r}")
            break
        u = d.get("usageMetadata", {})
        out_tok = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        usd = cost(a.model, u.get("promptTokenCount", ct), out_tok)
        with LEDGER.open("a") as fh:
            fh.write(json.dumps(dict(t=time.time(), model=a.model, id=q["id"], rep=rep, usd=usd, usage=u)) + "\n")
        try:
            text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"] if not p.get("thought"))
        except (KeyError, IndexError):
            text = ""
        with ans_p.open("a") as fh:
            fh.write(json.dumps(dict(id=q["id"], rep=rep, text=text, usage=u, sec=round(time.time() - t0, 1)), ensure_ascii=False) + "\n")
        n += 1
        print(f"[ask] {q['id']} rep{rep}  入力 {u.get('promptTokenCount')}  出力 {u.get('candidatesTokenCount')}  "
              f"考える {u.get('thoughtsTokenCount', 0)}  ${usd:.4f}  {time.time() - t0:.1f}s  {text[:80]}")
    print(f"[ask] {n} 回送った。残り {max(len(todo) - n, 0)} 問。合計 ${spent():.4f} / 上限 ${cap:.2f}")


def cmd_spend(a):
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []
    by = {}
    for r in rows:
        by.setdefault(r["model"], []).append(r)
    for mdl, rs in by.items():
        u = [r.get("usage", {}) for r in rs if "usage" in r]
        avg = lambda k: np.mean([x.get(k, 0) for x in u]) if u else 0   # noqa: E731
        print(f"{mdl}: {len(rs)} 回  ${sum(r['usd'] for r in rs):.4f}  平均 入力 {avg('promptTokenCount'):.0f}・出力 {avg('candidatesTokenCount'):.0f}"
              f"・考える {avg('thoughtsTokenCount'):.0f} トークン")
    print(f"合計 ${spent():.4f} / 上限 ${CAP_USD:.2f}")


# ---------------------------------------------------------------- Scoring
def parse_point(text):
    try:
        v = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(v, dict):
        v = [v]
    for x in v if isinstance(v, list) else []:
        if isinstance(x, dict) and isinstance(x.get("point"), list) and len(x["point"]) == 2:
            y, xx = x["point"]
            return float(y) * 256 / 1000, float(xx) * 256 / 1000          # pixels (y, x)
    return None


def cmd_score(a):
    qs = {q["id"]: q for q in json.loads((OUT / "questions.json").read_text())}
    rows = []
    for l in (OUT / f"answers_{a.model}.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        q = qs[r["id"]]
        pt = parse_point(r["text"])
        mk = np.load(OUT / "img" / f"{q['case']}_mask.npz")
        ok = on_t = None
        err = None
        if pt is not None and q["center"]["t"] is not None:
            y, x = int(np.clip(round(pt[0]), 0, 255)), int(np.clip(round(pt[1]), 0, 255))
            on_t = bool(mk["t"][y, x])
            err = float(np.hypot(pt[0] - q["center"]["t"][0], pt[1] - q["center"]["t"][1]))
            if "d" in mk.files and q["center"]["d"] is not None:
                if mk["d"][y, x]:
                    ok = False
                else:
                    ok = on_t or err < np.hypot(pt[0] - q["center"]["d"][0], pt[1] - q["center"]["d"][1])
            else:
                ok = on_t or err <= 15                                    # with no distractor: on the pixels, or within 15 px of the centroid
        rows.append(dict(q, rep=r["rep"], point=pt, ok=None if ok is None else bool(ok), on_target=on_t, err_px=err))
    (OUT / f"score_{a.model}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))

    def show(title, key):
        print(f"== {title}")
        for k in sorted({key(r) for r in rows}):
            v = [r for r in rows if key(r) == k]
            okv = [r["ok"] for r in v if r["ok"] is not None]
            ev = [r["err_px"] for r in v if r["err_px"] is not None]
            print(f"  {k!s:24} 正答 {np.mean(okv) if okv else float('nan'):.2f}（{sum(okv)}/{len(v)}、読めない {sum(r['point'] is None for r in v)}）"
                  f"  画素の上 {np.mean([r['on_target'] for r in v if r['on_target'] is not None]):.2f}"
                  f"  誤差の中央値 {np.median(ev) if ev else float('nan'):.1f} px")
    for r in rows:
        r["visible"] = r["pixels"]["t"] >= VISIBLE_PX
    show("全体", lambda r: "all")
    show(f"狙いの物が {VISIBLE_PX} 画素以上写っているか", lambda r: r["visible"])
    rows = [r for r in rows if r["visible"]]
    print(f"---- 以下は {VISIBLE_PX} 画素以上写っている問いだけ")
    show("紛らわしい物", lambda r: r["dis"])
    show("物の位置", lambda r: r["pos"])
    show("聞き方", lambda r: r["ask"])
    show("紛らわしい物 × 物の位置", lambda r: f"{r['dis']} × {r['pos']}")
    show("clip", lambda r: r["clip"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "ask", "score", "spend", "goals"])
    ap.add_argument("--model", default="gemini-robotics-er-2-preview", choices=sorted(PRICES))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="", help="問いの id の頭で絞る（カンマ区切り）")
    ap.add_argument("--cap", type=float, default=CAP_USD, help=f"費用の上限（USD、{CAP_USD} より上には上げられない）")
    ap.add_argument("--check", action="store_true", help="goals: VLM の代わりに正解の画素で 3D に戻す誤差を測る")
    ap.add_argument("--win", type=int, default=3, help="goals: 指した点のまわりで、いちばん手前の面を探す範囲（画素）")
    ap.add_argument("--ask", default="instr", choices=["instr", "name"], help="goals: どの聞き方の答えを使うか")
    a = ap.parse_args()
    dict(render=cmd_render, ask=cmd_ask, score=cmd_score, spend=cmd_spend, goals=cmd_goals)[a.cmd](a)


if __name__ == "__main__":
    main()
