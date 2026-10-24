#!/usr/bin/env python3
"""Remove the "small tremor while supposed to be still" from generated trajectories by post-processing.

GR00T generates by diffusion sampling, so every frame carries independent noise. On top of that it switches
to a new action chunk every 16 frames (`rollout` in `groot_eval.py`), so
**the motion jumps by 1.5x at the single boundary frame**. These two are what people see as "trembling when it should be still",
and they are almost absent from the teacher data (HumanML3D, mocap).

Measured (B-5, 10 texts x 8 clips = 80 clips, `ws/b5/results/rollouts.npz`):

| Still frames (stretches with no real motion within +-0.5 s) | Teacher | Generated | Corrected |
| --- | --- | --- | --- |
| Joint rotation per frame | 0.603 deg | 0.902 deg | 0.310 deg |
| Root movement per frame | 0.26 cm | 1.73 cm | 0.47 cm |

**Root jitter is the main culprit** (6.6x the teacher). A twitching pelvis is more noticeable than twitching joints.

What it does: a 1-euro filter (Casiez et al. 2012), run forward and then backward.
**Fast parts pass almost untouched and only slow parts are smoothed strongly**, so only the tremor when still disappears
and stride length and jump height are kept. It is offline, so two passes cancel the lag (phase shift).
Measured: 0-2.5 Hz content is kept 100%, 2.5-5 Hz 85-95%, and the pelvis's vertical travel in a jump goes
from 47.8 cm to 46.2 cm (-3%).

    # correct generated trajectories (117 dims) in bulk
    python3 smooth_motion.py work/b5/results/rollouts.npz -o work/b5/results/rollouts_smooth.npz \
        --teacher work/lerobot_humanml3d --report work/b5/smooth_report.json

    # render before/after side by side as mp4 (89 ms/frame locally at 256 px)
    MUJOCO_GL=osmesa python3 smooth_motion.py work/b5/results/rollouts.npz \
        --render work/b5/smooth_preview --clips waves_with_the:0 jumps_in_place:0

    # correct a motion.npz from the G1 adapter directly (for already retargeted output)
    python3 smooth_motion.py --g1 work/b5/g1/*/out/06_groot_g1/motion.npz --suffix _smooth

    # G1 before/after side by side as mp4 (also for comparing with smoothing the body first and retargeting again)
    MUJOCO_GL=osmesa python3 smooth_motion.py -o work/b5/g1_preview/waves_pair.mp4 \
        --g1-pair work/b5/g1/waves_with_the_init083/out/06_groot_g1/motion.npz \
                  work/b5/g1_smooth/waves_with_the_init083/out/06_groot_g1/motion.npz

**Fixing it on the generation side is the better approach** (overlap and blend chunks, more denoising steps,
average several samples). But each of those needs another A100 session,
whereas this post-processing works right away on npz files already collected.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FPS = 20.0
# Defaults were tuned on B-5's 80 clips (the weakest setting where tremor when still drops below the teacher
# while real motion stays above 99% preserved). Root is measured in m and rotations in quaternion components,
# so beta has different units.
ROOT = (1.2, 20.0)          # (mincutoff Hz, beta)
ROT = (1.2, 40.0)


# ---------------------------------------------------------------- 1-euro filter

def _alpha(cutoff, dt):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(x, fps=FPS, mincutoff=1.0, beta=20.0, dcutoff=1.0):
    """x: (T,D). A 1-euro filter per column. The cutoff rises with speed (fast parts pass through)."""
    dt = 1.0 / fps
    x = np.asarray(x, np.float64)
    out = np.empty_like(x)
    xh = x[0].copy()
    dxh = np.zeros(x.shape[1])
    out[0] = xh
    ad = _alpha(dcutoff, dt)
    for t in range(1, len(x)):
        dxh = ad * ((x[t] - xh) / dt) + (1 - ad) * dxh
        a = _alpha(mincutoff + beta * np.abs(dxh), dt)
        xh = a * x[t] + (1 - a) * xh
        out[t] = xh
    return out


def one_euro_zp(x, **kw):
    """Forward pass, then backward pass. The lags cancel, so the motion is not shifted later in time."""
    return one_euro(one_euro(x, **kw)[::-1], **kw)[::-1]


# ---------------------------------------------------------------- CBD's 117 dims

def _quats(state):
    from rotation import sixd_to_quat
    return sixd_to_quat(np.asarray(state[:, 3:], np.float64).reshape(len(state), -1, 6))


def _sign_align(q):
    """(T,J,4). Align signs between adjacent frames. Otherwise the double-cover flips get smoothed too."""
    q = q.copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    return q


def _blur_at(x, idx, win=2, passes=1):
    """Apply a 3-tap average only within +-win of the given indices (evens out steps)."""
    y = np.array(x, np.float64, copy=True)
    for _ in range(passes):
        z = y.copy()
        for k in idx:
            for t in range(max(1, k - win), min(len(y) - 1, k + win + 1)):
                z[t] = 0.25 * y[t - 1] + 0.5 * y[t] + 0.25 * y[t + 1]
        y = z
    return y


def dechunk(state, horizon=16, win=2, passes=1):
    """Even out only the steps at chunk boundaries. In `rollout`, traj[0] is the initial pose and traj[1] the first generated frame.

    On B-5's 80 clips, the rotation angle at boundary frames was 1.50x the rest (2.40 vs 1.74 deg).
    The 1-euro filter alone leaves 1.46 vs 1.32 deg (1.10x). With this pass it becomes 1.15 vs 1.32 deg,
    smaller at the boundaries, but 2.5-5 Hz content is cut from 92% to 88%. **Off by default.**
    """
    from rotation import quat_to_6d
    state = np.asarray(state)
    idx = list(range(1, len(state), horizon))
    out = state.astype(np.float64).copy()
    out[:, :3] = _blur_at(state[:, :3], idx, win, passes)
    q = _sign_align(_quats(state))
    qf = _blur_at(q.reshape(len(q), -1), idx, win, passes).reshape(q.shape)
    qf /= np.linalg.norm(qf, axis=-1, keepdims=True)
    out[:, 3:] = quat_to_6d(qf).reshape(len(state), -1)
    return out.astype(np.float32)


def smooth_states(state, fps=FPS, root=ROOT, rot=ROT, horizon=0):
    """(T,117) -> (T,117). Root translation is smoothed in m; bones are converted back to quaternions first."""
    from rotation import quat_to_6d
    state = np.asarray(state)
    if horizon:
        state = dechunk(state, horizon)
    out = state.astype(np.float64).copy()
    out[:, :3] = one_euro_zp(state[:, :3], fps=fps, mincutoff=root[0], beta=root[1])
    q = _sign_align(_quats(state))
    qf = one_euro_zp(q.reshape(len(q), -1), fps=fps, mincutoff=rot[0], beta=rot[1])
    qf = qf.reshape(q.shape)
    qf /= np.linalg.norm(qf, axis=-1, keepdims=True)     # smoothing breaks the norm, so restore it
    out[:, 3:] = quat_to_6d(qf).reshape(len(state), -1)
    return out.astype(np.float32)


# ---------------------------------------------------------------- Metrics

def _qang(a, b):
    return np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(-1)), 0, 1)))


def _speeds(state, win=5):
    """Return (raw_1frame, content).

    content is the displacement across `win` frames on either side, per frame. **Zero-mean jitter
    cancels out here**, so it indicates "how much it is really moving". The difference from raw is the jitter.
    """
    q = _sign_align(_quats(state))
    r = state[:, :3]
    raw_q = _qang(q[1:], q[:-1]).mean(1)
    raw_r = np.linalg.norm(np.diff(r, axis=0), axis=1) * 100
    con_q = _qang(q[2 * win:], q[:-2 * win]).mean(1) / (2 * win)
    con_r = np.linalg.norm(r[2 * win:] - r[:-2 * win], axis=1) / (2 * win) * 100
    n = min(len(con_q), len(raw_q) - win)
    s = slice(win, win + n)
    return raw_q[s], raw_r[s], con_q[:n], con_r[:n]


def quiet_jitter(states, thr_q=0.35, thr_r=0.5):
    """Collect only frames that are "essentially still" and measure the per-frame jitter there."""
    rows = []
    for s in states:
        raw_q, raw_r, con_q, con_r = _speeds(s)
        m = (con_q < thr_q) & (con_r < thr_r)
        if m.sum() >= 5:
            rows.append([m.mean(), raw_q[m].mean(), raw_r[m].mean()])
    if not rows:
        return None
    a = np.mean(rows, 0)
    return dict(n_clips=len(rows), quiet_frac=float(a[0]),
                bone_deg_per_frame=float(a[1]), root_cm_per_frame=float(a[2]))


def band_power(x, lo, hi, fps=FPS):
    x = np.asarray(x, np.float64)
    x = x - x.mean(0)
    p = (np.abs(np.fft.rfft(x, axis=0)) ** 2).sum(1)
    f = np.fft.rfftfreq(len(x), 1 / fps)
    return float(p[(f >= lo) & (f < hi)].sum())


def band_keep(before, after):
    """Power ratio "after / before correction" per frequency band. If the low band is 1.0, the content is intact."""
    out = {}
    for lo, hi in [(0, 2.5), (2.5, 5), (5, 10)]:
        b = np.mean([band_power(a[:, 3:], lo, hi) for a in before])
        f = np.mean([band_power(a[:, 3:], lo, hi) for a in after])
        out[f"{lo}-{hi}Hz"] = float(f / max(b, 1e-12))
    return out


def chunk_phase(states, horizon=16):
    """Look for jumps at chunk boundaries. Mean rotation angle (degrees) per difference index % horizon."""
    sd = np.stack([_qang(_quats(s)[1:], _quats(s)[:-1]).mean(1) for s in states])
    m = sd.mean(0)
    return [float(m[i::horizon].mean()) for i in range(horizon)]


def load_teacher(dataset, limit=120, min_len=80):
    import pyarrow.parquet as pq
    out = []
    for f in sorted(pathlib.Path(dataset).rglob("*.parquet"))[:limit]:
        col = pq.read_table(f, columns=["observation.state"]).column(0).combine_chunks()
        s = col.values.to_numpy().reshape(len(col), -1).astype(np.float32)
        if len(s) >= min_len:
            out.append(s[:200])
    return out


# ---------------------------------------------------------------- G1 motion.npz

def smooth_g1(src, dst, fps=None, root=ROOT, rot=ROT):
    """Smooth a motion.npz from the adapter (43 joint angles + pelvis pose) directly.

    **The default route is "correct the body side (117 dims) first, then retarget again"**
    (`state_to_cbd.py` -> `cbd_adapter_groot_g1.py`; 11 s per clip in the Codespace).
    Measuring both on B-5's 80 clips, jitter in still stretches is:

    | | Joints when still | Pelvis |
    | --- | --- | --- |
    | Teacher -> G1 | 0.387 deg/f | 0.64 cm/f |
    | Generated -> G1 (before correction) | 0.529 deg/f | 2.91 cm/f |
    | Smoothed after retargeting (this function) | 0.166 deg/f | 0.83 cm/f |
    | Body smoothed, then retargeted | 0.229 deg/f | 0.66 cm/f |

    Both remove the tremor, but **this function makes the joints smoother than the teacher** (0.166 vs 0.387 deg).
    Smoothing the body side is closer to the teacher's level, and retargeting quality does not drop
    (median direction residual 1.219 -> 1.222 deg, maximum time at joint limits 4.26% -> 3.33%).
    **Treat this as the fallback when only the retargeted npz is at hand.**
    """
    d = dict(np.load(src, allow_pickle=True))
    f = float(d["fps"][0]) if fps is None else fps
    ang = d["joint_angles_rad"].astype(np.float64)
    lim = d.get("joint_limits_rad")
    ang = one_euro_zp(ang, fps=f, mincutoff=rot[0], beta=rot[1] * 0.05)   # in rad
    if lim is not None:
        ang = np.clip(ang, lim[:, 0], lim[:, 1])     # keep within the range of motion
    d["joint_angles_rad"] = ang.astype(np.float32)
    d["pelvis_position_robot"] = one_euro_zp(
        d["pelvis_position_robot"].astype(np.float64), fps=f,
        mincutoff=root[0], beta=root[1]).astype(np.float32)
    q = _sign_align(d["pelvis_orientation_wxyz"].astype(np.float64)[:, None])[:, 0]
    q = one_euro_zp(q, fps=f, mincutoff=rot[0], beta=rot[1])
    d["pelvis_orientation_wxyz"] = (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **d)
    return dst


# ---------------------------------------------------------------- Rendering

def render_pair(before, after, out, xml, size=256, fps=FPS):
    """mp4 with before (left) and after (right) side by side. One Renderer is reused."""
    import imageio
    import mujoco
    from cbd_to_mujoco import frames, ground
    from humanml3d_to_video import state_to_qpos
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size, size)
    vids = []
    for s in (before, after):
        q = state_to_qpos(np.asarray(s, np.float64))
        q[:, 2] -= ground(m, d, q[:1].copy())       # ground set from the initial pose (same as at generation)
        vids.append(np.stack([f.copy() for f in frames(m, d, q, r)]))
    pair = np.concatenate(vids, axis=2)
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, list(pair), fps=fps, quality=8, macro_block_size=1)
    return out


def render_pair_g1(before_npz, after_npz, out, xml, size=(384, 256), fps=None):
    """mp4 with two G1 motion.npz files side by side. The camera framing is the same as g1_to_mujoco.py."""
    import imageio
    import mujoco
    from g1_to_mujoco import build_qpos, ground
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    w, h = size
    r = mujoco.Renderer(m, h, w)
    vids, f = [], fps
    for src in (before_npz, after_npz):
        npz = np.load(src, allow_pickle=True)
        f = f or float(npz["fps"][0])
        q = build_qpos(m, npz)
        ground(m, d, q)
        out_frames = []
        for t in range(len(q)):
            d.qpos[:] = q[t]
            mujoco.mj_forward(m, d)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.trackbodyid = -1
            cam.fixedcamid = -1
            cam.distance, cam.elevation, cam.azimuth = 2.6, -10, 135
            cam.lookat[:] = [q[t, 0], q[t, 1], 0.6]
            r.update_scene(d, cam)
            out_frames.append(r.render().copy())
        vids.append(np.stack(out_frames))
    n = min(len(v) for v in vids)
    pair = np.concatenate([v[:n] for v in vids], axis=2)
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, list(pair), fps=f, quality=8, macro_block_size=1)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", help="groot_eval.py が書いた rollouts.npz")
    ap.add_argument("-o", "--out", help="補正した npz の書き出し先")
    ap.add_argument("--g1", nargs="*", default=None, help="G1 アダプタの motion.npz（複数可）")
    ap.add_argument("--g1-pair", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="G1 の motion.npz 2本を横に並べた mp4 を焼く（書き出し先は -o）")
    ap.add_argument("--g1-xml", default=str(HERE / "g1/scene_with_hands.xml"))
    ap.add_argument("--suffix", default="_smooth", help="--g1 の書き出し名に付ける接尾辞")
    ap.add_argument("--teacher", help="比較する教師データセット（lerobot の parquet があるディレクトリ）")
    ap.add_argument("--report", help="指標を JSON で書き出す")
    ap.add_argument("--render", help="補正前後を並べた mp4 の書き出し先ディレクトリ")
    ap.add_argument("--clips", nargs="*", default=["waves_with_the:0"],
                    help="--render で焼くクリップ（<テキストのslug>:<本数の番号>）")
    ap.add_argument("--xml", default=str(HERE / "humanoid.xml"))
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--root", nargs=2, type=float, default=list(ROOT), metavar=("MINCUTOFF", "BETA"))
    ap.add_argument("--rot", nargs=2, type=float, default=list(ROT), metavar=("MINCUTOFF", "BETA"))
    ap.add_argument("--dechunk", type=int, default=0, metavar="HORIZON",
                    help="チャンク境界の段差もならす（16 を渡す。既定は通さない。dechunk() の注記を読むこと）")
    a = ap.parse_args()
    root, rot = tuple(a.root), tuple(a.rot)

    if a.g1_pair:
        out = render_pair_g1(a.g1_pair[0], a.g1_pair[1], a.out, a.g1_xml, fps=a.fps)
        print(f"-> {out}  (左: 補正前 / 右: 補正後)")
        return

    if a.g1:
        for p in a.g1:
            p = pathlib.Path(p)
            dst = p.with_name(p.stem + a.suffix + p.suffix)
            smooth_g1(p, dst, root=root, rot=rot)
            print(f"{p.parents[2].name:32s} -> {dst}")
        return

    if not a.src:
        raise SystemExit("rollouts.npz か --g1 のどちらかを渡すこと")

    src = np.load(a.src)
    texts = [k for k in src.files if k != "init_eps"]
    before = {t: src[t] for t in texts}
    after = {t: np.stack([smooth_states(v[b], a.fps, root, rot, a.dechunk) for b in range(len(v))])
             for t, v in before.items()}

    flat_b = [v[b] for v in before.values() for b in range(len(v))]
    flat_a = [v[b] for v in after.values() for b in range(len(v))]
    rep = dict(src=a.src, fps=a.fps, root=root, rot=rot, dechunk=a.dechunk,
               n_clips=len(flat_b),
               quiet_before=quiet_jitter(flat_b), quiet_after=quiet_jitter(flat_a),
               band_keep=band_keep(flat_b, flat_a),
               chunk_phase_before=chunk_phase(flat_b), chunk_phase_after=chunk_phase(flat_a))
    if a.teacher:
        rep["quiet_teacher"] = quiet_jitter(load_teacher(a.teacher))

    def line(name, q):
        if q:
            print(f"  {name:12s} 静止 {q['quiet_frac'] * 100:3.0f}% の区間で  "
                  f"関節 {q['bone_deg_per_frame']:.3f}°/f   root {q['root_cm_per_frame']:.2f} cm/f")
    print(f"[静止時の揺れ] {rep['n_clips']} 本")
    line("教師", rep.get("quiet_teacher"))
    line("補正前", rep["quiet_before"])
    line("補正後", rep["quiet_after"])
    print("[帯域の残り] " + "  ".join(f"{k} {v * 100:.0f}%" for k, v in rep["band_keep"].items()))
    cb, ca = rep["chunk_phase_before"], rep["chunk_phase_after"]
    print(f"[チャンク境界の跳ね] 補正前 {max(cb) / min(cb):.2f}倍 -> 補正後 {max(ca) / min(ca):.2f}倍")

    if a.out:
        np.savez_compressed(a.out, init_eps=src["init_eps"], **after)
        print(f"-> {a.out}")
    if a.report:
        pathlib.Path(a.report).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        print(f"-> {a.report}")
    if a.render:
        for spec in a.clips:
            t, _, i = spec.partition(":")
            i = int(i or 0)
            out = render_pair(before[t][i], after[t][i],
                              pathlib.Path(a.render) / f"{t}_{i}_pair.mp4", a.xml, fps=a.fps)
            print(f"-> {out}  (左: 補正前 / 右: 補正後)")


if __name__ == "__main__":
    main()
