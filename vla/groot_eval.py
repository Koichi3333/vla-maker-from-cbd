#!/usr/bin/env python3
"""Evaluate a GR00T N1.7 fine-tune (shared by B-2 / B-2b). Runs in the GR00T venv on the VM.

    cd /content/Isaac-GR00T
    MUJOCO_GL=egl .venv/bin/python /content/b2/gen/groot_eval.py \\
        --ckpt /content/b2/out/checkpoint-10000 --train /content/b2/train \\
        --heldout /content/b2/heldout --out /content/b2/eval/ckpt10000

Once per checkpoint. To load the model (about 1 minute) only once, it does the following two in a row.

1. **Open-loop evaluation on held-out** (in place of a validation loss)
   GR00T's launch_finetune.py cannot run validation (factory.py asserts
   eval_strategy == "no"). So, following the same steps as gr00t/eval/open_loop_eval.py, every 16 frames
   it predicts an action chunk from the true state and image and compares it with the true action.
   The same number of clips is taken from train and the same metrics computed (a sign of overfitting).
   For comparison, the trivial prediction that "holds the current pose for 16 frames" (hold) is listed too.
2. **Closed-loop generation** (unseen texts x held-out initial poses)
   Render the initial pose -> execute the 16 predicted frames -> render the last pose -> ... and repeat.
   Observation images use the same humanml3d_to_video.py rendering as training (same camera, same resolution).
   It records the root's horizontal displacement and "forward progress" (projection onto the initial heading) and writes mp4s.
   --blind makes observation images fully black (for B-2b models trained on black images,
   so the input matches training. mp4s are still rendered as usual for viewing).

With --object-eval, only V-5's H-2 runs (generate in closed loop from a frame with an object and measure the minimum hand-object distance).

With --check-dataset, the model is not loaded; it only checks that GR00T's loader can read
the dataset including video (a pre-training check).
"""
import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LANG_KEY = "annotation.human.task_description"
TEXTS = [
    "a person walks forward",
    "a person walks backward",
    "a person waves with the right hand",
    "a person squats down and stands up",
]


# ---------------------------------------------------------------- Metrics

def facing_xz(state):
    """(...,117) -> (...,2). Heading: the third column of the hips 6D (= +Z after rotation) projected onto the horizontal plane.

    HumanML3D aligns the initial heading to +Z (mean (0.01, 1.00) over the first frames of the real data).
    """
    f = np.cross(state[..., 3:6], state[..., 6:9])[..., [0, 2]]
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


def root_metrics(traj):
    """Horizontal root displacement of (T,117). Forward progress is the projection onto the initial heading (negative when moving backward).

    On the 31 "walks forward" clips of the real data, forward progress over the first 3 s has median +0.81 m and is positive in 87%.
    Over 10 s, "walk, turn around and come back" is common and the median shrinks to +0.15 m, so the 3 s value is reported too.
    """
    r = traj[:, [0, 2]]
    d = r[-1] - r[0]
    f0 = facing_xz(traj[0])
    return dict(disp=float(np.linalg.norm(d)),
                forward=float(d @ f0),
                forward_3s=float((r[min(60, len(r) - 1)] - r[0]) @ f0),
                path=float(np.linalg.norm(np.diff(r, axis=0), axis=1).sum()),
                max_dist=float(np.linalg.norm(r - r[0], axis=1).max()))


def bone_angle_deg(a, b):
    """Return the angle difference (degrees) of bone rotations between two (N,117) arrays as (N,19). 6D is fixed by Gram-Schmidt."""
    from rotation import sixd_to_quat
    qa = sixd_to_quat(a[:, 3:].reshape(len(a), -1, 6).astype(np.float64))
    qb = sixd_to_quat(b[:, 3:].reshape(len(b), -1, 6).astype(np.float64))
    dot = np.clip(np.abs((qa * qb).sum(-1)), 0, 1)
    return np.degrees(2 * np.arccos(dot))


def compare(pred, gt):
    return dict(mse=float(((pred - gt) ** 2).mean()),                      # same quantity as open_loop_eval.py
                root_err_m=float(np.linalg.norm(pred[:, :3] - gt[:, :3], axis=1).mean()),
                bone_err_deg=float(bone_angle_deg(pred, gt).mean()))


def project_6d(x):
    """Make a predicted 6D orthonormal again (the training state is always orthonormal, so bring it back before using it as the next input)."""
    from rotation import quat_to_6d, sixd_to_quat
    y = x.copy()
    q = sixd_to_quat(x[:, 3:].reshape(len(x), -1, 6).astype(np.float64))
    y[:, 3:] = quat_to_6d(q).reshape(len(x), -1)
    return y.astype(np.float32)


# ---------------------------------------------------------------- Rendering

class Scene:
    """Draw the same picture as the training observation images. One Renderer is reused within the process."""

    def __init__(self, xml, size=256, objects=False, object_scale=1.0):
        import mujoco
        if objects:                  # V-5's H-2: add object proxies (the same ones as vla/render_with_objects.py)
            from render_with_objects import AWAY, body_ids, xml_with_objects
            self.m = mujoco.MjModel.from_xml_string(xml_with_objects(xml, object_scale))
            self.mocap, self.hand = body_ids(self.m)
        else:
            self.m = mujoco.MjModel.from_xml_path(str(xml))
            self.mocap = None
        self.d = mujoco.MjData(self.m)
        if objects:
            for i in self.mocap.values():
                self.d.mocap_pos[i] = AWAY
        self.r = mujoco.Renderer(self.m, size, size)

    def qpos(self, states):
        from humanml3d_to_video import state_to_qpos
        return state_to_qpos(np.asarray(states, np.float64))

    def lift(self, state0):
        """Ground correction. Training used the lowest point over the whole clip, but generation has no future, so it is set from the initial pose."""
        from cbd_to_mujoco import ground
        return ground(self.m, self.d, self.qpos(state0[None]))

    def render(self, states, lo, obj=None):
        """If obj = (label, world coordinates (3,) or (T,3)), draw the object at that position."""
        from cbd_to_mujoco import frames
        q = self.qpos(states)
        q[:, 2] -= lo
        if obj is None:
            return np.stack([f.copy() for f in frames(self.m, self.d, q, self.r)])
        from render_with_objects import frames_with_object
        pos = np.asarray(obj[1], np.float64)
        pos = pos if pos.ndim == 2 else np.tile(pos, (len(q), 1))     # V-12: with (T,3) the object moves every frame (a held object)
        return np.stack([f.copy() for f in frames_with_object(self.m, self.d, self.r, self.mocap, q, pos, obj[0])])

    def goal_obs(self, state, lo, p):
        """V-8: one frame's state (117) and the object's world position p -> (goal (3,) = left, forward, up seen from the current pelvis; distance from p to the nearer hand).
        Same definition as vla/goal_labels.py (MuJoCo skeleton, horizontal heading from the hip joints, hand = midpoint of the wrist and the base of the middle finger)."""
        import mujoco
        q = self.qpos(np.asarray(state, np.float64)[None, :117])
        q[:, 2] -= lo
        self.d.qpos[:] = q[0]
        mujoco.mj_kinematics(self.m, self.d)
        x = lambda n: self.d.xpos[self.m.body(n).id]           # noqa: E731
        left = x("left_upper_leg") - x("right_upper_leg")
        left[2] = 0
        left /= np.linalg.norm(left) + 1e-9
        up = np.array([0.0, 0, 1])
        basis = np.stack([left, np.cross(left, up), up])
        g = basis @ (np.asarray(p, np.float64) - x("pelvis"))
        dist = min(np.linalg.norm((x(f"{s}_hand") + x(f"{s}_middle")) / 2 - p) for s in ("left", "right"))
        return g, float(dist)

    def waypoint_obs(self, state, prev, lo, p, side=None, reach=None):
        """V-11: the upper layer (planner). Current and previous state (117) and the object's world position p -> (waypoint (6,) = left and right hand 15 frames ahead minus now,
        in coordinates seen from the current pelvis; only the grasping hand moves and the other is 0; grasping hand side; distance from the grasping hand to p). If side is None, the hand currently nearer p.
        The path is plan() in vla/waypoint_labels.py (minimum jerk, duration estimated from a speed of 0.5 m/s)."""
        import mujoco
        from waypoint_labels import plan

        def fk(st):
            q = self.qpos(np.asarray(st, np.float64)[None, :117])
            q[:, 2] -= lo
            self.d.qpos[:] = q[0]
            mujoco.mj_kinematics(self.m, self.d)
            x = lambda n: self.d.xpos[self.m.body(n).id].copy()     # noqa: E731
            return x, {s: (x(f"{s}_hand") + x(f"{s}_middle")) / 2 for s in ("left", "right")}
        _, h_prev = fk(prev)
        x, h_now = fk(state)
        left = x("left_upper_leg") - x("right_upper_leg")
        left[2] = 0
        left /= np.linalg.norm(left) + 1e-9
        up = np.array([0.0, 0, 1])
        basis = np.stack([left, np.cross(left, up), up])
        body = lambda w: basis @ (np.asarray(w, np.float64) - x("pelvis"))   # noqa: E731
        if side is None:
            side = min(("left", "right"), key=lambda s: np.linalg.norm(h_now[s] - p))
        w = np.zeros(6)
        j = 0 if side == "left" else 3
        now, before, tgt = body(h_now[side]), body(h_prev[side]), body(p)
        if reach:                              # V-12: a goal farther than reach (m) from the shoulder is pulled into range (pulling a seated body toward an unreachable spot made it stand up)
            sh = body(x(f"{side}_upper_arm"))
            n = np.linalg.norm(tgt - sh)
            if n > reach:
                tgt = sh + (tgt - sh) * reach / n
        w[j:j + 3] = plan(now, before, tgt)
        path = np.stack([now + plan(now, before, tgt, t=k) for k in range(1, 17)])   # hands at action frames 0-15 (1-16 frames after the observation)
        return w, side, float(np.linalg.norm(h_now[side] - p)), path

    def hands(self, states, lo):
        """(T,117) -> positions of both hands (midpoint of the wrist and the base of the middle finger). Used for the H-2 distance."""
        from render_with_objects import anchors
        q = self.qpos(states)
        q[:, 2] -= lo
        return anchors(self.m, self.d, self.hand, q)


def write_mp4(path, frames, fps=20):
    from lerobot_add_video import ffmpeg_exe
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    p = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                          str(path)], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(frames).tobytes())
    p.stdin.close()
    assert p.wait() == 0, path


def tile(videos, cols):
    """[(T,H,W,3)] -> (T, rows*H, cols*W, 3). Empty cells are black."""
    T, H, W, _ = videos[0].shape
    rows = -(-len(videos) // cols)
    out = np.zeros((T, rows * H, cols * W, 3), np.uint8)
    for k, v in enumerate(videos):
        r, c = divmod(k, cols)
        out[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = v
    return out


# ---------------------------------------------------------------- Data

def layout(dataset):
    """Ranges in meta/modality.json -> {group: (start, end)}."""
    m = json.loads((pathlib.Path(dataset) / "meta/modality.json").read_text())
    return {k: (v["start"], v["end"]) for k, v in m["state"].items()}


def join(groups, lay, lead):
    """{group: (..., D)} -> (..., 117). Ranges not in groups (such as V-8's goal in the state) are left out."""
    out = np.zeros(lead + (max(e for k, (_, e) in lay.items() if k in groups),), np.float32)
    for k in groups:
        s, e = lay[k]
        out[..., s:e] = groups[k]
    return out


def check_dataset(path, config_py):
    """Without a model, check that GR00T's loader can read the data including video."""
    import importlib
    sys.path.insert(0, str(pathlib.Path(config_py).parent))
    importlib.import_module(pathlib.Path(config_py).stem)
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    mc = MODALITY_CONFIGS["new_embodiment"]
    loader = LeRobotEpisodeLoader(dataset_path=path, modality_configs=mc)
    for i in sorted({0, len(loader) - 1}):
        df = loader[i]
        v = np.stack(df["video.front_view"].tolist())
        print(f"episode pos {i}: T={len(df)} video={v.shape} {v.dtype} mean={v.mean():.1f} "
              f"std={v.std():.1f} lang={df['language.' + LANG_KEY].iloc[0]!r}")
        assert v.shape == (len(df), 256, 256, 3) and v.dtype == np.uint8
    print(f"LOADER OK: {path} ({len(loader)} episodes)")


# ---------------------------------------------------------------- Evaluation

def open_loop(policy, dataset, positions, horizon, lay, samples=1, smooth=False, seed=0):
    """Build inputs the same way as evaluate_single_trajectory in open_loop_eval.py and return only the metrics.

    With `samples` > 1, several diffusion samples are drawn from the same observation and **their mean over the 117 dims** is the prediction
    (averaging 6D breaks orthogonality, so it is restored with `project_6d`). The sampling noise has
    zero mean, so the mean moves the prediction toward the conditional mean. `smooth` applies
    `smooth_motion.smooth_states` to the predicted trajectory before comparing. **Both act only at inference;
    the weights are not touched.** The default (1 sample, no smoothing) is the same yardstick as B-2 / B-2b / B-5.
    """
    import torch
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=dataset, modality_configs=modality)
    mc = copy.deepcopy(loader.modality_configs)
    mc.pop("action")
    akeys = loader.modality_configs["action"].modality_keys
    skeys = loader.modality_configs["state"].modality_keys
    rows, lat = [], []
    for pos in positions:
        traj = loader[pos]
        T = len(traj)
        gt = join({k: np.stack(traj[f"action.{k}"].tolist()) for k in akeys}, lay, (T,))
        st = join({k: np.stack(traj[f"state.{k}"].tolist()) for k in skeys if k not in EXTRA_STATE}, lay, (T,))   # for hold (117)
        pred = []
        for s in range(0, T, horizon):
            # allow_padding=True: when a state delta is negative, the index goes negative at the start of a clip.
            # With the default, pandas iloc picks up **the last frames** instead (it breaks silently).
            # With no history (delta=[0]) nothing changes.
            dp = extract_step_data(traj, s, mc, policy.embodiment_tag, allow_padding=True)
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            obs.update({f"video.{k}": np.array(v) for k, v in dp.images.items()})
            for lk in loader.modality_configs["language"].modality_keys:
                obs[lk] = dp.text
            t0 = time.perf_counter()
            acc = None
            for k in range(samples):
                if samples > 1:
                    torch.manual_seed(seed * 1000 + k)     # different noise per sample
                a, _ = policy.get_action(parse_observation_gr00t(obs, loader.modality_configs))
                chunk = join({m: a[m][0] for m in akeys}, lay, (horizon,))
                acc = chunk if acc is None else acc + chunk
            lat.append((time.perf_counter() - t0) / samples)
            pred.append(acc / samples if samples == 1 else project_6d(acc / samples))
        pred = np.concatenate(pred)[:T]
        if smooth:
            from smooth_motion import smooth_states
            pred = smooth_states(pred)
        hold = st[(np.arange(T) // horizon) * horizon]
        ep = int(loader.episodes_metadata[pos]["episode_index"])
        rows.append(dict(ep=ep, T=T, text=dp.text, model=compare(pred, gt), hold=compare(hold, gt)))
    mean = {w: {m: float(np.mean([r[w][m] for r in rows])) for m in rows[0][w]}
            for w in ("model", "hold")}
    return dict(mean=mean, per_traj=rows, latency_s=float(np.median(lat)))


GOAL_REACHED_M = 0.08          # V-8: once the hand is within this distance of the object, "no goal" from then on (in the training data, too, the object goal disappears after the grasp)
EXTRA_STATE = ("goal", "waypoint")   # ranges in the state that are not part of CBD's 117 (V-8's goal, V-11's waypoint)


def rollout(policy, scene, inits, text, frames, horizon, lay, blind=False, samples=1, seed=0, objs=None, goal_mode="object",
            hand_sides=None, attach=False, overlap=0, reach=None, goals=None):
    """inits: (B,117). Runs B rollouts together in one batch. With blind, observation images are fully black.

    V-10: if `goals` (a sequence of B items, each (3,) or None) is given, the goal (goal, planner, guidance) uses it instead of the object's position.
    The object on screen and "held" (attach) stay at the object's true position. A None item means no goal (the VLM returned no point).

    With `samples` > 1, several diffusion samples are drawn and averaged at each step. **In closed loop
    the averaged pose becomes the next observation directly**, so unlike post-hoc smoothing the jitter does not accumulate.

    V-8: if the model's state has `goal`, a goal is built from the current pose every step. With `goal_mode="object"`, the position of the object (objs)
    is converted to coordinates seen from the current pelvis with flag 1, and once the hand comes within `GOAL_REACHED_M` the flag is 0 from then on. With no object or `goal_mode="off"` the flag is always 0
    (a control for whether the model can ground the target from the image without a goal).

    V-11: if the state has `waypoint`, with `goal_mode="object"` the planner (`Scene.waypoint_obs`) builds a waypoint 15 frames ahead every step from the path
    "current hand -> object" (the grasping hand is fixed at the first step to the one nearer the object). Once reached (`GOAL_REACHED_M`) the flag is 0 from then on.

    V-12: `hand_sides` ([B]) sets the grasping hand. With `attach`, from the frame the hand reaches the object, the object is attached to that hand and moves with it (the same as held
    in the training-data renderer `render_with_objects.object_track`; the held object also appears in the observation image). Reaching is checked every frame. With `overlap` = k > 0 and
    inference-time guidance (groot_guidance) installed, each inference advances only horizon - k frames and the remaining k frames are overlapped onto the start of the next chunk.
    `reach` is the arm length the planner uses to pull in goals. The third return value is dict(attach=[attach frame or None], side=[grasping hand]).
    """
    import torch
    B = len(inits)
    lo = [scene.lift(s) for s in inits]
    objs = objs or [None] * B
    tgt = [None if o is None else o[1] for o in objs] if goals is None else list(goals)   # goal position
    cur = project_6d(inits)
    traj = [cur]
    skeys = policy.get_modality_config()["state"].modality_keys
    akeys = policy.get_modality_config()["action"].modality_keys
    sdelta = list(policy.get_modality_config()["state"].delta_indices)   # with history, e.g. [-4,-2,0]
    use_goal = "goal" in skeys
    use_wp = "waypoint" in skeys
    try:                                       # V-12: with inference-time guidance installed, run the planner and pass the goal even for models without waypoints
        import groot_guidance
        guide = groot_guidance if policy.model.action_head.get_action_with_features.__func__.__module__ == "groot_guidance" else None
    except (ImportError, AttributeError):
        guide = None
    wp_side = list(hand_sides) if hand_sides is not None else [None] * B
    reached = [False] * B
    attached = [None] * B                       # V-12: frame at which the object was attached to the hand
    ov = overlap if (guide is not None and overlap > 0) else 0
    E = horizon - ov                            # frames advanced per inference
    prev_norm = None

    def obj_now(b):
        if objs[b] is None or attached[b] is None:
            return objs[b]
        return (objs[b][0], scene.hands(cur[b][None], lo[b])[wp_side[b]][0])

    while len(traj) < frames:
        if blind:
            img = np.zeros((B, 1, scene.r.height, scene.r.width, 3), np.uint8)
        else:
            img = np.stack([scene.render(cur[b:b + 1], lo[b], obj_now(b)) for b in range(B)])   # (B,1,H,W,3)
        # Stack the state for the policy's delta_indices. Missing frames (at the start of generation) are filled with the first one.
        hist = np.stack([traj[max(0, len(traj) - 1 + d)] for d in sdelta], 1)      # (B,T,117)
        state = {k: hist[:, :, lay[k][0]:lay[k][1]].copy() for k in skeys if k not in EXTRA_STATE}
        if use_goal:
            gl = np.zeros((B, len(sdelta), lay["goal"][1] - lay["goal"][0]), np.float32)
            for b in range(B):
                if goal_mode == "object" and tgt[b] is not None and not reached[b]:
                    g, dist = scene.goal_obs(cur[b], lo[b], tgt[b])
                    if dist <= GOAL_REACHED_M:
                        reached[b] = True
                    else:
                        gl[b, :, :3] = g
                        gl[b, :, 3] = 1.0
            state["goal"] = gl
        if use_wp or guide:
            wl = np.zeros((B, len(sdelta), 7), np.float32)
            active, path, sides = np.zeros(B, bool), np.zeros((B, 16, 3)), ["left"] * B
            prev = traj[-2] if len(traj) > 1 else cur
            for b in range(B):
                if goal_mode == "object" and tgt[b] is not None and not reached[b]:
                    w, wp_side[b], dist, pth = scene.waypoint_obs(cur[b], prev[b], lo[b], tgt[b], wp_side[b], reach)
                    if dist <= GOAL_REACHED_M:
                        reached[b] = True
                    else:
                        wl[b, :, :6] = w
                        wl[b, :, 6] = 1.0
                        active[b], path[b], sides[b] = True, pth, wp_side[b]
            if use_wp:
                state["waypoint"] = wl
            if guide:
                guide.CTX.update(active=active, path=path, side=sides, horizon=horizon, overlap=ov, prev_norm=prev_norm)
        obs = {"video": {"front_view": img},
               "state": state,
               "language": {LANG_KEY: [[text]] * B}}
        acc, acc_n = None, None
        for k in range(samples):
            if samples > 1:
                torch.manual_seed(seed * 1000 + k)
            a, _ = policy.get_action(obs)
            c = join({m: a[m] for m in akeys}, lay, (B, horizon))
            acc = c if acc is None else acc + c
            if ov:
                ln = guide.CTX["last_norm"]
                acc_n = ln if acc_n is None else acc_n + ln
        chunk = acc / samples
        if ov:
            prev_norm = acc_n / samples
        for t in range(E):
            traj.append(project_6d(chunk[:, t]))
            if attach:                          # every frame, check whether the grasping hand has reached the object
                for b in range(B):
                    if objs[b] is not None and attached[b] is None and wp_side[b] is not None:
                        h = scene.hands(traj[-1][b][None], lo[b])[wp_side[b]][0]
                        if np.linalg.norm(h - np.asarray(objs[b][1])) <= GOAL_REACHED_M:
                            attached[b] = len(traj) - 1
                            reached[b] = True
        cur = traj[-1]
    traj = np.stack(traj[:frames], 1)                                             # (B,T,117)
    return traj, lo, dict(attach=attached, side=wp_side, exec_frames=E)


def h4_slope(rows, key):
    """sum e.d / sum d.d (e = hand displacement, d = object offset)."""
    e = np.array([x[key] for x in rows])
    d = np.array([x["offset_m"] for x in rows])
    return float((e * d).sum() / (d * d).sum()) if len(rows) else float("nan")


def object_eval(policy, spec, xml, frames, horizon, lay, out, samples=1, seed=0, reps=4, object_scale=1.0, reps_h4=2,
                goal_mode="object", spec_hand=False, attach=False, overlap=0, reach=None, goal_key=None):
    """V-5's H-2. Generate in closed loop from a frame with an object and measure the minimum hand-object distance.

    For each video, the initial pose (first frame) is run `reps` times per condition, all conditions in one batch
    (even with the same initial pose, the diffusion noise differs per batch element, so the generations differ):
    - `orig`: put the object **where it was actually first grasped**
    - `shift`: put it `shift_m` away along the pelvis left-right axis, toward the side opposite the object (a control for **whether the model looks at the object on screen**)
    - `mirror` (when the spec has `obj_mirror_raw`; V-5b): put it on the other side of the body (mirrored about the midsagittal plane through the pelvis)
    The distance is between the nearer of the two hands (midpoint of the wrist and the base of the middle finger) and the object's center. For `shift` / `mirror`
    it is reported both to the drawn position and to the undrawn original position (closer to the drawn one = `closer_to_shown`).
    `closer_to_shown` compares the minimum distance over all 200 frames, so it fails if the hand passes near the original position after grasping, and clips whose hand starts at the original position
    always fail (confirmed 2026-10-10; HANDOFF_VLA.md "mirrored stuck at 0.64"). It is kept for comparison with past results, and fixed values are reported separately:
    `closer_to_shown_pre` (compare only up to the grasp) and `start_near_orig` (the hand starts within 15 cm of the original position).
    The summary's `frac_closer_to_shown_fixed` = the fraction of closer_to_shown_pre excluding start_near_orig. The pass bar is judged on this one.
    - `h4_k` (when the spec has `h4_offsets`; V-7's H-4): put the object `h4_offsets[k]` away from the original position (`reps_h4` runs each).
      **The hand position is measured at a fixed time**: t* = the median time at which `h4_hand` came closest to the object in orig. From the displacement e of `h4_hand` at that time
      relative to the orig mean, and the object offset d, compute the slope sum e.d / sum d.d (1 = the hand follows the object fully, 0 = ignores it).
      Measuring at the point nearest the object would pick a point near the object even when the trajectory is unrelated to it and give a positive slope, so that is reported only for reference (`slope_nearest`)

    V-12 stage 2b: with `spec_hand`, the grasping hand is the spec's `h4_hand` (the hand first used in the recording), swapped only for mirror (otherwise the planner picks the nearer hand).
    `attach` / `overlap` / `reach` go to rollout. The mp4 draws the held object together with the hand. Trajectories are saved to `traj_<clip>.npz`.
    V-10: with `goal_key` (e.g. "vlm_goal"), the spec's `it[goal_key][cond]` ((3,) in the same coordinates as the placement, or None) is used as the goal
    (the object is drawn at its true position; h4 conditions keep the true position).
    **Motion roughness** is measured too: the magnitude of the second difference (acceleration) of both hand positions, averaged over chunk-seam frames and the rest (`acc_seam` / `acc_inner`, m/s^2),
    and the mean of the third difference (jerk) (`jerk`, m/s^3). If the motion jumps at seams, acc_seam is larger than acc_inner
    """
    import torch
    scene = Scene(xml, objects=True, object_scale=object_scale)
    rows = []
    conds = ["orig", "shift"] + (["mirror"] if all("obj_mirror_raw" in it for it in spec["items"]) else [])
    use_h4 = all("h4_offsets" in it for it in spec["items"])
    for it in spec["items"]:
        init = np.asarray(it["init_state"], np.float32)
        lo0 = scene.lift(init)
        down = np.array([0, 0, lo0])
        place = dict(orig=np.asarray(it["obj_raw"]) - down)
        place["shift"] = place["orig"] + np.asarray(it["shift_dir"]) * it["shift_m"]
        if "mirror" in conds:
            place["mirror"] = np.asarray(it["obj_mirror_raw"]) - down
        h4c = [f"h4_{k}" for k in range(len(it["h4_offsets"]))] if use_h4 else []
        for k, c in enumerate(h4c):
            place[c] = place["orig"] + np.asarray(it["h4_offsets"][k])
        cb = [c for c in conds for _ in range(reps)] + [c for c in h4c for _ in range(reps_h4)]
        objs = [(it["label"], place[c]) for c in cb]
        torch.manual_seed(seed)
        t1 = time.perf_counter()
        sides = None
        if spec_hand:
            flip = {"left": "right", "right": "left"}
            sides = [flip[it["h4_hand"]] if c == "mirror" else it["h4_hand"] for c in cb]
        goals = None
        if goal_key:
            gk = it.get(goal_key, {})
            goals = [(None if gk.get(c) is None else np.asarray(gk[c])) if c in gk else place[c] for c in cb]
        traj, lo, info = rollout(policy, scene, np.stack([init] * len(cb)), it["text"], frames, horizon, lay,
                                 samples=samples, seed=seed, objs=objs, goal_mode=goal_mode, goals=goals,
                                 hand_sides=sides, attach=attach, overlap=overlap, reach=reach)
        E = info["exec_frames"]
        seam = np.zeros(frames, bool)
        seam[1::E] = True                                   # first frame of each inference (traj[0] is the initial pose)
        hds = []
        for b, c in enumerate(cb):
            hd = scene.hands(traj[b], lo[b])
            hds.append(hd)
            dist = lambda p: np.minimum(np.linalg.norm(hd["left"] - p, axis=1), np.linalg.norm(hd["right"] - p, axis=1))
            d_shown = dist(place[c])
            r = dict(clip=it["clip"], label=it["label"], cond=c, rep=b % reps,
                     min_dist_m=float(d_shown.min()), t_min_sec=float(d_shown.argmin() / 20),
                     reached_10cm=bool(d_shown.min() <= 0.10), side=info["side"][b],
                     attach_sec=None if info["attach"][b] is None else info["attach"][b] / 20)
            acc = np.mean([np.linalg.norm(np.diff(hd[s_], 2, axis=0), axis=1) * 400 for s_ in ("left", "right")], 0)   # (T-2,) m/s²
            jerk = np.mean([np.linalg.norm(np.diff(hd[s_], 3, axis=0), axis=1) * 8000 for s_ in ("left", "right")], 0)
            sm = seam[1:-1]                                  # acc[i] is the second difference centered on frame i+1
            r.update(acc_seam=float(acc[sm].mean()), acc_inner=float(acc[~sm].mean()), jerk=float(jerk.mean()))
            if c.startswith("h4_"):
                r.update(offset_m=np.asarray(it["h4_offsets"][int(c[3:])]).tolist())
            elif c != "orig":
                d_orig = dist(place["orig"])
                a0 = info["attach"][b]
                end = a0 if a0 else frames                    # before the grasp (the whole clip if never grasped)
                r.update(min_dist_to_unshown_orig_m=float(d_orig.min()),
                         closer_to_shown=bool(d_shown.min() < d_orig.min()),
                         closer_to_shown_pre=bool(d_shown[:end].min() < d_orig[:end].min()),
                         start_near_orig=bool(d_orig[0] < 0.15))
            rows.append(r)
        if use_h4:                      # H-4: displacement of the hand position at the fixed time t*
            hh = it["h4_hand"]
            bo = [b for b, c in enumerate(cb) if c == "orig"]
            tstar = int(np.median([np.linalg.norm(hds[b][hh] - place["orig"], axis=1).argmin() for b in bo]))
            ref = np.mean([hds[b][hh][tstar] for b in bo], 0)
            def near(b, p):             # reference: position of the hand nearest the object, at the time it came nearest
                dl, dr = (np.linalg.norm(hds[b][s_] - p, axis=1) for s_ in ("left", "right"))
                return hds[b]["left"][dl.argmin()] if dl.min() <= dr.min() else hds[b]["right"][dr.argmin()]
            ref_near = np.mean([near(b, place["orig"]) for b in bo], 0)
            base = len(rows) - len(cb)
            for b, c in enumerate(cb):
                if c.startswith("h4_"):
                    rows[base + b].update(t_star_sec=tstar / 20, hand=hh,
                                          disp_m=(hds[b][hh][tstar] - ref).tolist(),
                                          disp_nearest_m=(near(b, place[c]) - ref_near).tolist())
        slug = it.get("slug", it["clip"][-4:])          # V-12 stage 3 specs give mirrored items their own slug
        tracks = []
        for b in range(len(cb)):                             # a held object moves with the grasping hand (same as the training-data renderer)
            pos = np.tile(np.asarray(objs[b][1], np.float64), (frames, 1))
            if info["attach"][b] is not None:
                a0 = info["attach"][b]
                pos[a0:] = hds[b][info["side"][b]][a0:]
            tracks.append(pos)
        for c in conds:
            b = cb.index(c)
            write_mp4(out / "mp4_object" / f"{slug}_{c}.mp4", scene.render(traj[b], lo[b], (objs[b][0], tracks[b])))
        np.savez_compressed(out / f"traj_{slug}.npz", traj=traj.astype(np.float32), lo=np.asarray(lo), conds=np.array(cb),
                            obj=np.stack(tracks).astype(np.float32), side=np.array([str(x) for x in info["side"]]),
                            attach=np.array([-1 if x is None else x for x in info["attach"]]), exec_frames=E, label=it["label"])
        med = lambda c, k: np.median([x[k] for x in rows if x["clip"] == it["clip"] and x["cond"] == c])
        h4s = ""
        if use_h4:
            v = [x for x in rows if x["clip"] == it["clip"] and x["cond"].startswith("h4_")]
            h4s = f"  H-4 傾き {h4_slope(v, 'disp_m'):+.2f}（参考: 最接近点 {h4_slope(v, 'disp_nearest_m'):+.2f}）"
        print(f"[object] {it['clip'][-40:]:40s} " + "  ".join(
            f"{c} {med(c, 'min_dist_m'):.3f} m" + ("" if c == "orig" else f"（元の位置へ {med(c, 'min_dist_to_unshown_orig_m'):.3f}）")
            for c in conds) + h4s + f" [{time.perf_counter() - t1:.0f}s]", flush=True)
    summ = {}
    for c in conds:
        v = [x for x in rows if x["cond"] == c]
        summ[c] = dict(median_min_dist_m=float(np.median([x["min_dist_m"] for x in v])),
                       mean_min_dist_m=float(np.mean([x["min_dist_m"] for x in v])),
                       reached_10cm=float(np.mean([x["reached_10cm"] for x in v])), n=len(v))
        if c != "orig":
            summ[c]["frac_closer_to_shown"] = float(np.mean([x["closer_to_shown"] for x in v]))
            summ[c]["frac_closer_to_shown_pre"] = float(np.mean([x["closer_to_shown_pre"] for x in v]))
            vf = [x for x in v if not x["start_near_orig"]]
            summ[c]["frac_closer_to_shown_fixed"] = float(np.mean([x["closer_to_shown_pre"] for x in vf])) if vf else None
            summ[c]["n_start_near_orig"] = len(v) - len(vf)
            summ[c]["median_min_dist_to_unshown_orig_m"] = float(np.median([x["min_dist_to_unshown_orig_m"] for x in v]))
    summ["motion"] = dict(acc_seam=float(np.mean([x["acc_seam"] for x in rows])), acc_inner=float(np.mean([x["acc_inner"] for x in rows])),
                          jerk=float(np.mean([x["jerk"] for x in rows])),
                          attached=float(np.mean([x["attach_sec"] is not None for x in rows if not x["cond"].startswith("h4_")])))
    print(f"[motion] 手の加速度 継ぎ目 {summ['motion']['acc_seam']:.2f} / それ以外 {summ['motion']['acc_inner']:.2f} m/s²  "
          f"加加速度 {summ['motion']['jerk']:.1f} m/s³  物を持った割合 {summ['motion']['attached']:.2f}", flush=True)
    if use_h4:
        v = [x for x in rows if x["cond"].startswith("h4_")]
        per_clip = {c: h4_slope([x for x in v if x["clip"] == c], "disp_m") for c in dict.fromkeys(x["clip"] for x in v)}
        summ["h4"] = dict(slope=h4_slope(v, "disp_m"), slope_clip_median=float(np.median(list(per_clip.values()))),
                          slope_per_clip=per_clip, slope_nearest=h4_slope(v, "disp_nearest_m"),
                          frac_toward=float(np.mean([np.dot(x["disp_m"], x["offset_m"]) > 0 for x in v])), n=len(v))
    gg = sys.modules.get("groot_guidance")          # V-12: distance between the hand at action frame 14 and the goal, before and after guidance (per chunk x sample)
    if gg is not None and gg.STATS.get("all_before"):
        eb, ea = np.array(gg.STATS["all_before"]), np.array(gg.STATS.get("all_after", []))
        summ["guidance"] = dict(err_before_median_m=float(np.median(eb)), err_after_median_m=float(np.median(ea)) if len(ea) else None, n=len(eb))
        print(f"[guidance] キーフレームの手と計画器の経路の距離（中央値）: 誘導前 {np.median(eb) * 100:.1f} cm → 誘導後 "
              f"{np.median(ea) * 100 if len(ea) else float('nan'):.1f} cm（{len(eb)} 回）", flush=True)
    (out / "object_eval.json").write_text(json.dumps(dict(summary=summ, rows=rows, samples=samples, seed=seed, goal_mode=goal_mode,
                                                          reps=reps, reps_h4=reps_h4 if use_h4 else 0,
                                                          frames=frames, conds=conds),
                                                     indent=1, ensure_ascii=False))
    print("[object] 全体: " + "  ".join(
        f"{c} {summ[c]['median_min_dist_m']:.3f} m（10cm 以内 {summ[c]['reached_10cm']:.2f}"
        + ("" if c == "orig" else f"、描いた方へ近づいた割合 {summ[c]['frac_closer_to_shown']:.2f}") + "）"
        for c in conds), flush=True)
    fx = lambda v: "—" if v is None else f"{v:.2f}"    # noqa: E731
    print("[object] 近づいた割合（直した数え方: 掴む前で比べ、最初から元の位置にある本数を除く）: " + "  ".join(
        f"{c} {fx(summ[c]['frac_closer_to_shown_fixed'])}（掴む前 {summ[c]['frac_closer_to_shown_pre']:.2f}、"
        f"除いた {summ[c]['n_start_near_orig']} 本）" for c in conds if c != "orig"), flush=True)
    if use_h4:
        h = summ["h4"]
        print(f"[object] H-4: 傾き {h['slope']:+.3f}（clip の中央値 {h['slope_clip_median']:+.3f}、ずらした向きへ動いた割合 "
              f"{h['frac_toward']:.2f}、参考: 最接近点 {h['slope_nearest']:+.3f}、{h['n']} 本）", flush=True)
    print(f"OBJECT_EVAL_DONE {out / 'object_eval.json'}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--train")
    ap.add_argument("--heldout")
    ap.add_argument("--out")
    ap.add_argument("--xml", default=str(HERE / "humanoid.xml"))
    ap.add_argument("--n-train", type=int, default=21, help="train 側から等間隔に何本評価するか")
    ap.add_argument("--frames", type=int, default=200, help="生成の長さ（20fps で 10 秒）")
    ap.add_argument("--texts", nargs="*", default=TEXTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-inits", type=int, default=0,
                    help="閉ループの初期姿勢を等間隔に何本へ間引くか（0 なら held-out 全件）")
    ap.add_argument("--blind", action="store_true", help="閉ループの観測画像を全面黒にする（B-2b）")
    ap.add_argument("--samples", type=int, default=1,
                    help="拡散のサンプルを何本引いて平均するか（推論側だけの操作。既定 1 = 従来と同じ）")
    ap.add_argument("--denoising", type=int, default=0,
                    help="denoising ステップ数を上書きする（0 なら重みの既定値のまま。既定は 4）")
    ap.add_argument("--smooth", action="store_true",
                    help="予測した軌道に smooth_motion.smooth_states をかけてから突き合わせる")
    ap.add_argument("--sweep", nargs="*", default=None, metavar="SAMPLES:DENOISE:SMOOTH",
                    help="held-out の open-loop だけを複数条件で回して比べる"
                         "（例: 1:4:0 8:4:0 8:4:1 1:16:0）。閉ループは回さない")
    ap.add_argument("--check-dataset", nargs="*", default=None,
                    help="モデルを読まずにローダの確認だけする（データセットのパス）")
    ap.add_argument("--config", default=str(HERE / "groot_cbd_config.py"))
    ap.add_argument("--goal-key", default=None,
                    help="V-10: 仕様の項目のこのキー（例 vlm_goal）を目標に使う（物は本当の位置に描く）")
    ap.add_argument("--object-eval", default=None,
                    help="V-5 の H-2。vla/object_eval_spec.py の JSON。これだけを回して終わる")
    ap.add_argument("--object-reps", type=int, default=4)
    ap.add_argument("--goal", default="object", choices=["object", "off"],
                    help="V-8 / V-11: モデルの state に goal / waypoint があるとき、閉ループで物の位置（V-11 は計画器の経由点）を渡すか（off なら常になし）")
    ap.add_argument("--object-h4-reps", type=int, default=2, help="H-4 のずらし 1 通りあたりの本数（仕様に h4_offsets があるとき）")
    ap.add_argument("--object-scale", type=float, default=1.0,
                    help="H-2 の物の寸法の倍率（V-7: 学習を render_with_objects.py --object-scale で描いたときは同じ値を渡す）")
    ap.add_argument("--action-head", default=None,
                    help="groot_extract_trainable.py の safetensors。--ckpt を読んだあと strict=False で上書きする（V-5b）")
    ap.add_argument("--guide", default="none", choices=["none", "cfg", "fk", "both"],
                    help="V-12: 推論時に経由点を効かせる（groot_guidance.py）。cfg = 条件の増幅、fk = 順運動学による誘導。H-2 の閉ループでだけ効く")
    ap.add_argument("--cfg-w", type=float, default=3.0)
    ap.add_argument("--fk-eta", type=float, default=0.7)
    ap.add_argument("--overlap", type=int, default=0, help="V-12: チャンクの重ね（推論時の誘導の仕組みで。0 なら従来どおり 16 フレームずつ）")
    ap.add_argument("--attach", action="store_true", help="V-12: 届いたら物を手に付けて動かす（H-2 の閉ループ）")
    ap.add_argument("--spec-hand", action="store_true", help="V-12: 掴む手を仕様の h4_hand にする（mirror は逆の手）")
    ap.add_argument("--v-plan", type=float, default=None,
                    help="V-12 段 3: 計画器の手の平均の速さ（m/s。既定は waypoint_labels.V_PLAN = 0.5）。撮影は直線で約 0.14（v12_reach_timing.py）")
    ap.add_argument("--reach", type=float, default=None, help="V-12: 計画器が目標を縮める腕の長さ（m）。学習データの 99%% は 0.546")
    a = ap.parse_args()

    if a.check_dataset is not None:
        for p in a.check_dataset:
            check_dataset(p, a.config)
        return

    import torch
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lay = layout(a.train)

    t0 = time.perf_counter()
    policy = Gr00tPolicy(embodiment_tag="new_embodiment", model_path=a.ckpt, device="cuda")
    horizon = len(policy.get_modality_config()["action"].delta_indices)
    if a.action_head:
        # Replace only the trainable part (action_head). The VLM side is frozen and identical to the base (see the note in groot_extract_trainable.py)
        from safetensors.torch import load_file
        sd = load_file(a.action_head)
        missing, unexpected = policy.model.load_state_dict(sd, strict=False)
        if unexpected:                      # if the key prefix does not match, load directly into action_head (with the prefix stripped)
            pre = "action_head."
            sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            missing, unexpected = policy.model.action_head.load_state_dict(sub, strict=False)
            print(f"[action-head] 前置きを外して action_head へ直接読んだ（missing {len(missing)}）", flush=True)
        assert not unexpected, f"読み込めなかったキー {len(unexpected)} 件（例 {list(unexpected)[:3]}）"
        print(f"[action-head] {a.action_head} から {len(sd)} テンソルを上書きした", flush=True)
    if a.denoising:
        policy.model.action_head.num_inference_timesteps = a.denoising
    if a.guide != "none" or a.overlap > 0:
        import groot_guidance
        groot_guidance.install(policy, a.ckpt, a.xml, mode=a.guide, cfg_w=a.cfg_w, fk_eta=a.fk_eta)
    res = dict(ckpt=a.ckpt, action_head=a.action_head, seed=a.seed, horizon=horizon, blind=a.blind,
               samples=a.samples, smooth=a.smooth,
               denoising_steps=int(policy.model.action_head.num_inference_timesteps),
               t_load=time.perf_counter() - t0)
    print(f"loaded {a.ckpt} in {res['t_load']:.0f}s (horizon {horizon}, "
          f"{res['denoising_steps']} denoising steps)", flush=True)

    if a.v_plan:                                # plan() reads V_PLAN on every call, so replacing the module value takes effect
        import waypoint_labels
        waypoint_labels.V_PLAN = a.v_plan
        print(f"[plan] 計画器の速さ {a.v_plan} m/s", flush=True)
    if a.object_eval:
        object_eval(policy, json.loads(pathlib.Path(a.object_eval).read_text()), a.xml, a.frames, horizon, lay,
                    out, samples=a.samples, seed=a.seed, reps=a.object_reps, object_scale=a.object_scale,
                    reps_h4=a.object_h4_reps, goal_mode=a.goal,
                    spec_hand=a.spec_hand, attach=a.attach, overlap=a.overlap, reach=a.reach, goal_key=a.goal_key)
        return

    # 1. open-loop (all of held-out, and the same number from train)
    held_eps = [json.loads(l) for l in (pathlib.Path(a.heldout) / "meta/episodes.jsonl").read_text().splitlines() if l]

    if a.sweep is not None:
        # Vary only the inference settings and compare on held-out. **The weights are not touched.**
        # The yardstick is unchanged: "can it beat hold (keep the current pose for 16 frames)?"
        specs = a.sweep or ["1:4:0", "1:4:1", "4:4:0", "8:4:0", "8:4:1", "1:16:0"]
        rows = []
        for spec in specs:
            ns, nd, sm = (int(v) for v in spec.split(":"))
            policy.model.action_head.num_inference_timesteps = nd
            torch.manual_seed(a.seed)
            np.random.seed(a.seed)
            t1 = time.perf_counter()
            r = open_loop(policy, a.heldout, range(len(held_eps)), horizon, lay,
                          samples=ns, smooth=bool(sm), seed=a.seed)
            m, h = r["mean"]["model"], r["mean"]["hold"]
            win = sum(1 for x in r["per_traj"] if x["model"]["bone_err_deg"] < x["hold"]["bone_err_deg"])
            rows.append(dict(spec=spec, samples=ns, denoising=nd, smooth=bool(sm),
                             bone=m["bone_err_deg"], root=m["root_err_m"], mse=m["mse"],
                             hold_bone=h["bone_err_deg"], hold_root=h["root_err_m"],
                             win_bone=win, n=len(r["per_traj"]), sec=time.perf_counter() - t1))
            print(f"[sweep] {spec:8s} bone {m['bone_err_deg']:6.2f}° (hold {h['bone_err_deg']:.2f}) "
                  f"root {m['root_err_m'] * 100:5.1f}cm (hold {h['root_err_m'] * 100:.1f}) "
                  f"mse {m['mse']:.5f}  hold に勝ったクリップ {win}/{len(r['per_traj'])} "
                  f"[{time.perf_counter() - t1:.0f}s]", flush=True)
        (out / "sweep.json").write_text(json.dumps(dict(ckpt=a.ckpt, seed=a.seed, rows=rows),
                                                   indent=1, ensure_ascii=False))
        print(f"SWEEP_DONE {out / 'sweep.json'}", flush=True)
        return
    n_tr = sum(1 for l in (pathlib.Path(a.train) / "meta/episodes.jsonl").read_text().splitlines() if l)
    for name, path, pos in [("heldout", a.heldout, range(len(held_eps))),
                            ("train", a.train, np.linspace(0, n_tr - 1, a.n_train).round().astype(int))]:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        t1 = time.perf_counter()
        res[name] = open_loop(policy, path, [int(p) for p in pos], horizon, lay,
                              samples=a.samples, smooth=a.smooth, seed=a.seed)
        m, h = res[name]["mean"]["model"], res[name]["mean"]["hold"]
        print(f"[{name}] {len(res[name]['per_traj'])} traj in {time.perf_counter() - t1:.0f}s  "
              f"mse {m['mse']:.5f} (hold {h['mse']:.5f})  root {m['root_err_m'] * 100:.1f}cm "
              f"(hold {h['root_err_m'] * 100:.1f})  bone {m['bone_err_deg']:.2f}deg "
              f"(hold {h['bone_err_deg']:.2f})", flush=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # 2. Closed-loop generation (the first frame of each held-out clip is the initial pose)
    # --n-inits thins the clips (B-5 used 10 texts x 8 clips). The open-loop part (1) stays complete,
    # keeping the same yardstick as B-2 / B-2b. Thinning is evenly spaced, and **the chosen episode_index values are always recorded**
    # (so B-2's per_init can be re-aggregated on the same 8 clips for comparison).
    import pyarrow.parquet as pq
    roll_eps = held_eps
    if a.n_inits and a.n_inits < len(held_eps):
        idx = np.linspace(0, len(held_eps) - 1, a.n_inits).round().astype(int)
        roll_eps = [held_eps[i] for i in idx]
    res["rollout_init_eps"] = [e["episode_index"] for e in roll_eps]
    print(f"[rollout] 初期姿勢 {len(roll_eps)}/{len(held_eps)} 本: {res['rollout_init_eps']}", flush=True)
    info = json.loads((pathlib.Path(a.heldout) / "meta/info.json").read_text())
    inits = []
    for e in roll_eps:
        i = e["episode_index"]
        f = pathlib.Path(a.heldout) / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
        col = pq.read_table(f, columns=["observation.state"]).column(0).combine_chunks()
        inits.append(col.values.to_numpy().reshape(len(col), -1)[0])
    inits = np.stack(inits).astype(np.float32)
    scene = Scene(a.xml)
    res["rollout"] = {}
    all_traj = {}
    for text in a.texts:
        torch.manual_seed(a.seed)
        t1 = time.perf_counter()
        traj, lo, _ = rollout(policy, scene, inits, text, a.frames, horizon, lay, a.blind,
                              samples=a.samples, seed=a.seed)
        rm = [root_metrics(t) for t in traj]
        summ = {k: dict(median=float(np.median([r[k] for r in rm])),
                        mean=float(np.mean([r[k] for r in rm])),
                        min=float(np.min([r[k] for r in rm])),
                        max=float(np.max([r[k] for r in rm]))) for k in rm[0]}
        # On the teacher data's 36 "walk... forward" clips this is 0.75, wave 0.11, squat 0.00 (ws/b2/reference_root.json)
        summ["frac_forward_3s_gt_0.3m"] = float(np.mean([r["forward_3s"] > 0.3 for r in rm]))
        res["rollout"][text] = dict(summary=summ, per_init=[dict(init_ep=e["episode_index"], **r)
                                                            for e, r in zip(roll_eps, rm)],
                                    sec=time.perf_counter() - t1)
        all_traj[text] = traj
        print(f"[rollout] {text!r}: forward_3s median {summ['forward_3s']['median']:+.2f} m "
              f"(>0.3m {summ['frac_forward_3s_gt_0.3m']:.2f})  forward median {summ['forward']['median']:+.2f} m "
              f"(min {summ['forward']['min']:+.2f}, max {summ['forward']['max']:+.2f})  "
              f"disp median {summ['disp']['median']:.2f} m  path median {summ['path']['median']:.2f} m  "
              f"[{time.perf_counter() - t1:.0f}s]", flush=True)
        slug = "_".join(text.split()[2:5])
        vids = [scene.render(t, l) for t, l in zip(traj, lo)]
        for e, v in zip(roll_eps, vids):
            write_mp4(out / "mp4" / slug / f"init_{e['episode_index']:03d}.mp4", v)
        write_mp4(out / f"tile_{slug}.mp4", tile(vids[:6], 3))
    np.savez_compressed(out / "rollouts.npz", init_eps=np.array([e["episode_index"] for e in roll_eps]),
                        **{"_".join(t.split()[2:5]): v for t, v in all_traj.items()})
    (out / "metrics.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"EVAL_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
