"""V-12: apply a goal (waypoints) at inference time by replacing the flow sampling of GR00T's action head (no training).

    import groot_guidance as gg
    gg.install(policy, ckpt_dir, xml, mode="fk", fk_eta=0.7)          # guidance by forward kinematics
    gg.CTX.update(active=..., side=..., path=..., horizon=16, overlap=4, prev_norm=...)   # every step, before policy.get_action

V-11's arm W was trained with waypoints in the state, but its sensitivity per chunk was a weak 0.02 (closed-loop H-4: +0.12 with waypoints / -0.03 without).
Instead of making the model use them through training, apply them at generation time:

- **cfg (classifier-free guidance):** arm W was trained with 25% of frames "without waypoints", so the same model can output velocities both "with" and "without".
  Every step v = v_without + w (v_with - v_without). Only usable with models that have waypoints in the state. In V-12 stage 2 it had almost no effect on goals on the far side of the body
- **fk (guidance by forward kinematics):** every step, estimate the finished action x_hat = x + (1 - t) v from the current estimate, and compute the grasping hand's position at each frame in `KEYS`
  (seen from the pelvis at that time; same definition as `waypoint_labels.py`) by MuJoCo forward kinematics. The difference from the planner's path (CTX["path"], hand position per frame) is reduced
  by a fraction `fk_eta` using one damped Gauss-Newton step on the grasping arm's dims (dims constant in the training data excluded).
  Corrections between keyframes are interpolated; before the first keyframe they ramp up from 0 (right after the overlapped frames); after the last keyframe the same correction is held.
  v is moved by delta / (1 - t) so that x_hat moves by delta. After the last step, the correction is applied `final_iters` times directly to the finished action (corrections in the middle of the
  steps get pulled back by the model: in stage 2, W_fk10 only got from 27 to 7 cm, and even the fake model only from 18 to 6.5 cm). Also usable with models without waypoints (arm B etc.).
  The first version in stage 2 pushed hard on a single keyframe (the 14th), and the motion jumped at chunk seams (seen by eye)
- **both:** both. **none:** no guidance (when only chunk overlap is used)
- **Chunk overlap (RTC; the same inpainting as GR00T's original implementation):** if CTX["overlap"] = k > 0, the first k frames of the current chunk are fixed (velocity 0) to the last k unused frames
  of the previous chunk (normalized, CTX["prev_norm"]). root_translation (0-2) is relative to the current pose and its reference changes with every chunk,
  so it is not fixed. groot_eval's rollout advances only horizon - k frames per inference

Normalization (q01-q99 -> [-1, 1]) is read from the checkpoint's statistics.json. Action frame j = the state j + 1 frames after the observation (`cbd_to_lerobot.py`).
"""
import json
import pathlib
import types

import numpy as np

KEYS = (3, 7, 11, 14)        # Action frames that guidance acts on (14 = 15 frames after the observation = the waypoint definition)
ARM = {"left": slice(33, 57), "right": slice(57, 81)}
WP = slice(117, 124)
ROOT = 3                     # 0-2 are root_translation (relative). Not fixed by chunk overlap
CTX = {}                     # Set by groot_eval every step: active (B,) bool, side [B], path (B,16,3), horizon, overlap, prev_norm (B,16,117) or None
STATS = {}                   # Hand-to-goal distance before and after guidance (mean over keyframes)


class HandFK:
    """117-d state -> hand position seen from the pelvis at that time (same definition as goal_labels.kinematics / waypoint_labels). MuJoCo."""

    def __init__(self, xml):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        self.ids = {n: self.m.body(n).id for n in ("pelvis", "left_upper_leg", "right_upper_leg",
                                                    "left_hand", "left_middle", "right_hand", "right_middle")}

    def local(self, states, side):
        """states (N,117) → (N,3)。"""
        from humanml3d_to_video import state_to_qpos
        q = state_to_qpos(np.asarray(states, np.float64))
        out = np.zeros((len(q), 3))
        up = np.array([0.0, 0, 1])
        for n in range(len(q)):
            self.d.qpos[:] = q[n]
            self.mj.mj_kinematics(self.m, self.d)
            x = self.d.xpos
            left = x[self.ids["left_upper_leg"]] - x[self.ids["right_upper_leg"]]
            left[2] = 0
            left /= np.linalg.norm(left) + 1e-9
            basis = np.stack([left, np.cross(left, up), up])
            hand = (x[self.ids[f"{side}_hand"]] + x[self.ids[f"{side}_middle"]]) / 2
            out[n] = basis @ (hand - x[self.ids["pelvis"]])
        return out

    def jacobian(self, state, side, eps=1e-3):
        """Hand position (3,) and its numerical derivative (3,24) with respect to the grasping arm's 24 dims."""
        sl = ARM[side]
        S = np.repeat(np.asarray(state, np.float64)[None], 1 + (sl.stop - sl.start), 0)
        for k in range(sl.stop - sl.start):
            S[1 + k, sl.start + k] += eps
        h = self.local(S, side)
        return h[0], (h[1:] - h[0]).T / eps


def _load_norm(ckpt):
    """Checkpoint statistics.json -> action q01 / q99 (laid out over the 117 dims), and the normalized values for "no waypoint"."""
    s = json.loads((pathlib.Path(ckpt) / "statistics.json").read_text())["new_embodiment"]
    groups = ["root_translation", "hips", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
    a_lo = np.concatenate([s["action"][g]["q01"] for g in groups])
    a_hi = np.concatenate([s["action"][g]["q99"] for g in groups])
    out = dict(a_lo=a_lo, a_hi=a_hi)
    if "waypoint" in s["state"]:
        lo, hi = np.array(s["state"]["waypoint"]["q01"]), np.array(s["state"]["waypoint"]["q99"])
        out["wp_off"] = np.clip(2 * (0 - lo) / np.maximum(hi - lo, 1e-8) - 1, -1, 1)   # normalized values for "no waypoint" (all zeros)
    return out


def install(policy, ckpt, xml, mode="fk", cfg_w=3.0, fk_eta=0.7, fk_damp=1e-4, final_iters=2):
    """Replace policy.model.action_head.get_action_with_features. mode: none / cfg / fk / both."""
    import torch
    head = policy.model.action_head
    norm = _load_norm(ckpt)
    if mode in ("cfg", "both"):
        assert "wp_off" in norm, "cfg は経由点を state に持つモデル（arm W）でだけ使える"
    fk = HandFK(xml) if mode in ("fk", "both") else None
    a_lo, a_hi = norm["a_lo"], norm["a_hi"]
    half = (a_hi - a_lo) / 2                                   # real value per unit of normalization
    inv_half = np.where(half > 1e-6, 1.0 / np.maximum(half, 1e-6), 0.0)   # dims constant in the training data (width 0) are not moved (division by 0 gave NaN)

    def raw(x117):
        return a_lo + (np.asarray(x117, np.float64) + 1) * half   # no clip (do not cut the guidance gradient)

    def gn_step(state117, side, target):
        """Arm correction (normalized) that moves one frame's hand toward target, and the error before correction. None if forward kinematics breaks."""
        sl = ARM[side]
        h, J = fk.jacobian(state117, side)
        e = h - np.asarray(target)
        if not (np.isfinite(J).all() and np.isfinite(e).all()):
            return None, None
        J = J * (inv_half[sl] > 0)[None]                      # dims that are not moved are also dropped from the Jacobian
        d_raw = -fk_eta * J.T @ np.linalg.solve(J @ J.T + fk_damp * np.eye(3), e)   # damped Gauss-Newton
        return d_raw * inv_half[sl], float(np.linalg.norm(e))

    def correction(xn, b, ov, H, keys, gain=1.0):
        """Arm correction for one batch element (normalized, (H, 24)) and the mean keyframe error. Keyframe corrections are joined, starting from 0 right after
        the overlapped frames, by cubic Hermite interpolation (Catmull-Rom tangents). Linear joins made the hand speed step at keyframes, and a per-segment smoothstep
        dropped the speed to 0 at keyframes, giving "stop and go" (checked with the fake model)."""
        side = CTX["side"][b]
        corr, errs = {}, []
        for k in keys:
            d, e = gn_step(raw(xn[k, :117]), side, CTX["path"][b][k])
            if d is not None:
                corr[k] = d * gain
                errs.append(e)
        if not corr:
            return None, None
        ks = sorted(corr)
        xs = np.array([ov - 1] + ks, float)
        ys = np.stack([np.zeros_like(corr[ks[0]])] + [corr[k] for k in ks])
        m = np.zeros_like(ys)                               # tangents: 0 at both ends, from neighboring points inside (Catmull-Rom)
        for i in range(1, len(xs) - 1):
            m[i] = (ys[i + 1] - ys[i - 1]) / (xs[i + 1] - xs[i - 1])
        out = np.zeros((H, ys.shape[1]))
        for j in range(ov, H):
            if j >= ks[-1]:
                out[j] = ys[-1]
                continue
            i = int(np.searchsorted(xs, j, side="right") - 1)
            h = xs[i + 1] - xs[i]
            u = (j - xs[i]) / h
            h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
            out[j] = h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]
        return out, float(np.mean(errs))

    @torch.no_grad()
    def get_action_with_features(self, backbone_features, state_features, embodiment_id, backbone_output, action_input, options=None):
        from transformers.feature_extraction_utils import BatchFeature
        vl = backbone_features
        B = vl.shape[0]
        dev = vl.device
        actions = torch.randn(size=(B, self.config.action_horizon, self.action_dim), dtype=vl.dtype, device=dev)
        dt = 1.0 / self.num_inference_timesteps
        H = int(CTX.get("horizon", 16))
        ov = int(CTX.get("overlap", 0))
        prev = CTX.get("prev_norm")
        vel_strength = torch.ones_like(actions)
        if ov > 0 and prev is not None:                         # chunk overlap: fix the start to the last ov unused frames of the previous chunk
            p = torch.as_tensor(np.asarray(prev)[:, H - ov:H, ROOT:117], dtype=actions.dtype, device=dev)
            actions[:, :ov, ROOT:117] = p
            vel_strength[:, :ov, ROOT:117] = 0.0
        else:
            ov = 0
        active = np.asarray(CTX.get("active", np.zeros(B, bool)), bool)
        use_cfg = mode in ("cfg", "both") and active.any()
        use_fk = mode in ("fk", "both") and active.any()
        keys = [k for k in KEYS if k >= ov]
        if use_cfg:
            st = action_input.state.view(B, 1, -1).clone()
            st[:, :, WP] = torch.as_tensor(norm["wp_off"], dtype=st.dtype, device=dev)
            sf = torch.cat([state_features, self.state_encoder(st, embodiment_id)], 0)
            vl2 = torch.cat([vl, vl], 0)
            emb2 = torch.cat([embodiment_id, embodiment_id], 0)
            extra = {}
            if self.config.use_alternate_vl_dit:
                extra = dict(image_mask=torch.cat([backbone_output.image_mask] * 2, 0),
                             backbone_attention_mask=torch.cat([backbone_output.backbone_attention_mask] * 2, 0))
        err_before, err_after = [], []
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            tb = int(t_cont * self.num_timestep_buckets)
            if use_cfg:
                ts = torch.full((2 * B,), tb, device=dev)
                af = self.action_encoder(torch.cat([actions, actions], 0), ts, emb2)
            else:
                ts = torch.full((B,), tb, device=dev)
                af = self.action_encoder(actions, ts, embodiment_id)
            if self.config.add_pos_embed:
                af = af + self.position_embedding(torch.arange(af.shape[1], dtype=torch.long, device=dev)).unsqueeze(0)
            if use_cfg:
                sa = torch.cat((sf, af), 1)
                if self.config.use_alternate_vl_dit:
                    out = self.model(hidden_states=sa, encoder_hidden_states=vl2, timestep=ts, **extra)
                else:
                    out = self.model(hidden_states=sa, encoder_hidden_states=vl2, timestep=ts)
                v2 = self.action_decoder(out, emb2)[:, -self.action_horizon:]
                v_on, v_off = v2[:B], v2[B:]
                w = torch.as_tensor(np.where(active, cfg_w, 1.0), dtype=v_on.dtype, device=dev)[:, None, None]
                v = v_off + w * (v_on - v_off)
            else:
                sa = torch.cat((state_features, af), 1)
                if self.config.use_alternate_vl_dit:
                    out = self.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts,
                                     image_mask=backbone_output.image_mask, backbone_attention_mask=backbone_output.backbone_attention_mask)
                else:
                    out = self.model(hidden_states=sa, encoder_hidden_states=vl, timestep=ts)
                v = self.action_decoder(out, embodiment_id)[:, -self.action_horizon:]
            if use_fk and keys:
                rem = 1.0 - t_cont
                x1 = (actions + rem * v * vel_strength).float().cpu().numpy()   # estimate of the finished action (normalized)
                dv = np.zeros_like(x1)
                for b in np.nonzero(active)[0]:
                    c, e = correction(x1[b], b, ov, H, keys)
                    if c is None:
                        continue
                    if t == 0:
                        err_before.append(e)
                    dv[b, :H, ARM[CTX["side"][b]]] = c / rem
                v = v + torch.as_tensor(dv, dtype=v.dtype, device=dev)
            actions = actions + dt * v * vel_strength
        x1 = actions.float().cpu().numpy()
        if use_fk and keys:
            for b in np.nonzero(active)[0]:                     # finishing pass: applied directly to the finished action (the model can no longer pull it back)
                for _ in range(final_iters):
                    c, _e = correction(x1[b], b, ov, H, keys, gain=1.0 / fk_eta)
                    if c is None:
                        break
                    x1[b, :H, ARM[CTX["side"][b]]] += c
            actions = torch.as_tensor(x1, dtype=actions.dtype, device=dev)
            for b in np.nonzero(active)[0]:
                side = CTX["side"][b]
                h = fk.local(raw(x1[b, keys, :117]), side)
                err_after.append(float(np.mean(np.linalg.norm(h - np.asarray(CTX["path"][b])[keys], axis=1))))
        CTX["last_norm"] = x1[:, :H, :117]                       # rollout uses the mean over samples for the next chunk's overlap
        STATS.update(err_before=err_before, err_after=err_after)
        STATS.setdefault("all_before", []).extend(err_before)
        STATS.setdefault("all_after", []).extend(err_after)
        return BatchFeature(data={"action_pred": actions, "backbone_features": vl, "state_features": state_features})

    head.get_action_with_features = types.MethodType(get_action_with_features, head)
    print(f"[guidance] mode={mode} cfg_w={cfg_w} fk_eta={fk_eta}（キーフレーム {KEYS} の手を計画器の経路へ）", flush=True)
    return head
