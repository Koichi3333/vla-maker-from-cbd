#!/usr/bin/env python3
"""End-to-end test without GR00T or a GPU (CPU, 0 CU). Checks inference-time guidance (groot_guidance.py) and the closed-loop evaluation (groot_eval.object_eval).

    MUJOCO_GL=osmesa python3 tests/smoke_test.py head          # one chunk: does the hand move onto the planner's path; chunk overlap at the seam
    MUJOCO_GL=osmesa python3 tests/smoke_test.py rollout       # closed loop, 2 clips x 60 frames (holding the object, mp4, traj_*.npz)
    MUJOCO_GL=osmesa python3 tests/smoke_test.py rollout vlm   # goal = the VLM's point (vlm_goal in data/eval/object_eval_vlm_box.json)

Instead of GR00T's action head, plug in a fake head that returns a "hold the current pose" flow (v = (A - x) / (1 - t)).
No weights needed. Needs: torch (CPU build), transformers, mujoco, pyarrow, imageio (requirements.txt).
Output goes to work/smoke/.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VLA = ROOT / "vla"
STATS = str(ROOT / "data/stats")                     # statistics.json of checkpoint-47000 (q01 / q99 of the actions)
XML = str(VLA / "humanoid.xml")
DATA = ROOT / "data/video_reach"
EVAL = ROOT / "data/eval"


def test_head():
    import sys, types, numpy as np, torch, pyarrow.parquet as pq
    sys.path.insert(0, str(VLA))
    import groot_guidance as gg
    from waypoint_labels import plan
    norm = gg._load_norm(STATS); lo_, hi_ = norm["a_lo"], norm["a_hi"]; half = (hi_-lo_)/2
    st = np.stack(pq.read_table(str(DATA / "data/chunk-000/episode_000003.parquet")).column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float64)[20,:117]
    A = torch.zeros(1, 40, 132); A[0, :, :117] = torch.tensor(np.clip((st - lo_)/np.maximum(half,1e-8) - 1, -1, 1), dtype=torch.float32)
    class Cfg: action_horizon=40; use_alternate_vl_dit=False; add_pos_embed=False
    class Head:
        config=Cfg(); action_dim=132; num_inference_timesteps=4; num_timestep_buckets=1000; action_horizon=40
        def action_encoder(self, x, ts, emb): return x
        def model(self, hidden_states, encoder_hidden_states, timestep):
            t = timestep.float()[:, None, None] / 1000.0
            x = hidden_states[:, 1:]
            return (A.expand_as(x) - x) / (1 - t)
        def action_decoder(self, out, emb): return out
        def state_encoder(self, s, e): return torch.zeros(s.shape[0], 1, 132)
    head = Head()
    pol = types.SimpleNamespace(model=types.SimpleNamespace(action_head=head))
    gg.install(pol, STATS, XML, mode="fk", fk_eta=0.7)
    fk = gg.HandFK(XML)
    side = "left"; now = fk.local(st[None], side)[0]; tgt = now + np.array([0.15, 0.10, 0.12])
    path = np.stack([now + plan(now, now, tgt, t=k) for k in range(1, 17)])
    torch.manual_seed(0)
    def run(ov, prev):
        gg.CTX.update(active=np.array([True]), side=[side], path=path[None], horizon=16, overlap=ov, prev_norm=prev)
        out = head.get_action_with_features(torch.zeros(1, 5, 8), torch.zeros(1, 1, 132), torch.zeros(1, dtype=torch.long), None,
                                            types.SimpleNamespace(state=torch.zeros(1, 1, 132)))
        x = out["action_pred"][0, :16, :117].float().numpy()
        return x, fk.local(lo_ + (x + 1) * half, side)
    x, h = run(0, None)
    print("誘導前→後（キーフレーム平均）", np.round(gg.STATS["err_before"], 3), "→", np.round(gg.STATS["err_after"], 3), "m")
    print("各フレームの手と経路の距離 cm:", np.round(np.linalg.norm(h - path, axis=1) * 100, 1))
    print("手の動き（フレーム間 cm）  :", np.round(np.linalg.norm(np.diff(np.vstack([now, h]), axis=0), axis=1) * 100, 1))
    print("経路の動き（フレーム間 cm）:", np.round(np.linalg.norm(np.diff(np.vstack([now, path]), axis=0), axis=1) * 100, 1))
    prev = gg.CTX["last_norm"]
    x2, h2 = run(4, prev)
    print("重ね 4: 先頭 4 フレームが前のチャンクの 12〜15 と一致:", np.allclose(x2[:4, 3:], prev[0, 12:16, 3:], atol=1e-5))


def test_rollout():
    import sys, types, json, pathlib, numpy as np, torch
    sys.path.insert(0, str(VLA))
    import groot_eval as ge, groot_guidance as gg
    norm = gg._load_norm(STATS); lo_, hi_ = norm["a_lo"], norm["a_hi"]; half = (hi_-lo_)/2
    GROUPS = ["root_translation", "hips", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
    lay = ge.layout(str(DATA))
    class Cfg: action_horizon=40; use_alternate_vl_dit=False; add_pos_embed=False
    class Head:
        config=Cfg(); action_dim=132; num_inference_timesteps=4; num_timestep_buckets=1000; action_horizon=40
        A = None
        def action_encoder(self, x, ts, emb): return x
        def model(self, hidden_states, encoder_hidden_states, timestep):
            t = timestep.float()[:, None, None] / 1000.0
            x = hidden_states[:, 1:]
            return (self.A - x) / (1 - t)
        def action_decoder(self, out, emb): return out
        def state_encoder(self, s, e): return torch.zeros(s.shape[0], 1, 132)
    head = Head()
    class MC:
        def __init__(s, keys, d): s.modality_keys, s.delta_indices = keys, d
    class Policy:
        model = types.SimpleNamespace(action_head=head)
        def get_modality_config(self):
            return {"state": MC(GROUPS, [0]), "action": MC(GROUPS, list(range(16)))}
        def get_action(self, obs):
            st = np.concatenate([obs["state"][g][:, -1] for g in GROUPS], -1)      # (B,117)
            B = len(st)
            A = torch.zeros(B, 40, 132); A[:, :, :117] = torch.tensor(np.clip((st - lo_) / np.maximum(half, 1e-8) - 1, -1, 1), dtype=torch.float32)[:, None]
            A[:, :, :3] = 0
            head.A = A
            out = head.get_action_with_features(torch.zeros(B, 5, 8), torch.zeros(B, 1, 132), torch.zeros(B, dtype=torch.long), None,
                                                types.SimpleNamespace(state=torch.zeros(B, 1, 132)))
            x = out["action_pred"][:, :16, :117].float().numpy()
            raw = lo_ + (x + 1) * half
            raw[:, :, :3] = st[:, None, :3]                                     # root unchanged (fake)
            return {g: raw[:, :, lay[g][0]:lay[g][1]] for g in GROUPS}, {}
    pol = Policy()
    gg.install(pol, STATS, XML, mode="fk", fk_eta=0.7)
    goal_key = None
    if len(sys.argv) > 2 and sys.argv[2] == "vlm":   # V-10: the VLM's point as the goal (2243 as is, 2239 a missed point, 2243 mirror set to None to test no goal)
        spec = json.load(open(EVAL / "object_eval_vlm_box.json"))
        spec["items"] = [it for it in spec["items"] if it["clip"][-4:] in ("2243", "2239")]
        spec["items"][0]["vlm_goal"]["mirror"] = None
        goal_key = "vlm_goal"
    elif len(sys.argv) > 2:                   # Stage 3: train-side spec (2 items picked by slug) and output folder
        spec = json.load(open(sys.argv[2]))
        spec["items"] = [it for it in spec["items"] if it["slug"] in ("ep02", "ep02m")]
    else:
        spec = json.load(open(EVAL / "object_eval_heldout.json"))
        spec["items"] = [it for it in spec["items"] if it["clip"][-4:] in ("2247", "2243")]
    for it in spec["items"]:
        if "h4_offsets" in it: it["h4_offsets"] = it["h4_offsets"][:2]
    out = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else str(ROOT / ("work/smoke/vlm" if goal_key else "work/smoke/object")))
    ge.object_eval(pol, spec, XML, 60, 16, lay, out, samples=2, seed=0, reps=1, reps_h4=1,
                   goal_mode="object", spec_hand=True, attach=True, overlap=4, reach=0.52, goal_key=goal_key)
    d = json.load(open(out / "object_eval.json"))
    for r in d["rows"]: print(r["clip"][-4:], r["cond"], r["side"], "最小距離", round(r["min_dist_m"], 3), "持った", r["attach_sec"], "加速度 継ぎ目/他", round(r["acc_seam"], 2), round(r["acc_inner"], 2))
    print(sorted(p.name for p in out.rglob("*") if p.is_file())[:12])
    for f in sorted(out.glob("traj_*.npz")):
        z = np.load(f); print(f.name, {k: z[k].shape for k in z.files}, "attach", z["attach"].tolist())


if __name__ == "__main__":
    {"head": test_head, "rollout": test_rollout}[sys.argv[1] if len(sys.argv) > 1 else "head"]()
