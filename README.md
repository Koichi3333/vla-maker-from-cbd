<div align="center">

# vla-maker-from-cbd

**Build an AI model that moves a humanoid (a VLA) from human motion data (CBD).**

[日本語](README.ja.md) · [Results](docs/results.md) · [Contributing](CONTRIBUTING.md) · [Common Behavior Data](https://github.com/Koichi3333/common-behavior-data)

<img src="docs/media/hero_video_to_vla.gif" width="544" alt="Left: a person on a chair reaches for a glass on a table and drinks, with the tracked skeleton overlaid. Right: the simulated humanoid driven by the VLA reaches for a magenta glass and brings it to its mouth">

*Left: an ordinary video turned into body-motion data (CBD).<br>
Right: the VLA trained on CBD, told "a seated person picks up a glass from the table, drinks from it, and puts it back". It reaches for the glass (magenta) and brings it to its mouth.*

<sub>The left is a sample from [Common Behavior Data](https://github.com/Koichi3333/common-behavior-data); that person is not in the training data.
The right is a kinematic MuJoCo simulation (poses played back in sequence, no forces computed). There are no fingers; the glass attaches to the hand once the hand reaches it.
Until the hand reaches the glass, an outside planner shows it the way (guidance); the lift to the mouth comes from the model itself.
This is one of 12 trials of this scene; in the other 11 the glass stops at chin height or below.
Small jitter in the motion is smoothed afterwards with [`vla/smooth_motion.py`](vla/smooth_motion.py).</sub>

</div>

---

A **VLA** (vision-language-action model) takes a camera image and a text instruction and outputs body motion.
This repository builds its training data from **Common Behavior Data (CBD)**, an open format that records human motion
as bone rotations, independent of any particular body (a person, a robot or an avatar).

The whole thing is three steps:

1. **Turn human video or mocap into CBD** (the generators in [common-behavior-data](https://github.com/Koichi3333/common-behavior-data))
2. **Turn CBD into training data and fine-tune NVIDIA GR00T N1.7** (this repository)
3. **Turn the model's motion back into CBD and retarget it to a robot or avatar** (CBD's adapters)

The model does not output a robot's joint angles. It outputs **CBD bone rotations directly** (19 bones plus the hip position,
117 numbers per frame). So putting the motion on a MuJoCo humanoid, a Unitree G1 or a VRM avatar
**needs no retraining for each body**.

> **Status: research preview, looking for collaborators.** Everything runs end to end, and every number below is measured.
> Several hard problems are still open, and they are listed [below](#help-wanted) on purpose.

## Help wanted

This is a one-person project so far. These are the places where your work would move it the most.
Each one can be taken on by itself. Comment on (or open) an issue to claim one; see [CONTRIBUTING.md](CONTRIBUTING.md).

| | Problem | Why it matters | Needs a GPU? |
| --- | --- | --- | --- |
| 🔴 | [#1](https://github.com/Koichi3333/vla-maker-from-cbd/issues/1) **Make the VLA reach objects on its own.** Today the hand reaches the object precisely because an outside planner shows it the way while the model runs (inference-time guidance). Without that, the model still turns the hand toward the object, but the hand reaches it only about half as often (0.93 → 0.51) | It becomes a real VLA that works without outside help | Yes (A100, ~30 CU per try) |
| 🔴 | [#2](https://github.com/Koichi3333/vla-maker-from-cbd/issues/2) **Record scenes with objects and contribute them as CBD.** Film seated or standing object handling from a fixed camera and convert it with the CBD video generator | Data is what is missing most, and only data you own leads to a model that can be used commercially | No |
| 🟠 | [#3](https://github.com/Koichi3333/vla-maker-from-cbd/issues/3) **Beat "just hold the current pose" at predicting video motion (H-3).** When asked to continue clips made from video, the best error so far is 5.53°, still worse than the 5.05° of simply holding the current pose | It is the test that cannot be gamed: does the model actually predict human motion? | Yes (~1 CU) |
| 🟠 | [#4](https://github.com/Koichi3333/vla-maker-from-cbd/issues/4) **Run on GPUs other than Colab.** Port `colab/vm_*.sh` to a local GPU, RunPod, Lambda or Modal | Today every run needs a browser click to connect Google Drive | Your own |
| 🟡 | [#5](https://github.com/Koichi3333/vla-maker-from-cbd/issues/5) **Check with physics.** Replay the generated G1 motion in a simulator that computes contact and gravity (Isaac, MuJoCo with contacts) and report what breaks | Everything so far is kinematic (poses in sequence, no forces) | Optional |
| 🟡 | [#6](https://github.com/Koichi3333/vla-maker-from-cbd/issues/6) **Add fingers.** CBD has no finger bones yet, so "holding" is a check of whether the hand is close to the object | Real grasping needs fingers | No |
| 🟢 | [#7](https://github.com/Koichi3333/vla-maker-from-cbd/issues/7) **Translate log messages.** Code comments are in English, but many printed messages and command-line help texts are still in Japanese | Lowers the barrier for everyone else | No |
| 🟢 | [#8](https://github.com/Koichi3333/vla-maker-from-cbd/issues/8) **Put the CPU smoke test in CI (automated tests)**, packaging, type hints | Keeps the pipeline from quietly breaking | No |

New to the project? Run the [CPU smoke test](#1-cpu-smoke-test-no-gpu-no-weights) first, then pick a 🟢 item.

## How it fits together

```
 your video ─┐                         (common-behavior-data)
 mocap ──────┼──► CBD clip ──────────────────────────────────────────────┐
             │    19 bone rotations, root, objects, captions             │
             │                                                           ▼
             │    vla/cbd_to_lerobot.py · vla/video_to_lerobot.py   CBD adapters
             │      117-d state/action (6D rotations) + caption     MuJoCo · Isaac G1 · VRM
             │    vla/humanml3d_to_video.py · render_with_objects.py     ▲
             │      256 px MuJoCo render of the body (+ object proxies)  │
             ▼                                                           │
        LeRobot dataset ──► GR00T N1.7 fine-tune (action head) ──► 117-d actions
                                     ▲                                   │
              goal ─► planner ─► inference-time guidance (vla/groot_guidance.py)
              ▲                                              vla/state_to_cbd.py
   VLM points at the object in the same image (vla/vlm_target.py, vla/vlm_goal_depth.py)
```

- **The model outputs CBD skeleton motion, not robot motion.** Retargeting to a robot happens after the model,
  in CBD's adapters. Swapping the robot means no new data and no retraining.
- **The model sees a re-rendered frame, not the original video.** It is a neutral MuJoCo body in the same pose,
  plus simple stand-ins for objects (cylinders, boxes). People's faces and rooms never enter the training data.
- **The VLM picks the object; the VLA moves the body.** A VLM (a language model that can read images; here Gemini Robotics-ER)
  points at the requested object in the frame. That point becomes a 3D goal, a planner draws a path for the hand, and while the
  VLA generates motion, the arm is pulled toward that path (inference-time guidance).

## What works today

**In short: with guidance, the hand reaches the object about 9 times in 10, wherever the object is placed. Without guidance (the model alone), the hand turns toward the object but reaches it only about half the time.**

We took 7 scenes not used in training, placed the object in 3 positions (original / shifted 25 cm / mirrored to the other side),
and ran each 4 times. Measured in a kinematic MuJoCo simulation.

| | Guided, goal = true position | Guided, goal = position the VLM picked | Unguided (model alone) | Pass bar |
| --- | ---: | ---: | ---: | ---: |
| Hand came within 10 cm of the object: original / shifted / mirrored | 1.00 / 1.00 / 0.89 | **1.00 / 1.00 / 0.89** | 0.86 / 0.86 / 0.61 | |
| Hand moved toward the object shown: shifted / mirrored | 1.00 / 1.00 | 1.00 / 1.00 | 0.96 / 0.71 | ≥ 0.7 |
| Hand reached the object ("held") | 0.94 | 0.93 | 0.51 | |
| When the object moves, does the hand's endpoint follow? (H-4 slope; closer to 1 = follows more closely) | +0.613 | – | +0.577 | ≥ 0.5 |
| Error predicting the original motion (HumanML3D): does mixing in video hurt it? (H-1; lower is better) | 9.55° | same weights | 9.32° | ≤ 9.65° |

<sub>"Guided" is `checkpoint-47000` with the arm H action head, run with guidance strength fk 0.3. This repository calls it the "usable form".
"Unguided" is arm J (an action head trained for 30,000 steps) run without guidance.</sub>

- The VLM picked the requested object in **all 378** questions, including scenes with a look-alike distractor.
- Mixing video-derived CBD into up to about half of the training data does **not** make the original motion prediction worse (H-1).
- Generated walking carries over to a Unitree G1 through the CBD adapter, with a smaller median direction error than the motions it learned from (0.96° vs 1.54°).

**What we cannot claim yet:**

- Success in a physics simulator or on a real robot (everything so far is kinematic)
- Grasping with fingers ("held" means the hand came within reach and the object was attached to it)
- That the VLA alone, without guidance, reaches objects precisely
- Beating "just hold the current pose" at predicting video motion (H-3)

Every number is in [docs/results.md](docs/results.md) (English summary) and [docs/results.ja.md](docs/results.ja.md) (full record, Japanese).

<img src="docs/media/reach_three_positions.gif" width="720" alt="The same scene with a magenta glass at three positions (original, shifted 25 cm, mirrored to the other side); the hand goes to the glass each time, and the glass moves with the hand once it is reached">

*Same instruction and same starting pose; only the position of the glass (magenta) changes. Each time, the hand goes where the glass is, and once it is reached the glass moves with the hand.*

<sub>Kinematic MuJoCo simulation, guided. The reach target is the point a VLM picked from the image (not drawn).
Smoothed afterwards with [`vla/smooth_motion.py`](vla/smooth_motion.py), as above.</sub>

<img src="docs/media/g1_walk.gif" width="360" alt="Generated 'a person walks forward' retargeted to a Unitree G1"> <img src="docs/media/gen_crawl.gif" width="360" alt="Generated 'a person crawls on the floor'">

*Left: motion generated from "a person walks forward", put on a G1 (kinematic). Right: "a person crawls on the floor".<br>
The right sentence is not in the training data, and only 16 training clips show crawling at all.*

## Quick start

### 1. CPU smoke test (no GPU, no weights)

It swaps the real model for a fake one that just holds the current pose, and checks that guidance, evaluation,
attaching objects to the hand and video output all run. It takes a few minutes on a laptop.

Rendering uses OSMesa (software OpenGL), so install it first (`sudo apt-get install libosmesa6` on Debian/Ubuntu).

```bash
pip install -r requirements.txt
MUJOCO_GL=osmesa python3 tests/smoke_test.py head          # guidance pulls the hand onto the planned path (17.8 cm → 0)
MUJOCO_GL=osmesa python3 tests/smoke_test.py rollout       # 2 scenes × 60 frames, writes mp4 + trajectories to work/smoke/
MUJOCO_GL=osmesa python3 tests/smoke_test.py rollout vlm   # same, with goals from the VLM
```

### 2. Run the usable form on Colab

You need four things:

- A Colab plan with A100 access
- The [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) (pin `jupyter-kernel-client<1.0`)
- A Hugging Face token with access to `nvidia/GR00T-N1.7-3B`
- The weights in your Google Drive (see [Weights](#weights))

```bash
bash colab/start.sh infer myrun        # creates an A100 session, uploads code, stops at the Drive consent URL
# open the URL in /tmp/dm_url.txt, approve
bash colab/go.sh infer myrun           # ~20 min per goal file, ~1.4 CU
colab download -s myrun /content/run/results.tgz work/myrun/results.tgz
colab stop -s myrun                    # always stop it
```

[docs/colab.md](docs/colab.md) covers the Colab traps. The main three: tokens expire after 60 minutes, the GPU allocation
disappears after about 25 minutes without traffic, and copies to Drive report success before the upload finishes.
The scripts handle all three; the doc explains why.

### 3. Continue training

This is how arm H (the action head of the usable form) was made: start from `checkpoint-47000`, mix object-bearing CBD into
HumanML3D (mocap motion data) (`DUP=40` ≈ 50% of frames), and train the action head (the part that outputs motion).

```bash
bash colab/start.sh train t1                           # VIDEO_DATA=<your LeRobot dataset> to use your own clips
STEPS=5000 DUP=40 RUN=t1 bash colab/go.sh train t1     # ~1.5 h, ~9 CU; weights land in Drive runs/t1/
ACTION_HEAD=runs/t1/action_head.safetensors bash colab/go.sh infer ...   # evaluate your head in the usable form
```

### 4. Bring your own CBD

1. Record a clip and convert it with the CBD video generator in
   [common-behavior-data](https://github.com/Koichi3333/common-behavior-data).
2. Run `python3 vla/video_to_lerobot.py --cbd-dir <your CBD dir> --out work/my_video`. It writes the 117-d
   LeRobot dataset plus per-frame object states.
3. Train with `VIDEO_DATA=work/my_video bash colab/start.sh train ...`.

Mocap or any other CBD source goes through `vla/cbd_to_lerobot.py` the same way.

## Repository layout

| Path | What |
| --- | --- |
| `vla/` | Everything that runs: CBD ↔ LeRobot conversion, rendering, the GR00T modality config, evaluation, guidance, VLM goals |
| `colab/` | `start.sh` / `go.sh` (Codespace side), `vm_train.sh` / `vm_infer.sh` / `vm_data.sh` (VM side), `watch.sh`, `autostop.sh` |
| `data/video_reach/` | Object-bearing training clips: 13 real (the author's own videos) + 658 synthetic reaches made with guidance (11 kept aside for evaluation), 671 episodes, no images |
| `data/eval/` | Evaluation specs: the 7 evaluation scenes, 7 training-side scenes, the goals the VLM picked |
| `data/stats/` | Normalization statistics of `checkpoint-47000` (used by guidance and the smoke test) |
| `tools/` | Colab helpers (CU balance, session re-attach, secrets, pty wrapper), Hugging Face metadata, the README GIFs |
| `tests/smoke_test.py` | The CPU smoke test above |
| `hf/README.md` | Model card of the Hugging Face repository |
| `docs/` | Results, training data and its terms (`data.md`), Colab operations, media |

## Weights

The usable form is made of two weight files. Both are derived from GR00T N1.7 (3B parameters).

- `checkpoint-47000`: the full fine-tuned checkpoint, ~12 GB without optimizer state.
- `action_head_bf16_v12e_armH.safetensors`: the arm H action head, 3.2 GB.

They are public on Hugging Face at **[Koichi3333/vla-maker-from-cbd](https://huggingface.co/Koichi3333/vla-maker-from-cbd)**.
To download them, log in to Hugging Face and agree to the terms below (non-commercial research or evaluation only); access is gated.

The Colab scripts still read them from Google Drive under `MyDrive/VLA/VLA_1_0/`
(`ckpt/checkpoint-47000/`, `v12e/…armH.safetensors`, `data/b3_dataset.tar`; set `DRIVE=` to change that).
Switching them to Hugging Face is planned.

- Re-upload from Drive to Hugging Face: `colab/hf_upload.sh` (the model card is `hf/README.md`)
- Rebuild the rendered training data `b3_dataset.tar`: `colab/go.sh data`. The HumanML3D it starts from is something you build yourself from AMASS ([docs/data.md](docs/data.md))

## License

**In short: the code is free to use (Apache-2.0). The weights and the training data are for non-commercial use (research or evaluation) only.**

- **Code** in this repository: Apache License 2.0 ([LICENSE](LICENSE)).
- **The left half of `docs/media/hero_video_to_vla.gif`** comes from the Common Behavior Data sample video and is not covered by
  this repository's license; see that repository's [THIRD_PARTY_NOTICES.md](https://github.com/Koichi3333/common-behavior-data/blob/main/THIRD_PARTY_NOTICES.md).
- **GR00T N1.7 and the weights derived from it**: GR00T N1.7's model card names the **NVIDIA Open Model License**, but the
  `LICENSE` file shipped with it is the non-commercial **NVIDIA License**. To satisfy both, the weights are distributed for
  non-commercial use (research or evaluation) only. The terms are in the model repository's `NOTICE` (source: [`hf/NOTICE`](hf/NOTICE)).
  **Built on NVIDIA Cosmos**: the part that reads images and language (the backbone) is Cosmos-Reason2-2B.
- **Training data**: HumanML3D/AMASS and SMPL (used to fit the torso in video clips) are licensed for
  **non-commercial research**, and so are the weights trained on them and `data/video_reach/`. Training for non-commercial
  research is allowed, but AMASS-derived motion data must never be redistributed, so each user obtains it from AMASS directly.
  Commercial use needs data you own, which is exactly why contributed CBD clips matter.
  See [docs/data.md](docs/data.md) and [data/README.md](data/README.md).

## Citation

If you use this work, please also cite what it is built on: **AMASS**, **SMPL**, **HumanML3D**, **HumanAct12** and **GR00T N1**.
The AMASS and SMPL licenses require citing their papers in any publication that uses them. BibTeX: [docs/data.md](docs/data.md#citations).

## Acknowledgements

NVIDIA Isaac GR00T (Built on NVIDIA Cosmos) · HumanML3D · AMASS · SMPL · MuJoCo · LeRobot · SAM 2 · MediaPipe · ROMP · Gemini Robotics-ER.
