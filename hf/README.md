---
license: other
license_name: nvidia-open-model-license-noncommercial-derivative
license_link: https://huggingface.co/Koichi3333/vla-maker-from-cbd/blob/main/NOTICE
extra_gated_heading: Non-commercial research use only
extra_gated_prompt: >-
  These weights are derivative models of NVIDIA GR00T N1.7, licensed by NVIDIA Corporation under the NVIDIA Open
  Model License (the LICENSE file shipped with GR00T N1.7 is the non-commercial NVIDIA License; both are included).
  They were trained on data licensed for non-commercial research (HumanML3D/AMASS, SMPL).
  See NOTICE for the full terms.
extra_gated_fields:
  I will use these weights for non-commercial research or evaluation only: checkbox
  I agree to both NVIDIA licenses included in this repository and to the terms in NOTICE: checkbox
  I will use the model consistently with NVIDIA's Trustworthy AI terms: checkbox
  Affiliation: text
extra_gated_button_content: Agree and access
base_model: nvidia/GR00T-N1.7-3B
pipeline_tag: robotics
tags:
  - robotics
  - vla
  - humanoid
  - gr00t
  - lerobot
  - motion-generation
  - common-behavior-data
---

# vla-maker-from-cbd: GR00T N1.7 fine-tuned on Common Behavior Data

A humanoid vision-language-action model whose action space is **Common Behavior Data (CBD)**, an open,
embodiment-independent behavior format (19 bone rotations + root motion, 117 numbers per frame).
Because the output is CBD, it can be retargeted to MuJoCo, a Unitree G1 or a VRM avatar with the CBD
adapters, without retraining per body.

- Code, training and evaluation: **[vla-maker-from-cbd](https://github.com/Koichi3333/vla-maker-from-cbd)**
- Data format: [common-behavior-data](https://github.com/Koichi3333/common-behavior-data)

> **Research preview.** All evaluation is kinematic simulation (MuJoCo). No physics, no real robot.
> **Non-commercial research or evaluation use only** (see [License and terms](#license-and-terms)).

**Built on NVIDIA Cosmos.** Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

## Files

| Path | What | Size |
| --- | --- | --- |
| `checkpoint-47000/` | GR00T N1.7-3B fine-tuned on 10,391 standing HumanML3D clips (via CBD, rendered in MuJoCo), 47,000 steps = one epoch. Optimizer state removed | ~12 GB (fp32) |
| `action_heads/arm_H.safetensors` | Action head of **arm H**: `checkpoint-47000` further trained 5,000 steps (lr 3e-5) with object-bearing CBD from video mixed in at ~50% of frames | 3.2 GB (bf16) |
| `NOTICE` | Required attribution, the list of derivative models, and the additional (non-commercial) terms | |
| `LICENSE-NVIDIA-Open-Model-License-2025-10-24.pdf` | Copy of the NVIDIA Open Model License Agreement | |
| `LICENSE-NVIDIA-GR00T-N1.7.txt` | Unmodified copy of the LICENSE file shipped with GR00T N1.7 | |

Only the action head (projector + DiT + VL layer norm, 1.62 B parameters) was trained; the vision-language
backbone is the frozen GR00T N1.7 one.

**The "usable form"** is `checkpoint-47000` with its action head replaced by `arm_H`, run with
inference-time forward-kinematics guidance toward a planned path (`--guide fk --fk-eta 0.3`). The goal can
be the true object position or a point picked by a VLM in the same rendered image.

## Inputs and outputs

- **Observation:** one 256×256 front-view render of a neutral MuJoCo humanoid in the current pose (plus simple
  proxies for objects), the 117-d state, and an English instruction (HumanML3D style, e.g.
  `a seated person picks up a glass from the table and puts it back down.`).
- **Action:** 16 future frames at 20 fps of the 117-d CBD state: root translation (3, relative) and 19 bones
  as 6D rotations (absolute), grouped `root_translation, hips, torso, left_arm, right_arm, left_leg, right_leg`.
- Embodiment tag `NEW_EMBODIMENT`, modality config `vla/groot_cbd_config.py` in the code repository.

## How to use

```bash
git clone https://github.com/Koichi3333/vla-maker-from-cbd && cd vla-maker-from-cbd
huggingface-cli download Koichi3333/vla-maker-from-cbd --local-dir weights
# On a machine with the Isaac-GR00T environment (colab/setup_vm.sh) and the HumanML3D render (colab/vm_data.sh):
MUJOCO_GL=egl python vla/groot_eval.py --ckpt weights/checkpoint-47000 \
    --action-head weights/action_heads/arm_H.safetensors \
    --train <b3>/train --heldout <b3>/heldout \
    --object-eval data/eval/object_eval_vlm_box.json --goal-key vlm_goal \
    --goal object --samples 8 --frames 200 --overlap 4 --attach --spec-hand --reach 0.52 \
    --guide fk --fk-eta 0.3 --out work/eval
```

The Colab scripts in the code repository (`colab/start.sh infer`) currently read these weights from Google
Drive; switching them to this repository is planned.

## Results

7 held-out scenes × 3 object positions (original / shifted 25 cm / mirrored) × 4 samples, kinematic MuJoCo.

| | Usable form, true goal | Usable form, VLM goal | Bar |
| --- | ---: | ---: | ---: |
| Hand within 10 cm: orig / shift / mirror | 1.00 / 1.00 / 0.89 | 1.00 / 1.00 / 0.89 | |
| Moved toward the shown object: shift / mirror | 1.00 / 1.00 | 1.00 / 1.00 | ≥ 0.7 |
| Hand reached the object ("held") | 0.94 | 0.93 | |
| H-4: hand endpoint follows the object (slope) | +0.613 | – | ≥ 0.5 |
| H-1: HumanML3D held-out bone error | 9.55° (`checkpoint-47000`: 9.35°) | same weights | ≤ 9.65° |

`checkpoint-47000` alone follows unseen instructions (`walks forward` +3.06 m, `walks backward` −1.79 m in 10 s),
and its generated walking retargets to a Unitree G1 through the CBD adapter. Full numbers:
[docs/results.md](https://github.com/Koichi3333/vla-maker-from-cbd/blob/main/docs/results.md).

## Limitations

- **Kinematic only.** No physics, contacts, torques or real-robot tests.
- **No fingers.** "Held" means the hand came within reach and the object was attached to it; CBD has no finger bones yet.
- **Precise reaching relies on inference-time guidance.** Without it, the model learns the direction toward the
  object, but reaches it about half as often.
- **Does not beat a "hold the pose" baseline on open-loop video prediction** (5.53° vs 5.05°).
- Objects appear only as simple proxy shapes; the model never saw real images.
- Torso pose of the video-derived clips comes from an SMPL fit, which is not ground-truth-verified.

## Training data

- **HumanML3D** (built on AMASS, plus 363 HumanAct12 clips): standing subset, converted to CBD and rendered.
  AMASS motion data is not redistributed here; to rebuild the training data, obtain AMASS yourself and follow
  [docs/data.md](https://github.com/Koichi3333/vla-maker-from-cbd/blob/main/docs/data.md).
- **Object-bearing clips from video:** 13 real clips recorded by the author (MediaPipe, SMPL via ROMP, SAM 2),
  plus 658 synthetic reaches from guided closed-loop rollouts. Included in the code repository as `data/video_reach/`.

## Citation

Please also cite the data and models this work is built on. The AMASS and SMPL licenses require it in any publication.

```bibtex
@inproceedings{AMASS:2019,
  title     = {AMASS: Archive of Motion Capture as Surface Shapes},
  author    = {Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
@article{SMPL:2015,
  title     = {{SMPL}: A Skinned Multi-Person Linear Model},
  author    = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
  journal   = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
  volume    = {34}, number = {6}, pages = {248:1--248:16}, year = {2015}
}
@InProceedings{Guo_2022_CVPR,
  title     = {Generating Diverse and Natural 3D Human Motions From Text},
  author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}, pages = {5152-5161}
}
@inproceedings{gr00tn1_2025,
  title     = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author    = {NVIDIA and others},
  booktitle = {ArXiv Preprint}, eprint = {2503.14734}, archivePrefix = {arxiv}, year = {2025}
}
```

HumanAct12: C. Guo et al., *Action2Motion: Conditioned Generation of 3D Human Motions*, ACM Multimedia 2020.

## License and terms

**Use these weights for non-commercial research or evaluation only.**

- **Base model.** GR00T N1.7's model card states it is released under the **NVIDIA Open Model License**
  (copy included). The `LICENSE` file shipped in the GR00T N1.7 repository is the **NVIDIA License**,
  whose Section 3.3 limits use to research or evaluation (copy included, unmodified). These derivative
  models are distributed so as to comply with both: the attribution notice and the derivative-model list are in
  [`NOTICE`](NOTICE), and the non-commercial limitation applies to them.
- **Built on NVIDIA Cosmos.** The vision-language backbone (Cosmos-Reason2-2B) is included unchanged.
- **Training data.** HumanML3D (built on AMASS) and the SMPL body model (used to fit the torso of the video clips)
  are licensed for non-commercial research.
- **AI ethics.** Use must be consistent with
  [NVIDIA's Trustworthy AI terms](https://www.nvidia.com/en-us/agreements/trustworthy-ai/terms/).
- The code that trains and runs the model is licensed under the Apache License 2.0 ([code repository](https://github.com/Koichi3333/vla-maker-from-cbd)).

NVIDIA, GR00T and Cosmos are trademarks of NVIDIA Corporation, used here only to describe the origin of the model.

## 日本語の要約

CBD（体に依存しない人の動作の形式）を行動空間にした、GR00T N1.7 の追加学習版です。`checkpoint-47000` は
HumanML3D（CBD 経由）で 1 エポック学習したもの、`action_heads/arm_H.safetensors` は動画由来の物つき CBD を
約 50% 混ぜて 5,000 ステップ追加学習した action head です。推論時の誘導と組み合わせると、運動学の
シミュレーションで、物の位置へ手を伸ばします。物理・実機は未確認、指は無く、用途は非商用の研究・評価に限ります（NVIDIA Open Model License と GR00T N1.7 同梱の NVIDIA License の両方に従う形で配布。条件は `NOTICE`）。Built on NVIDIA Cosmos。
コードと詳細は [vla-maker-from-cbd](https://github.com/Koichi3333/vla-maker-from-cbd)。
