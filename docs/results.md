# Results

The full record, with every table, is in Japanese: [results.ja.md](results.ja.md). This page summarizes it.
All numbers were measured by the author between 2026-09 and 2026-10 on Colab A100s. All evaluation is
**kinematic MuJoCo**: no physics, no real robot. Experiment names (B-4, V-5, arm H, …) are the labels used during development.

## Weight lineage

| Name | What | Training | Cost |
| --- | --- | --- | --- |
| `checkpoint-47000` (stage B) | GR00T N1.7-3B on 10,391 standing HumanML3D clips, converted through CBD to 117-d and rendered in MuJoCo, one epoch | 47,000 steps, action head only (1.62 B params) | 48.85 CU |
| **arm H (usable form)** | `checkpoint-47000` + object-bearing CBD from video (`data/video_reach`) at ~50% of frames | 5,000 steps, lr 3e-5 | ~9 CU |
| arm J | Same mix as arm H, 30,000 steps | Best at following the object *without* guidance | ~30 CU |

## Stage B: train a VLA on mocap-derived CBD

- **Follows instructions it never saw.** Ten test sentences with zero verbatim matches in the 9,905 training captions, 8 starting poses, 10 s closed loop. `walks forward` +3.06 m, `walks backward` −1.79 m, `turns around and walks back` −1.48 m. The six in-place actions stay within ±0.34 m. `crawls on the floor` (16 supporting clips) really goes on all fours in 6 of 8 samples.
- **Held-out prediction, 16 frames ahead, 21 clips:** beats the "hold the current pose" baseline on MSE (0.01865 vs 0.01883) and root error (0.107 vs 0.119 m), but not yet on bone angle (10.86° vs 9.54°).
- **Retargets to a Unitree G1 through the CBD adapter without retraining.** Median direction residual is 0.957° for generated `walks forward` vs 1.543° for the teacher walks. In 6 of 7 comparable texts it is below the teacher median. The G1 waist (±30°) limits deep bends.
- **Vision matters, moderately.** Retraining with black frames costs +0.0068 loss and +1.18° held-out bone error, but closed-loop differences were mixed.

## Stage V: add object-bearing CBD from video, reach for the object

| Hypothesis | Bar | Result | Verdict |
| --- | --- | --- | --- |
| H-1: adding video does not hurt HumanML3D held-out | ≤ 9.35° + 0.3° | 9.24–9.55° across arms, even at 50% video and 30k steps | **Pass** |
| H-2: with an object in the instruction, the hand goes toward it | hand–object distance drops | 0.160 → 0.077 m, but it went to a *memorized* spot | Pass (content tested by H-2') |
| H-3: beat hold on open-loop video clips | lower mean error | best 5.53° vs hold 5.05° | **Not met** |
| H-2': move the object, the hand follows | shift and mirror ≥ 0.7 | guided: 1.00 / 1.00; unguided arm J: 0.96 / 0.71 | **Pass with guidance**; borderline without |
| H-4: hand endpoint tracks object position | slope ≥ 0.5 | unguided +0.049 (5% video) → +0.241 → +0.446 → **+0.577** (arm J); guided +0.613 | **Pass** |
| E-1/E-2: give the goal as extra state input | H-4 ≥ 0.5 | +0.057, sensitivity 0.02; same with other training settings and waypoints | **Not met** |
| V-10: a VLM picks the goal; the usable form moves to it | ≥ 0.9 | picks 378/378; within 10 cm 1.00 / 1.00 / 0.89, held 0.93 (same as the true goal) | **Pass** |

### What we learned

1. **Video-derived CBD does not break the mocap side** (H-1), even at 50% of frames and 30,000 steps.
2. **With a little video, the model ignores where the object is.** It reaches to a memorized place. Mirroring, synthetic paired clips and putting the goal into the state all failed. Given the goal as input, the model's sensitivity to it is 0.02. Only 0.91% of the variance in synthetic data is information that the goal alone carries.
3. **Inference-time guidance works without training.** Pull the arm keyframes toward a planner's path with forward kinematics, overlap chunk seams, attach the object near the hand.
4. **More video and longer training teach direction without guidance** (H-4 +0.241 → +0.446 → +0.577). Precision still comes from guidance: mirrored-side grasps are 0.42 unguided vs 0.83 guided.
5. **The VLM picks correctly every time; turning a 2D point into a 3D grasp target is the hard part.** A ray hits the object's near surface, which is 7–12 cm short for bowls and books. Asking for a bounding box and pushing the goal half the object size deeper matched the true-goal result.
6. **Check the metric before believing it.** Two counting bugs (passing near the old spot after grasping, clips that start at the object) made the mirrored case look stuck at 0.64. Results use the fixed count.

### Open problems (the README's help-wanted list)

| Problem | What is known | Training needed | Estimate |
| --- | --- | --- | --- |
| VLA uses the goal by itself | Synthesize clips where the goal changes mid-motion | Yes | ~30 CU per try |
| Train-side mirrored reaching is low (0.29 vs 0.71 held-out) | Guided rollouts at the mirrored spot are missing for three training episodes; suspected lost when the reach-only dataset was cut | Maybe | Investigation is CPU-only, about half a day |
| Arm J + guidance at 0.3 | Can more training allow weaker guidance? | No | ~1 CU |
| H-3 open-loop | Not measured after arm H | No | ~1 CU |
| Fingers, physics, G1 dynamics | CBD has no finger bones; G1 is kinematic only | — | Out of current scope |

## Not yet verified in this repository

The code was moved here from the development repository on 2026-10-19. The CPU smoke test (`head`, `rollout`,
`rollout vlm`) was re-run after the move and passes. `vla/vlm_target.py render` reproduces the 63 VLM
images and 126 questions of the original run pixel for pixel. The Codespace side of `colab/start.sh` and `go.sh`
was dry-run against a stubbed `colab`. The VM-side scripts `colab/vm_train.sh`, `vm_infer.sh` and `vm_data.sh`
consolidate scripts that ran successfully during development, but **have not been run on a real A100
since the move**. If you are the first to run one, please report it.
