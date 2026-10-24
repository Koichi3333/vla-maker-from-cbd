# Working rules for agents — vla-maker-from-cbd

This repository builds a GR00T N1.7 VLA from Common Behavior Data (CBD). The overview is in `README.md`,
the numbers in `docs/results.md` (full record in Japanese: `docs/results.ja.md`), the training data and its
terms in `docs/data.md`, and the Colab traps and the reasons behind each workaround in `docs/colab.md`.

## Where things go

- **Generated output goes in `work/`** (not tracked by git): smoke-test output, results downloaded from Colab,
  VLM images and answers (`vla/vlm_target.py` uses `$VLA_WORK`, default `work/`).
- To change behavior, edit the sources in `vla/`, `colab/` and `tools/`. `data/` is training and evaluation input;
  if you change it, write down why in `data/README.md`.
- Never commit weights, videos or archives (already in `.gitignore`). Large files go to Hugging Face or a release.
- **Never put AMASS-derived motion data (such as HumanML3D's `new_joints`) in this repository or on Hugging Face,
  and never download it from third-party mirrors.** Each user obtains it from AMASS through the official steps
  (`docs/data.md`, which also summarizes the terms).

## Run this first (CPU, no GPU, no cost)

```bash
sudo apt-get install -y libosmesa6                     # once, for software OpenGL
pip install -r requirements.txt
MUJOCO_GL=osmesa python3 tests/smoke_test.py head      # before -> after guidance [0.178] -> [0.] m, overlap 4 is True
MUJOCO_GL=osmesa python3 tests/smoke_test.py rollout   # 2 scenes x 5 conditions, mp4 and traj_*.npz in work/smoke/
```

After touching anything under `vla/`, always run both and include the last lines in your report.

## Using Colab (A100)

```bash
bash colab/start.sh <train|infer|data> <name>   # stops at the Drive consent URL (/tmp/dm_url.txt)
bash colab/go.sh <train|infer|data> <name>      # after consent; the VM side is colab/vm_<job>.sh
colab stop -s <name>                             # always
```

1. **Before starting**, check that no session is left over (`colab sessions`). Long runs depend on the machine running
   the CLI: if it stops, the keep-alive stops and the VM goes with it.
2. **Never forget to stop.** `.claude/hooks/colab-session-guard.sh` warns at the end of each turn while a session exists.
   For unattended runs, start `colab/autostop.sh` right after launch.
3. **The CLI can drop a session after 60 minutes** (`appears to be lost (404/401)`). The VM is still alive: re-attach
   with `tools/colab_reattach.py <name>` and restart `colab/watch.sh`.
4. **The Drive consent URL is for the user.** Hand it over; do not try to complete the consent yourself.
5. **Always pass `--timeout` explicitly** (the default is 30 s for the whole call). Detach long work on the VM side.
6. **Collect results before stopping** (`colab download -s <name> /content/run/results.tgz work/<RUN>/results.tgz`).
   For weights written to Drive, wait for `VLA_WEIGHTS_VERIFIED` before stopping.
7. **Never put secrets in `colab exec` code or output.** `HF_TOKEN` is passed by `tools/colab_secrets.py`. Check that
   a value is set by its length only
   (`python3 -c 'import os;v=os.environ.get("HF_TOKEN");print(f"set len={len(v)}" if v else "NOT SET")'`).

## Cost

Colab compute units cost real money. An A100 uses about 5.3 CU per hour: training 5,000 steps is about 9 CU,
one inference evaluation about 1.4 CU. **Before any paid run, show an estimate and get the user's approval.**

## Writing about results

Stay within what is measured. Follow "What we cannot claim yet" in `README.md` (and the section
「書いてよいこと・書いてはいけないこと」 in `docs/results.ja.md`). In particular, never write that the VLA alone sees
and grasps objects, or that anything succeeded in physics, on a real robot or on a G1 beyond kinematics.
"Held" means the hand came within reach and the object was attached; there are no fingers.

## Language

Code comments and docstrings are in English. Many log messages and command-line help texts are still in Japanese;
translating them is an open task.

## References

- `.claude/skills/colab-operator/SKILL.md` — the official guide to the colab CLI
- `docs/colab.md` — the list of traps, and what to change when moving to another GPU environment
