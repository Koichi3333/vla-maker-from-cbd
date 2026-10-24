# Contributing

[日本語](CONTRIBUTING.ja.md)

Thanks for looking. This project is small enough that a single contribution changes its direction, and
it is open about what is unsolved. Five ways to help follow, from no setup at all to a GPU experiment.

## Ways to help

### 1. Try it and tell us what broke (10 minutes)

Run the [CPU smoke test](README.md#1-cpu-smoke-test-no-gpu-no-weights). If anything fails, whether it is
an install problem, an unclear error or a missing dependency, open an issue with the command and the
output. Reports from fresh machines are exactly what we lack.

### 2. Pick a "help wanted" item

The table in the [README](README.md#help-wanted) is the roadmap. To take one:

1. Comment on its issue (or open one titled after the row) saying you are on it. This avoids two people doing the same work.
2. Post a short plan before a large change: which files, and how you will show it works.
3. Open a draft PR early. Unfinished is fine.

🟢 items need no GPU and no weights. They are the best first PR.

### 3. Contribute CBD clips (no code)

Data is the bottleneck, and data whose rights are clear is what makes a commercially usable model possible.
Useful clips:

- **Manipulation from a fixed camera:** reach, pick up, move, put down, drink. Seated at a table or standing. Whole body in frame.
- **The same action with the object in different places** (left, right, near, far, high). This is exactly what the model needs to learn "go where the object is".
- 5 to 15 seconds each, one person in frame, real recordings only (no generated video; the pipeline rejects it on purpose).

Convert them with the video generator in
[common-behavior-data](https://github.com/Koichi3333/common-behavior-data) and open a PR or an issue with
a link. **Only submit recordings you have the rights to, of people who agreed to it.** State the license
you grant (we suggest CC BY 4.0). We store skeletons and object states, never the original frames.

### 4. Run an experiment (GPU)

The open research problems (🔴 / 🟠) need A100-class GPUs. If you have access:

- Start from the recipe in `colab/vm_train.sh`. Its steps are plain shell, so they port to any machine with an A100.
- Report results with the evaluator in `vla/groot_eval.py` on `data/eval/object_eval_heldout.json`, so numbers are comparable to the [results table](docs/results.md).
- Always report **H-1** (no regression on HumanML3D held-out). A change that buys reaching by breaking ordinary motion does not count.
- Negative results are welcome. Several of the most useful findings so far are failures (see "what did not work" in the results).

### 5. Improve the code

Translation of Japanese log messages and help texts, tests, CI, packaging, a non-Colab runner. See the 🟢 rows.

## Pull requests

- Keep a PR to one topic. Describe what changed and **how you checked it**: a command and its output, or a number.
- If you touch anything under `vla/`, run the smoke test (`head` and `rollout`) and paste the last lines.
- Do not commit weights, rendered videos or anything under `work/`. Large files go to Hugging Face or a release.
- Never commit secrets (`HF_TOKEN`, API keys). The Colab helpers pass them through files on purpose. See [docs/colab.md](docs/colab.md).
- Code: Python 3.10+, standard library plus what is in `requirements.txt`; keep VM-only imports (torch, gr00t) inside functions, as the existing files do.

## Credit

Every merged contributor is listed in the README. Data contributors are listed with their clips and license.
If a result is written up, people whose work it depends on will be credited by name.

## Conduct

Be kind and specific. Critique numbers and code, not people. Harassment is not tolerated; the maintainer
may remove content or block participants to keep the space usable.

## Questions

Open an issue with the `question` label. Japanese or English are both fine.
