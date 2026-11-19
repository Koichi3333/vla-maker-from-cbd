# `colab-cli/` — run this VLA on Colab from your terminal, no GPU machine needed

[日本語](README.ja.md) · [Field guide](guide/field_guide.md) ·
[Traps specific to the VLA jobs](../docs/colab.md)

Everything needed to drive Google Colab from a shell with the official
[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli): install, sign-in, handing keys over,
an end-to-end check, and a warning when a session is left running. It has the same layout and the same
commands as `colab-cli/` in [Common Behavior Data](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli),
so a machine set up for one works for the other.

**You do not need a GPU of your own.** The terminal can be a GitHub Codespace in your browser, and the
3B model trains and evaluates on a Colab A100 that you drive from that terminal. A local machine
(macOS / Linux; Windows inside WSL) works the same way.

## Quick start

From the repository root:

```bash
colab-cli/setup.sh        # install uv + google-colab-cli 0.7.5 (safe to re-run)
colab-cli/auth.sh         # sign in once per machine — the page opens in your browser
colab-cli/smoke_test.sh   # start a CPU VM, run code, fetch a file, stop it (~1 min)
```

All you need is a Google account (Workspace accounts work too). No `gcloud`, no GCP project.
The free Colab tier covers everything up to here.

## What each step needs

| Step | Runs on | Needs | Time and cost |
|---|---|---|---|
| CPU smoke test ([`tests/smoke_test.py`](../tests/smoke_test.py)) | your machine, a Codespace, GitHub Actions | nothing | minutes, free |
| [`smoke_test.sh`](smoke_test.sh) (the CLI end to end) | a Colab CPU runtime | a Google account | ~1 min, free tier |
| Inference and evaluation (`colab/start.sh infer`) | a Colab A100 | a paid Colab plan, `HF_TOKEN` (weights come from Hugging Face), `b3_dataset.tar` in Drive | ~20 min and ~1.4 CU per goal file |
| Continued training (`colab/start.sh train`) | a Colab A100 | the same | ~1.5 h and ~9 CU for 5,000 steps |
| Rebuild the rendered training data (`colab/go.sh data`) | a Colab A100 | the same, plus HumanML3D built from AMASS yourself | the author need not (it is in Drive) |

- **A100s need a paid Colab plan**, at about 5.3 compute units an hour. A free T4 cannot train GR00T N1.7
  (flash-attn 2 needs Ampere or newer). L4 is untested.
- `colab-cli/tools/colab_ccu.py` shows your balance, the current burn rate and the GPUs you can get
  (`colab/start.sh` prints it before it starts).
- The weights are downloaded from Hugging Face (gated: accept the terms with the account of your `HF_TOKEN`).
  Drive holds only `data/b3_dataset.tar` and the run output ([Weights](../README.md#weights)).

## Cloud or local: what differs

Three things; the scripts absorb the first.

| | Cloud (Codespaces, SSH) | Local (macOS / Linux / WSL) |
|---|---|---|
| **Sign-in** | `auth.sh` opens the page in **your own browser** through `$BROWSER` (Codespaces, VS Code Remote). Over plain SSH it writes `ws/.auth/auth_url.txt` and a one-link `index.html`, or serves it with `--serve 8765` + port forwarding | `auth.sh` opens your default browser |
| **`HF_TOKEN`** | A Codespaces secret: GitHub → Settings → Codespaces → Secrets, add **this repository** under *Repository access*, then **Rebuild** the Codespace. A secret shared only with another repository does not arrive here. Check which names arrived with `python3 colab-cli/tools/colab_secrets.py --list`. Short-lived: [`--set` below](#a-short-lived-hf_token) | `cp colab-cli/.env.example colab-cli/.env` and fill it in (gitignored), or just `export` it |
| **What survives** | Rebuilding or recreating the Codespace deletes `~/.local` and `~/.config`: the CLI and the token are gone. Re-run `setup.sh` and `auth.sh` | nothing to redo |

Either way the sign-in itself is the CLI's own flow (approve in a browser, paste the code Google shows).
`auth.sh` exists because the authorization URL is about 700 characters long: copied out of a terminal it gets
truncated and Google answers *Error 400: Missing required parameter: scope*
([details](guide/field_guide.md#the-one-real-trap-the-url-is-cut)).
For agents and scripts there is a two-step form: `auth.sh --url-only`, then `auth.sh --code <CODE>`.

### A short-lived `HF_TOKEN`

Instead of leaving the token in a secret, you can create one for a run and delete it afterwards:

```bash
python3 colab-cli/tools/colab_secrets.py --set HF_TOKEN     # paste it; the input is not echoed or kept in history
bash colab/start.sh infer myrun                             # → approve → bash colab/go.sh infer myrun
python3 colab-cli/tools/colab_secrets.py --unset HF_TOKEN   # safe once go.sh has launched the job
```

- `--set` writes `colab-cli/.env` (gitignored) with mode 600. `colab/start.sh` stops **before allocating a VM**
  when a key is missing.
- The job keeps the token from the environment it started in, so removing it here after `go.sh` reports the launch
  does not affect the run. Only after re-attaching a session (`colab_reattach.py`) would you need to `--set` it again.
- The token stays on the VM until the job ends (`colab stop` wipes the VM). When the run is over, delete the token
  itself on Hugging Face (Settings → Access Tokens). *Read* scope (or fine-grained access to the gated models) is enough.
- Run `--set` in your own terminal (such as the VS Code terminal): it prompts for input. Never paste the token
  into a chat or a command line.

## Running the VLA jobs

The VLA jobs are not notebooks: they are shell scripts sent to the VM with `colab exec`.
The scripts in `colab/` build an A100 session with the tools in this directory.

```bash
bash colab/start.sh infer myrun        # get an A100 → ship HF_TOKEN → upload the code → stop at the Drive consent URL
# open the URL in /tmp/dm_url.txt (or http://localhost:8899) and approve within 600 s
bash colab/go.sh infer myrun           # finish the mount, start colab/vm_infer.sh detached on the VM
tail -f /tmp/watch_myrun.log           # progress; VLA_DONE at the end, VLA_EXIT=<non-zero> on failure
colab download -s myrun /content/run/results.tgz work/myrun/results.tgz
colab stop -s myrun && colab sessions  # always stop, and confirm
```

Training is the same with `train` in place of `infer` (`STEPS=5000 DUP=40 RUN=t1 bash colab/go.sh train t1`;
`VIDEO_DATA=<your LeRobot dataset> bash colab/start.sh train t1` to use your own clips).

The `colab/` scripts handle the traps that only long A100 jobs hit. The reasons and the fixes are in
[`docs/colab.md`](../docs/colab.md):

- The CLI never refreshes the 60-minute token it got from `colab new` → `tools/colab_reattach.py --refresh` every 45 min
- An allocation with no traffic vanishes after ~25 min → `colab/watch.sh` sends a light `exec` every 2 min
- The Drive mount needs a human for every session → wrapped in `tools/ptyrun.py`; the URL goes to a file and a local page
- `colab exec` can hang, ignoring `--timeout` (0.7.5) → CLI calls are wrapped in coreutils `timeout`
- When you step away, `colab/autostop.sh` collects the results and stops the session at the end of the job or at a deadline

## Five rules

1. **Always `colab stop -s <name>`.** Closing the terminal does not release the VM, and an A100 bills until it is
   stopped. Confirm with `colab sessions`. With Claude Code in this repository, a warning appears at the end of each
   reply while a session is left running (registered in [`.claude/settings.json`](../.claude/settings.json); no setup).
2. **Keep the machine running the CLI awake.** Keep-alive, token refresh and the watcher all run there, so if the
   Codespace stops, the VM goes with it. Raise the Codespace idle timeout before a long run.
3. **Always pass `--timeout`.** The default is 30 s, and it is a deadline for the whole run, not for silence.
   Detach long work on the VM.
4. **Do not trust the exit code.** `colab exec` can exit 0 after a failure. Read `VLA_EXIT` and grep for `Traceback`.
5. **Collect before stopping.** `/content` goes with the session. Use `colab download` (files only). For weights
   written to Drive, wait for `VLA_WEIGHTS_VERIFIED`.

Hand keys over with `colab_secrets.py` and **never write one into code passed to `colab exec`**: that command
appends both the code and its output to `~/.config/colab-cli/history/<session>.jsonl` in plain text.

## What's in this directory

| Path | What it is |
|---|---|
| [`setup.sh`](setup.sh) | Installs `uv` and `google-colab-cli` (pinned to 0.7.5; `COLAB_CLI_VERSION=latest` to change), creates `ws/`, writes the Claude Code skill |
| [`auth.sh`](auth.sh) → [`tools/auth.py`](tools/auth.py) | Sign-in that never passes the URL through the terminal |
| [`smoke_test.sh`](smoke_test.sh) | new → exec → download → stop; always stops, even on failure. Extra arguments go to `colab new` (e.g. `--gpu T4`) |
| [`tools/colab_secrets.py`](tools/colab_secrets.py) | Ships keys (environment → `.env` → Codespaces secrets) to a session. `--set` / `--unset` store and remove a short-lived key. Its `find_secret()` also serves local scripts (`vla/vlm_target.py`) |
| [`tools/ptyrun.py`](tools/ptyrun.py) | Lets `colab drivemount` / `colab auth` finish without a terminal |
| [`tools/colab_reattach.py`](tools/colab_reattach.py) | Refreshes the 60-minute token ahead of time (`--refresh`); re-attaches a session the CLI dropped to its still-running VM |
| [`tools/colab_ccu.py`](tools/colab_ccu.py) | Compute-unit balance, burn rate, GPUs you can get |
| [`hooks/colab_session_guard.py`](hooks/colab_session_guard.py) | Claude Code Stop hook: warns about sessions left running |
| [`examples/drive/`](examples/drive/) | Writing to a mounted Drive from a `.py` and reading it from an `.ipynb` |
| [`guide/field_guide.md`](guide/field_guide.md) · [日本語](guide/field_guide.ja.md) | Measured CLI behaviour: sign-in, notebooks, secrets, artifacts, Drive, troubleshooting |
| [`CLAUDE.md`](CLAUDE.md) | Rules for Claude Code when it uses these tools |
| [`.env.example`](.env.example) | Template for local keys |
| `ws/` | Scratch directory; everything but `.gitkeep` is gitignored |
| `.claude/skills/colab-operator/` | Generated by `setup.sh` from `colab skill` (always matches the installed CLI; gitignored) |

The VLA jobs themselves (`start.sh`, `go.sh`, `vm_*.sh`, `watch.sh`, `autostop.sh`) are in [`../colab/`](../colab/).

## Status — what has and has not been run

- **On an A100 with 0.7.5 (2026-11-05, in a Codespace):** `colab/start.sh` → consent → `colab/go.sh train` ran
  500 training steps and two evaluations through to `VLA_DONE` and `VLA_EXIT=0`. At the time the helpers lived in
  `tools/` and the CLI was installed by hand (with an extra `jupyter-kernel-client<1.0` pin; 0.7.5 pins `==0.9.0`
  itself, so it is the same version).
- **Checked after the move into this directory (2026-11-09, in a Codespace):** re-running `setup.sh` (no reinstall,
  skill generated), `auth.sh` with an existing token (`colaboratory` scope present), the Stop hook (session left,
  none left, no CLI installed), `colab_secrets.py --list`, and `colab/common.sh` finding the CLI's Python and
  `colab-cli/tools/`. `smoke_test.sh` passed too (CPU runtime; the VM had Python 3.13.18 and no GPU; the file came
  back and the session was stopped, with nothing left in `colab sessions`).
- **A100 after the move (2026-11-09, in a Codespace):** with a short-lived `HF_TOKEN` (`--set`), `colab/start.sh infer`
  → consent → `colab/go.sh infer` → collection and stop by `autostop.sh`, ending in `VLA_DONE` and `VLA_EXIT=0`. The
  goal file gave the same numbers as the README's "guided, goal chosen by the VLM" column (within 10 cm 1.00 / 1.00 /
  0.89, held 0.93). The A100 was refused twice with `503 Service Unavailable` and granted on the third try, 30 minutes
  later. The extract `colab exec` hung and was re-run by hand (the extract step of `start.sh` now has `timeout` and
  retries, and so do `colab_secrets.py` and `smoke_test.sh`; a fake `colab` whose first `exec` hangs showed them giving
  up, retrying and stopping, but a real hang has not been through them yet).
- **Weights from Hugging Face (2026-11-19, in a Codespace):** `colab/start.sh infer` → consent → `colab/go.sh infer`
  with the default `WEIGHTS=hf`. `checkpoint-47000` and arm H came from `Koichi3333/vla-maker-from-cbd` with a
  short-lived `HF_TOKEN` (73 s), and the run ended in `VLA_DONE` and `VLA_EXIT=0`, collected and stopped by
  `autostop.sh`. The evaluation summary was identical to the 2026-11-09 run that read the weights from Drive.
  The A100 was refused once with `503` and granted five minutes later. About 1.9 CU.
- **Not checked yet:**
  - Training (`train`) after the move.
  - `setup.sh` on a clean machine and a first `auth.sh` sign-in (both passed on 0.7.5 on the CBD side).
  - A local machine.

If any of it does not work for you, that is a finding: please open an issue.

## Where this came from

It is based on [`colab-cli/` in common-behavior-data](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli).
`setup.sh`, `auth.sh`, `tools/auth.py`, the Stop hook, the field guide and `examples/drive/` are the same. Changes:

- The VLA's CLI helpers that lived in `tools/` (`colab_reattach.py`, `colab_ccu.py`, `colab_secrets.py`,
  `ptyrun.py`) moved here. `colab_secrets.py` and `ptyrun.py` were replaced by the CBD versions, and
  `colab_secrets.py` gained `find_secret()` for local scripts.
- The Stop hook is CBD's Python version instead of a shell script that needed `jq`. The official skill is no
  longer committed; `setup.sh` generates it.
- `smoke_test.sh` answers `--help` itself instead of passing it to `colab new`, and its CLI calls have deadlines
  (the same fix went into CBD).
- `colab_secrets.py` gives its `upload` and `exec` a deadline and up to three tries (against the 0.7.5 hang).
- `.env.example` and `CLAUDE.md` are written for the VLA (`HF_TOKEN`, A100s, cost).
