# Colab CLI — operating rules for Claude Code

How to drive Google Colab from here. Human-facing setup is in [`README.md`](README.md); the measured
behaviour behind each rule is in [`guide/field_guide.md`](guide/field_guide.md), and the traps of long
A100 jobs in [`../docs/colab.md`](../docs/colab.md). Paths below are from the repository root.
The repository-wide rules in the root `CLAUDE.md` (cost approval, collecting results, secrets) still apply.

## Setup and sign-in

- Install / repair: `colab-cli/setup.sh`. Never `uv tool upgrade google-colab-cli` bare; re-run
  `setup.sh` (it pins the version the notes and `colab/` scripts were checked against).
- **Do not run any `colab` command before a token exists** (`~/.config/colab-cli/token.json`):
  without one, `colab sessions` / `whoami` / `new` start an interactive sign-in and wait on stdin.
- Signing in from an agent is two steps, and the human does the middle:
  1. `colab-cli/auth.sh --url-only` writes `colab-cli/ws/.auth/auth_url.txt` and `index.html`.
     In Codespaces, open it for the user with `"$BROWSER" "$(cat colab-cli/ws/.auth/auth_url.txt)"`.
     Otherwise point them at `index.html`. **Never print the URL for them to copy**: it is ~700 characters and arrives
     truncated (`Error 400: Missing required parameter: scope`).
  2. The user approves and gives you the code → `colab-cli/auth.sh --code <CODE>`.
     It checks that the `colaboratory` scope was granted.
- The generated skill (`colab-cli/.claude/skills/colab-operator/SKILL.md`) may say the default auth is
  `adc` and recommend `gcloud`. Here the default is `oauth2`, and `auth.sh` is the way. Ignore its auth section.

## Running things

- **VLA jobs go through `colab/`, not hand-written sessions.** `bash colab/start.sh <train|infer|data> <name>`
  stops at the Drive consent URL (`/tmp/dm_url.txt`) — hand it to the user, never complete it yourself —
  then `bash colab/go.sh <job> <name>`. These start the token refresh loop and the watcher, which a bare
  `colab new` does not. **Before any A100 run, show the estimate (README "What each step needs") and get approval.**
- `colab-cli/smoke_test.sh` allocates a CPU runtime for about a minute. Extra arguments go to `colab new`.
- **Always pass `--timeout`.** Default 30 s, total run time not silence. Detach anything long on the VM
  (`subprocess.Popen(..., start_new_session=True)`), as `colab/go.sh` does.
- **The exit code is not the result.** Read the log (`VLA_EXIT`, `Traceback`).
- **`.ipynb` only runs through `colab exec -f`**, which writes `X_output.ipynb` next to the input: copy
  notebooks into `colab-cli/ws/` first. `colab run` treats the file as Python.
- **Collect before stopping.** `/content` is gone after `colab stop`. `colab download` takes files only
  (tar/zip directories on the VM first). For weights written to Drive, wait for `VLA_WEIGHTS_VERIFIED`.
- If `colab exec` says the session is lost (404/401) after ~60 min, the VM is usually still alive:
  `$CLIPY colab-cli/tools/colab_reattach.py <name>` (CLIPY is set in `colab/common.sh`), then restart `colab/watch.sh`.

## Always stop

- `colab stop -s <name>` when done, then `colab sessions` to confirm. Closing the shell does not release
  the VM. Stop GPU sessions first.
- The Stop hook (`colab-cli/hooks/colab_session_guard.py`, registered in `.claude/settings.json`) warns
  about leftovers when a reply ends. It is a safety net, not a substitute.
- For unattended runs, start `colab/autostop.sh` right after launch.

## Keys

- Ship keys with `python3 colab-cli/tools/colab_secrets.py -s <name> NAME ...`, once per session, after
  `colab new` and **before** the first `colab exec` (`colab/start.sh` does this for `HF_TOKEN`).
  It resolves from the environment, then `colab-cli/.env`, then Codespaces secrets.
- **Short-lived tokens:** the user stores one with `python3 colab-cli/tools/colab_secrets.py --set HF_TOKEN`
  **in their own terminal** (hidden prompt) and removes it with `--unset HF_TOKEN` once `colab/go.sh` has launched
  the job. Never ask the user to paste a token into the chat, and never pass one on a command line.
  Check presence without the value: `colab_secrets.py --list` (names only).
- If `colab_secrets.py --list` does not show a name in a Codespace, the secret is not shared with this
  repository (or the Codespace was not rebuilt since). Ask the user; you cannot fix it from inside.
- **Never put a key in code passed to `colab exec`** (logged in plain text to
  `~/.config/colab-cli/history/<session>.jsonl`), never `print` one, never `echo` one while debugging.
  Check presence only: `[ -n "$NAME" ] && echo set || echo "NOT SET"`. Note
  `echo "${X:+set}${X:-NOT SET}"` **prints the value** when it is set.

## Things that hang or fail without a browser

| In the code | Headless behaviour | Do this |
|---|---|---|
| `input()` | `StdinNotImplementedError` | Replace `input` **in the same cell** (ipykernel rebinds it before every cell) |
| `files.upload()` | Hangs, then the whole run dies | `colab upload` first, then replace `files.upload` |
| `files.download()` | Returns instantly, does nothing | `colab download` before stopping |
| `userdata.get()` | `TimeoutException` after 10 s | `colab_secrets.py` |
| `drive.mount()` / `colab drivemount` | Waits on `/dev/tty` | `colab-cli/tools/ptyrun.py` (as `colab/common.sh` does); the user approves in a browser for **every** session |

Never start `colab repl`, `colab console`, `colab auth` or `colab drivemount` directly; they need a TTY.
Recipes: [field guide](guide/field_guide.md#what-breaks-without-a-browser).
