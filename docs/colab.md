# Running on Colab

Everything that needs a GPU runs on a Colab A100 driven from a terminal by
[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli). The scripts in `colab/` handle the
problems below. This page explains why they look the way they do, so that you can port them to other platforms.

## Setup

The CLI is installed, signed in and checked through [`colab-cli/`](../colab-cli/README.md), the same layout as
[Common Behavior Data](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli):

```bash
colab-cli/setup.sh        # uv + google-colab-cli 0.7.5 (pinned; safe to re-run)
colab-cli/auth.sh         # sign in once per machine; the page opens in your browser
colab-cli/smoke_test.sh   # CPU VM for ~1 min: new → exec → download → stop
```

- **Re-run `setup.sh` instead of `uv tool upgrade google-colab-cli`.** The scripts here were checked against the
  pinned version. 0.7.x pins its own `jupyter-kernel-client`; on 0.6.x, `jupyter-kernel-client` 1.0 renamed
  `KernelClient` and `colab exec` failed with an `AttributeError`, so that version needed `<1.0`.
- The Hugging Face token (`HF_TOKEN`, needed for GR00T's gated Cosmos backbone and for this project's gated weights) is passed by
  `colab-cli/tools/colab_secrets.py` through a temporary file, from the environment, `colab-cli/.env` or a
  Codespaces secret. **Never put secrets in `colab exec` code.** The CLI logs executed code and output in plain
  text under `~/.config/colab-cli/history/`.
- `colab-cli/tools/colab_ccu.py` prints the compute-unit balance, the current burn rate (A100 ≈ 5.3 CU/h) and the GPUs you can get.
- General CLI behaviour without a browser (`input()`, `files.upload()`, `userdata.get()`, Drive) is in the
  [field guide](../colab-cli/guide/field_guide.md). This page is about what long A100 jobs add on top.

## The flow

```
colab/start.sh <job> <name>   create A100 session → secrets → token-refresh loop → watcher
                              → upload code (20 MB parts) → start env setup → Drive mount up to the consent URL
(you)                         open /tmp/dm_url.txt (or http://localhost:8899), approve within 600 s
colab/go.sh <job> <name>      finish mount → wait for setup → launch colab/vm_<job>.sh detached → watcher
colab/autostop.sh ...         optional: when the job ends or a deadline passes, fetch results and stop the session
colab stop -s <name>          always
```

Progress markers (`VLA_*`) appear in `/tmp/watch_<name>.log`. The VM job copies its log and `health.txt`
to Drive (`runs/<RUN>/live/`) every minute, so you can still diagnose a run whose VM disappeared.

## Traps, and what the scripts do about them

| Trap | Symptom | Countermeasure |
| --- | --- | --- |
| **Tokens expire after 60 min.** The CLI never refreshes the runtime proxy token from `colab new` | Around minute 60: `Session 'X' appears to be lost (404/401)`. The CLI then **forgets the session while the VM keeps billing** | Refresh loop every 45 min (`colab-cli/tools/colab_reattach.py --refresh`). If it still drops, `colab_reattach.py <name>` re-attaches to the living VM; restart the watcher |
| **Idle allocations vanish after ~25 min**, even with keep-alive, including while waiting for Drive consent | keep-alive returns 404 at exactly 1,532 s; the VM is gone | `colab/watch.sh` sends a light `colab exec` every 2 min from session creation on |
| **The Codespace stops** | keep-alive, refresh and watcher die with it, then the VM goes too | Make sure your Codespace stays awake for long runs |
| **Long jobs die with the kernel** | — | VM jobs run via `subprocess.Popen(..., start_new_session=True)` |
| **Drive mount needs a human, every session** | `colab drivemount` waits on `/dev/tty` for consent | `colab-cli/tools/ptyrun.py` wraps it; the URL goes to a file and a local page |
| **`ValueError: mount failed` right after consent** | DriveFS start race | Re-run `colab drivemount`; the second time needs no consent (`go.sh` does it once) |
| **Drive copies return before upload completes** | `ls` shows the full size; stopping the VM loses the file (a 12.9 GB file was lost this way) | `vla/drive_verify.py` checks server-side sizes via the Drive API before reporting `VLA_WEIGHTS_VERIFIED` |
| **Mount can drop mid-run** (`Transport endpoint is not connected`) | Writes go to a local stub, not Drive | Check `mount | grep /content/drive` before and after long steps |
| **`colab upload` fails on large files** | 112 MB → `SSLEOFError`; missing target dir → 500 | Bundles are split into 20 MB parts; target dirs are created first |
| **`colab exec` can hang for good** (seen with colab-cli 0.7.5, 2026-11) when its websocket connects ~10 s late | The request goes out before the socket is ready, the reply is missed, `--timeout` never fires. The code may never run on the VM (no `job.log`), and a hung watcher stops the 2-minute exec, so the allocation can vanish | Every CLI call in `watch.sh`, `go.sh` and `autostop.sh` is wrapped in coreutils `timeout`, and the upload steps of `start.sh` and `colab-cli/tools/colab_secrets.py` are retried up to 3 times (their VM-side code is idempotent; a hang in the upload step was seen on 2026-11-09). Files can still be read while it hangs: `colab download` and `colab console` do not use the kernel |
| **`--timeout` is a total deadline**, default 30 s | Long cells die no matter how much they print | Every call passes an explicit timeout; long work is detached |
| **`files.download()` does nothing from the CLI** | Cell succeeds, no file | Use `colab download` before `colab stop`; VM disks are wiped on stop |
| **T4 cannot train GR00T N1.7** | flash-attn 2 needs Ampere+ | Use A100 (L4 is untested) |

## Porting to another GPU host

`colab/vm_*.sh` only assume:

- an environment from `colab/setup_vm.sh` in `/content/Isaac-GR00T`;
- code and data in `/content/vla`;
- the Drive layout under `$DRIVE`: `data/b3_dataset.tar`, and `runs/` for output (with `WEIGHTS=drive`, also
  `ckpt/checkpoint-47000/` and action heads; by default `colab/vm_fetch_base.py` downloads those from Hugging Face).

On a machine with an A100, run `setup_vm.sh`, put the files in place, set `DRIVE` to a local directory, and run
the job script. You do not need `start.sh` and `go.sh` at all. A port like that is a welcome contribution.
