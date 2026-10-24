# Running on Colab

Everything that needs a GPU runs on a Colab A100 driven from a terminal by
[`google-colab-cli`](https://github.com/googlecolab/google-colab-cli). The scripts in `colab/` handle the
problems below. This page explains why they look the way they do, so that you can port them to other platforms.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install google-colab-cli --with 'jupyter-kernel-client<1.0'   # the pin is required
colab sessions                                                        # first run starts OAuth
```

- **Pin `jupyter-kernel-client<1.0`.** 1.0 renamed `KernelClient` and `colab exec` fails with an
  `AttributeError`. Do not run `uv tool upgrade google-colab-cli` without the pin.
- The Hugging Face token (`HF_TOKEN`, needed for GR00T's gated Cosmos backbone) is passed by
  `tools/colab_secrets.py` through a temporary file. **Never put secrets in `colab exec` code.** The CLI logs
  executed code and output in plain text under `~/.config/colab-cli/history/`.
- `tools/colab_ccu.py` prints the compute-unit balance, the current burn rate (A100 ≈ 5.3 CU/h) and the GPUs you can get.

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
| **Tokens expire after 60 min.** The CLI never refreshes the runtime proxy token from `colab new` | Around minute 60: `Session 'X' appears to be lost (404/401)`. The CLI then **forgets the session while the VM keeps billing** | Refresh loop every 45 min (`tools/colab_reattach.py --refresh`). If it still drops, `colab_reattach.py <name>` re-attaches to the living VM; restart the watcher |
| **Idle allocations vanish after ~25 min**, even with keep-alive, including while waiting for Drive consent | keep-alive returns 404 at exactly 1,532 s; the VM is gone | `colab/watch.sh` sends a light `colab exec` every 2 min from session creation on |
| **The Codespace stops** | keep-alive, refresh and watcher die with it, then the VM goes too | Make sure your Codespace stays awake for long runs |
| **Long jobs die with the kernel** | — | VM jobs run via `subprocess.Popen(..., start_new_session=True)` |
| **Drive mount needs a human, every session** | `colab drivemount` waits on `/dev/tty` for consent | `tools/ptyrun.py` wraps it; the URL goes to a file and a local page |
| **`ValueError: mount failed` right after consent** | DriveFS start race | Re-run `colab drivemount`; the second time needs no consent (`go.sh` does it once) |
| **Drive copies return before upload completes** | `ls` shows the full size; stopping the VM loses the file (a 12.9 GB file was lost this way) | `vla/drive_verify.py` checks server-side sizes via the Drive API before reporting `VLA_WEIGHTS_VERIFIED` |
| **Mount can drop mid-run** (`Transport endpoint is not connected`) | Writes go to a local stub, not Drive | Check `mount | grep /content/drive` before and after long steps |
| **`colab upload` fails on large files** | 112 MB → `SSLEOFError`; missing target dir → 500 | Bundles are split into 20 MB parts; target dirs are created first |
| **`--timeout` is a total deadline**, default 30 s | Long cells die no matter how much they print | Every call passes an explicit timeout; long work is detached |
| **`files.download()` does nothing from the CLI** | Cell succeeds, no file | Use `colab download` before `colab stop`; VM disks are wiped on stop |
| **T4 cannot train GR00T N1.7** | flash-attn 2 needs Ampere+ | Use A100 (L4 is untested) |

## Porting to another GPU host

`colab/vm_*.sh` only assume:

- an environment from `colab/setup_vm.sh` in `/content/Isaac-GR00T`;
- code and data in `/content/vla`;
- the Drive layout under `$DRIVE`: `ckpt/checkpoint-47000/`, `data/b3_dataset.tar`, action heads.

On a machine with an A100, run `setup_vm.sh`, put the files in place, set `DRIVE` to a local directory, and run
the job script. You do not need `start.sh` and `go.sh` at all. A port like that is a welcome contribution.
