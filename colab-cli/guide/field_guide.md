# Driving Google Colab from your terminal

A field guide to the official [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli):
how to run notebooks on Colab's GPU/TPU runtimes from a shell — your laptop, a
remote box, CI, or a coding agent — instead of a browser tab.

**The behaviour notes below were measured against v0.6.0**, not copied from the
docs. Where the official documentation and the observed behaviour disagree, that
is called out explicitly. The interesting parts are not the commands; they are
the five things that silently behave differently when no browser is attached.

> **Version status.** [`../setup.sh`](../setup.sh) installs **0.7.5**. Installation
> and sign-in in this guide are written against 0.7.5's source. The rest was
> measured on 0.6.0. Where the 0.7.5 source shows a change, the guide says so.
>
> On 2026-11-07 a CPU run on 0.7.5 re-checked these:
> - the full sign-in (oauth2 by default);
> - `new` / `upload` / `exec -f … --timeout` / `download` / `stop`;
> - `X_output.ipynb` written next to the input;
> - kernel state carrying over between `exec` calls (the secrets prelude from
>   one call was seen by the next);
> - no key left in the CLI history when it is shipped with `colab_secrets.py`.
>
> Not re-checked on 0.7.5: a failing cell's exit code, the no-browser recipes,
> GPU runtimes. If you find a behaviour that differs, that is a finding —
> please report it.
>
> Setup itself is in [`../README.md`](../README.md). This guide is the *why*.
>
> **In this repository.** This guide is a copy of `colab-cli/guide/` in
> [common-behavior-data](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli),
> which wrote it for notebooks. The VLA jobs are shell scripts driven by `colab exec`, and they
> ran on an **A100 runtime with 0.7.5** on 2026-11-05 (training and evaluation, through `colab/`).
> The traps that only long A100 jobs hit — the 60-minute token, the allocation that vanishes after
> ~25 idle minutes, Drive copies that return early, `colab exec` hanging — are in
> [`../../docs/colab.md`](../../docs/colab.md).

- [Why bother](#why-bother)
- [Install](#install)
- [Authenticate](#authenticate)
- [The core loop](#the-core-loop)
- [Command reference](#command-reference)
- [What breaks without a browser](#what-breaks-without-a-browser)
- [Secrets and API keys](#secrets-and-api-keys)
- [Getting your results back](#getting-your-results-back)
- [Google Drive](#google-drive)
- [Rules worth internalising](#rules-worth-internalising)
- [Troubleshooting](#troubleshooting)
- [Directory contents](#directory-contents)

---

## Why bother

A Colab session is a live Jupyter kernel on a VM Google allocates for you. The
CLI talks to the same backend as the web UI (`colab.research.google.com/tun/m/*`),
so sessions you create either way show up in both.

What that buys you:

- **Version control that works.** Notebooks live in your repo, get diffed and
  reviewed like code, and run unchanged on the GPU.
- **Scriptability.** `colab exec` in a shell script, a Makefile, or CI.
- **Agents.** A coding agent can run the notebook, read the output, and iterate,
  which a browser tab cannot offer.
- **A free GPU** for workloads that fit, without leaving your editor.

What it costs you: no browser means no widgets, no `files.upload()`, no Colab
Secrets. That is the whole difficulty, and [most of this guide](#what-breaks-without-a-browser).

---

## Install

```bash
colab-cli/setup.sh
```

It installs [`uv`](https://docs.astral.sh/uv/) if missing, then
`uv tool install google-colab-cli==0.7.5`, and is safe to re-run. The same
command works in Codespaces, on any remote Linux box, and on a local macOS or
Linux machine (Windows: inside WSL). `COLAB_CLI_VERSION=latest colab-cli/setup.sh`
takes the newest release instead.

**One pin that used to matter.** 0.6.0 declared `jupyter-kernel-client` unpinned,
and that package's 1.0.0 renamed `KernelClient` to `JupyterKernelClient`. Installed
without a constraint, every `colab exec` died with:

```
AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
```

0.7.5 pins `jupyter-kernel-client==0.9.0` itself and accepts all three class
names, so the extra `--with 'jupyter-kernel-client<1.0'` is only needed if you
deliberately install 0.6.x — `setup.sh` adds it in that case.

You need a Google account (Workspace accounts are fine). You do **not** need
`gcloud`, a GCP project, or your own OAuth client — the CLI bundles one.

---

## Authenticate

```bash
colab-cli/auth.sh
```

The CLI's real default is `--auth=oauth2` (its own `SKILL.md` still says `adc`;
`cli.py` sets `AuthProvider.OAUTH2`). Since 0.7.x that flow is the same one
`gcloud auth application-default login` uses: Google shows the authorization code
on a registered landing page and you paste it back. It is identical on a laptop
and in a container, so there is no separate "local" flow any more. The token is
cached with a refresh token in `~/.config/colab-cli/token.json`, so this is a
one-time step per machine.

### The one real trap: the URL is cut

**The authorization URL is about 700 characters and wraps across terminal lines.**
Copying it out of a terminal usually truncates it, `&scope=…` goes missing, and
Google answers:

```
Access blocked: Authorization Error
Missing required parameter: scope
Error 400: invalid_request
```

**When you see that error, suspect a truncated URL before anything else** — not
your Workspace admin policy, not your browser. A tell: if the `aes=` value in
the error page URL is identical on every retry, no new request is reaching
Google; your browser is serving a cached error page.

`auth.sh` runs the CLI's own flow (same client, scopes and landing page, imported
from `colab_cli.auth`) and only changes how the URL reaches a browser — **it is
never copied out of the terminal**:

| Where you are | What `auth.sh` does |
| --- | --- |
| Codespaces / VS Code Remote | Opens it through `$BROWSER`, which lands in the browser on *your* machine |
| Local macOS / Linux desktop | Opens your default browser |
| SSH or anything else | Writes `colab-cli/ws/.auth/auth_url.txt` and a one-link `index.html`; open either from your editor, or use `--serve 8765` and a port forward |
| An agent, or no TTY | `--url-only`, then `--code <CODE>` in a second call |

Without `auth.sh`, the manual equivalent is to send the URL to a file — it goes
to stderr, the prompt to stdout:

```bash
colab whoami 2> /tmp/colab_auth_url.txt
```

> **0.6.0 note.** 0.6.0 redirected to a different page, and the original notes
> used a loopback-redirect helper (`http://127.0.0.1:8085/`) as a fallback. 0.7.x
> made that unnecessary, so it is not carried here.

### Verify

```bash
colab whoami      # undocumented, but it exists
```

**Check that the scope list contains `…/auth/colaboratory`.** Without it the
keep-alive RPC returns 403 and sessions cannot be held open.

> `colab auth` is a *different thing*. It injects GCP credentials into the VM.
> It will not fix a 401/403 on the CLI itself.

---

## The core loop

```bash
colab new  -s work                                    # allocate a VM
colab exec -s work -f notebook.ipynb --timeout 1800   # run it
colab download -s work /content/out.zip ./out.zip     # collect results
colab stop -s work                                    # release it  (REQUIRED)
```

Add hardware at creation time: `colab new -s work --gpu T4`. Availability
depends on your plan — on the free tier, A100/H100 are simply never granted.

Three things about this loop are worth knowing before you write the script:

**A session holds its VM until `colab stop`.** **Closing your terminal does not
release it.** (0.6.0 ran a detached keep-alive daemon for up to 24 hours; 0.7.5's
skill says the backend now maintains liveness itself. Either way, stopping is your
job.) More under [Rules worth internalising](#rules-worth-internalising).

**Kernel state persists across `colab exec` calls** in the same session. That is
what makes the CLI usable interactively: run part 1, look at the output, inject a
value, run part 2.

```bash
colab exec -s work -f part1.ipynb --timeout 300
colab exec -s work <<< "threshold = 0.85"
colab exec -s work -f part2.ipynb --timeout 300
```

**`.ipynb` is `colab exec` only.** `colab run` reads its argument as raw Python
(`commands/run.py`, `_build_script_payload`), so handing it a notebook makes it
try to execute JSON and raise `SyntaxError`. Keep `colab run` for throwaway
`.py` scripts — its virtue is a `finally` block that always destroys the session,
so it cannot leak one.

### Running a notebook

```bash
colab exec -s work -f notebook.ipynb --timeout 1800    # cell by cell
colab exec -s work -f script.py      --timeout 1800    # one block
echo "print(1+1)" | colab exec -s work                 # stdin
colab exec -s work --timeout 60 <<'PY'
x = 42
print(x)
PY
```

Notebooks are parsed locally with `nbformat`; **only the code cells** travel over
the WebSocket to the kernel. The file is never uploaded anywhere.

Output streams back per cell with a header:

```
[colab] Executing cell 3/8 - Load the model
```

That label comes from the first line of the cell matching `^\s*#\s*@title\s+(.*)`,
so **give your cells `# @title` names** — otherwise you read a wall of cell-ID
UUIDs.

Afterwards the CLI writes an executed copy **next to the input file**:
`os.path.splitext(file)[0] + "_output.ipynb"` (`commands/execution.py` — line 238 in 0.6.0, 280 in 0.7.5).
Note *next to the input*, not the working directory — `-f ../nb.ipynb` writes
`../nb_output.ipynb`. Keep a scratch directory and copy notebooks into it if you
care where artefacts land. These files carry every printed value, so add
`*_output.ipynb` to `.gitignore`.

---

## Command reference

| Command | What it does |
| --- | --- |
| `colab new -s <name> [--gpu T4\|L4\|G4\|H100\|A100] [--tpu v5e1\|v6e1]` | Create a session. No accelerator flag = CPU. Availability is plan-dependent |
| `colab exec -s <name> -f <file> --timeout <sec>` | Run `.ipynb` / `.py` / stdin |
| `colab run [--gpu T4] <script.py>` | One-shot run; always destroys the session afterwards |
| `colab stop -s <name>` | Release the VM |
| `colab sessions` | Everything alive server-side (including browser-created sessions) |
| `colab status -s <name>` | Hardware, IDLE/BUSY, last execution |
| `colab upload -s <name> <local> <remote>` | Local → VM |
| `colab download -s <name> <remote> <local>` | VM → local (**remote path comes first**) |
| `colab ls -s <name> <path>` | List files on the VM |
| `colab restart-kernel -s <name>` | Reset kernel state |
| `colab log -s <name> [-n 20]` | Structured event log |
| `colab url -s <name>` | A URL that attaches the web UI to this session |
| `colab whoami` | Auth status and scopes |

0.7.5 also has `rm`, `edit`, `install`, `ssh`, `update` and `pay`; see
`colab help <command>`. They are not covered here.

`-s` may be omitted when exactly one session exists, but it is picked
automatically and that gets ambiguous fast. **Always pass it.**

`upload` / `download` / `ls` target the VM's Jupyter Contents API. **Not Drive.**

---

## What breaks without a browser

`colab exec` calls `runtime.execute_code()` without `allow_stdin`, so it defaults
to `False` and any request for stdin from the kernel fails. (`allow_stdin=True`
is passed only by `colab auth` and `colab drivemount` — not even by `colab repl`
or `colab console`.)

Five things behave differently, and they fail in four *different* ways:

| Call | What actually happens | Fix |
| --- | --- | --- |
| `input()` | `StdinNotImplementedError`, immediately | Patch it **in the same cell** ([recipe](#input)) |
| `files.upload()` | **Hangs until the timeout and kills the whole run.** No later cell executes | Stage the file with `colab upload`, patch `files.upload` ([recipe](#filesupload)) |
| `files.download()` | **Silently succeeds and does nothing.** Returns in 0s, no error | Ignore it; use `colab download` before stopping ([details](#getting-your-results-back)) |
| `userdata.get()` | `TimeoutException` after 10 s. **Unobtainable from the CLI, by design** | Ship the key yourself ([recipe](#secrets-and-api-keys)) |
| `drive.mount()` | Prints an auth URL, then blocks on `/dev/tty` waiting for Enter | Wrap in a pty ([details](#google-drive)) |

`files.upload()` is the dangerous one: it does not raise, it waits for a browser
widget response that will never come, and takes the entire run down with it.
`files.download()` is the *sneaky* one: everything reports success and you are
left wondering where your file went.

### `input()`

Feed the answers in and replace `input`:

```python
# @title Supply the answers up front
import builtins
_ANSWERS = iter(["Alice", "30", "yes"])
def input(prompt=""):
    v = next(_ANSWERS)
    print(f"{prompt}{v}")      # echo it, the way Colab does
    return v
builtins.input = input

name = input("Your name? ")
age  = int(input("Age? "))
ok   = input("Confirm? (yes/no) ")
```

**The patch must be in the same cell as the calls.** Patching in an earlier cell
does nothing: ipykernel rebinds `builtins.input` to its own `raw_input` right
before each cell executes, overwriting whatever you set. (Measured, and
genuinely surprising the first time.)

### `files.upload()`

Stage the file first, then hand back a dict shaped like the real return value:

```bash
colab upload -s work ./data.csv /content/data.csv
```

```python
# @title Prelude: replace files.upload()
from google.colab import files as _f
def _fake_upload():
    out = {}
    for n in ["data.csv"]:
        with open(f"/content/{n}", "rb") as fh:
            out[n] = fh.read()
        print(f"Saving {n} to {n}")     # same log shape as the real thing
    return out
_f.upload = _fake_upload
```

Downstream cells need no edits:

```python
from google.colab import files
up = files.upload()
df = pd.read_csv(io.BytesIO(up["data.csv"]))
```

**Unlike `input()`, this one may live in a separate cell** — and even a separate
`colab exec`. `files` is a module and `upload` an attribute of it, so ipykernel's
rebinding (which only touches `builtins.input`) never sees it.

---

## Secrets and API keys

`userdata.get()` is `_message.blocking_request('GetSecret', …)`: it emits a
`colab_request` on iopub and waits for a `colab_reply` **from the browser
frontend** over the stdin channel. Nothing in the CLI answers that message, so
after 10 seconds you always get:

```
google.colab.userdata.TimeoutException: Requesting secret FOO timed out.
Secrets can only be fetched when running from the Colab UI.
```

This is not a bug to work around upstream — Colab Secrets live in the browser
session. **Ship the key yourself.**

### The general pattern

Write the keys to a JSON file, `colab upload` it, and patch `userdata.get`:

```bash
python3 - <<'PY'
import json, os
json.dump({k: os.environ[k] for k in ["GEMINI_API_KEY", "HF_TOKEN"]},
          open("/tmp/secrets.json", "w"))
PY
chmod 600 /tmp/secrets.json
colab upload -s work /tmp/secrets.json /content/.secrets.json
```

```python
# @title Prelude: patch userdata only when running headless
import json, os
_P = "/content/.secrets.json"
if os.path.exists(_P):
    from google.colab import userdata as _ud
    with open(_P) as _fh:
        _SECRETS = json.load(_fh)
    os.remove(_P)                      # no plaintext left on the VM
    def _get(key):
        if key not in _SECRETS:
            raise _ud.SecretNotFoundError(key)
        return _SECRETS[key]
    _ud.get = _get
    os.environ.update(_SECRETS)        # for libraries that read the env directly
else:
    pass                               # Colab UI: real Colab Secrets, untouched
```

Every later cell works unmodified, in both worlds:

```python
from google.colab import userdata
token = userdata.get("HF_TOKEN")
```

**Branch on the file's existence, never on the exception.** A
`try: userdata.get(...) except TimeoutException:` guard costs you a 10-second
stall *per key*. Checking for a file is instant and unambiguous.

Both branches are verified: no file → `userdata.get` stays stock and times out;
file present → the patch takes and the value comes through.

### Keep it out of the notebook

Better still, do not put the prelude in the `.ipynb` at all. Push it in as an
independent `colab exec`; because `userdata` is a module, the patch survives
across `exec` calls for the life of the session, and the notebook stays byte-for-byte
the same file that opens correctly in the browser.

```bash
python3 colab-cli/tools/colab_secrets.py -s work GEMINI_API_KEY   # patches kernel state only
colab exec -s work -f notebook.ipynb --timeout 1800                 # notebook unmodified
```

`colab-cli/tools/colab_secrets.py` does exactly that: resolve names, upload a
0600 JSON payload, run the prelude, delete the local temp file. It looks a name
up in this order, so **it works on any machine**:

1. the environment of the calling shell;
2. `colab-cli/.env` (or `--env-file PATH`) — the local option, gitignored;
   start from `colab-cli/.env.example`;
3. GitHub Codespaces' secret store, when present (see below).

```bash
python3 colab-cli/tools/colab_secrets.py -s work GEMINI_API_KEY MAPBOX_TOKEN
python3 colab-cli/tools/colab_secrets.py -s work HF_TOKEN=MY_LOCAL_NAME   # rename in transit
python3 colab-cli/tools/colab_secrets.py --list                           # names in .env and Codespaces, never values
```

**Run it once per session, before your first `colab exec`.** The patch lives in
kernel state: it persists across `exec` calls, and it disappears the moment you
`colab new` a fresh session. "It worked yesterday, now the key is missing" is
almost always a new session that never got the prelude.

### Never put a key in the code you execute

**`colab exec` appends both the executed code and its output, in plaintext, to
`~/.config/colab-cli/history/<session>.jsonl`.**

```bash
colab exec -s work <<< 'import os; os.environ["API_KEY"] = "sk-..."'   # ← leaked
```

`colab upload` records only the path, which is why keys travel as files. For the
same reason, **never `print` a key**: it lands in that log *and* gets baked into
`<name>_output.ipynb` on disk.

While debugging, check for presence, never value:

```bash
[ -n "$GEMINI_API_KEY" ] && echo set || echo "NOT SET"
```

And beware this seductive one-liner, which **prints the key** whenever it is set —
`:+` yields `set`, then `:-` yields *the value itself*:

```bash
echo "${GEMINI_API_KEY:+set}${GEMINI_API_KEY:-NOT SET}"    # ← don't
```

### On GitHub Codespaces

Codespaces secrets are dropped, base64-encoded, into
`/workspaces/.codespaces/shared/.env-secrets`. `/etc/profile.d/codespaces.sh`
only exports them **when `$SSH_CONNECTION` is set**, so a VS Code terminal (and
anything it spawns) often cannot see them as environment variables at all —
`colab-cli/tools/colab_secrets.py` reads the file directly to sidestep this.

A user-level Codespaces secret is delivered **only to the repositories you
explicitly share it with**. If it is not shared, it is absent from `.env-secrets`
and nothing inside the Codespace can recover it. Three layers, check top down:

| Layer | Check | If it fails |
| --- | --- | --- |
| GitHub sharing | `gh api /user/codespaces/secrets/<NAME>/repositories --jq '[.repositories[].full_name]\|join(", ")'` | Add the repo under Settings → Codespaces → Repository access, then **rebuild the Codespace** |
| Codespace | `python3 colab-cli/tools/colab_secrets.py --list` | Sharing added but never rebuilt — `.env-secrets` is written at creation time |
| Colab VM | `colab exec -s work --timeout 60 <<< 'import os; print("set" if os.environ.get("NAME") else "NOT SET")'` | You have not run `colab_secrets.py -s work NAME` for *this* session |

The third row is the one that catches people: a key being visible in your shell
says nothing about whether the VM has it.

---

## Getting your results back

**The VM filesystem is volatile. `/content` disappears the instant you stop the
session.** Retrieve everything first.

And again, because it is the single most common confusion: **`files.download()`
does nothing here.** It displays JavaScript that streams bytes over a browser
comm; with no browser, it returns in 0 s having accomplished nothing, without an
error. "Every cell passed but there is no file" is nearly always this. Leave the
call in your notebook — it works fine in the UI — and take the file with
`colab download` instead.

| Method | Use for |
| --- | --- |
| `print` to stdout | Numbers, text, small JSON — also preserved in `_output.ipynb` |
| `colab download` (**before** `colab stop`) | The default for real files |
| Write into mounted Drive | Anything that must outlive the session |

```bash
colab download -s work /content/model.pt ./model.pt
```

Measured properties:

- **Binaries survive intact.** 20 MB of random bytes, sha256 match, 2.9 s.
- **Directories are rejected** — `Cannot download a directory: /content/out`.
  Zip on the VM first:

```bash
colab exec -s work --timeout 300 <<'EOF'
import shutil
print(shutil.make_archive("/content/out_bundle", "zip", "/content/output"))
EOF
colab download -s work /content/out_bundle.zip ./out_bundle.zip
colab stop -s work
```

---

## Google Drive

### First decide whether you need it at all

**Default to `colab upload` / `colab download`.** Mounting requires a human in a
browser **once per session**, which stops any automated pipeline dead.

| Situation | Choice |
| --- | --- |
| Input is local, output comes back local | **upload / download** |
| Fully automated runs | **upload / download** |
| Input already lives in Drive | **mount** |
| Output must be shared with people or other tools | **mount** |
| Too many or too large files to transfer one at a time | **mount** |

In short: mount only when Drive is genuinely *where the data lives*. Moving bytes
to and from the VM does not require it.

### Mounting from a non-interactive shell

`colab drivemount` prints an auth URL and then blocks reading `/dev/tty`, so wrap
it in a pty — `colab-cli/tools/ptyrun.py` does this:

```bash
colab new -s work
python3 colab-cli/tools/ptyrun.py colab-cli/ws/dm.log colab-cli/ws/dm.in \
    colab drivemount -s work /content/drive &
sleep 30 && cat colab-cli/ws/dm.log        # the authorization URL appears here
# → approve in a browser
printf '\n' > colab-cli/ws/dm.in           # send the Enter it is waiting for
# → [colab] Credentials propagated. Resuming mount... / Mounted at /content/drive
```

Then use it normally, including from `colab upload` / `download`:

```bash
colab download -s work /content/drive/MyDrive/out.json ./out.json
colab upload   -s work ./in.csv /content/drive/MyDrive/in.csv
```

Measured:

- **The mount outlives the `drivemount` process.** It is a FUSE mount on the VM,
  so it stays valid for the whole session — other `colab exec` calls, other
  files, read and write.
- **`colab upload` / `download` accept Drive paths directly**, which effectively
  gives you local ↔ Drive transfer and sidesteps `/content`'s volatility.
- **Consent is required again for every new session.** The propagated credential
  is a per-VM ephemeral one (`dfs_ephemeral` / `gsession`) and the auth URL is
  hardcoded to `prompt=consent`. **A Drive-based workflow cannot be fully
  automated.**
- **You have 600 seconds to approve** (`INTERACTIVE_AUTOMATION_TIMEOUT_SEC`),
  after which the mount must be restarted.
- The CLI's own OAuth scopes are insufficient (it holds `drive.file`; mounting
  wants full `drive` plus `drive.activity.readonly`), which is why the
  credential-propagation dry-run fails and it falls back to browser consent.

Mechanically: the VM requests `authType: dfs_ephemeral`, the CLI intercepts that
and relays it with its own credentials to
`colab.research.google.com/tun/m/credentials-propagation/<endpoint>`
(`commands/automation.py`, `drivefs_hook`).

---

## Rules worth internalising

**1. Always pass `--timeout`.** The default is **30 seconds**, which is useless
for real work. It is a deadline on *total execution*, not an idle timeout —
printing progress every 5 seconds does not extend it. For a notebook it applies
**per cell** (each cell is its own `execute_code` call), not to the whole file.
Pick a generous number up front.

**2. Always `colab stop`.** Closing the terminal does not release the VM (on
0.6.0 a detached keep-alive daemon held it for up to 24 hours). Verify
with `colab sessions`, which shows the server's actual state — including sessions
you created in the browser (they appear with `?` as the name). The web UI's
Runtime → Manage sessions works too; it is the same backend.

On the free tier there is no monetary charge, but idle sessions burn your usage
allowance and get GPU requests denied. On Pro or Pay-As-You-Go they consume real
compute units. If you use Claude Code in this repository, a Stop hook
(`colab-cli/hooks/colab_session_guard.py`, registered in the repository's
`.claude/settings.json`) warns when the local session store still has entries.

**3. Errors do not stop the run — and the exit code lies.** A failing cell is
reported and execution continues to the next one. Worse, **`colab exec` still
exits 0.** Do not trust the exit status in a script; grep the log:

```bash
colab exec -s work -f notebook.ipynb --timeout 1800 2>&1 | tee run.log
grep -qE "^(Traceback|[A-Za-z]*Error:)" run.log && echo "a cell failed"
```

**4. Do not run TTY-dependent commands non-interactively.** `colab repl`,
`colab console`, `colab auth` and `colab drivemount` all assume a terminal and
will hang. `auth` and `drivemount` can be rescued with `colab-cli/tools/ptyrun.py`; `repl`
and `console` are interactive by definition, so there is nothing to rescue.

**5. If you genuinely need interactivity**, `colab console` connects a raw TTY to
a remote shell (tmux + bash) over WebSocket, bypassing the Jupyter kernel
protocol entirely. Start `python3` there and `input()` behaves normally. It is
not a way to run notebooks, but it is the escape hatch.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Missing required parameter: scope` / `Error 400` | Truncated auth URL. Use `colab-cli/auth.sh`, which never puts it through the terminal |
| `AttributeError: … has no attribute 'KernelClient'` | 0.6.x with `jupyter-kernel-client` ≥ 1.0. Re-run `colab-cli/setup.sh` (installs 0.7.5) |
| `TimeoutError: Timeout waiting for output` | `--timeout` too small. It is a total-time deadline; raise it |
| `StdinNotImplementedError` | `input()` in the notebook. Patch it in the same cell |
| Run hangs mid-notebook, then dies | `files.upload()`. Stage with `colab upload` and patch it |
| `TimeoutException: Requesting secret … timed out` | Colab Secrets are browser-only. Ship the key as a file and patch `userdata.get` |
| Every cell passed but the file is missing | `files.download()` is inert here. Use `colab download` before stopping |
| `Cannot download a directory` | Zip it on the VM first |
| `403`, keep-alive dies | Missing `colaboratory` scope (a box left unticked on the consent screen). Check `colab whoami`, then `colab-cli/auth.sh --force` |
| `Session … appears to be lost (404/401)` | The VM is gone. `colab new` again |

Logs: `colab log -s <name>` and `~/.config/colab-cli/colab.log`.

---

## Directory contents

See the file table in [`../README.md`](../README.md#whats-in-this-directory).

## References

- [googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)

---

*Behaviour measured against `google-colab-cli` 0.6.0; install and sign-in written
against 0.7.5, and the core run cycle re-checked on 0.7.5 (2026-11-07). Corrections welcome — especially if a release changes any of the
behaviours above.*
