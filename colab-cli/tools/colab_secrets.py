#!/usr/bin/env python3
"""Hand API keys to a Colab CLI session and make `userdata.get()` work there.

Colab Secrets (`google.colab.userdata`) cannot be read from the CLI at all:
`userdata.get()` waits for a reply from the browser frontend and always times
out after 10 s. So the keys travel from *this* machine instead, and
`userdata.get()` is replaced in the running kernel -- the notebook itself is
not modified and still uses the real Colab Secrets when opened in the browser.

    python3 colab-cli/tools/colab_secrets.py --list                 # names only, never values
    python3 colab-cli/tools/colab_secrets.py -s vla HF_TOKEN
    python3 colab-cli/tools/colab_secrets.py -s vla HF_TOKEN=MY_HF   # rename in transit

Short-lived keys (create one for a run, delete it afterwards) can be kept in
colab-cli/.env only while they are needed:

    python3 colab-cli/tools/colab_secrets.py --set HF_TOKEN     # hidden prompt; never echoed
    python3 colab-cli/tools/colab_secrets.py --unset HF_TOKEN   # remove it from colab-cli/.env

--set reads the value without echoing it (or one line from stdin when stdin is
not a terminal), so it stays out of the shell history and the screen.

colab/start.sh runs it for you (HF_TOKEN by default; SECRETS=... to change).
Local scripts that need a key on this machine (vla/vlm_target.py) call
find_secret(NAME), which looks in the same places.

Where a NAME is looked up, first match wins:

  1. the environment of this process         (local: `export GEMINI_API_KEY=...`)
  2. colab-cli/.env, or the file in --env-file (local: KEY=value lines, gitignored)
  3. GitHub Codespaces secrets                (cloud: Settings -> Codespaces -> Secrets)

Run it once per session, after `colab new` and before the first `colab exec`.
The values reach the VM only through `colab upload`, which logs the path but
not the content. They never appear in code passed to `colab exec`: that command
appends both the code and its output to ~/.config/colab-cli/history/<session>.jsonl
in plain text.
"""

import argparse
import base64
import getpass
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENV_FILE = os.path.join(HERE, ".env")
CODESPACES_SECRETS = "/workspaces/.codespaces/shared/.env-secrets"
REMOTE_PATH = "/content/.secrets.json"


def load_env_file(path):
    """KEY=value lines; `export ` prefixes, comments and surrounding quotes allowed."""
    found = {}
    if not path or not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key and value:
                found[key] = value
    return found


def load_codespaces_secrets(path=CODESPACES_SECRETS):
    """Codespaces secrets, stored base64-encoded.

    /etc/profile.d/codespaces.sh only exports these when $SSH_CONNECTION is
    set, so a VS Code terminal (and anything it starts, Claude Code included)
    often cannot see them as environment variables. Read the file directly.
    """
    found = {}
    if not os.path.exists(path):
        return found
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                found[key] = base64.b64decode(value).decode("utf-8")
            except Exception:
                print(f"[secrets] {key}: could not base64-decode; skipped", file=sys.stderr)
    return found


def sources(env_file=DEFAULT_ENV_FILE):
    """Where keys are looked up, in order: environment, .env file, Codespaces secrets."""
    return [
        ("environment", dict(os.environ)),
        (os.path.relpath(env_file), load_env_file(env_file)),
        ("Codespaces secrets", load_codespaces_secrets()),
    ]


def find_secret(name, env_file=DEFAULT_ENV_FILE):
    """The value of one key from the first place that has it, or None. Never print it."""
    for _, pool in sources(env_file):
        if pool.get(name):
            return pool[name]
    return None


def _env_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines()


def _is_line_for(line, name):
    key = line.strip().removeprefix("export ").split("=", 1)[0].strip()
    return "=" in line and not line.lstrip().startswith("#") and key == name


def _write_private(path, lines):
    """Replace the file atomically, readable by the owner only."""
    tmp = path + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def set_secret(name, path):
    """Store NAME in the .env file, read without echo. Prints the length only."""
    if sys.stdin.isatty():
        value = getpass.getpass(f"{name} (input is hidden): ")
    else:
        value = sys.stdin.readline()
    value = value.strip()
    if not value:
        print(f"[secrets] nothing entered; {name} not saved", file=sys.stderr)
        return 1
    lines = [l for l in _env_lines(path) if not _is_line_for(l, name)]
    lines.append(f"{name}={value}")
    _write_private(path, lines)
    print(f"[secrets] saved {name} (len={len(value)}) to {os.path.relpath(path)} (mode 600). "
          f"Delete it when done: --unset {name}")
    return 0


def unset_secret(name, path):
    """Remove NAME from the .env file; delete the file if nothing but comments is left."""
    lines = _env_lines(path)
    kept = [l for l in lines if not _is_line_for(l, name)]
    if len(kept) == len(lines):
        print(f"[secrets] {name} is not in {os.path.relpath(path)}")
    elif any(l.strip() and not l.lstrip().startswith("#") for l in kept):
        _write_private(path, kept)
        print(f"[secrets] removed {name} from {os.path.relpath(path)}")
    else:
        os.remove(path)
        print(f"[secrets] removed {name}; {os.path.relpath(path)} held nothing else and was deleted")
    if os.environ.get(name):
        print(f"[secrets] note: {name} is still exported in this shell (unset {name})")
    if name in load_codespaces_secrets():
        print(f"[secrets] note: {name} is also a Codespaces secret (GitHub settings)")
    return 0


def resolve(specs, pools):
    """`NAME` or `REMOTE=LOCAL` -> {remote name: value}, plus the names not found."""
    resolved, missing = {}, []
    for spec in specs:
        remote, _, local = spec.partition("=")
        local = local or remote
        for _, pool in pools:
            if pool.get(local):
                resolved[remote] = pool[local]
                break
        else:
            missing.append(local)
    return resolved, missing


PRELUDE = '''# @title Prelude: serve userdata.get() from keys shipped by colab_secrets.py
import json, os
from google.colab import userdata as _ud

with open({remote!r}) as _fh:
    _SECRETS = json.load(_fh)
os.remove({remote!r})            # leave no plaintext on the VM

def _get(key):
    if key not in _SECRETS:
        raise _ud.SecretNotFoundError(key)
    return _SECRETS[key]

_ud.get = _get
os.environ.update(_SECRETS)      # for libraries that read os.environ directly
print("[secrets] userdata.get() available for:", sorted(_SECRETS))
'''


# colab-cli 0.7.5 can hang for good after sending an exec (the websocket connects late and the reply is
# missed; --timeout never fires), and then the code may not have run on the VM. So every call has a hard
# deadline and the pair is retried. Upload and prelude together are idempotent: each try sends the file again.
DEADLINE = int(os.environ.get("COLAB_SECRETS_DEADLINE", "180"))
TRIES = 3


def ship(session, payload, prelude):
    for attempt in range(1, TRIES + 1):
        for cmd in (
            ["colab", "upload", "-s", session, payload, REMOTE_PATH],
            ["colab", "exec", "-s", session, "-f", prelude, "--timeout", "120"],
        ):
            try:
                ok = subprocess.run(cmd, timeout=DEADLINE).returncode == 0
            except subprocess.TimeoutExpired:
                print(f"[secrets] no reply within {DEADLINE} s: {' '.join(cmd[:4])}", file=sys.stderr)
                ok = False
            if not ok:
                break
        else:
            return True
        if attempt < TRIES:
            print(f"[secrets] retrying ({attempt + 1}/{TRIES})", file=sys.stderr)
    print(f"[secrets] failed after {TRIES} tries", file=sys.stderr)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--session", help="Colab CLI session name")
    ap.add_argument("names", nargs="*", metavar="NAME[=SOURCE]",
                    help="secret names to ship; REMOTE=LOCAL renames in transit")
    ap.add_argument("--env-file", default=DEFAULT_ENV_FILE,
                    help="KEY=value file to read (default: colab-cli/.env)")
    ap.add_argument("--list", action="store_true",
                    help="print the names available from the .env file and Codespaces, never the values")
    ap.add_argument("--set", metavar="NAME",
                    help="store NAME in the .env file; the value is read without echo")
    ap.add_argument("--unset", metavar="NAME", help="remove NAME from the .env file")
    args = ap.parse_args()

    if args.set:
        return set_secret(args.set, args.env_file)
    if args.unset:
        return unset_secret(args.unset, args.env_file)

    pools = sources(args.env_file)

    if args.list:
        for label, pool in pools[1:]:
            print(f"# {label}: {'(none)' if not pool else ''}")
            for name in sorted(pool):
                print(name)
        print("# plus anything exported in this shell's environment")
        return 0
    if not args.session or not args.names:
        ap.error("need -s/--session and at least one NAME (or --list)")

    resolved, missing = resolve(args.names, pools)
    if missing:
        print(f"[secrets] not found: {', '.join(missing)}", file=sys.stderr)
        print("[secrets] looked in: environment, "
              f"{os.path.relpath(args.env_file)}, Codespaces secrets (see --list)", file=sys.stderr)
        return 1

    tmp_dir = tempfile.mkdtemp(prefix="colab-secrets-")
    payload = os.path.join(tmp_dir, "secrets.json")
    prelude = os.path.join(tmp_dir, "prelude.py")
    try:
        # Create as 0600 before writing, so there is no window where others can read it.
        fd = os.open(payload, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(resolved, fh)
        with open(prelude, "w", encoding="utf-8") as fh:
            fh.write(PRELUDE.format(remote=REMOTE_PATH))

        if not ship(args.session, payload, prelude):
            return 1
    finally:
        for path in (payload, prelude):
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(tmp_dir)

    print(f"[secrets] shipped {len(resolved)} key(s) to '{args.session}'. "
          "Later cells can call userdata.get() / os.environ unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
