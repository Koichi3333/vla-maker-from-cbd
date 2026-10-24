#!/usr/bin/env python3
"""Pass Codespaces secrets to a Colab session and replace userdata.get().

Colab Secrets (google.colab.userdata) cannot be read from the CLI, by design
(it waits for the frontend's colab_reply, so it always ends in TimeoutException).
Instead, carry the Codespaces secrets to the VM and replace userdata.get(),
so notebook code runs unmodified.

    python3 tools/colab_secrets.py -s work GEMINI_API_KEY MAPBOX_TOKEN
    python3 tools/colab_secrets.py -s work HF_TOKEN=GEMINI_API_KEY   # pass under a different name

Secret values reach the VM only through colab upload (which does not log the content).
They are never put in code passed to colab exec, because colab exec appends both the code and
its output in plain text to ~/.config/colab-cli/history/<session>.jsonl.
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile

ENV_SECRETS = "/workspaces/.codespaces/shared/.env-secrets"
REMOTE_PATH = "/content/.secrets.json"


def load_codespaces_secrets(path=ENV_SECRETS):
    """Read Codespaces secrets stored in base64.

    /etc/profile.d/codespaces.sh exports this file only in SSH sessions,
    so the VS Code terminal and Claude Code's child processes do not see them
    as environment variables. Read the file directly instead.
    """
    secrets = {}
    if not os.path.exists(path):
        return secrets
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                secrets[key] = base64.b64decode(value).decode("utf-8")
            except Exception:
                print(f"[secrets] {key}: base64 デコードに失敗。スキップ", file=sys.stderr)
    return secrets


def resolve(specs, pool):
    """Resolve a list of `NAME` or `REMOTE=LOCAL` into {remote name: value}."""
    resolved, missing = {}, []
    for spec in specs:
        remote, _, local = spec.partition("=")
        local = local or remote
        value = os.environ.get(local) or pool.get(local)
        if value is None:
            missing.append(local)
        else:
            resolved[remote] = value
    return resolved, missing


PRELUDE = '''# @title 前口上: Colab Secrets (userdata) を Codespaces の Secret で差し替える
import json, os
from google.colab import userdata as _ud

with open({remote!r}) as _fh:
    _SECRETS = json.load(_fh)
os.remove({remote!r})            # VM 上に平文を残さない

def _get(key):
    if key not in _SECRETS:
        raise _ud.SecretNotFoundError(key)
    return _SECRETS[key]

_ud.get = _get
os.environ.update(_SECRETS)      # os.environ 経由で読むライブラリにも通す
print("[secrets] userdata.get() available for:", sorted(_SECRETS))
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--session", help="Colab セッション名")
    ap.add_argument("names", nargs="*", metavar="NAME[=SOURCE]",
                    help="渡す Secret 名。REMOTE=LOCAL で名前を付け替えられる")
    ap.add_argument("--list", action="store_true",
                    help="値は出さず、渡せる Secret 名だけを表示して終了")
    args = ap.parse_args()

    pool = load_codespaces_secrets()
    if args.list:
        for name in sorted(pool):
            print(name)
        return 0
    if not args.session or not args.names:
        ap.error("-s/--session と 1つ以上の NAME が要る（一覧だけなら --list）")

    resolved, missing = resolve(args.names, pool)
    if missing:
        print(f"[secrets] 見つからない: {', '.join(missing)}", file=sys.stderr)
        print(f"[secrets] 利用可能な名前は --list で確認できる", file=sys.stderr)
        return 1

    tmp_dir = tempfile.mkdtemp(prefix="colab-secrets-")
    payload = os.path.join(tmp_dir, "secrets.json")
    prelude = os.path.join(tmp_dir, "prelude.py")
    try:
        # Create it with 0600 first, then write (no window between creation and chmod for others to open it)
        fd = os.open(payload, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(resolved, fh)
        with open(prelude, "w", encoding="utf-8") as fh:
            fh.write(PRELUDE.format(remote=REMOTE_PATH))

        for cmd in (
            ["colab", "upload", "-s", args.session, payload, REMOTE_PATH],
            ["colab", "exec", "-s", args.session, "-f", prelude, "--timeout", "120"],
        ):
            if subprocess.call(cmd) != 0:
                print(f"[secrets] 失敗: {' '.join(cmd[:4])}", file=sys.stderr)
                return 1
    finally:
        for path in (payload, prelude):
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(tmp_dir)

    print(f"[secrets] {len(resolved)} 件を '{args.session}' に渡した。"
          " 以降のセルは userdata.get() をそのまま使える。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
