#!/usr/bin/env python3
"""Re-attach an allocation (VM) still alive on the server to a local session name.

    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_reattach.py vla2 [endpoint]
    ~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_reattach.py --refresh vla2

**When to use it:** `colab exec` returns "Session 'X' not found", but `colab sessions` still lists
`[?] gpu-a100-... | Hardware: A100` (observed 2026-09-13).

**The cause is the runtime proxy token expiring (1 hour)** (identified 2026-09-14). The CLI keeps using the token it
got from `colab new`, so a `colab exec` after an hour gets a 401, calls `prune_session()`,
and **deletes the local record and stops the keep-alive daemon**. The kernel is not gone (the old kernel process is still alive).
Every call to `list_assignments()` returns a fresh token (`tokenExpiresInSeconds` 3600).

**`--refresh` gets ahead of it.** For a session that runs past an hour, swapping just the token every 50 minutes
keeps the record, and the kernel (os.environ set by `colab_secrets.py`, etc.) stays usable.

The rest is for after the record is already gone. The next `colab exec`
gets a 401, calls `prune_session()`, and **deletes the local record and stops the keep-alive daemon**.
The VM itself stays allocated and keeps billing, and a detached training process keeps running. But with no record,
neither `colab stop -s X` nor the Stop hook works (invisible billing).

This rebuilds the SessionState the same way `colab new` does and starts keep-alive.
**It does not take a new allocation.** kernel_id is left empty, so the next `colab exec` creates a new kernel.
The kernel's state (os.environ set by `colab_secrets.py`, etc.) is gone, so send it again.
"""
import sys

from colab_cli.commands.session import spawn_keep_alive
from colab_cli.common import state
from colab_cli.state import SessionState


def refresh(name):
    """Keep the local record and swap only the runtime proxy token for a new one.

    kernel_id is kept, so you can continue in the same kernel (os.environ, variables).
    """
    s = state.store.get(name)
    if not s:
        raise SystemExit(f"'{name}' がローカルに無い。--refresh ではなく付け直す（引数なし）こと")
    a = [a for a in state.client.list_assignments() if a.endpoint == s.endpoint]
    if not a:
        raise SystemExit(f"'{name}' の割当（{s.endpoint}）がサーバーに無い")
    s.token = a[0].runtime_proxy_info.token
    state.store.add(s)
    state.history.log_event(name, "token_refreshed", {"endpoint": s.endpoint})
    print(f"[refresh] '{name}' のトークンを更新（期限 {a[0].runtime_proxy_info.token_expires_in_seconds} 秒）")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--refresh":
        return refresh(sys.argv[2])
    name = sys.argv[1]
    if state.store.get(name):
        raise SystemExit(f"'{name}' はローカルに既にある。付け直す必要は無い")
    asg = state.client.list_assignments()
    known = {s.endpoint for s in state.store.list().values()}
    cand = [a for a in asg if a.endpoint not in known]
    if len(sys.argv) > 2:
        cand = [a for a in cand if a.endpoint == sys.argv[2]]
    if len(cand) != 1:
        for a in asg:
            print(f"  {a.endpoint}  {getattr(a, 'accelerator', '')}")
        raise SystemExit(f"ローカルに無い割当が {len(cand)} 件。endpoint を第2引数で指定すること")
    a = cand[0]
    s = SessionState(name=name, token=a.runtime_proxy_info.token, url=a.runtime_proxy_info.url,
                     endpoint=a.endpoint, variant=str(getattr(a.variant, "name", a.variant)),
                     accelerator=str(getattr(a.accelerator, "value", a.accelerator)))
    state.store.add(s)             # The daemon reads the store right after it starts, so write it first (same as colab new)
    s.keep_alive_pid = spawn_keep_alive(a.endpoint, name, auth_provider=state.auth_provider,
                                        config_path=state.config_path)
    state.store.add(s)
    state.history.log_event(name, "session_created", {"endpoint": a.endpoint, "note": "reattached"})
    print(f"[reattach] '{name}' -> {a.endpoint}（keep-alive pid {s.keep_alive_pid}）")


if __name__ == "__main__":
    main()
