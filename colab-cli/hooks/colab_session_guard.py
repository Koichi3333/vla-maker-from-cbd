#!/usr/bin/env python3
"""Claude Code Stop hook: warn when a Colab CLI session was left running.

Reads only the CLI's local session store, so it makes no network call and
returns at once. `colab stop` removes the entry, so "the store is not empty"
is exactly "a session was not stopped".

Registered for everyone in the repository's .claude/settings.json. Prints
nothing (and exits 0) when no session is left or the Colab CLI is not installed.
"""
import json
import os
import sys

STORE = os.path.expanduser("~/.config/colab-cli/sessions.json")

try:
    with open(STORE, encoding="utf-8") as fh:
        names = sorted(json.load(fh) or {})
except Exception:
    sys.exit(0)

if names:
    print(json.dumps({"systemMessage": (
        f"⚠ Colab session(s) still running: {', '.join(names)}\n"
        "  stop: colab stop -s <name>   /   check: colab sessions")}))
