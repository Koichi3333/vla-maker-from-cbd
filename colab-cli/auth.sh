#!/usr/bin/env bash
# Sign the Colab CLI in (one time per machine). See tools/auth.py for details.
#
#   colab-cli/auth.sh                 # open the sign-in page, then paste the code
#   colab-cli/auth.sh --serve 8765    # also serve a one-link page (port-forwarded setups)
#   colab-cli/auth.sh --url-only      # step 1 for agents: write the link, exit
#   colab-cli/auth.sh --code 4/0A...  # step 2: exchange the code
#   colab-cli/auth.sh --force         # sign in again
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"

# auth.py imports colab_cli, so it must run on the CLI's own interpreter.
PY=""
if command -v uv >/dev/null 2>&1; then
  PY="$(uv tool dir 2>/dev/null)/google-colab-cli/bin/python"
fi
if [ ! -x "$PY" ] && command -v colab >/dev/null 2>&1; then
  PY="$(head -1 "$(command -v colab)" | sed -n 's/^#!\(.*python[0-9.]*\).*/\1/p')"   # pipx / venv installs
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "[auth] google-colab-cli not found. Run colab-cli/setup.sh first." >&2
  exit 1
fi

"$PY" -I "$HERE/tools/auth.py" "$@" || exit $?

# --code / interactive success: show who we are and check the Colab scope.
case " $* " in *" --url-only "*) exit 0 ;; esac
if [ -f "$HOME/.config/colab-cli/token.json" ]; then
  out="$(colab whoami 2>&1 </dev/null || true)"
  printf '%s\n' "$out" | sed -n '1,3p'
  if printf '%s' "$out" | grep -q 'auth/colaboratory'; then
    echo "[auth] OK — 'colaboratory' scope present."
  else
    echo "[auth] 'colaboratory' scope missing — sessions will 403. Re-run with --force." >&2
    exit 1
  fi
fi
