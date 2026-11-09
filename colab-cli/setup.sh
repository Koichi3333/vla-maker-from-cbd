#!/usr/bin/env bash
# Install and prepare the Google Colab CLI for this repository.
#
#   colab-cli/setup.sh                 # install / repair, then say what is next
#
# The Claude Code Stop hook that warns about unstopped sessions needs no setup:
# it is registered for everyone in the repository's .claude/settings.json.
#
# Works the same in GitHub Codespaces (or any remote Linux box) and on a local
# macOS / Linux machine. On Windows, run it inside WSL.
#
# Safe to re-run: it only reinstalls when the installed version differs.
# Override the CLI version with COLAB_CLI_VERSION=<x.y.z>, or =latest.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null || dirname "$HERE")"

# The version this directory's tools and notes were checked against.
# 0.7.x pins its own jupyter-kernel-client; 0.6.x needs the pin added below.
COLAB_CLI_VERSION="${COLAB_CLI_VERSION:-0.7.5}"

for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[1m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[setup]\033[0m %s\n' "$*" >&2; }

# --- where are we? ----------------------------------------------------------
# "cloud" = the browser you will approve in is on another machine.
if [ "${CODESPACES:-}" = "true" ]; then
  ENV_KIND="codespaces"
elif [ -n "${SSH_CONNECTION:-}" ]; then
  ENV_KIND="remote"
elif [ "$(uname -s)" = "Darwin" ] || [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || grep -qi microsoft /proc/version 2>/dev/null; then
  ENV_KIND="local"
else
  ENV_KIND="remote"   # headless Linux with no display: treat like a remote box
fi
say "environment: $ENV_KIND"

# --- 1. uv ------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (https://astral.sh/uv) into ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
say "uv $(uv --version | awk '{print $2}')"

# --- 2. google-colab-cli ----------------------------------------------------
installed="$(colab version 2>/dev/null | awk '/Version/{print $2}' || true)"
if [ "$COLAB_CLI_VERSION" = "latest" ]; then
  spec="google-colab-cli"
else
  spec="google-colab-cli==$COLAB_CLI_VERSION"
fi
with=()
case "$COLAB_CLI_VERSION" in
  # 0.6.x left jupyter-kernel-client unpinned and broke on its 1.0 rename.
  0.5.*|0.6.*) with=(--with 'jupyter-kernel-client<1.0') ;;
esac

if [ -n "$installed" ] && { [ "$COLAB_CLI_VERSION" = "latest" ] || [ "$installed" = "$COLAB_CLI_VERSION" ]; }; then
  say "google-colab-cli $installed already installed"
else
  say "installing $spec${with:+ ${with[*]}}"
  uv tool install --force "$spec" "${with[@]}"
fi

if ! command -v colab >/dev/null 2>&1; then
  warn "'colab' is installed but not on PATH. Add this to your shell profile:"
  warn "  export PATH=\"$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin"):\$PATH\""
  exit 1
fi
say "$(colab version)"

# --- 3. working directory and agent skill -----------------------------------
mkdir -p "$HERE/ws"

# The official agent guide ships inside the CLI; regenerate it so it always
# matches the installed version (gitignored -- it is a build product).
skill_dir="$HERE/.claude/skills/colab-operator"
mkdir -p "$skill_dir"
if colab skill > "$skill_dir/SKILL.md" 2>/dev/null && head -1 "$skill_dir/SKILL.md" | grep -q '^---$'; then
  say "wrote colab-cli/.claude/skills/colab-operator/SKILL.md (from 'colab skill')"
else
  rm -f "$skill_dir/SKILL.md"
  warn "'colab skill' gave no usable output; skipped the Claude Code skill"
fi

# --- 4. what is next --------------------------------------------------------
echo
# Only ask whoami when a token exists: without one it starts the sign-in flow.
if [ -f "$HOME/.config/colab-cli/token.json" ] && colab whoami >/dev/null 2>&1 </dev/null; then
  say "already authenticated:"
  colab whoami 2>/dev/null | sed -n '1,3p' | sed 's/^/         /'
  echo
  say "next: colab-cli/smoke_test.sh     (allocates a CPU VM for ~1 min, then stops it)"
  echo "       then the VLA jobs on an A100 (paid Colab plan): bash colab/start.sh infer <name>"
  echo "       — see colab-cli/README.md for what they need and cost."
else
  say "next: authenticate once with"
  echo "         colab-cli/auth.sh"
  case "$ENV_KIND" in
    codespaces) echo "       It opens the sign-in page in the browser you are using for this Codespace." ;;
    remote)     echo "       It writes the sign-in link to colab-cli/ws/.auth/ so you never copy it out of the terminal." ;;
    local)      echo "       It opens the sign-in page in your default browser." ;;
  esac
fi
if [ "$ENV_KIND" = "codespaces" ]; then
  echo
  say "note: rebuilding/recreating the Codespace deletes ~/.local and ~/.config —"
  echo "       re-run colab-cli/setup.sh and colab-cli/auth.sh after a rebuild."
fi
