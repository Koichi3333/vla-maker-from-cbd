#!/usr/bin/env bash
# Stop hook: warn if a colab session is still left running.
#
# It only reads the local session store, so there is no network round trip and it returns at once.
# `colab stop` removes the entry from the store, so
# "the store is not empty" means "forgot to stop", which is exactly the warning condition.
STORE="$HOME/.config/colab-cli/sessions.json"
[ -f "$STORE" ] || exit 0

names=$(jq -r 'keys | join(", ")' "$STORE" 2>/dev/null) || exit 0
[ -n "$names" ] || exit 0

jq -n --arg n "$names" '{
  systemMessage: ("⚠ Colab セッションが停止されていません: " + $n
    + "\n  停止: colab stop -s <name>   /   確認: colab sessions")
}'
