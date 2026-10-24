#!/usr/bin/env bash
# Upload the weights in Drive to Hugging Face (Codespace side). A CPU session sends Drive -> VM -> HF directly (not through the Codespace).
#
#   bash colab/hf_upload.sh start <session name>   # CPU session -> stops at the Drive mount consent URL
#   bash colab/hf_upload.sh go <session name>      # after consent: sends the token and starts the VM side (colab/vm_hf_upload.py) detached
#   colab stop -s <session name>                    # after you see VLA_HF_VERIFIED
#
#   HF_REPO=<user>/<repo> (default Koichi3333/vla-maker-from-cbd). The repository is created private (an existing one keeps its visibility)
#   The write token is read from ~/.hf_write.json ({"HF_TOKEN": "..."}, mode 600). It is sent as a file, so the value stays out of colab's records
set -euo pipefail
source "$(dirname "$0")/common.sh"
STEP=${1:?start か go}; NAME=${2:?セッション名}
HF_REPO=${HF_REPO:-Koichi3333/vla-maker-from-cbd}
TOKEN_FILE=${TOKEN_FILE:-$HOME/.hf_write.json}

case $STEP in
start)
  [ -f "$TOKEN_FILE" ] || { echo "$TOKEN_FILE が無い"; exit 1; }
  preflight
  GPU=cpu SECRETS= create_session $NAME
  start_watch $NAME
  echo "=== コードとモデルカードを送る"
  colab exec -s $NAME --timeout 60 <<PYEOF | tail -1
import os
for d in ("$VM/colab", "$VM/hf", "$RUNDIR"): os.makedirs(d, exist_ok=True)
print("[mkdir] ok")
PYEOF
  colab upload -s $NAME $REPO/colab/vm_hf_upload.py $VM/colab/vm_hf_upload.py | tail -1
  for f in $REPO/hf/*; do colab upload -s $NAME "$f" $VM/hf/$(basename "$f") | tail -1; done
  echo "=== Drive マウント（認可 URL を待つ）"
  mount_drive_start $NAME
  echo
  echo "  → 認可 URL: /tmp/dm_url.txt。期限 600 秒"
  echo "  → 同意したら: bash colab/hf_upload.sh go $NAME"
  ;;
go)
  mount_drive_finish $NAME
  echo "=== トークンをファイルで送る（VM 側は読み込んだ直後に消す）"
  colab upload -s $NAME "$TOKEN_FILE" /content/.hf_write.json | tail -1
  colab exec -s $NAME --timeout 60 <<PYEOF | tail -2
import os, subprocess
os.chmod("/content/.hf_write.json", 0o600)
log = open("$RUNDIR/job.log", "w")
env = {**os.environ, "DRIVE": "$DRIVE", "HF_REPO": "$HF_REPO"}
subprocess.Popen(["bash", "-c", "python3 $VM/colab/vm_hf_upload.py; echo VLA_EXIT=\$?"], env=env,
                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("[go] vm_hf_upload.py を起動 → $HF_REPO")
PYEOF
  start_watch $NAME
  echo
  echo "  → 進み具合: tail -f /tmp/watch_$NAME.log（VLA_HF_STAGED → VLA_HF_UPLOADED → VLA_HF_VERIFIED → VLA_EXIT=0）"
  echo "  → **終わったら colab stop -s $NAME**"
  ;;
*) echo "start か go"; exit 2 ;;
esac
