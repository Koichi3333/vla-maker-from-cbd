#!/usr/bin/env bash
# Safety net for when you step away (Codespace side): once VLA_EXIT shows up in the watcher log, or the deadline passes, collect the results and stop the session.
#   nohup setsid bash colab/autostop.sh <session name> <destination> "<deadline (a time for date -d)>" > /tmp/autostop_<name>.log 2>&1 < /dev/null &
# Example: bash colab/autostop.sh t1 work/t1 "now + 3 hours"
set -uo pipefail
source "$(dirname "$0")/common.sh"
NAME=${1:?セッション名}; DEST=${2:?回収先}; DEADLINE=$(date -d "${3:?期限}" +%s)
while :; do
  grep -q "VLA_EXIT\|appears to be lost" /tmp/watch_$NAME.log 2>/dev/null && { echo "$(date -u +%T) 終了の印"; break; }
  [ $(date +%s) -ge $DEADLINE ] && { echo "$(date -u +%T) 期限を過ぎた"; break; }
  sleep 30
done
mkdir -p $DEST
colab exec -s $NAME --timeout 120 <<'PY' 2>&1 | grep -v "new version\|colab update\|silence this"
import subprocess
print(subprocess.run("grep -E 'VLA_|TIMING' /content/run/job.log; cd /content/run && [ -f results.tgz ] || tar czf results.tgz --exclude=base --exclude=mix --exclude=ckpt --exclude=train_video *.log eval* *.json 2>/dev/null",
                     shell=True, capture_output=True, text=True).stdout)
PY
colab download -s $NAME /content/run/results.tgz $DEST/results.tgz 2>&1 | tail -1
colab stop -s $NAME 2>&1 | tail -1
stop_helpers $NAME
colab sessions 2>&1 | tail -1
$CLIPY $REPO/tools/colab_ccu.py 2>&1 | head -1
echo "$(date -u +%T) AUTOSTOP_DONE"
