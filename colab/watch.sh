#!/usr/bin/env bash
# Watcher: every 2 minutes, send a light colab exec and record the last line of job.log and of the latest evaluation log (Codespace side).
#   nohup setsid bash colab/watch.sh <session name> > /tmp/watch_<name>.log 2>&1 < /dev/null &
# Left alone with keep-alive only, the allocation disappeared after about 25 minutes; an exec every 2 minutes got past that (docs/colab.md).
# Stops when the VM disappears or VLA_EXIT shows up. colab/start.sh and go.sh start it automatically.
NAME=${1:?セッション名}
while sleep 120; do
  out=$(colab exec -s $NAME --timeout 60 2>&1 <<'PYEOF'
import subprocess
print(subprocess.run("tail -1 /content/run/job.log 2>/dev/null; f=$(ls -t /content/run/*.log 2>/dev/null | grep -v job.log | head -1); "
                     "[ -n \"$f\" ] && tail -c 300 $f | tr '\r' '\n' | tail -1", shell=True, capture_output=True, text=True).stdout)
PYEOF
)
  echo "== $(date -u +%T) $(echo "$out" | grep -v '^$' | grep -v 'new version\|colab update\|silence this' | tr '\n' ' ')"
  echo "$out" | grep -q 'appears to be lost\|VLA_EXIT' && break
done
echo "== $(date -u +%T) 見張りを終了"
