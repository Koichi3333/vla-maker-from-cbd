# Shared part sourced by colab/start.sh, go.sh, watch.sh and autostop.sh (Codespace side).
# Why things are done this way (why the watcher starts right after creation, why the token is swapped every 45 minutes, etc.) is in docs/colab.md.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CLIPY=${CLIPY:-$HOME/.local/share/uv/tools/google-colab-cli/bin/python}   # Python of the uv tool that colab-cli is installed in
DRIVE=${DRIVE:-/content/drive/MyDrive/VLA/VLA_1_0}                        # Drive folder as seen from the VM (weights, rendered HumanML3D data)
VM=/content/vla        # Code and data on the VM
RUNDIR=/content/run    # Working directory on the VM (logs, evaluation and training output)
JOBS="train infer data"

job_ok() { [[ " $JOBS " == *" $1 "* ]] || { echo "JOB は train・infer・data のどれか: '$1'"; exit 2; }; }

# ---------------------------------------------------------------- Session
preflight() {
  echo "=== 残存セッションと CU 残高"
  colab sessions 2>&1 | tail -3
  $CLIPY $REPO/tools/colab_ccu.py 2>&1 | head -2
  echo "  → Codespace が止まると VM も道連れになる。長い実行の前に Codespace 側の keep-alive を確認すること"
}

create_session() {   # $1=name
  local NAME=$1
  local ACC="--gpu ${GPU:-A100}"
  [ "${GPU:-}" = cpu ] && ACC=""          # GPU=cpu for a CPU runtime (uploads etc.)
  echo "=== セッション作成（${GPU:-A100}）"
  colab new -s $NAME $ACC
  if [ "${SECRETS-HF_TOKEN}" != "" ]; then   # SECRETS= (empty) to pass none
    echo "=== ${SECRETS-HF_TOKEN}（exec より前に流す）"
    python3 $REPO/tools/colab_secrets.py -s $NAME ${SECRETS-HF_TOKEN} | tail -1
  fi
  echo "=== トークン更新ループ（45 分ごと。CLI は 60 分で切れるトークンを更新しない）"
  local OLD; OLD=$(pgrep -f "colab_reattach.py --refresh $NAME" || true)
  [ -n "$OLD" ] && kill $OLD || true
  nohup setsid bash -c "while sleep 2700; do $CLIPY $REPO/tools/colab_reattach.py --refresh $NAME || true; done" \
      > /tmp/refresh_$NAME.log 2>&1 < /dev/null &
}

start_watch() {   # $1=name. An exec every 2 minutes (with keep-alive alone the allocation disappears after about 25 minutes)
  local NAME=$1 OLD
  OLD=$(pgrep -f "colab/watch[.]sh $NAME" || true); [ -n "$OLD" ] && kill $OLD || true
  nohup setsid bash $REPO/colab/watch.sh $NAME > /tmp/watch_$NAME.log 2>&1 < /dev/null &
  echo "  見張り: /tmp/watch_$NAME.log"
}

stop_helpers() {   # $1=name
  pkill -f "colab_reattach.py --refresh $1" || true
  pkill -f "while sleep 2700; do .*$1" || true
  pkill -f "colab/watch[.]sh $1" || true
}

# ---------------------------------------------------------------- Payload
upload_bundle() {   # $1=name $2=tgz. colab upload fails on one large file, so send it in 20 MB parts
  local NAME=$1 TGZ=$2 TMP=/tmp/vla_upload_$1
  rm -rf $TMP && mkdir -p $TMP
  split -b 20m $TGZ $TMP/bundle.tgz.part
  colab exec -s $NAME --timeout 60 <<PYEOF | tail -1
import os
for d in ("$VM", "$RUNDIR"): os.makedirs(d, exist_ok=True)
print("[mkdir] $VM $RUNDIR")
PYEOF
  for f in $TMP/bundle.tgz.part*; do colab upload -s $NAME $f $VM/$(basename $f) | tail -1; done
  colab exec -s $NAME --timeout 300 <<PYEOF | tail -2
import subprocess
subprocess.run("cd $VM && cat bundle.tgz.part* > bundle.tgz && tar xzf bundle.tgz && rm -f bundle.tgz bundle.tgz.part*", shell=True, check=True)
subprocess.Popen("nohup bash $VM/colab/setup_vm.sh > /content/groot_setup.log 2>&1 &", shell=True)
print("[bundle] 展開した。環境構築を裏で開始（約 2 分）")
PYEOF
  rm -rf $TMP
}

# ---------------------------------------------------------------- Drive mount (needs consent in a browser for every session)
mount_drive_start() {   # $1=name. Writes the authorization URL to /tmp/dm_url.txt and http://localhost:8899, then stops
  local NAME=$1 try
  for try in 1 2; do   # It sometimes fails with "Connection was lost" before printing the URL. Retry once
    rm -f /tmp/dm.log /tmp/dm.in
    nohup setsid python3 $REPO/tools/ptyrun.py /tmp/dm.log /tmp/dm.in \
        colab drivemount -s $NAME /content/drive > /tmp/dm_wrap.log 2>&1 < /dev/null &
    for i in $(seq 1 24); do grep -q "accounts.google.com\|Connection was lost\|ptyrun\] exit" /tmp/dm.log 2>/dev/null && break; sleep 5; done
    grep -q "accounts.google.com" /tmp/dm.log && break
    echo "  マウント $try 回目: 認可 URL が出ない。やり直す"; sleep 5
  done
  grep -q "accounts.google.com" /tmp/dm.log || { echo "認可 URL が出ない。/tmp/dm.log を見ること"; exit 1; }
  python3 - <<'PY'
import html, pathlib, re
url = re.search(r"https://accounts\.google\.com/\S+", pathlib.Path("/tmp/dm.log").read_text()).group(0)
pathlib.Path("/tmp/dm_url.txt").write_text(url + "\n")
pathlib.Path("/tmp/dmpage").mkdir(exist_ok=True)
pathlib.Path("/tmp/dmpage/index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Drive consent</title>
<style>body{{font:16px/1.7 system-ui;margin:0;display:grid;place-items:center;min-height:100vh}}a{{display:inline-block;background:#2a6df5;
color:#fff;padding:14px 26px;border-radius:10px;text-decoration:none;font-weight:600}}</style>
<div><h1>Google Drive consent</h1><p>Approve, close this tab, then run colab/go.sh (within 600 s).</p>
<a href="{html.escape(url)}" target="_blank" rel="noopener">Authorize with Google</a></div>""")
PY
  local P; P=$(pgrep -f 'http[.]server 8899' || true); [ -n "$P" ] && kill $P || true
  nohup setsid python3 -m http.server 8899 --directory /tmp/dmpage > /tmp/dmpage.log 2>&1 < /dev/null &
  sleep 1
}

mount_drive_finish() {   # $1=name. Sends Enter after consent. On "mount failed", simply running it again goes through without consent
  local NAME=$1
  printf '\n' > /tmp/dm.in
  for i in $(seq 1 24); do grep -q "Mounted at\|mount failed" /tmp/dm.log 2>/dev/null && break; sleep 5; done
  if ! grep -q "Mounted at" /tmp/dm.log && grep -q "mount failed" /tmp/dm.log; then
    echo "  mount failed → colab drivemount をやり直す（同意は不要のはず）"
    rm -f /tmp/dm.log /tmp/dm.in
    nohup setsid python3 $REPO/tools/ptyrun.py /tmp/dm.log /tmp/dm.in colab drivemount -s $NAME /content/drive > /tmp/dm_wrap.log 2>&1 < /dev/null &
    for i in $(seq 1 36); do grep -q "Mounted at\|accounts.google.com\|mount failed\|ptyrun\] exit" /tmp/dm.log 2>/dev/null && break; sleep 5; done
  fi
  tail -3 /tmp/dm.log
  grep -q "Mounted at" /tmp/dm.log || { echo "マウントできていない。colab drivemount を再実行すること（2 回目は認可不要）"; exit 1; }
  local P; P=$(pgrep -f 'http[.]server 8899' || true); [ -n "$P" ] && kill $P || true
  rm -rf /tmp/dmpage /tmp/dm_url.txt
}
