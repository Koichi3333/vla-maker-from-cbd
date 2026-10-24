#!/usr/bin/env bash
# Run after the Drive consent. Finishes the mount, waits for the environment setup, then starts the VM-side job detached.
#
#   bash colab/go.sh train <session name>
#   RUN=myrun STEPS=5000 DUP=40 bash colab/go.sh train <session name>
#   ACTION_HEAD=runs/myrun/action_head.safetensors bash colab/go.sh infer <session name>
#   bash colab/go.sh data <session name>         # rebuild the HumanML3D training data (colab/vm_data.sh)
#
# Variables passed to the VM (defaults at the top of colab/vm_*.sh): RUN STEPS DUP LR EVAL_GUIDES H1 ACTION_HEAD FK_ETA JOINTS
set -euo pipefail
source "$(dirname "$0")/common.sh"
JOB=${1:?train・infer・data のどれか}; NAME=${2:?セッション名}
job_ok $JOB
RUN=${RUN:-$JOB-$(date -u +%Y%m%d-%H%M)}

mount_drive_finish $NAME

# Turn the variables for the VM into JSON (keeps values with spaces such as "none fk" intact)
ENVS=$(DRIVE=$DRIVE RUN=$RUN python3 -c '
import json, os
keys = ["DRIVE", "RUN", "STEPS", "DUP", "LR", "EVAL_GUIDES", "H1", "ACTION_HEAD", "FK_ETA", "JOINTS"]
print(json.dumps({k: os.environ[k] for k in keys if os.environ.get(k)}))')

colab exec -s $NAME --timeout 900 <<PYEOF | tail -4
import json, os, subprocess, time
ENV = json.loads(r"""$ENVS""")
D = ENV["DRIVE"]
if "$JOB" != "data":
    assert os.path.isdir(f"{D}/ckpt/checkpoint-47000"), "checkpoint-47000 が Drive に見えない"
    assert os.path.isfile(f"{D}/data/b3_dataset.tar"), "b3_dataset.tar（HumanML3D の描画済みデータ）が Drive に見えない。先に colab/go.sh data"
for _ in range(90):
    if "SETUP_DONE" in open("/content/groot_setup.log").read(): break
    time.sleep(10)
assert "SETUP_DONE" in open("/content/groot_setup.log").read(), "setup_vm.sh が終わっていない（/content/groot_setup.log）"
log = open("$RUNDIR/job.log", "w")
subprocess.Popen(["bash", "$VM/colab/vm_$JOB.sh"], env={**os.environ, **ENV}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("[go] $JOB を起動:", ENV)
PYEOF

start_watch $NAME

echo
echo "  → 進み具合: tail -f /tmp/watch_$NAME.log（VLA_* の印が順に出る。終わりは VLA_DONE、異常は VLA_EXIT=非0）"
echo "  → VM が消えたとき: Drive の runs/$RUN/live/（ログと health.txt を 1 分ごとに写している）"
echo "  → 60 分で CLI がセッションを捨てたら: \$CLIPY tools/colab_reattach.py $NAME で付け直し、bash colab/watch.sh を張り直す"
echo "  → 席を外すなら: nohup setsid bash colab/autostop.sh $NAME work/$RUN '<期限>' > /tmp/autostop_$NAME.log 2>&1 < /dev/null &"
echo "  → 回収: colab download -s $NAME $RUNDIR/results.tgz work/$RUN/results.tgz（Drive の runs/$RUN/ にも写す）"
echo "  → **終わったら colab stop -s $NAME**"
