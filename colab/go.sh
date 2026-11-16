#!/usr/bin/env bash
# Run after the Drive consent. Finishes the mount, waits for the environment setup, then starts the VM-side job detached.
#
#   bash colab/go.sh train <session name>
#   RUN=myrun STEPS=5000 DUP=40 bash colab/go.sh train <session name>
#   ACTION_HEAD=runs/myrun/action_head.safetensors bash colab/go.sh infer <session name>
#   bash colab/go.sh data <session name>         # rebuild the HumanML3D training data (colab/vm_data.sh)
#
#   WEIGHTS=drive bash colab/go.sh infer <session name>   # read checkpoint-47000 / arm H from Drive instead of Hugging Face
#
# Variables passed to the VM (defaults at the top of colab/vm_*.sh): RUN WEIGHTS STEPS DUP LR EVAL_GUIDES H1 ACTION_HEAD FK_ETA JOINTS
set -euo pipefail
source "$(dirname "$0")/common.sh"
JOB=${1:?train・infer・data のどれか}; NAME=${2:?セッション名}
job_ok $JOB
RUN=${RUN:-$JOB-$(date -u +%Y%m%d-%H%M)}

mount_drive_finish $NAME

# Turn the variables for the VM into JSON (keeps values with spaces such as "none fk" intact)
ENVS=$(DRIVE=$DRIVE RUN=$RUN python3 -c '
import json, os
keys = ["DRIVE", "RUN", "WEIGHTS", "STEPS", "DUP", "LR", "EVAL_GUIDES", "H1", "ACTION_HEAD", "FK_ETA", "JOINTS"]
print(json.dumps({k: os.environ[k] for k in keys if os.environ.get(k)}))')

# Hard local deadline: colab-cli 0.7.5 can hang after sending when the websocket connects late, ignoring --timeout
# (the code then never runs on the VM). The watcher starts either way so the allocation is not lost while you retry.
LAUNCHED=1
timeout -k 5 960 colab exec -s $NAME --timeout 900 <<PYEOF | tail -4 || LAUNCHED=0
import json, os, subprocess, time
ENV = json.loads(r"""$ENVS""")
D = ENV["DRIVE"]
W = ENV.get("WEIGHTS", "hf")
assert W in ("hf", "drive"), f"WEIGHTS は hf か drive: {W!r}"
if "$JOB" != "data" and W == "drive":
    assert os.path.isdir(f"{D}/ckpt/checkpoint-47000"), "checkpoint-47000 が Drive に見えない（WEIGHTS=hf なら Hugging Face から取る）"
if "$JOB" != "data" and W == "hf":
    # Check access with one small file before a ~15 GB download: the repository is gated (accept its terms on HF first)
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download("Koichi3333/vla-maker-from-cbd", "checkpoint-47000/config.json", token=os.environ.get("HF_TOKEN"))
    except Exception as e:
        raise SystemExit(f"Koichi3333/vla-maker-from-cbd を読めない（{type(e).__name__}）。HF_TOKEN のアカウントで "
                         "https://huggingface.co/Koichi3333/vla-maker-from-cbd の条件に同意したか確かめる。Drive から読むなら WEIGHTS=drive")
if "$JOB" != "data":
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
[ $LAUNCHED = 1 ] || echo "  !! exec が返らなかった。colab download -s $NAME $RUNDIR/job.log /tmp/job.log で起動を確かめ、無ければ go.sh をもう一度"

echo
echo "  → 進み具合: tail -f /tmp/watch_$NAME.log（VLA_* の印が順に出る。終わりは VLA_DONE、異常は VLA_EXIT=非0）"
echo "  → VM が消えたとき: Drive の runs/$RUN/live/（ログと health.txt を 1 分ごとに写している）"
echo "  → 60 分で CLI がセッションを捨てたら: \$CLIPY colab-cli/tools/colab_reattach.py $NAME で付け直し、bash colab/watch.sh を張り直す"
echo "  → 席を外すなら: nohup setsid bash colab/autostop.sh $NAME work/$RUN '<期限>' > /tmp/autostop_$NAME.log 2>&1 < /dev/null &"
echo "  → 回収: colab download -s $NAME $RUNDIR/results.tgz work/$RUN/results.tgz（Drive の runs/$RUN/ にも写す）"
echo "  → **終わったら colab stop -s $NAME**"
# A short-lived key kept in colab-cli/.env is no longer needed here: the job got it from the kernel's environment
if grep -qs '^HF_TOKEN=' "$REPO/colab-cli/.env" && [ $LAUNCHED = 1 ]; then
  echo "  → ジョブは HF_TOKEN を持って起動済み。手元から消すなら: python3 colab-cli/tools/colab_secrets.py --unset HF_TOKEN（使い捨てなら、終わった後に Hugging Face でトークン自体も削除）"
fi
