#!/usr/bin/env bash
# Create an A100 session, send the code and data, and go as far as the Drive mount consent (Codespace side).
#
#   bash colab/start.sh train <session name>      # continued training (colab/vm_train.sh)
#   bash colab/start.sh infer <session name>      # inference and evaluation of the usable form (colab/vm_infer.sh)
#   bash colab/start.sh data <session name>       # rebuild the HumanML3D training data (colab/vm_data.sh)
#
#   VIDEO_DATA=<LeRobot dataset>  object-bearing data mixed into HumanML3D for training (default data/video_reach; must have objects/)
#   GOALS="<spec json> ..."         specs with goals to run at inference (default data/eval/object_eval_vlm_box.json)
#
# After consent, run colab/go.sh with the same arguments.
set -euo pipefail
source "$(dirname "$0")/common.sh"
JOB=${1:?train・infer・data のどれか}; NAME=${2:?セッション名}
job_ok $JOB
VIDEO_DATA=${VIDEO_DATA:-$REPO/data/video_reach}
GOALS=${GOALS:-$REPO/data/eval/object_eval_vlm_box.json}

preflight

echo "=== 資材の確認と梱包（Codespace 側）"
STAGE=/tmp/vla_stage_$NAME
rm -rf $STAGE && mkdir -p $STAGE/data
cp -r $REPO/vla $REPO/colab $STAGE/
cp -r $REPO/data/eval $REPO/data/stats $STAGE/data/
if [ $JOB = train ]; then
  python3 - "$VIDEO_DATA" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
info = json.loads((d / "meta/info.json").read_text())
n = len(list((d / "data").rglob("episode_*.parquet")))
assert n == info["total_episodes"], f"parquet が {n} 本、info.json は {info['total_episodes']} 本"
assert (d / "objects").is_dir(), "objects/ が無い（物を描くのに要る。vla/video_to_lerobot.py が書く）"
assert (d / "meta/split.json").exists(), "meta/split.json が無い（train / heldout の分け方）"
print(f"[local] {d.name}: {n} 本・{info['total_frames']} フレーム")
PY
  mkdir -p $STAGE/data/train_video
  cp -r "$VIDEO_DATA"/{data,meta,objects} $STAGE/data/train_video/
  rm -rf $STAGE/data/train_video/videos   # re-rendered on the VM
elif [ $JOB = infer ]; then
  mkdir -p $STAGE/data/goals
  for g in $GOALS; do
    python3 - "$g" <<'PY'
import json, sys
spec = json.loads(open(sys.argv[1]).read())
assert spec["items"], "items が空"
print(f"[local] {sys.argv[1]}: {len(spec['items'])} 場面、目標 = {'VLM の点（vlm_goal）' if 'vlm_goal' in spec['items'][0] else '物の本当の位置'}")
PY
    cp "$g" $STAGE/data/goals/
  done
fi
find $STAGE -name __pycache__ -prune -exec rm -rf {} +
tar czf /tmp/vla_bundle_$NAME.tgz -C $STAGE .
echo "  梱包: $(du -h /tmp/vla_bundle_$NAME.tgz | cut -f1)"

create_session $NAME
start_watch $NAME            # Keep sending exec while waiting for consent (the allocation has vanished during the wait before)

echo "=== 資材の upload と環境構築の開始"
upload_bundle $NAME /tmp/vla_bundle_$NAME.tgz
rm -rf $STAGE /tmp/vla_bundle_$NAME.tgz

echo "=== Drive マウント（認可 URL を待つ）"
mount_drive_start $NAME
echo
echo "  → 認可 URL: /tmp/dm_url.txt（ブラウザで http://localhost:8899 からも開ける）。期限 600 秒"
echo "  → 同意したら: bash colab/go.sh $JOB $NAME"
