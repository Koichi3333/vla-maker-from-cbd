#!/usr/bin/env bash
# Continued training (VM side; started by colab/go.sh train). The same recipe that produced arm H (the usable form's weights):
# from checkpoint-47000, mix object-bearing video data (data/train_video), duplicated DUP times, into HumanML3D (b3_dataset.tar in Drive),
# and train the action head for STEPS steps. Also runs the evaluation (closed loop with moved objects) and writes the weights out.
#
#   Markers: VLA_RENDER_OK / VLA_DATA_OK / VLA_BASE_OK / VLA_MIX_OK / VLA_TRAIN_DONE / VLA_EVAL_<name>_DONE /
#       VLA_WEIGHTS_VERIFIED / VLA_H1_DONE / VLA_DONE / VLA_EXIT=<exit code>
#
# Rough cost (A100, measured 2026-10): STEPS=5000 trains in about 46 minutes, about 1.5 hours and 9 CU in total. 30,000 steps train in about 4.6 hours.
set -euo pipefail
DRIVE=${DRIVE:?DRIVE を渡すこと}
RUN=${RUN:?RUN を渡すこと}
STEPS=${STEPS:-5000}
LR=${LR:-3e-5}
DUP=${DUP:-40}                    # Copies of the video data. 40 makes about 50% of training frames video (arm H); 2 makes about 5%
EVAL_GUIDES=${EVAL_GUIDES:-"none fk"}   # none = model alone, fk = the same inference-time guidance as the usable form
FK_ETA=${FK_ETA:-0.3}
H1=${H1:-1}                       # 1 = measure whether HumanML3D heldout got worse (H-1)
V=/content/vla; G=$V/vla; B=/content/b3; W=/content/run
OUT=$DRIVE/runs/$RUN
H2_FLAGS="--goal object --samples 8 --frames 200 --overlap 4 --attach --spec-hand --reach 0.52"
T0=$(date +%s)
t() { echo "TIMING $1 $(( $(date +%s) - T0 ))s"; }
mkdir -p $W $OUT

# ---------------------------------------------------------------- 0. Copy the logs to Drive every minute (so the cause can be traced even if the VM disappears)
L=$OUT/live
sync_live() {
  mkdir -p $L
  { echo "== $(date -u +%FT%TZ)"; free -m | sed -n 2p; df -h /content | tail -1
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    dmesg 2>/dev/null | grep -iE 'out of memory|killed process' | tail -3; } >> $L/health.txt 2>&1 || true
  rsync -a --max-size=5m --exclude='/mix/' --exclude='/base/' --exclude='/ckpt/' --exclude='/video_heldout/' \
    --include='*/' --include='*.log' --include='*.json' --include='*.txt' --exclude='*' $W/ $L/ 2>/dev/null || true
}
( while :; do sync_live; sleep 60; done ) &
MIRROR=$!
trap 'rc=$?; echo "VLA_EXIT=$rc"; kill $MIRROR 2>/dev/null; sync_live' EXIT
cd /content/Isaac-GR00T

# ---------------------------------------------------------------- 1. Rendering setup (MuJoCo on EGL)
uv pip install --python .venv/bin/python mujoco > $W/mujoco_install.log 2>&1
python3 - <<'PY'
import json, os
p = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(p):
    json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "/usr/lib64-nvidia/libEGL_nvidia.so.0"}}, open(p, "w"))
print("[egl] ICD ok")
PY

# ---------------------------------------------------------------- 2. Render the object-bearing video data (observation = body + object proxies)
VID=$V/data/train_video
MUJOCO_GL=egl .venv/bin/python $G/render_with_objects.py --dataset $VID --xml $G/humanoid.xml \
    --workers 8 --gl egl --only-missing > $W/render.log 2>&1
tail -1 $W/render.log
python3 - <<PY
import json, pathlib
w = pathlib.Path("$VID")
info = json.loads((w / "meta/info.json").read_text())
n = len(list((w / "videos").rglob("*.mp4")))
assert n == info["total_episodes"], (n, info["total_episodes"])
print(f"[render] {n} 本 / {info['total_frames']} フレーム")
PY
echo VLA_RENDER_OK; t render

# ---------------------------------------------------------------- 3. HumanML3D (rendered train / heldout)
if [ ! -d $B/heldout ]; then
  cp $DRIVE/data/b3_dataset.tar $W/b3_dataset.tar
  tar xf $W/b3_dataset.tar -C $(dirname $B) && rm -f $W/b3_dataset.tar
fi
echo VLA_DATA_OK; t data

# ---------------------------------------------------------------- 4. Starting point checkpoint-47000 (without optimizer etc.)
python3 - <<PY
import pathlib, shutil, time
src, dst = pathlib.Path("$DRIVE/ckpt/checkpoint-47000"), pathlib.Path("$W/base")
t0 = time.time()
for p in sorted(src.rglob("*")):
    if p.is_dir() or p.name in ("optimizer.pt", "scheduler.pt") or p.name.startswith("rng_state"):
        continue
    q = dst / p.relative_to(src)
    q.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, q)
print(f"[base] checkpoint-47000 を {time.time() - t0:.0f} 秒でコピー")
PY
echo VLA_BASE_OK; t base

# ---------------------------------------------------------------- 5. Mix (normalization statistics stay those of HumanML3D train)
.venv/bin/python $G/mix_datasets.py --hml $B/train --video $VID --dup $DUP --out $W/mix --heldout-out $W/video_heldout 2>&1 | tail -3
MUJOCO_GL=egl .venv/bin/python $G/groot_eval.py --check-dataset $W/mix $W/video_heldout 2>&1 | tail -4
echo VLA_MIX_OK; t mix

# ---------------------------------------------------------------- 6. Train (only the action head is trained, the N1.7 default)
echo "[train] HumanML3D + 動画 × $DUP、$STEPS ステップ、lr $LR"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python gr00t/experiment/launch_finetune.py \
  --base-model-path $W/base --dataset-path $W/mix \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path $G/groot_cbd_config.py \
  --num-gpus 1 --output-dir $W/ckpt --max-steps $STEPS \
  --learning-rate $LR --save-steps $STEPS --save-total-limit 1 \
  --global-batch-size 32 --dataloader-num-workers 8 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 > $W/train.log 2>&1
CK=$W/ckpt/checkpoint-$STEPS
echo VLA_TRAIN_DONE; t train

# ---------------------------------------------------------------- 7. Weights (trainable part only, bf16, about 3.2 GB) to Drive
.venv/bin/python $G/groot_extract_trainable.py --ckpt $CK --out $W/action_head.safetensors 2>&1 | tail -2
cp $W/action_head.safetensors $OUT/
cp $CK/trainer_state.json $OUT/
python3 $G/drive_verify.py --wait 2400 $OUT && echo VLA_WEIGHTS_VERIFIED \
  || echo "VLA_VERIFY_WARNING $OUT が Drive 側と一致しない。停止する前に確認すること"
t weights

# ---------------------------------------------------------------- 8. Evaluation: 7 heldout clips x 3 object positions (original / shifted 25 cm / mirrored) and H-4
for g in $EVAL_GUIDES; do
  n=eval_$g; GF="--guide none"; [ $g = fk ] && GF="--guide fk --fk-eta $FK_ETA"
  MUJOCO_GL=egl .venv/bin/python $G/groot_eval.py --ckpt $CK --train $B/train --heldout $B/heldout \
    --object-eval $V/data/eval/object_eval_heldout.json $H2_FLAGS $GF --out $W/$n/h2 > $W/$n.log 2>&1 \
    || echo "VLA_EVAL_FAILED $n（$W/$n.log）"
  grep "全体\|直した数え方\|H-4\|guidance\]\|^\[object\] [a-z]" $W/$n.log | grep -v "^\[guidance\] mode" | sed "s|^|[$n] |" || true
  echo VLA_EVAL_${n^^}_DONE; t $n
done

# ---------------------------------------------------------------- 9. H-1: has HumanML3D heldout got worse? (checkpoint-47000 is 9.35 deg)
if [ "$H1" = 1 ]; then
  MUJOCO_GL=egl .venv/bin/python $G/groot_eval.py --ckpt $CK --train $B/train --heldout $B/heldout \
    --out $W/eval_h1 --sweep 8:4:1 > $W/eval_h1.log 2>&1 || echo "VLA_EVAL_FAILED h1（$W/eval_h1.log）"
  grep "\[sweep\]" $W/eval_h1.log | sed "s|^|[h1] |" || true
  echo VLA_H1_DONE; t h1
fi

# ---------------------------------------------------------------- 10. Results to collect
python3 - <<PY
import json, pathlib
st = json.loads(pathlib.Path("$CK/trainer_state.json").read_text())
out = [dict(step=h["step"], loss=h.get("loss"), lr=h.get("learning_rate")) for h in st["log_history"] if "loss" in h]
pathlib.Path("$W/loss.json").write_text(json.dumps(out))
print("[loss]", len(out), "点  最後の 10 点の平均", round(sum(x["loss"] for x in out[-10:]) / max(len(out[-10:]), 1), 5))
PY
cd $W && tar czf $W/results.tgz loss.json *.log eval_* mix/meta/info.json 2>/dev/null || true
cp $W/results.tgz $OUT/ || true
echo "[bundle] $(du -h $W/results.tgz | cut -f1) → $OUT"
echo VLA_DONE
