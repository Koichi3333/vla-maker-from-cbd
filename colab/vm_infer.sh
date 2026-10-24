#!/usr/bin/env bash
# Inference and evaluation of the usable form (VM side; started by colab/go.sh infer). No training.
# Swap checkpoint-47000's action head for ACTION_HEAD and run closed loop with inference-time guidance (fk), once per goal spec.
# If the spec has vlm_goal, the goal is the VLM's point; otherwise the object's true position. The object is always drawn, and "held" always judged, at its true position.
#
#   Markers: VLA_DATA_OK / VLA_BASE_OK / VLA_EVAL_<name>_DONE (per spec) / VLA_DONE / VLA_EXIT=<exit code>
#
# Rough cost (A100): one spec (7 scenes x 3 conditions) takes about 20 minutes, about 1.4 CU.
set -euo pipefail
DRIVE=${DRIVE:?DRIVE を渡すこと}
RUN=${RUN:?RUN を渡すこと}
ACTION_HEAD=${ACTION_HEAD:-v12e/action_head_bf16_v12e_armH.safetensors}   # Path relative to $DRIVE in Drive. Default is arm H (the usable form)
FK_ETA=${FK_ETA:-0.3}
V=/content/vla; G=$V/vla; B=/content/b3; W=/content/run
OUT=$DRIVE/runs/$RUN
H2_FLAGS="--goal object --samples 8 --frames 200 --overlap 4 --attach --spec-hand --reach 0.52 --guide fk --fk-eta $FK_ETA"
T0=$(date +%s)
t() { echo "TIMING $1 $(( $(date +%s) - T0 ))s"; }
mkdir -p $W $OUT

L=$OUT/live
sync_live() {
  mkdir -p $L
  { echo "== $(date -u +%FT%TZ)"; free -m | sed -n 2p
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; } >> $L/health.txt 2>&1 || true
  rsync -a --max-size=5m --exclude='/base/' --include='*/' --include='*.log' --include='*.json' --include='*.txt' --exclude='*' $W/ $L/ 2>/dev/null || true
}
( while :; do sync_live; sleep 60; done ) &
MIRROR=$!
trap 'rc=$?; echo "VLA_EXIT=$rc"; kill $MIRROR 2>/dev/null; sync_live' EXIT
cd /content/Isaac-GR00T

uv pip install --python .venv/bin/python mujoco > $W/mujoco_install.log 2>&1
python3 - <<'PY'
import json, os
p = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(p):
    json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "/usr/lib64-nvidia/libEGL_nvidia.so.0"}}, open(p, "w"))
print("[egl] ICD ok")
PY

# HumanML3D train / heldout (groot_eval uses them to build observations and statistics)
if [ ! -d $B/heldout ]; then
  cp $DRIVE/data/b3_dataset.tar $W/b3_dataset.tar
  tar xf $W/b3_dataset.tar -C $(dirname $B) && rm -f $W/b3_dataset.tar
fi
echo VLA_DATA_OK; t data

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
shutil.copy("$DRIVE/$ACTION_HEAD", "$W/action_head.safetensors")
print(f"[base] checkpoint-47000 と $ACTION_HEAD を {time.time() - t0:.0f} 秒でコピー")
PY
echo VLA_BASE_OK; t base

for S in $V/data/goals/*.json; do
  n=eval_$(basename $S .json)
  GK=$(python3 -c "import json,sys; it=json.load(open(sys.argv[1]))['items'][0]; print('--goal-key vlm_goal' if 'vlm_goal' in it else '')" $S)
  MUJOCO_GL=egl .venv/bin/python $G/groot_eval.py --ckpt $W/base --train $B/train --heldout $B/heldout \
    --action-head $W/action_head.safetensors --object-eval $S $H2_FLAGS $GK \
    --out $W/$n/h2 > $W/$n.log 2>&1 || echo "VLA_EVAL_FAILED $n（$W/$n.log）"
  grep "全体\|直した数え方\|guidance\]\|^\[object\] [a-z]" $W/$n.log | grep -v "^\[guidance\] mode" | sed "s|^|[$n] |" || true
  echo VLA_EVAL_${n^^}_DONE; t $n
done

cd $W && tar czf $W/results.tgz eval_* *.log 2>/dev/null || true
cp $W/results.tgz $OUT/ || true
echo "[bundle] $(du -h $W/results.tgz | cut -f1) → $OUT"
echo VLA_DONE
