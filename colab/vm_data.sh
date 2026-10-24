#!/usr/bin/env bash
# Rebuild HumanML3D -> CBD -> LeRobot (117-d) -> renders on the VM and put it in Drive as data/b3_dataset.tar (started by colab/go.sh data).
# This builds the HumanML3D train / heldout used by training and inference. The author's Drive already has it, so the author need not run this.
#
#   Markers: VLA_FETCH_OK / VLA_PARQUET_OK / VLA_RENDER_<part>_OK / VLA_STATS_OK / VLA_DONE / VLA_EXIT=<exit code>
#
# The procedure that ran in B-3 on 2026-09-15 (all 10,427 clips, rendering took 28 minutes on an A100 VM), moved to this repository's layout.
# Not run since the move ("unverified" in docs/results.md). HumanML3D / AMASS are for non-commercial research only (README license section, docs/data.md).
set -euo pipefail
DRIVE=${DRIVE:?DRIVE を渡すこと}
V=/content/vla; G=$V/vla; B=/content/b3; W=/content/run
T0=$(date +%s)
t() { echo "TIMING $1 $(( $(date +%s) - T0 ))s"; }
trap 'echo "VLA_EXIT=$?"' EXIT
mkdir -p $B $W

# 0. mujoco and the EGL ICD (Colab's glvnd only registers mesa)
python3 -m pip install -q mujoco
python3 - <<'PY'
import json, os
p = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(p):
    json.dump({"file_format_version": "1.0.0", "ICD": {"library_path": "/usr/lib64-nvidia/libEGL_nvidia.so.0"}}, open(p, "w"))
PY

# 1. HumanML3D joint positions and text. The list of standing clips is vla/standing.json
#    The joint positions (new_joints) are not distributed (AMASS terms). Build them yourself from AMASS with HumanML3D's official steps
#    and put them in your own Drive as $JOINTS (default $DRIVE/data/humanml3d_new_joints.zip) (docs/data.md).
#    Text annotations and lists come from the HumanML3D authors' GitHub
JOINTS=${JOINTS:-$DRIVE/data/humanml3d_new_joints.zip}
[ -e "$JOINTS" ] || { echo "VLA_NO_JOINTS $JOINTS が無い（docs/data.md）"; exit 1; }
cd $G
python3 -c "import pathlib, fetch_and_curate as f; f.fetch(pathlib.Path('$B/humanml3d'), '$JOINTS')"
cp standing.json $B/humanml3d/
echo VLA_FETCH_OK; t fetch

# 2. Through CBD into the 117-d LeRobot format (all standing clips)
[ -f $B/lerobot_standing_all/meta/info.json ] || \
  python3 cbd_to_lerobot.py --source humanml3d --humanml3d-root $B/humanml3d --out $B/lerobot_standing_all
echo VLA_PARQUET_OK; t parquet

# 3. Rendering (split per vla/full_split.json; smaller part first; if it dies, starting it again renders only what is left)
for p in heldout excluded train; do
  MUJOCO_GL=egl python3 lerobot_add_video.py --src $B/lerobot_standing_all --dst $B/$p \
      --pattern render --workers 8 --xml humanoid.xml --split full_split.json --part $p \
      --skip-existing --report $W/report_$p.json
  echo VLA_RENDER_${p^^}_OK; t render_$p
done

# 4. GR00T normalization statistics (computed on train, the same copy placed in heldout) and a loader check
for i in $(seq 1 60); do grep -q SETUP_DONE /content/groot_setup.log 2>/dev/null && break; sleep 10; done
cd /content/Isaac-GR00T
.venv/bin/python gr00t/data/stats.py --dataset-path $B/train \
    --embodiment-tag NEW_EMBODIMENT --modality-config-path $G/groot_cbd_config.py
for p in heldout excluded; do cp $B/train/meta/{stats,relative_stats}.json $B/$p/meta/; done
.venv/bin/python $G/groot_eval.py --check-dataset $B/train $B/heldout
echo VLA_STATS_OK; t stats

# 5. To Drive (training and inference unpack $DRIVE/data/b3_dataset.tar; it holds b3/train and b3/heldout)
mkdir -p $DRIVE/data
tar cf $W/b3_dataset.tar -C $(dirname $B) b3/train b3/heldout
cp $W/b3_dataset.tar $DRIVE/data/
python3 $G/drive_verify.py --wait 2400 $DRIVE/data || echo "VLA_VERIFY_WARNING Drive 側と一致しない。停止する前に確認すること"
t upload
echo VLA_DONE
