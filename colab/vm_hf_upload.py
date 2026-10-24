#!/usr/bin/env python3
"""Upload the weights in Drive to Hugging Face (VM side; started by colab/hf_upload.sh go).

1. Read the write token from /content/.hf_write.json and **delete the file at once** (the value is never printed)
2. Copy from Drive to the VM's local disk (reading Drive directly is slow). Skip what inference does not need, such as the optimizer
     checkpoint-47000/                 <- $DRIVE/ckpt/checkpoint-47000
     action_heads/arm_H.safetensors    <- $DRIVE/v12e/action_head_bf16_v12e_armH.safetensors
     action_heads/arm_H_trainer_state.json (if present)
     README.md, NOTICE, LICENSE-*      <- everything in /content/vla/hf/ (model card, attribution notice, license copies)
3. Create the private repository (or keep an existing one) and upload with upload_large_folder (resumes if interrupted)
4. Compare the file list and sizes on HF with the local copy

Markers: VLA_HF_STAGED / VLA_HF_UPLOADED / VLA_HF_VERIFIED
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

DRIVE = pathlib.Path(os.environ["DRIVE"])
REPO = os.environ["HF_REPO"]
STAGE = pathlib.Path("/content/hfstage")
SKIP = ("optimizer.pt", "scheduler.pt")
T0 = time.time()


def t(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def main():
    tf = pathlib.Path("/content/.hf_write.json")
    token = json.loads(tf.read_text())["HF_TOKEN"]
    tf.unlink()
    t(f"トークンを読んでファイルを消した（len={len(token)}）")

    # 2.x has no upload_large_folder (2026-10-19, AttributeError on 2.2.0). Pin to 1.x, whose arguments were checked
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[hf_xet]>=1.23,<2"], check=True)
    import huggingface_hub
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    t(f"huggingface_hub {huggingface_hub.__version__}、アカウント {api.whoami()['name']}")

    # ---------------------------------------------------------------- 2. Copy
    # On later runs, files with the same size are not copied again. STAGE/.cache is for resuming upload_large_folder, so keep it
    src = DRIVE / "ckpt/checkpoint-47000"
    assert src.is_dir(), f"{src} が無い"
    for p in sorted(src.rglob("*")):
        if p.is_dir() or p.name in SKIP or p.name.startswith("rng_state"):
            continue
        q = STAGE / "checkpoint-47000" / p.relative_to(src)
        if q.exists() and q.stat().st_size == p.stat().st_size:
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, q)
    heads = STAGE / "action_heads"
    heads.mkdir(parents=True, exist_ok=True)
    shutil.copy(DRIVE / "v12e/action_head_bf16_v12e_armH.safetensors", heads / "arm_H.safetensors")
    ts = DRIVE / "v12e/trainer_state_armH.json"
    if ts.exists():
        shutil.copy(ts, heads / "arm_H_trainer_state.json")
    for f in pathlib.Path("/content/vla/hf").iterdir():      # Model card, NOTICE and license copies
        shutil.copy(f, STAGE / f.name)
    local = {str(p.relative_to(STAGE)): p.stat().st_size for p in STAGE.rglob("*")
             if p.is_file() and ".cache" not in p.parts}
    t(f"写した: {len(local)} ファイル・{sum(local.values()) / 1e9:.2f} GB")
    for k in sorted(local):
        print(f"    {local[k] / 1e6:10.1f} MB  {k}")
    print("VLA_HF_STAGED", flush=True)

    # ---------------------------------------------------------------- 3. Upload
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(REPO, repo_type="model")
    t(f"リポジトリ {REPO}（private={info.private}）")
    api.upload_large_folder(repo_id=REPO, folder_path=str(STAGE), repo_type="model", private=True,
                            ignore_patterns=[".cache/**"], print_report_every=120)
    t("アップロード完了")
    print("VLA_HF_UPLOADED", flush=True)

    # ---------------------------------------------------------------- 4. Compare
    remote = {f.path: f.size for f in api.list_repo_tree(REPO, repo_type="model", recursive=True)
              if getattr(f, "size", None) is not None}
    missing = [k for k in local if k not in remote]
    wrong = [k for k in local if k in remote and remote[k] != local[k]]
    t(f"HF 側 {len(remote)} ファイル。足りない {len(missing)}、サイズ違い {len(wrong)}")
    for k in missing + wrong:
        print("    NG", k, local[k], remote.get(k))
    assert not missing and not wrong, "HF 側がローカルと一致しない"
    print(f"VLA_HF_VERIFIED {len(remote)} files, private={api.repo_info(REPO, repo_type='model').private}", flush=True)


if __name__ == "__main__":
    main()
