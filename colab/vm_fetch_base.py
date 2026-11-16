"""Put checkpoint-47000 (and optionally an action head) into the VM's working directory (VM side).

Called by colab/vm_infer.sh and colab/vm_train.sh. Runs with the kernel's python3 (it has huggingface_hub).

  --source hf     download from the Hugging Face model repository (default; needs HF_TOKEN of an account that
                  accepted the repository's terms, the access is gated)
  --source drive  copy from Drive ($DRIVE/ckpt/checkpoint-47000), the layout used before the weights were on HF

The action head:
  --action-head ''          (default) take none (training)
  --action-head hf          arm H from Hugging Face (action_heads/arm_H.safetensors), the usable form
  --action-head <path>      a file relative to --drive in Drive (e.g. runs/myrun/action_head.safetensors)

Writes <out>/base/ (checkpoint without optimizer etc.) and, with an action head, <out>/action_head.safetensors.
"""
import argparse
import os
import pathlib
import shutil
import sys
import time

REPO = "Koichi3333/vla-maker-from-cbd"
CKPT = "checkpoint-47000"
ARM_H = "action_heads/arm_H.safetensors"
SKIP = ("optimizer.pt", "scheduler.pt")


def copy_ckpt(src, dst):
    for p in sorted(src.rglob("*")):
        if p.is_dir() or p.name in SKIP or p.name.startswith("rng_state") or ".cache" in p.parts:
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, q)


def from_hf(out, want_head):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("[base] HF_TOKEN が無い。colab/start.sh が渡す（colab-cli/README.md の「A short-lived HF_TOKEN」）")
    patterns = [f"{CKPT}/*"] + ([ARM_H] if want_head else [])
    local = out / "hf"
    try:
        snapshot_download(REPO, allow_patterns=patterns, local_dir=local, token=token)
    except (GatedRepoError, RepositoryNotFoundError) as e:
        sys.exit(f"[base] {REPO} を読めない（{type(e).__name__}）。そのトークンのアカウントで "
                 f"https://huggingface.co/{REPO} の条件に同意したか確かめること")
    base = out / "base"
    if base.exists():
        shutil.rmtree(base)
    (local / CKPT).rename(base)          # same filesystem: no second copy of ~12 GB
    shutil.rmtree(base / ".cache", ignore_errors=True)
    if want_head:
        (local / ARM_H).rename(out / "action_head.safetensors")
    shutil.rmtree(local, ignore_errors=True)   # only download metadata is left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "drive"], default="hf")
    ap.add_argument("--drive", required=True)
    ap.add_argument("--action-head", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, drive = pathlib.Path(a.out), pathlib.Path(a.drive)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    head_from_hf = a.action_head == "hf"
    if a.source == "hf":
        from_hf(out, head_from_hf)
        where = f"Hugging Face {REPO}"
    else:
        if head_from_hf:
            sys.exit("[base] --action-head hf は --source hf のときだけ")
        copy_ckpt(drive / "ckpt" / CKPT, out / "base")
        where = f"Drive {drive / 'ckpt'}"
    msg = f"[base] {CKPT} を {where} から"
    if a.action_head and not head_from_hf:
        shutil.copy(drive / a.action_head, out / "action_head.safetensors")
        msg += f"、action head を Drive の {a.action_head} から"
    elif head_from_hf:
        msg += f"（action head は {ARM_H}）"
    n = sum(1 for p in (out / "base").rglob("*.safetensors"))
    assert n >= 3, f"{out / 'base'} の safetensors が {n} 個しか無い"
    print(f"{msg} {time.time() - t0:.0f} 秒で用意")


if __name__ == "__main__":
    main()
