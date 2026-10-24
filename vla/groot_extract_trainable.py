#!/usr/bin/env python3
"""Extract only the trainable weights (action_head = projector + DiT + vlln) from a fine-tuned checkpoint.

    .venv/bin/python groot_extract_trainable.py --ckpt /content/b2/out/checkpoint-10000 \\
        --out /content/b2/action_head_bf16.safetensors

N1.7 has no LoRA, and with launch_finetune.py's defaults (tune_llm/tune_visual=False,
tune_projector/tune_diffusion_model/tune_vlln=True, tune_top_llm_layers=0)
only what is under action_head is trained (set_trainable_parameters in gr00t_n1d7.py).
Measured in B-0: 1,620,515,968 parameters. 3.2 GB in bf16, 1/7 of the full checkpoint (24 GB).

To restore, load the N1.7 base and then overwrite it with this state_dict using strict=False.
The VLM side is frozen, so it is identical to the base. Collect processor/ (statistics and modality config) separately.
"""
import argparse
import collections
import json
import pathlib

EXPECTED = 1_620_515_968   # "Trainable parameters:" in the B-0 log


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="action_head.")
    a = ap.parse_args()

    ck = pathlib.Path(a.ckpt)
    idx = ck / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    sd = {}
    for f in files:
        with safe_open(ck / f, "pt") as fh:
            for k in fh.keys():
                if k.startswith(a.prefix):
                    sd[k] = fh.get_tensor(k).to(torch.bfloat16).contiguous()
    n = sum(t.numel() for t in sd.values())
    print(f"{len(sd)} tensors / {n:,} params under {a.prefix!r} (expected {EXPECTED:,})")
    by = collections.Counter()
    for k, t in sd.items():
        by[".".join(k.split(".")[:2])] += t.numel()
    for k, v in by.most_common():
        print(f"  {k:40s} {v:>14,}")
    save_file(sd, a.out, metadata={"source": str(ck), "dtype": "bfloat16", "prefix": a.prefix})
    print(f"wrote {a.out} ({pathlib.Path(a.out).stat().st_size / 1e9:.2f} GB)")
    assert n == EXPECTED, f"パラメータ数が B-0 の学習対象数と合わない: {n:,}"


if __name__ == "__main__":
    main()
