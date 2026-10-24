#!/usr/bin/env python3
"""Upload the model card, NOTICE and license copies (everything in hf/) to the Hugging Face model repository and set gating.
The weights are not touched (they are uploaded from Drive by colab/hf_upload.sh). Runs directly from the Codespace.

    python3 tools/hf_update_meta.py                     # upload hf/ and set gated=auto (visibility unchanged)
    python3 tools/hf_update_meta.py --gated manual      # approve requests one by one
    python3 tools/hf_update_meta.py --public            # make it public (still gated); undo with --private

The write token is read from ~/.hf_write.json ({"HF_TOKEN": "..."}, mode 600). The value is never printed.
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Koichi3333/vla-maker-from-cbd")
    ap.add_argument("--token-file", default=str(pathlib.Path.home() / ".hf_write.json"))
    ap.add_argument("--gated", choices=["auto", "manual", "off"], default="auto")
    vis = ap.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true")
    vis.add_argument("--private", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=json.loads(pathlib.Path(a.token_file).read_text())["HF_TOKEN"])
    print("account:", api.whoami()["name"])

    files = sorted(p for p in (ROOT / "hf").iterdir() if p.is_file())
    assert {"README.md", "NOTICE"} <= {p.name for p in files}, "hf/README.md と hf/NOTICE が要る"
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(ROOT / "hf"),
                      commit_message="Add NOTICE and license copies; update model card terms")
    print("uploaded:", [p.name for p in files])

    api.update_repo_settings(a.repo, repo_type="model", gated=False if a.gated == "off" else a.gated)
    if a.public or a.private:
        api.update_repo_settings(a.repo, repo_type="model", private=a.private)

    info = api.model_info(a.repo, files_metadata=True)
    names = {s.rfilename for s in info.siblings}
    missing = [p.name for p in files if p.name not in names]
    print(f"private={info.private} gated={info.gated} files={len(names)} missing={missing}")
    assert not missing


if __name__ == "__main__":
    main()
