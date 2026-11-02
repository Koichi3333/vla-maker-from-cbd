#!/usr/bin/env python3
"""Upload work/hf_dataset (made by tools/make_hf_dataset.py) to a Hugging Face dataset repository, gated, private at first.
Runs directly from the Codespace.

    python3 tools/hf_upload_dataset.py                 # create it private + gated=auto and upload
    python3 tools/hf_upload_dataset.py --public        # then make it public (still gated)

The write token is read from ~/.hf_write.json ({"HF_TOKEN": "..."}, mode 600). The value is never printed.
"""
import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Koichi3333/vla-maker-from-cbd-data")
    ap.add_argument("--src", default=str(ROOT / "work" / "hf_dataset"))
    ap.add_argument("--token-file", default=str(pathlib.Path.home() / ".hf_write.json"))
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()

    src = pathlib.Path(a.src)
    assert (src / "README.md").exists() and (src / "meta/episodes.jsonl").exists(), "run tools/make_hf_dataset.py first"
    assert not any(json.loads(l).get("synthetic") is None for l in (src / "meta/episodes.jsonl").read_text().splitlines()), \
        "a non-synthetic (real) episode is in the dataset"

    from huggingface_hub import HfApi
    api = HfApi(token=json.loads(pathlib.Path(a.token_file).read_text())["HF_TOKEN"])
    print("account:", api.whoami()["name"])
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(src),
                      commit_message="Add the synthetic reaching clips (CBD, LeRobot format)")
    api.update_repo_settings(a.repo, repo_type="dataset", gated="auto")
    if a.public:
        api.update_repo_settings(a.repo, repo_type="dataset", private=False)
    print("done:", f"https://huggingface.co/datasets/{a.repo}", "(public)" if a.public else "(private)")


if __name__ == "__main__":
    main()
