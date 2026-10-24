#!/usr/bin/env python3
"""Check through the API that files written via the mount have really finished uploading to Drive (run on the VM).

    python3 drive_verify.py /content/drive/MyDrive/VLA/VLA_1_0/ckpt/checkpoint-15000
    python3 drive_verify.py --wait 2400 <same>     # wait until fully uploaded

**Why `ls` / `du` are not enough (learned on 2026-09-19 by losing 12.9 GB).**
A `cp` to mounted Drive **returns as soon as it is written to DriveFS's local cache**.
The upload happens in the background, and **starts minutes later** (measured: `active_uploads: 1`
appeared more than 10 minutes after `cp` finished). `ls` / `du` read the local metadata DB, so
**a file with not a single byte uploaded yet looks the same size as the real thing**.
A `colab stop` in that state loses whatever has not been sent.

So **`drive_flush_wait.py` (which only reads DriveFS's log) is a helper at best**.
Right after a copy the queue is still empty, so it **wrongly concludes "quiet, therefore done"**.
Asking the server, as this script does, is the reliable path.

**The check.** List the parent folder's children with files.list and compare `size` with the local files.
If any local file is missing from the API or has a different size, the upload is not complete.
Google Docs formats (which have no `size`) are out of scope and skipped.

Exit code 0 = all match / 1 = timed out while still incomplete / 2 = API unavailable.
"""
import argparse
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_purge as dp  # noqa: E402  borrows token() and find_id()


def api_children(tok, folder_id):
    """Return {name: size (int or None)}. Follows pagination."""
    out, page = {}, None
    while True:
        r = requests.get(dp.API, params={
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id,name,size,mimeType)",
            "pageSize": 1000, **({"pageToken": page} if page else {}),
        }, headers={"Authorization": f"Bearer {tok}"}, timeout=60)
        r.raise_for_status()
        j = r.json()
        for f in j["files"]:
            out[f["name"]] = int(f["size"]) if "size" in f else None
        page = j.get("nextPageToken")
        if not page:
            return out


def compare(tok, path):
    """Return (list of incomplete files, total count). Incomplete entries are (name, local, API) tuples."""
    fid = dp.find_id(tok, path)
    remote = api_children(tok, fid)
    bad, n = [], 0
    for name in sorted(os.listdir(path)):
        local_path = os.path.join(path, name)
        if os.path.isdir(local_path):
            continue          # Only one level. The contents of checkpoint-N are flat, so that is enough
        n += 1
        want = os.path.getsize(local_path)
        got = remote.get(name, "無し")
        if got != want:
            bad.append((name, want, got))
    return bad, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="確かめたい Drive 上のフォルダ（マウント越しのパス）")
    ap.add_argument("--wait", type=int, default=0, help="上がりきるまで待つ上限（秒）。0 なら1回見るだけ")
    ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()

    tok, why = dp.token()
    if not tok:
        print(f"DRIVE_VERIFY_ERROR トークンが取れない: {why}")
        raise SystemExit(2)
    print(f"[verify] token ok ({why})")

    t0 = time.time()
    while True:
        try:
            bad, n = compare(tok, a.path)
        except Exception as e:  # noqa: BLE001  on token expiry etc., fetch a new one and continue
            print(f"[verify] 一時的な失敗: {type(e).__name__}: {str(e)[:150]}")
            tok, why = dp.token()
            if not tok:
                print(f"DRIVE_VERIFY_ERROR トークンを取り直せない: {why}")
                raise SystemExit(2)
            bad, n = [("(再試行中)", 0, 0)], 0

        el = int(time.time() - t0)
        if not bad:
            print(f"DRIVE_VERIFY_OK {a.path} のファイル {n} 件がすべて Drive 側と一致（{el}秒）")
            raise SystemExit(0)
        short = ", ".join(f"{nm}: ローカル{lo} / API {gt}" for nm, lo, gt in bad[:3])
        print(f"[verify] t={el}s 未完了 {len(bad)}/{n} — {short}", flush=True)
        if el >= a.wait:
            print(f"DRIVE_VERIFY_INCOMPLETE {len(bad)}/{n} 件が未完了のまま（{el}秒）")
            for nm, lo, gt in bad:
                print(f"  {nm}: ローカル {lo} / API {gt}")
            raise SystemExit(1)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
