# 別ファイル(.py)から Drive に保存できるかの確認
import os, json, datetime
D = "/content/drive/MyDrive/ColabData"
os.makedirs(D, exist_ok=True)
p = os.path.join(D, "colab_cli_roundtrip.json")
payload = {"written_at": datetime.datetime.now().isoformat(),
           "by": "drive_save.py", "rows": [1, 2, 3]}
with open(p, "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("[save] ismount:", os.path.ismount("/content/drive"))
print("[save] wrote:", p, os.path.getsize(p), "bytes")
