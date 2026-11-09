# Google Colaboratory CLI 運用ガイド

公式の [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli)
で Colab の GPU/TPU/CPU ランタイムをコマンドラインから操作するための手引き。
Codespaces などのクラウド環境からでも、ローカルの macOS / Linux からでも同じ手順で動く。

**挙動に関する記述は v0.6.0 での実測に基づく。** 公式ドキュメントと食い違う箇所は、
その旨を明記している。

> **バージョンについて。** [`../setup.sh`](../setup.sh) が入れるのは **0.7.5**。
> インストールと認証の節は 0.7.5 のソースに合わせて書いてある。それ以外は 0.6.0 での実測で、
> 0.7.5 のソースで変化が確認できた箇所にはその旨を書いた。
>
> 2026-11-07 に 0.7.5 の CPU ランタイムで次を再確認した:
> - 認証の完了（既定は oauth2）。
> - `new` / `upload` / `exec -f … --timeout` / `download` / `stop`。
> - `X_output.ipynb` が入力の隣に書き出されること。
> - `exec` をまたいでカーネルの状態が残ること（ある `exec` で入れた secrets の前処理が次の `exec` から見えた）。
> - `colab_secrets.py` で渡したキーが CLI の履歴に残らないこと。
>
> 0.7.5 ではまだ再確認していないもの: セルが失敗したときの終了コード、ブラウザなしでの対処手順、GPU ランタイム。
> 挙動の違いを見つけたら、それ自体が知見なので報告してほしい。
>
> セットアップ手順そのものは [`../README.ja.md`](../README.ja.md)。このガイドは「なぜそうするか」。
>
> **このリポジトリでは。** このガイドは
> [common-behavior-data](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli) の
> `colab-cli/guide/` の写しで、もとはノートブック向けに書かれている。VLA のジョブは `colab exec` で動かすシェルスクリプトで、
> 2026-11-05 に **0.7.5 の A100 ランタイム**で `colab/` から学習と評価を通した。
> 長い A100 のジョブでだけ踏む落とし穴（60 分で切れるトークン、通信が無いと約 25 分で消える割り当て、
> 完了前に返る Drive へのコピー、`colab exec` の固まり）は [`../../docs/colab.md`](../../docs/colab.md) にある。

英語版は [`field_guide.md`](field_guide.md)。

---

## 目次

- [前提](#前提)
- [セットアップ](#セットアップ)
- [認証](#認証)
- [基本操作](#基本操作)
- [推奨の使い方](#推奨の使い方)
- [対話コード（`input()` など）の扱い](#対話コードinput-などの扱い)
- [注意点](#注意点)
- [トラブルシューティング](#トラブルシューティング)
- [関連ファイル](#関連ファイル)

---

## 前提

- GitHub Codespaces などのクラウド環境、またはローカルの macOS / Linux（Windows は WSL 内）
- Google アカウント。Workspace アカウントでも可
- Colab の契約は**無料枠**を想定（Colab Pro なし）

---

## セットアップ

```bash
colab-cli/setup.sh
```

`uv` が無ければ入れ、`uv tool install google-colab-cli==0.7.5` を行う。何度流してもよい。
最新版を入れたいときは `COLAB_CLI_VERSION=latest colab-cli/setup.sh`。

### かつて必須だった固定: `jupyter-kernel-client<1.0`

0.6.0 は `jupyter-kernel-client` 依存をバージョン未固定にしており、同パッケージは
1.0.0 で `KernelClient` を `JupyterKernelClient` にリネームした。新しい版が入ると
`colab exec` / `colab run` がこう落ちた:

```
AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
```

**0.7.5 は `jupyter-kernel-client==0.9.0` を自分で固定し、3通りのクラス名すべてに
対応している**ので、この制約が要るのは 0.6.x をあえて入れる場合だけ
（`setup.sh` はその場合に自動で付ける）。

---

## 認証

```bash
colab-cli/auth.sh
```

### 方式

CLI の実際のデフォルトは **`--auth=oauth2`**（CLI 同梱の `SKILL.md` には今も `adc` と
書かれているが、`cli.py` の既定値は `AuthProvider.OAUTH2`）。

0.7.x の oauth2 は `gcloud auth application-default login` と同じ方式で、
同意後に Google の登録済みページに認可コードが表示され、それを貼り付ける。
**ローカルでもリモートでも同じ手順**なので、「ローカル用の別方式」はもう無い。

`gcloud` のインストールや独自 OAuth クライアントの用意は**不要**。
取得したトークンは `~/.config/colab-cli/token.json` にキャッシュされ、
リフレッシュトークンを含むため、マシンごとに1回やれば以降は不要。

> **`colab auth` は別物。** こちらは VM 側（Colab ランタイム）に GCP 認証情報を
> 注入するコマンドで、CLI 自体の認証とは無関係。401/403 の解決には使わない。

### ⚠ 最大の落とし穴: URLの切れ

認可URLは **約700文字**ある。ターミナルから直接コピーすると折り返しで切れやすく、
`&scope=...` が欠落すると Google はこう返す:

```
アクセスをブロック: 認証エラーです
Missing required parameter: scope
エラー 400: invalid_request
```

**このエラーが出たら、まず URL が完全にコピーできているかを疑うこと。**
Workspace の管理者ポリシーやブラウザ設定を疑う前に、ここを確認する。

見分け方: エラーページの URL に含まれる `aes=` の値が**毎回同じ**なら、
新しいリクエストが送られていない（同一の不正リクエスト →
ブラウザが同じエラーページをキャッシュ返却している）。

`auth.sh` は CLI 自身の認証フロー（クライアント・スコープ・着地ページを
`colab_cli.auth` から import）をそのまま使い、**URL をブラウザへ届ける経路だけ**を変える。
ターミナルからコピーさせることは一度もない。

| 環境 | `auth.sh` の動き |
| --- | --- |
| Codespaces / VS Code Remote | `$BROWSER` 経由で開く。**手元のマシンのブラウザ**に開く |
| ローカルの macOS / Linux デスクトップ | 既定のブラウザで開く |
| SSH などそれ以外 | `colab-cli/ws/.auth/auth_url.txt` とリンク1つだけの `index.html` を書く。エディタから開くか、`--serve 8765` とポート転送で開く |
| エージェント・TTY なし | `--url-only` で URL を書き出し、`--code <コード>` で2回目に交換 |

`auth.sh` を使わない場合の手動版は、URL（stderr に出る）だけをファイルへ落とす方法:

```bash
colab whoami 2> /tmp/colab_auth_url.txt
```

> **0.6.0 のころの話。** 0.6.0 は別の着地ページを使っており、元の手順書では
> loopback リダイレクト（`http://127.0.0.1:8085/`）の補助スクリプトを代替手段にしていた。
> 0.7.x で不要になったので、ここには持ち込んでいない。

### 確認

```bash
colab whoami        # 隠しコマンド
```

```
Auth provider: oauth2
Email:         your-address@example.com
Expires in:    59m
Scopes:
  - https://www.googleapis.com/auth/cloud-platform
  - https://www.googleapis.com/auth/colaboratory
  ...
```

**`colaboratory` スコープが含まれていること**を確認する。
これが無いと keep-alive RPC が 403 になり、セッションを維持できない。

---

## 基本操作

### セッションのライフサイクル

セッション = Colab が確保した VM 上の生きた Jupyter カーネル。

```bash
colab new -s work                                    # 作成
colab exec -s work -f notebook.ipynb --timeout 300   # 実行
colab stop -s work                                   # 停止（必須）
```

`colab new` は VM を確保する。**`colab stop` を呼ぶまで解放されない**
（0.6.0 ではバックグラウンドの keep-alive デーモンが維持していた）。

### コマンド一覧

| コマンド | 用途 |
| --- | --- |
| `colab new -s <name> [--gpu T4\|L4\|G4\|H100\|A100] [--tpu v5e1\|v6e1]` | セッション作成。アクセラレータ省略で CPU。**利用可否は契約プランによる**（無料枠では A100/H100 は割り当てられない） |
| `colab exec -s <name> -f <file> --timeout <秒>` | `.ipynb` / `.py` / stdin を実行 |
| `colab run [--gpu T4] <script.py>` | 使い捨て実行。終了時に必ずセッションを破棄する |
| `colab stop -s <name>` | セッション停止 |
| `colab sessions` | サーバー側の全セッション一覧 |
| `colab status -s <name>` | ハードウェア構成・IDLE/BUSY・最終実行 |
| `colab upload -s <name> <local> <remote>` | ローカル → VM |
| `colab download -s <name> <remote> <local>` | VM → ローカル（**引数は remote が先**） |
| `colab ls -s <name> <path>` | VM 上のファイル一覧 |
| `colab restart-kernel -s <name>` | カーネル再起動（状態をリセット） |
| `colab log -s <name> [-n 20]` | 構造化イベントログ |
| `colab url -s <name>` | Web UI をこのセッションに接続する URL を出力 |
| `colab whoami` | 認証状態の確認 |

### `colab exec` の入力形式

拡張子で挙動が変わる。

```bash
colab exec -s work -f notebook.ipynb --timeout 300   # コードセルごとに分割実行
colab exec -s work -f script.py --timeout 300        # ファイル全体を1ブロックとして実行
echo "print(1+1)" | colab exec -s work               # stdin
colab exec -s work --timeout 60 <<'PY'
x = 42
print(x)
PY
```

0.7.5 には `rm` / `edit` / `install` / `ssh` / `update` / `pay` もある（`colab help <command>`）。
このガイドでは扱っていない。

`-s` は、有効なセッションが1つだけなら省略できる（自動採用される）。
複数あると曖昧になるので、**常に明示するのが安全。**

---

## 推奨の使い方

### 標準ワークフロー

```bash
colab new -s work --gpu T4
colab upload -s work ./data.csv /content/data.csv        # 入力ファイルを先に置く
colab exec   -s work -f analysis.ipynb --timeout 1800
colab download -s work /content/result.csv ./result.csv  # 停止前に回収する
colab stop -s work
```

`colab exec` の実行前に必ず `/content` へ `chdir` されるため、
VM 上では相対パスで `data.csv` を読める。

### `.ipynb` を扱えるのは `colab exec` だけ

`colab run` はファイルを**そのまま Python として読む**ため、
`.ipynb` を渡すと JSON を実行しようとして SyntaxError になる。
`colab run` は「単発の `.py` を投げて捨てる」用途に限る。

### セルには `#@title` を付ける

```python
# @title データ読み込み
```

実行時のヘッダに反映される。

```
[colab] Executing cell 3/8 - データ読み込み...
```

付けないとセルの UUID が表示されて追いにくい。

### `--timeout` は必ず明示する

**既定 30 秒。しかも総実行時間の締切であって、無音時間の上限ではない。**
5秒ごとに出力していても総時間を超えれば `TimeoutError` になるので、
進捗を print しても延命はできない。最初から余裕のある値を渡すこと。

`.ipynb` の場合はセルごとに実行が呼ばれるため、**この締切は1セルあたり**に
適用される（ノートブック全体の合計ではない）。

### 出力を見てから続きを流す

同一セッションではカーネル状態（import・変数）が `colab exec` を跨いで保持される。
これを使えば対話的な進め方ができる。

```bash
colab exec -s work -f part1.ipynb --timeout 300   # 出力を確認
colab exec -s work <<< "threshold = 0.85"         # 値を注入
colab exec -s work -f part2.ipynb --timeout 300   # 続きを流す
```

### 成果物の回収

**VM のファイルシステム（`/content` など）は揮発性で、停止と同時に消える。**

**`files.download()` は CLI では何もしない。** ノートブックの末尾によくある
`files.download(path)` は、ブラウザ側の comm でバイト列を受け取る JavaScript を
display するだけなので、CLI から実行すると**黙って成功して 0 秒で return する**。
「セルが全部通ったのにファイルが無い」の原因はほぼこれ。
セルはそのまま残してよい（UI では正しく動く）。CLI では無視して `colab download` で取る。

| 方法 | 使いどころ |
| --- | --- |
| 標準出力に `print` | 数値・テキスト・小さな JSON |
| `colab download -s <name> <remote> <local>`（**停止前に**実行） | 生成したファイル |
| スクリプト内で Drive にマウントして書き出す | 大きなファイル・モデル重み |

`colab download` について実測で分かっていること:

- **バイナリは壊れない。** 20MB のランダムバイナリで sha256 一致、所要 2.9 秒。
- **ディレクトリは渡せない。** `Cannot download a directory: ...` で失敗するので、
  VM 側で zip にしてから取る。

```bash
colab exec -s work --timeout 300 <<'EOF'
import shutil
print(shutil.make_archive("/content/out_bundle", "zip", "/content/output"))
EOF
colab download -s work /content/out_bundle.zip ./out_bundle.zip
colab stop -s work
```

`.ipynb` 実行時は、出力入りの `<元の名前>_output.ipynb` が**ローカルに**書かれるため、
print した内容はそこにも残る。書き出し先は
`os.path.splitext(file)[0] + "_output.ipynb"`（`commands/execution.py` の 0.6.0 では 238 行目、0.7.5 では 280 行目）、つまり
**カレントディレクトリではなく入力ファイルの隣**。`-f ../nb.ipynb` と書くと
`../nb_output.ipynb` に落ちる。

このリポジトリでは生成物を `colab-cli/ws/`（`.gitignore` 済み）に隔離する運用にしている。
`ws/` に入って**ノートブックをコピーしてから** `-f` に渡すこと。詳細は
[`../README.ja.md`](../README.ja.md) と [`../CLAUDE.md`](../CLAUDE.md)。

---

## 対話コード（`input()` など）の扱い

`colab exec` は内部で `allow_stdin` を渡していない（既定 `False`）ため、
カーネル側の stdin 要求はそのままでは失敗する。

| 対象 | 素の挙動 | 対処 |
| --- | --- | --- |
| `input()` | `StdinNotImplementedError` で即失敗 | 同一セル内で差し替える（下記） |
| `files.upload()` | **ハングしてタイムアウトし、ラン全体が異常終了** | 事前に `colab upload` + `files.upload` を差し替える（下記） |
| `files.download()` | **何も起こらず、エラーも出ずに成功する**（0秒で return） | 無視してよい。停止前に `colab download` で取る（[成果物の回収](#成果物の回収)） |
| `userdata.get()`（Colab Secrets） | 10秒待って `TimeoutException`。**CLI からは取得できない** | 鍵をファイルで渡して差し替える（下記） |
| `drive.mount()` | 認可URLを出したあと `/dev/tty` から Enter を待つ | `colab-cli/tools/ptyrun.py` で pty に包めば非対話でも完走できる |

### `input()` を含むノートブックを実行するレシピ

答えを先に用意して `input` を差し替える。**同一セル内なら何回呼んでも動く。**

```python
# @title 入力を事前に与えて実行
import builtins
_ANSWERS = iter(["Alice", "30", "yes"])
def input(prompt=""):
    v = next(_ANSWERS)
    print(f"{prompt}{v}")   # Colab と同じく入力値をエコーする
    return v
builtins.input = input

name = input("お名前は? ")
age  = int(input("年齢は? "))
ok   = input("よろしいですか? (yes/no) ")
```

**差し替えは必ず `input()` を呼ぶセルと同じセルに書く。**
前のセルで `builtins.input` を差し替えても効かない ——
ipykernel が各セル実行の直前に `builtins.input` を自前の `raw_input` へ
再バインドし直すため、前セルのパッチは毎回上書きされる。

元のノートブックのロジックには手を入れず、冒頭にこのブロックを足すだけでよい。

### `files.upload()` を含むノートブックを実行するレシピ

事前に `colab upload` で VM に置き、`files.upload` を差し替える。
`files.upload()` は `{ファイル名: bytes}` の dict を返すだけなので、
それを模した関数に置き換えれば元のコードはそのまま動く。

```bash
colab upload -s work ./data.csv /content/data.csv
```

```python
# @title 前口上: files.upload() を差し替える
from google.colab import files as _f
def _fake_upload():
    out = {}
    for n in ["data.csv"]:                  # 必要なファイル名を並べる
        with open(f"/content/{n}", "rb") as fh:
            out[n] = fh.read()
        print(f"Saving {n} to {n}")         # Colab と同じ体裁のログ
    return out
_f.upload = _fake_upload
```

以降のセルは無改造でよい。

```python
from google.colab import files
up = files.upload()
df = pd.read_csv(io.BytesIO(up["data.csv"]))
```

**こちらは `input()` と違って別セルに書いてよい。**
`files` はモジュールで `upload` はその属性なので、ipykernel の再バインド
（`builtins.input` のみが対象）の影響を受けない。

### Colab Secrets（`userdata.get()`）を含むノートブックを実行するレシピ

`userdata.get()` の実体は `_message.blocking_request('GetSecret', ...)` で、
iopub に `colab_request` を投げて**ブラウザの Colab フロントエンドからの応答を待つ**。
CLI はこのメッセージに応答しないため、10秒後に必ず失敗する。

```
google.colab.userdata.TimeoutException: Requesting secret FOO timed out.
Secrets can only be fetched when running from the Colab UI.
```

例外なので `files.upload()` のようにラン全体を巻き込むことはないが、そのセルは必ず落ちる。
**鍵はファイルで渡す。**

```bash
echo '{"API_KEY": "..."}' > /tmp/secrets.json
colab upload -s work /tmp/secrets.json /content/.secrets.json   # 中身はログに残らない
```

```python
# @title 前口上: CLI のときだけ userdata を差し替える（UI ではそのまま）
import json, os
_P = "/content/.secrets.json"
if os.path.exists(_P):
    from google.colab import userdata as _ud
    with open(_P) as _fh:
        _SECRETS = json.load(_fh)
    os.remove(_P)                     # VM 上に平文を残さない
    def _get(key):
        if key not in _SECRETS:
            raise _ud.SecretNotFoundError(key)
        return _SECRETS[key]
    _ud.get = _get
    os.environ.update(_SECRETS)       # os.environ を読むライブラリにも通す
```

`userdata` はモジュールなので、`files.upload` と同じく**別セル・別 `colab exec`
でも効き続ける**。以降のセルは無改造でよい。

**ファイルの有無で分岐させれば Colab UI での汎用性は失われない。**
UI では `/content/.secrets.json` が存在せず差し替えが走らないので、本物の
Colab Secrets がそのまま使われる。逆向き（`try: userdata.get() except TimeoutException:`）
で判定すると鍵ごとに10秒待たされるので避ける。

差し替えをノートブックに書き込みたくない場合は、独立した `colab exec` として
流し込めば `.ipynb` は一行も変わらない。`colab-cli/tools/colab_secrets.py` がそれをやる。
鍵は次の順に探すので、**クラウドでもローカルでも同じコマンドで動く**:

1. 呼び出したシェルの環境変数
2. `colab-cli/.env`（`--env-file PATH` で変更可）— ローカル向け。gitignore 済み。
   `colab-cli/.env.example` をコピーして作る
3. GitHub Codespaces の Secret（下記）

```bash
python3 colab-cli/tools/colab_secrets.py --list                     # .env と Codespaces の名前（値は出さない）
python3 colab-cli/tools/colab_secrets.py -s work API_KEY            # カーネル状態にだけ効く
python3 colab-cli/tools/colab_secrets.py -s work NEW=OLD            # 名前を付け替える
```

Codespaces の Secret は `/workspaces/.codespaces/shared/.env-secrets` に base64 で
置かれるが、`/etc/profile.d/codespaces.sh` は **`$SSH_CONNECTION` があるときしか
export しない**ため、VS Code のターミナルからは環境変数として見えないことがある
（`colab-cli/tools/colab_secrets.py` はファイルを直接読む）。
`--list` に出ない場合は**リポジトリへの共有設定が抜けている**。

```bash
gh api /user/codespaces/secrets --jq '.secrets[].name'
gh api /user/codespaces/secrets/<NAME>/repositories --jq '[.repositories[].full_name]|join(", ")'
```

共有先は Settings → Codespaces → Repository access で追加し、
**Codespace を Rebuild しないと反映されない**（`.env-secrets` は環境作成時に書かれる）。

**Codespace 側で見えること ≠ VM 側で使えること。** 3段あるので上から潰す。

| 見る場所 | 確認 | 出ないときの原因 |
| --- | --- | --- |
| GitHub の共有設定 | `gh api /user/codespaces/secrets/<NAME>/repositories --jq '[.repositories[].full_name]\|join(", ")'` | Repository access に未追加。追加後 **Rebuild が必要** |
| Codespace 内 | `python3 colab-cli/tools/colab_secrets.py --list` | 共有設定が未反映（Rebuild していない） |
| Colab VM 内 | `colab exec -s <name> --timeout 60 <<< 'import os; print("set" if os.environ.get("<NAME>") else "NOT SET")'` | `colab_secrets.py -s <name> <NAME>` を流していない |

**`colab_secrets.py` はセッションごとに1回、`colab exec` の前に流す。** 差し替えは
カーネル状態に効くので同一セッション内では持続するが、`colab new` で作り直せば消える。
headless では `userdata.get()` を避けて `os.environ` を読む作りのノートブックもあるため、
**渡し忘れると `not set` のまま静かに先へ進む。**

切り分け中も**鍵の値を `echo` しないこと**（端末の出力はログに残る）。
特に次は set のとき**値がそのまま出る** —— `:+` が `set`、`:-` が値を返すため。

```bash
echo "${GEMINI_API_KEY:+set}${GEMINI_API_KEY:-NOT SET}"    # ← NG
[ -n "$GEMINI_API_KEY" ] && echo set || echo "NOT SET"     # ← こちら
```

### 秘密情報をコードに書かない

`colab exec` は**実行したコードと出力の両方**を
`~/.config/colab-cli/history/<session>.jsonl` に平文で追記する。

```bash
colab exec -s work <<< 'import os; os.environ["API_KEY"] = "..."'   # ← 鍵がログに残る
```

`colab upload` が記録するのはパスだけなので、鍵はファイルで渡すこと。
同じ理由で**鍵を `print` してもいけない** —— 出力側からログに残り、
`.ipynb` 実行なら `_output.ipynb` にも焼き付く。

### 本当に対話が必要な場合

`colab console` は WebSocket で VM 上のリモートシェル（tmux + bash）に
生 TTY で接続する。Jupyter カーネルプロトコルを経由しないため、
ここで `python3` を起動すれば `input()` も普通に動く。
ノートブックの実行手段ではないが、対話が不可欠ならこの経路になる。

---

## 注意点

### セッションの停止忘れ

**最重要。** セッションは `colab stop` するまで VM を握り続け、
**ターミナルを閉じても解放されない。**（0.6.0 ではバックグラウンドの keep-alive デーモンが
24 時間上限まで走り続けた。0.7.5 の同梱 skill によれば生存維持はバックエンド側に移ったが、
止めるのが利用者の責任である点は変わらない。）

このリポジトリで Claude Code を使う場合は、安全網として Stop フック
（`colab-cli/hooks/colab_session_guard.py`）がリポジトリの `.claude/settings.json` に登録してある。ローカルの session store にエントリが残っていると、
Claude Code の応答終了時に警告が出る。

```bash
colab sessions            # サーバー側の実際の状態
colab stop -s <name>
```

Web UI（colab.research.google.com）の「ランタイム」→「セッションの管理」からも
同じセッションを確認・終了できる。CLI と Web UI は同一バックエンド
（`colab.research.google.com/tun/m/*`）を共有しているため、
**Web UI で作ったセッションも `colab sessions` に出てくる**（名前は `?` で表示される）。

### 課金・利用制限

無料枠では金銭的な課金は発生しない。ただしセッションの放置は利用制限枠を消費し、
GPU の割り当てが受けられなくなる。Colab Pro / Pay-As-You-Go 契約がある場合は
コンピューティングユニットを実際に消費する。

### エラーで停止しない

`.ipynb` 実行時、**セルがエラーになっても中断せず次のセルへ進む。**
fail-fast ではないので、長いノートブックでは途中のエラーを見落としやすい。

**しかも `colab exec` の終了コードはこの場合も 0 になる。**
スクリプトから回すときは終了コードを信用せず、ログを見て判定すること。

```bash
colab exec -s work -f notebook.ipynb --timeout 1800 2>&1 | tee run.log
grep -qE "^(Traceback|[A-Za-z]*Error:)" run.log && echo "セルエラーあり"
```

長時間ジョブをバックグラウンドに回すときは、**二重にバックグラウンド化しない**こと
（`nohup ... &` をさらにバックグラウンド実行すると、即座に終了するラッパーの
終了コードを本体の完了と取り違える）。

### 非対話環境で実行してはいけないコマンド

`colab repl` / `colab console` / `colab auth` / `colab drivemount` は TTY 前提。
非対話環境（スクリプト・CI・エージェント）から起動するとハングする。

### Google Drive は経由しない

`.ipynb` はローカルで `nbformat` によりパースされ、コードセルのみが
WebSocket 経由で VM のカーネルへ送られる。**Drive にアップロードされることはない。**
`colab upload` / `download` / `ls` の宛先も VM の Jupyter Contents API であって Drive ではない。
Drive が関わるのは `colab drivemount` とスクリプト内の `drive.mount()` のみ。

---

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| `Missing required parameter: scope` / `エラー 400` | 認可URLが途中で切れている。ターミナルを経由させない `colab-cli/auth.sh` を使う |
| `AttributeError: ... has no attribute 'KernelClient'` | 0.6.x に `jupyter-kernel-client` 1.0 以上が入っている。`colab-cli/setup.sh` を流し直す（0.7.5 になる） |
| `TimeoutError: Timeout waiting for output` | `--timeout` 不足。総実行時間の締切なので大きめの値を渡す |
| `StdinNotImplementedError` | `input()` を使っている。同一セル内での差し替えレシピを使う |
| セルの途中でハングして全体が異常終了 | `files.upload()` が原因。`colab upload` で事前配置し、`files.upload` を差し替える |
| `TimeoutException: Requesting secret ... timed out` | Colab Secrets は CLI からは取得できない。鍵をファイルで渡して `userdata.get` を差し替える |
| セルは全部通ったのにファイルが無い | `files.download()` は CLI では無反応。`colab download` で取る |
| `Cannot download a directory` | `colab download` はディレクトリ非対応。VM 側で zip にしてから取る |
| `403` / keep-alive が止まる | `colaboratory` スコープ不足（同意画面でチェックを外した）。`colab whoami` で確認し `colab-cli/auth.sh --force` |
| `Session ... appears to be lost (404/401)` | VM が失われている。`colab new` で作り直す |

ログは `colab log -s <name>` と `~/.config/colab-cli/colab.log` で追える。

---

## 関連ファイル

ディレクトリ内のファイル一覧は [`../README.ja.md`](../README.ja.md#このディレクトリの中身) を参照。

## 参考

- [公式リポジトリ](https://github.com/googlecolab/google-colab-cli)
- [参考にした記事](https://note.com/npaka/n/n3ad5e2331e00)
