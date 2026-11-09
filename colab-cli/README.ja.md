# `colab-cli/` — GPU マシン無しで、ターミナルから Colab でこの VLA を動かす

[English](README.md) · [運用ガイド](guide/field_guide.ja.md) ·
[VLA のジョブ固有の落とし穴](../docs/colab.md)

公式の [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) で
Google Colab をコマンドラインから操作するための一式です。インストール、ログイン、キーの受け渡し、
疎通の確認、止め忘れの警告までがこのディレクトリに入っています。
[Common Behavior Data](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli) の
`colab-cli/` と同じ構成・同じコマンドなので、片方で用意した環境はもう片方でもそのまま使えます。

**手元に GPU は要りません。** ターミナルはブラウザで開いた GitHub Codespace で十分で、
3B のモデルの学習と評価は、そのターミナルから操作する Colab の A100 で動きます。
ローカル（macOS / Linux。Windows は WSL 内）でも同じ手順です。

## クイックスタート

リポジトリのルートで:

```bash
colab-cli/setup.sh        # uv と google-colab-cli 0.7.5 を入れる（何度流してもよい）
colab-cli/auth.sh         # ログイン。マシンごとに1回。ブラウザでページが開く
colab-cli/smoke_test.sh   # CPU の VM を起動 → 実行 → ファイル回収 → 停止（約1分）
```

必要なのは Google アカウントだけです（Workspace アカウントも可）。`gcloud` も GCP のプロジェクトも要りません。
ここまでは Colab の無料枠で足ります。

## 何に何が要るか

| 工程 | 動かす場所 | 要るもの | 時間と費用の目安 |
|---|---|---|---|
| CPU の通しテスト（[`tests/smoke_test.py`](../tests/smoke_test.py)） | 手元・Codespace・GitHub Actions | なし | 数分・無料 |
| [`smoke_test.sh`](smoke_test.sh)（CLI の疎通） | Colab の CPU | Google アカウント | 約1分・無料枠で可 |
| 推論と評価（`colab/start.sh infer`） | Colab の A100 | 有料の Colab プラン・`HF_TOKEN`・Drive の重み | 目標ファイル1本で約20分・約 1.4 CU |
| 続きの学習（`colab/start.sh train`） | Colab の A100 | 同上 | 5,000 ステップで約1.5時間・約 9 CU |
| 学習用の描画データを作り直す（`colab/go.sh data`） | Colab の A100 | 同上と、各自が AMASS から作った HumanML3D | 作者は不要（Drive にある） |

- **A100 には有料の Colab プランが要ります。** A100 は約 5.3 CU/時です。無料枠の T4 では GR00T N1.7 を学習できません
  （flash-attn 2 が Ampere 以降を要求する）。L4 は未確認です。
- 残高・消費の速さ・確保できる GPU は `colab-cli/tools/colab_ccu.py` で見られます（`colab/start.sh` が始める前に表示します）。
- 重みは今のところ Google Drive の `MyDrive/VLA/VLA_1_0/` から読みます（[重み](../README.ja.md#重み)）。

## クラウドとローカルで違うところ

違いは3つです。1つ目はスクリプトが吸収します。

| | クラウド（Codespaces・SSH） | ローカル（macOS / Linux / WSL） |
|---|---|---|
| **ログイン** | `auth.sh` が `$BROWSER` 経由で**手元のブラウザ**に開く（Codespaces・VS Code Remote）。素の SSH では `ws/.auth/auth_url.txt` とリンク1つの `index.html` を書くか、`--serve 8765` + ポート転送で開く | `auth.sh` が既定のブラウザで開く |
| **`HF_TOKEN`** | Codespaces の Secret: GitHub → Settings → Codespaces → Secrets で、*Repository access* に**このリポジトリ**を加えてから Codespace を **Rebuild**。別のリポジトリにだけ共有した Secret は届かない。届いた名前は `python3 colab-cli/tools/colab_secrets.py --list` で確かめる。使い捨てなら[下の `--set`](#hf_token-を使い捨てにする) | `cp colab-cli/.env.example colab-cli/.env` して記入（gitignore 済み）。または `export` するだけ |
| **残るもの** | Codespace を Rebuild・作り直しすると `~/.local` と `~/.config` が消え、CLI もトークンも無くなる。`setup.sh` と `auth.sh` を流し直す | やり直し不要 |

ログインの中身はどちらも CLI 自身のフローです（ブラウザで同意 → 表示されたコードを貼る）。
`auth.sh` があるのは、認可 URL が約700文字あり、ターミナルからコピーすると途中で切れて
*エラー 400: Missing required parameter: scope* になるためです
（[詳細](guide/field_guide.ja.md#-最大の落とし穴-urlの切れ)）。
エージェントやスクリプト向けには2段階版があります: `auth.sh --url-only` → `auth.sh --code <コード>`。

### `HF_TOKEN` を使い捨てにする

トークンを Secret として置きっぱなしにせず、実行のたびに作って消すこともできます。

```bash
python3 colab-cli/tools/colab_secrets.py --set HF_TOKEN     # 貼り付ける。入力は画面にも履歴にも残らない
bash colab/start.sh infer myrun                             # → 同意 → bash colab/go.sh infer myrun
python3 colab-cli/tools/colab_secrets.py --unset HF_TOKEN   # go.sh がジョブを起動したら消してよい
```

- `--set` は `colab-cli/.env`（gitignore 済み）に権限 600 で書きます。`colab/start.sh` は、キーが見つからないと
  **VM を確保する前に**止まります。
- ジョブは起動したときの環境でトークンを持つので、`go.sh` が起動を報告した後なら手元から消しても実行は続きます。
  セッションを付け直した（`colab_reattach.py`）後にもう一度渡す必要があるときだけ、`--set` し直してください。
- VM の上にはジョブが終わるまでトークンが残ります（`colab stop` で VM ごと消える）。実行が終わったら、
  Hugging Face の Settings → Access Tokens でトークン自体も削除してください。権限は *Read*（または gated のモデルを読める
  fine-grained）で足ります。
- `--set` は端末で入力を受けるので、自分のターミナル（VS Code のターミナルなど）で実行してください。
  トークンをチャットやコマンドの引数に貼らないでください。

## VLA のジョブを流す

VLA のジョブはノートブックではなく、`colab exec` で VM に送るシェルスクリプトです。
`colab/` のスクリプトが、ここの道具を使って A100 のセッションを組み立てます。

```bash
bash colab/start.sh infer myrun        # A100 を確保 → HF_TOKEN を渡す → コードを送る → Drive の同意 URL で止まる
# /tmp/dm_url.txt の URL（または http://localhost:8899）を開いて同意。期限 600 秒
bash colab/go.sh infer myrun           # マウントを済ませ、VM 側の colab/vm_infer.sh を切り離して起動
tail -f /tmp/watch_myrun.log           # 進み具合。終わりは VLA_DONE、異常は VLA_EXIT=非0
colab download -s myrun /content/run/results.tgz work/myrun/results.tgz
colab stop -s myrun && colab sessions  # 必ず止めて、確認する
```

学習は `infer` を `train` に替えるだけです（`STEPS=5000 DUP=40 RUN=t1 bash colab/go.sh train t1`。
自分の clip を使うなら `VIDEO_DATA=<LeRobot データ> bash colab/start.sh train t1`）。

`colab/` のスクリプトは、長い A100 のジョブで踏む次の落とし穴に対処しています。理由と対処の詳細は
[`docs/colab.md`](../docs/colab.md) にあります。

- CLI は `colab new` で得た 60 分のトークンを更新しない → 45 分ごとに `tools/colab_reattach.py --refresh`
- 通信が無いと約 25 分で割り当てが消える → `colab/watch.sh` が2分ごとに軽い `exec` を送る
- Drive の同意はセッションごとに人が要る → `tools/ptyrun.py` で包み、URL をファイルとローカルのページに出す
- `colab exec` が `--timeout` を無視して固まることがある（0.7.5）→ CLI の呼び出しを coreutils の `timeout` で包む
- 席を外すときは `colab/autostop.sh` が、ジョブの終わりか期限で結果を回収して止める

## 守ること

1. **必ず `colab stop -s <name>`。** ターミナルを閉じても VM は解放されず、A100 は止めるまで課金が続きます。
   `colab sessions` で確かめてください。このリポジトリで Claude Code を使うと、止め忘れたセッションがあれば
   応答が終わるたびに警告が出ます（[`.claude/settings.json`](../.claude/settings.json) に登録済みで、設定は要りません）。
2. **CLI を動かしているマシンを止めない。** keep-alive・トークンの更新・見張りはすべて CLI 側のマシンで動くので、
   Codespace が止まると VM も道連れになります。長い実行の前に Codespace の idle timeout を延ばしてください。
3. **`--timeout` を必ず付ける。** 既定は 30 秒で、無音の時間ではなく総実行時間の締切です。長い処理は VM 側で切り離します。
4. **終了コードを信じない。** 失敗しても `colab exec` は 0 で終わることがあります。ログの `VLA_EXIT` と `Traceback` を見てください。
5. **止める前に回収する。** `/content` はセッションと一緒に消えます。`colab download` で取ります（ファイルのみ）。
   Drive に書いた重みは `VLA_WEIGHTS_VERIFIED` が出るまで止めないでください。

キーは `colab_secrets.py` で渡し、**`colab exec` に渡すコードには絶対に書かないでください。**
`colab exec` は実行したコードと出力の両方を `~/.config/colab-cli/history/<session>.jsonl` に平文で追記します。

## このディレクトリの中身

| パス | 内容 |
|---|---|
| [`setup.sh`](setup.sh) | `uv` と `google-colab-cli`（0.7.5 に固定。`COLAB_CLI_VERSION=latest` で変更可）を入れ、`ws/` を作り、Claude Code 用の skill を書き出す |
| [`auth.sh`](auth.sh) → [`tools/auth.py`](tools/auth.py) | URL をターミナルに通さないログイン |
| [`smoke_test.sh`](smoke_test.sh) | new → exec → download → stop。失敗しても必ず止める。追加の引数は `colab new` へ（例: `--gpu T4`） |
| [`tools/colab_secrets.py`](tools/colab_secrets.py) | キー（環境変数 → `.env` → Codespaces Secret の順）をセッションへ渡す。`--set` / `--unset` で使い捨てのキーを保存・削除する。`find_secret()` は手元のスクリプト（`vla/vlm_target.py`）も使う |
| [`tools/ptyrun.py`](tools/ptyrun.py) | `colab drivemount` / `colab auth` を端末なしで完走させる |
| [`tools/colab_reattach.py`](tools/colab_reattach.py) | 60 分のトークンを先回りで更新する（`--refresh`）。CLI が捨てたセッションを、生きている VM に付け直す |
| [`tools/colab_ccu.py`](tools/colab_ccu.py) | CU 残高・消費の速さ・確保できる GPU |
| [`hooks/colab_session_guard.py`](hooks/colab_session_guard.py) | Claude Code の Stop フック。止め忘れたセッションを警告する |
| [`examples/drive/`](examples/drive/) | マウントした Drive 経由で `.py` から書き、`.ipynb` から読む例 |
| [`guide/field_guide.ja.md`](guide/field_guide.ja.md) · [英語版](guide/field_guide.md) | CLI の挙動の実測メモ: ログイン・ノートブック・秘密情報・成果物・Drive・トラブルシューティング |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code がこのディレクトリの道具を使うときの運用ルール |
| [`.env.example`](.env.example) | ローカル用のキーのひな形 |
| `ws/` | 作業ディレクトリ。`.gitkeep` 以外は git 管理外 |
| `.claude/skills/colab-operator/` | `setup.sh` が `colab skill` から生成（インストールした CLI と常に一致。gitignore 済み） |

VLA のジョブそのもの（`start.sh`・`go.sh`・`vm_*.sh`・`watch.sh`・`autostop.sh`）は [`../colab/`](../colab/) にあります。

## 状態 — 何を動かし、何をまだ動かしていないか

- **A100 での実行（0.7.5、2026-11-05、Codespace 上）:** `colab/start.sh` → 同意 → `colab/go.sh train` で、
  500 ステップの学習と評価2本が `VLA_DONE`・`VLA_EXIT=0` まで通りました。このときは道具が `tools/` にあり、
  CLI は手で入れていました（`jupyter-kernel-client<1.0` を余分に固定。0.7.5 は自分で `==0.9.0` に固定しているので同じ版）。
- **このディレクトリへ移した後に確かめたこと（2026-11-09、Codespace 上）:** `setup.sh` の再実行（入れ直さずに skill を生成）、
  ログイン済みでの `auth.sh`（`colaboratory` スコープあり）、Stop フック（セッションあり・なし・CLI なしの3通り）、
  `colab_secrets.py --list`、`colab/common.sh` が CLI の Python と `colab-cli/tools/` を見つけること。
  `smoke_test.sh` も通った（CPU ランタイム、VM は Python 3.13.18・GPU なし、ファイルを回収して停止。`colab sessions` で残りなし）。
- **移した後の A100 での通し（2026-11-09、Codespace 上）:** 使い捨ての `HF_TOKEN`（`--set`）で `colab/start.sh infer` → 同意 →
  `colab/go.sh infer` → `autostop.sh` による回収と停止まで通り、`VLA_DONE`・`VLA_EXIT=0`。目標ファイル1本の結果は
  README の「誘導あり・目標は VLM が選んだ位置」と同じ（10 cm 以内 1.00 / 1.00 / 0.89、持てた割合 0.93）。
  A100 の確保は2回 `503 Service Unavailable` で断られ、30分後の3回目で取れた。展開の `colab exec` が固まり、手で流し直した
  （その後、`start.sh` の展開の段と `colab_secrets.py` と `smoke_test.sh` にも締切とやり直しを入れた。
  最初の `exec` だけ固まる偽の `colab` で、打ち切ってやり直す・止めるところまでは確かめたが、本物の固まりではまだ通していない）。
- **まだ確かめていないこと:**
  - 学習（`train`）を移した後に流すこと。
  - まっさらな環境での `setup.sh` と、初回の `auth.sh`（CBD 側では 0.7.5 で通っている）。
  - ローカルマシン。

どれかが動かなかったら、それ自体が知見なので issue にしてください。

## 由来

[common-behavior-data の `colab-cli/`](https://github.com/Koichi3333/common-behavior-data/tree/main/colab-cli) をもとにしました。
`setup.sh`・`auth.sh`・`tools/auth.py`・Stop フック・運用ガイド・`examples/drive/` は同じものです。変えたところ:

- `tools/` にあった VLA 用の CLI 補助（`colab_reattach.py`・`colab_ccu.py`・`colab_secrets.py`・`ptyrun.py`）をここに集めた。
  `colab_secrets.py` と `ptyrun.py` は CBD 版に置き換え、`colab_secrets.py` には手元のスクリプト用の `find_secret()` を足した。
- Stop フックを jq を使うシェルから CBD の Python 版に替えた。公式 skill はコミットせず、`setup.sh` が生成する。
- `smoke_test.sh` は `--help` を `colab new` に渡さず使い方を出して終わり、CLI の呼び出しに締切がある（CBD 側にも同じ修正を入れた）。
- `colab_secrets.py` の `upload` と `exec` に締切と3回までのやり直しを入れた（0.7.5 の固まり対策）。
- `.env.example` と `CLAUDE.md` を VLA 向け（`HF_TOKEN`・A100・費用）に書いた。
