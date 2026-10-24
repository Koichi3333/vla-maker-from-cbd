# 学習データ: 出所・規約・入手の仕方

[English](data.md)

このプロジェクトの重みは 3 種類のデータで学習しています。それぞれに規約があります。このページはそれを
（2026-10-19 に原文で確認して）まとめ、**公式の経路での**入手の仕方を説明します。規約の本文を読んだ整理で、法的な助言ではありません。

| データ | 使いどころ | 規約の要点 | 入手先 |
| --- | --- | --- | --- |
| **AMASS**（HumanML3D 経由） | 学習した 10,427 本のうち 10,064 本（KIT・CMU・BMLmovi・Eyes_Japan・HDM05 など） | 非商用の研究・教育・芸術に限り、**利用者本人に**許諾。学習の禁止は「商用目的の学習」だけ。データ（改変したものを含む）を第三者に渡すのは禁止。AMASS の論文の引用が義務 | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de)（登録と同意） |
| **HumanAct12**（HumanML3D 経由） | 363 本 | 作者が公開で配っている（[action-to-motion](https://github.com/EricGuo5513/action-to-motion)、MIT のリポジトリ） | HumanML3D のリポジトリに同梱（`pose_data/`） |
| **HumanML3D** のテキスト注釈とスクリプト | キャプション・一覧・変換のスクリプト | MIT のリポジトリ。動きのデータそのものは、AMASS の方針で配れないため作者も配っていない | [EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D) |
| **SMPL** の人体モデル | 動画由来の clip の胴の当てはめ（ROMP） | 非商用。派生物の作成は可、本体の再配布は不可、商用目的の学習は不可。SMPL の論文の引用が義務 | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de)（登録と同意） |
| 作者の動画 | `data/video_reach/` の実写 13 本 | 作者が撮影し、写っているのは作者だけ。保存しているのは骨格と物の状態だけ | — |

**使う人にとっての意味**

- **非商用の研究として**モデルを学習・共有することは、これらの規約と合っています。商用の製品のための学習は認められていません。
- **AMASS 由来の動きのデータを再配布しないでください**（HumanML3D の `new_joints` を含む）。第三者のミラーから取ってもいけません。
  各自が、自分の許諾のもとで AMASS から入手します。
- このリポジトリにも、モデルのリポジトリにも、AMASS の動きや SMPL のモデルは入っていません。`vla/standing.json` と
  `vla/full_split.json` は clip の番号・フレーム数と HumanML3D のキャプション（MIT のテキスト注釈）だけで動きは入っておらず、`data/stats/statistics.json` は集計した統計だけです。
- 商用に使うなら、権利者（AMASS・SMPL は ps-license@tue.mpg.de、GR00T は NVIDIA）に確認するか、権利を持つデータで学習し直してください。
  提供してもらう CBD の clip が、そのための道です（[CONTRIBUTING.ja.md](../CONTRIBUTING.ja.md)）。

## HumanML3D を公式の経路で作る

HumanML3D の学習用の描画（`b3_dataset.tar`）を自分で作り直すときだけ要ります。

1. [AMASS](https://amass.is.tue.mpg.de) で**登録して規約に同意**し、[HumanML3D の README](https://github.com/EricGuo5513/HumanML3D) にある
   サブデータセットの **SMPL+H** 版をダウンロードする。
2. HumanML3D が使う**人体モデル**を入手する: Extended SMPL+H（[MANO のサイト](https://mano.is.tue.mpg.de/download.php)）と
   DMPL（[SMPL のサイト](https://smpl.is.tue.mpg.de/download.php)）。
3. HumanML3D のリポジトリで**スクリプトを回す**: `raw_pose_processing.ipynb` → `motion_representation.ipynb`。
   要るのは `HumanML3D/new_joints/`（3D の関節位置、SMPL の 22 関節、20fps）。
4. **そのフォルダを zip にして、自分の保存先に置く。** Colab の段取りで使うなら、Google Drive の
   `MyDrive/VLA/VLA_1_0/data/humanml3d_new_joints.zip` に置く（または `JOINTS=<VM 上のパス>` で渡す）。
5. 学習用の描画を作る:

   ```bash
   bash colab/start.sh data mydata
   bash colab/go.sh data mydata      # HumanML3D → CBD → 117 次元の LeRobot → MuJoCo の描画 → Drive の data/b3_dataset.tar
   ```

   手元では `python3 vla/fetch_and_curate.py --joints <new_joints のフォルダか zip> --all` が、その最初の段です。
   キャプションと一覧を HumanML3D のリポジトリから取り、自分で作った `new_joints` を所定の場所に置きます。

## 引用

このモデルやデータの段取りを使って論文などを書くときは、土台にしたデータとモデルを引用してください。
BibTeX は [英語版](data.md#citations) にあります（AMASS・SMPL・HumanML3D・GR00T N1）。
HumanAct12 は C. Guo et al., *Action2Motion: Conditioned Generation of 3D Human Motions*, ACM Multimedia 2020。
