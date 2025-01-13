# natsume-simple

[![CI](https://github.com/borh/natsume-simple/actions/workflows/ci.yaml/badge.svg)](https://github.com/borh/natsume-simple/actions/workflows/ci.yaml)

## 概要

natsume-simpleは日本語の係り受け関係を検索できるシステム

## 開発環境のセットアップ

本プロジェクトには以下の3つの開発環境のセットアップ方法があります：

### Dev Container を使用する場合

[VSCode](https://code.visualstudio.com/)と[Dev Containersの拡張機能](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)をインストールした後：

1. このリポジトリをクローン：
```bash
git clone https://github.com/borh/natsume-simple.git
```

2. VSCodeでフォルダを開く
3. 右下に表示される通知から、もしくはコマンドパレット（F1）から「Dev Containers: Reopen in Container」を選択（natsume-simple）
4. コンテナのビルドが完了すると、必要な開発環境が自動的に設定されます

#### Codespaces

上記はローカルですが、Codespacesで作り、Githubクラウド上の仮想マシンで入ることもできます。
その場合は、Github上の「Code」ボタンから「Codespaces」を選択し、「Create codespace on main」でブラウザ経由でDev Containerに入れます。
また、同じことはVSCode内の[GitHub Codespaces](https://marketplace.visualstudio.com/items?itemName=GitHub.codespaces)拡張機能でもできます。

### Nixを使用する場合

1. [Determinate Nix Installer](https://github.com/DeterminateSystems/nix-installer)でNixをインストール：
```bash
curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | \
  sh -s -- install
```

2. プロジェクトのセットアップ：
```bash
git clone https://github.com/borh/natsume-simple.git
cd natsume-simple
nix develop
```

### 手動セットアップの場合

以下のツールを個別にインストールする必要があります：

- git
- Python 3.12以上
- [uv](https://github.com/astral/uv)
- Node.js
- pandoc

その後：

```bash
git clone https://github.com/borh/natsume-simple.git
cd natsume-simple
uv sync --extra backend
cd natsume-frontend && npm install && npm run build && cd ..
```

## Nixフレークの使用方法

このプロジェクトはNixフレークを使用して開発環境とビルドを管理しています。以下のコマンドが利用可能です：

### 開発環境

```bash
# 開発環境に入る
nix develop

# または direnvを使用している場合（推奨）
direnv allow
```

### ビルドとテスト

```bash
# テストを実行
nix run .#test

# リンター実行
nix run .#lint

# 開発サーバー起動（ホットリロード有効）
nix run .#dev-server

# プロダクションサーバー起動
nix run .#server
```

### その他の便利なコマンド

```bash
# フォーマット
nix fmt

# ビルド
nix build

# 開発環境のシェル情報を表示
nix develop --print-build-logs
```

## CLI使用方法

### 処理の順序

データの準備から検索システムの利用まで、以下の順序で処理を行う必要があります：

1. データの準備（JNLPコーパスとTEDコーパスのダウンロードと前処理）
2. パターン抽出（各コーパスからのNPVパターンの抽出）
3. サーバーの起動（検索インターフェースの提供）

### 1. データの準備

```bash
python src/natsume_simple/data.py --prepare
```

オプション：
- `--data-dir PATH` - データを保存するディレクトリ（デフォルト: ./data）
- `--seed INT` - 乱数シードの設定（デフォルト: 42）

この処理で以下が実行されます：
- JNLPコーパスのダウンロードと変換
- TEDコーパスのダウンロードと前処理
- 前処理済みデータの保存

### 2. データの読み込み

準備済みデータを読み込む場合：

```bash
python src/natsume_simple/data.py --load \
    --jnlp-sample-size 3000 \
    --ted-sample-size 30000
```

オプション：
- `--jnlp-sample-size INT` - JNLPコーパスからのサンプルサイズ（デフォルト: 3000）
- `--ted-sample-size INT` - TEDコーパスからのサンプルサイズ（デフォルト: 30000）
- `--data-dir PATH` - データディレクトリの指定（デフォルト: ./data）
- `--seed INT` - 乱数シードの設定（デフォルト: 42）

この処理で以下が実行されます：
- 準備済みコーパスの読み込み
- 指定されたサイズでのサンプリング
- サンプリングされたコーパスの保存（{コーパス名}-corpus.txt形式）

注意：
- 各コマンドは必要な依存関係がインストールされていることを前提としています
- エラーが発生した場合は、依存関係のインストール状態を確認してください
- コーパスファイルは標準的な命名規則（{コーパス名}-corpus.txt）で保存されます

### 3. パターン抽出

```bash
# JNLPコーパス用
python src/natsume_simple/pattern_extraction.py \
    --model ja_ginza_bert_large \
    --corpus-name "JNLP" \
    data/jnlp-corpus.txt \
    data/jnlp_npvs_ja_ginza_bert_large.csv

# TEDコーパス用
python src/natsume_simple/pattern_extraction.py \
    --model ja_ginza_bert_large \
    --corpus-name "TED" \
    data/ted-corpus.txt \
    data/ted_npvs_ja_ginza_bert_large.csv
```

オプション：
- `--model NAME` - 使用するspaCyモデル（オプション、デフォルト: `ja_ginza_bert_large`）
- `--corpus-name NAME` - コーパス名の指定（デフォルト: "Unknown"）
- `--seed INT` - 乱数シードの設定（デフォルト: 42）

この処理で以下が実行されます：
- コーパスの読み込み
- NPVパターンの抽出
- 結果のCSVファイルへの保存

### 4. サーバーの起動

```bash
uv run fastapi dev src/natsume_simple/server.py
```

このコマンドでは，server.pyをFastAPIでウエブサーバで起動し，ブラウザからアクセス可能にする。

オプション：
- `--reload` - コード変更時の自動リロード（開発時のみ）
- `--host HOST` - ホストの指定（デフォルト: 127.0.0.1）
- `--port PORT` - ポートの指定（デフォルト: 8000）

注意：
- `server.py`では，モデルの指定があるのでご注意。
- サーバを起動後は，出力される手順に従い，<http://127.0.0.1:8000/>にアクセスする。
- FastAPIによるドキュメンテーションは<http://127.0.0.1:8000/docs>にある。
- 環境によっては<http://0.0.0.0:8000>が<http://127.0.0.1:8000>と同様ではない

## 機能

- 特定の係り受け関係（名詞ー格助詞ー動詞，名詞ー格助詞ー形容詞など）における格助詞の左右にある語から検索できる
- 検索がブラウザを通して行われる
- 特定共起関係のジャンル間出現割合
- 特定共起関係のコーパスにおける例文表示

### CLIコマンド

以下のコマンドが利用可能です：

#### プロジェクト外から実行する場合
```bash
nix run github:borh-lab/natsume-simple#コマンド名
```

#### プロジェクト内から実行する場合
以下のコマンドを直接実行できます：

#### 開発ワークフロー
- `watch` - 開発サーバー（バックエンド＋フロントエンド）の起動

#### フロントエンド
- `build-frontend` - プロダクション用フロントエンドのビルド
- `watch-frontend` - 開発モードでのフロントエンド起動（ホットリロード有効）

#### サーバー
- `watch-dev-server` - 開発モードでのバックエンドサーバー起動
- `watch-prod-server` - プロダクションモードでのバックエンドサーバー起動

#### データ管理
- `prepare-data` - コーパスサンプルの準備と読み込み
- `extract-patterns` - すべてのコーパスからのパターン抽出

#### テストと品質管理
- `lint` - すべてのリンターとフォーマッターの実行
- `run-tests` - pytestによるテストスイートの実行

#### メイン
- `run-all` - データ準備、パターン抽出、サーバー起動の一括実行

注意：
- 各コマンドは自動的に必要な依存関係を設定しています
- `nix develop`で入る開発環境には以下が含まれています：
  - Python 3.12
  - Node.js (Svelteのフロントエンド)
  - uv（Pythonパッケージマネージャー）
  - pandoc
  - その他開発に必要なツール

### 開発サーバー

フロントエンドとバックエンドの開発サーバーを同時に起動するには：

```bash
nix run .#watch
```

これにより両サーバーがホットリロード機能付きで起動します。

### 環境変数

- `ACCELERATOR` - 現在のアクセラレータタイプ（cpu/cuda/rocm）
- `PC_PORT_NUM` - プロセスコンポーズのポート番号（デフォルト：10011）

### 開発環境

開発環境に入るには：

```bash
# 開発環境に入る
nix develop

# または direnvを使用している場合（推奨）
direnv allow
```

注意：
- 各コマンドは自動的に必要な依存関係をインストールします
- `nix develop`で入る開発環境には以下が含まれています：
  - Python 3.12
  - Node.js
  - uv（Pythonパッケージマネージャー）
  - pandoc
  - その他開発に必要なツール

## プロジェクト構造

このプロジェクトは以下のファイルを含む：

## プロジェクト構造

```
.
├── data/                        # コーパスとパターン抽出結果
│   ├── ted_corpus.txt           # TEDコーパス
│   ├── jnlp_npvs_*.csv          # 自然言語処理コーパスのパターン抽出結果
│   └── ted_npvs_*.csv           # TEDコーパスのパターン抽出結果
│
├── notebooks/                   # 分析・可視化用Jupyterノートブック
│   ├── pattern_extraction.ipynb # パターン抽出処理の開発用
│   └── visualization.ipynb      # データ可視化用
│
├── natsume-frontend/            # Svelteベースのフロントエンド
│   ├── src/                     # アプリケーションソース
│   │   ├── routes/              # ページルーティング
│   │   └── tailwind.css         # スタイル定義
│   ├── static/                  # 静的アセット
│   └── tests/                   # フロントエンドテスト
│
├── src/natsume_simple/          # バックエンドPythonパッケージ
│   ├── server.py                # FastAPIサーバー
│   ├── data.py                  # データ処理
│   ├── pattern_extraction.py    # パターン抽出ロジック
│   └── utils.py                 # ユーティリティ関数
│
├── scripts/                     # データ準備スクリプト
│   ├── get-jnlp-corpus.py       # コーパス取得
│   └── convert-jnlp-corpus.py   # コーパス変換
│
├── tests/                       # バックエンドテスト
│   └── test_models.py           # モデルテスト
│
├── pyproject.toml               # Python依存関係定義
├── flake.nix                    # Nix開発環境定義
└── README.md                    # プロジェクトドキュメント
```

### data

各種のデータはdataに保存する。
特にscriptsやnotebooks下で行われる処理は，最終的にdataに書き込むようにする。

### notebooks

特に動的なプログラミングをするときや，データの性質を確認したいときに活用する。
ここでは，係り受け関係の抽出はすべてノートブック上で行う。

VSCodeなどでは，使用したいPythonの環境を選択の上，実行してください。
Google Colabで使用する場合は，[リンク](https://colab.research.google.com/drive/1pb7MXf2Q-4MkadWHmzUrb-qXsAVG4--T?usp=sharing)から開くか，`pattern_extraction_colab.ipynb`のファイルをColabにアップロードして利用する。

Jupyter Notebook/JupyterLabでは使用したPythonの環境をインストールの上，Jupterを立ち上げてください。

```bash
jupyter lab
```

右上のメニューに選択できない場合は環境に入った上で下記コマンドを実行するとインストールされる：

```bash
python -m ipykernel install --user --name=$(basename $VIRTUAL_ENV)
```

### natsume-frontend

[Svelte 5](https://svelte.dev/)で書かれた検索インターフェース。

Svelteのインターフェース（html, css, jsファイル）は以下のコマンドで生成できる：
（`natsume-frontend/`フォルダから実行）

```bash
npm install
npm run build
```

Svelteの使用にはnodejsの環境整備が必要になる。

#### static

サーバ読む静的ファイルを含むファルダ。
上記の`npm run build`コマンド実行で`static/`下にフロントエンドのファイルが作成される。
ここに置かれるものは基本的にAPIの`static/`URL下で同一ファイル名でアクセス可能。

# 開発向け情報

## GiNZA/spaCyのモデル使用（Pythonコードから）

係り受け解析に使用されるモデルを利用するために以下のようにloadする必要がある。
環境設定が正常かどうかも以下のコードで検証できる。

```python
import spacy
nlp = spacy.load('ja_ginza_bert_large')
```

あるいは

```python
import spacy
nlp = spacy.load('ja_ginza')
```

notebooksにあるノートブックでは，優先的に`ja_ginza_bert_large`を使用するが，インストールされていない場合は`ja_ginza`を使用する。

## ノートブックからのプログラム改良

プロジェクト環境内でノートブックを作れば，`from natsume_simple.pattern_extraction import normalize_verb_span`など個別に関数をインポートし，動的にテストすることができる。
