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

## 開発環境の入り方

開発環境に入るには以下のコメントを実行：

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


## CLI Commands

The following commands are available after entering the development environment:

### Development Workflow
- `watch-all` - Start development servers (backend + frontend)

### Frontend
- `build-frontend` - Build the frontend for production
- `watch-frontend` - Start frontend in development mode with hot reload

### Server
- `watch-dev-server` - Start backend server in development mode
- `watch-prod-server` - Start backend server in production mode

### Setup
- `initial-setup` - Initialize Python environment and dependencies

### Testing & QC
- `lint` - Run all linters and formatters
- `run-tests` - Run the test suite with pytest

### Environment Variables
- `ACCELERATOR` - Current accelerator type (cpu/cuda)
- `PC_PORT_NUM` - Process compose port (default: 10011)

Type `h` to see this command overview again

Note: The default command (`nix run`) will start the backend server in production mode (`watch-prod-server`).

## Corpus pipeline

The offline pipeline in `corpus_pipeline.py` converts a local JNLP archive,
adapts converted JNLP and pinned local Wikipedia Parquet files, segments and
extracts them, and writes a fresh immutable schema-v1 artifact. It never updates
the deployed DuckDB file in place. The small executable walkthrough is covered
by `tests/test_corpus_adapters.py` and `tests/test_teaching_examples.py`.

### Server (api.py)

```bash
# Serve a published schema-v1 artifact in development mode
NATSUME_ARTIFACT_DIR=deploy/current uvicorn natsume_simple.api:app --reload

# Production mode
NATSUME_ARTIFACT_DIR=deploy/current uvicorn natsume_simple.api:app

# Available API endpoints:
# GET /api/corpora - List corpora and their counts
# GET /api/suggestions?q=...&pos=noun - Suggest lemmas
# GET /api/collocations?term=...&pos=noun&rankBy=raw - Search collocations
# GET /api/examples?noun=...&particle=...&verb=... - Get example sentences

# Example API calls:
curl http://localhost:8000/api/corpora
curl 'http://localhost:8000/api/collocations?term=本&pos=noun&rankBy=meanPerMillion'
curl 'http://localhost:8000/api/examples?noun=本&particle=を&verb=読む'
```

## Nix Flake Usage

This project uses Nix flakes to manage the development environment and builds.
The following commands are available:

```bash
# Format
nix fmt

# Build
nix build .#command-name
# Build results are linked in ./results

# Show development shell info
nix develop --print-build-logs
```

## 機能

- 特定の係り受け関係（名詞ー格助詞ー動詞，名詞ー格助詞ー形容詞など）における格助詞の左右にある語から検索できる
- 検索がブラウザを通して行われる
- 特定共起関係のジャンル間出現割合
- 特定共起関係のコーパスにおける例文表示

## プロジェクト構造

このプロジェクトは以下のファイルを含む：

## プロジェクト構造

```
.
├── data/                        # ローカルのコーパス入力
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
│   ├── api.py                   # FastAPIサーバー
│   ├── artifact_builder.py      # DuckDBアーティファクト生成
│   ├── artifact_registry.py     # 公開・ロールバック
│   ├── corpus_pipeline.py       # コーパス変換パイプライン
│   ├── data.py                  # データ処理
│   └── pattern_extraction.py    # パターン抽出ロジック
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

`npm run build` writes the deployable frontend to `natsume-frontend/build/`.

# 開発向け情報

## GiNZA/spaCyのモデル使用（Pythonコードから）

係り受け解析に使用されるモデルを利用するために以下のようにloadする必要がある。
環境設定が正常かどうかも以下のコードで検証できる。

```python
import spacy

nlp = spacy.load("ja_ginza_bert_large")
```

あるいは

```python
import spacy

nlp = spacy.load("ja_ginza")
```

notebooksにあるノートブックでは，優先的に`ja_ginza_bert_large`を使用するが，インストールされていない場合は`ja_ginza`を使用する。

## ノートブックからのプログラム改良

プロジェクト環境内でノートブックを作れば，`from natsume_simple.pattern_extraction import normalize_verb_span`など個別に関数をインポートし，動的にテストすることができる。
