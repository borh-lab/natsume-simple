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
- `ensure-database` - Ensure database exists and is up-to-date
- `initial-setup` - Initialize Python environment and dependencies

### Data Management
- `prepare-data` - Prepare corpus data and load into database
- `extract-patterns` - Extract patterns (collocations)

### Testing & QC
- `lint` - Run all linters and formatters
- `run-tests` - Run the test suite with pytest

### Main
- `run-all` - Initialize database, prepare data, extract patterns and start server

### Environment Variables
- `ACCELERATOR` - Current accelerator type (cpu/cuda/rocm)
- `PC_PORT_NUM` - Process compose port (default: 10011)

Type `h` to see this command overview again

Note: The default command (`nix run`) will start the backend server in production mode (`watch-prod-server`).

## Deployment

To deploy and run the application:

```bash
nix run github:borh-lab/natsume-simple
```

Requirements:
- At least 3.6GB of free disk space (for the database and application files)
- Nix package manager installed (see Setup section above)

The application will:
1. Create a clean distribution directory
2. Download and set up all dependencies
3. Build the frontend
4. Download the corpus database
5. Start the server

By default, the server runs on `localhost:8000`. You can override these with environment variables:

```bash
env NATSUME_HOST=0.0.0.0 NATSUME_PORT=9000 nix run github:borh-lab/natsume-simple
```

## Python Module CLI Examples

### Data Processing (data.py)

```bash
# Process all standard corpora
python src/natsume_simple/data.py --corpus-type all --data-dir data

# Process specific corpus types
python src/natsume_simple/data.py --corpus-type jnlp --name "自然言語処理" --data-dir data
python src/natsume_simple/data.py --corpus-type ted --name "TED" --data-dir data
python src/natsume_simple/data.py --corpus-type wikipedia --name "Wikipedia" --data-dir data

# Process generic corpus with custom directory
python src/natsume_simple/data.py --corpus-type generic --name "my-corpus" --dir path/to/corpus --data-dir data
```

### Pattern Extraction (pattern_extraction.py)

```bash
# Process all unprocessed sentences
python src/natsume_simple/pattern_extraction.py --data-dir data

# Process specific corpus with options
python src/natsume_simple/pattern_extraction.py \
    --data-dir data \
    --model ja_ginza \
    --corpus ted \
    --sample 0.1 \
    --batch-size 1000 \
    --clean \
    --debug

# Process only unprocessed sentences
python src/natsume_simple/pattern_extraction.py \
    --data-dir data \
    --unprocessed-only

# Set custom random seed
python src/natsume_simple/pattern_extraction.py \
    --data-dir data \
    --seed 42
```

### Database Management (database.py)

```bash
# Show row counts for all tables
python src/natsume_simple/database.py \
    --data-dir data \
    --action show-counts

# Clean pattern data while preserving corpus data
python src/natsume_simple/database.py \
    --data-dir data \
    --action clean-patterns
```

### Server (server.py)

```bash
# Run with FastAPI CLI in development mode
uvicorn src.natsume_simple.server:app --reload

# Run with FastAPI CLI in production mode
uvicorn src.natsume_simple.server:app

# Available API endpoints:
# GET /corpus/stats - Get corpus statistics
# GET /corpus/norm - Get normalization factors
# GET /npv/{search_type}/{term} - Search for collocations
# GET /sentences/{n}/{p}/{v}/{limit} - Get example sentences
# GET /search/{query} - Search for terms

# Example API calls:
curl http://localhost:8000/corpus/stats
curl http://localhost:8000/npv/noun/本
curl http://localhost:8000/npv/verb/読む
curl http://localhost:8000/sentences/本/を/読む/5
curl http://localhost:8000/search/読
```

### Utility Modules (no CLI interface)

The following modules provide functionality used by other modules but don't have direct CLI interfaces:

- `log.py` - Logging configuration
- `utils.py` - Utility functions like random seed setting

Example usage in Python code:
```python
from natsume_simple.log import setup_logger
from natsume_simple.utils import set_random_seed

logger = setup_logger(__name__)
set_random_seed(42)
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
├── data/                        # コーパスとパターン抽出結果
│   └── corpus.db                # データベース（ensure-databaseで取得可）
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
│   ├── database.py              # データベース関連
│   ├── data.py                  # データ処理
│   ├── pattern_extraction.py    # パターン抽出ロジック
│   ├── log.py                   # ログ設定
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
