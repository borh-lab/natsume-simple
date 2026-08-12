# natsume-simple

[![CI](https://github.com/borh/natsume-simple/actions/workflows/ci.yaml/badge.svg)](https://github.com/borh/natsume-simple/actions/workflows/ci.yaml)

日本語コーパスから名詞・格助詞・動詞の共起を検索し、コーパス別の頻度と
例文を比較するための小さな公開サービスです。Nix flake が開発環境、検査、
パッケージ、OCI イメージの正本です。

## Quick start

Nix をインストールしてリポジトリを取得します。

```bash
git clone https://github.com/borh/natsume-simple.git
cd natsume-simple
nix flake check
```

既存の schema-v1 corpus artifact を起動するには、読み取り専用の artifact
directory を明示します。

```bash
nix run .#serve -- --artifact-dir deploy/current --port 8000
```

`http://127.0.0.1:8000/api/health/ready` が `200` なら検索 UI も同じ port で
利用できます。artifact は `manifest.json` と `corpus.duckdb` を含みます。
公開用 artifact にはさらに `LICENSE-CONTENT.txt` と `ATTRIBUTION.md` を置き、
ソフトウェアの MIT license と corpus content の license を混同しないでください。

## Development shells

用途ごとに closure を分けています。shell へ入っても依存同期、build、server
起動、checkout 内の virtualenv 作成は行いません。

```bash
nix develop .#server
nix develop .#builder
nix develop .#frontend
nix develop                 # 上記の union
```

Dev Container / Codespaces は `.devcontainer/devcontainer.json` から同じ flake を
使います。別の Dockerfile、rootless variant、ROCm variant はありません。

### Backend and frontend development

Backend:

```bash
nix develop .#server --command \
  uvicorn natsume_simple.api:app --reload --host 127.0.0.1 --port 8000
```

`NATSUME_ARTIFACT_DIR` の既定値は `deploy/current` です。別の artifact を使う
場合は環境変数で指定してください。

Frontend（別 terminal）:

```bash
nix develop .#frontend --command bash -lc \
  'cd natsume-frontend && npm ci && VITE_API_URL=http://127.0.0.1:8000 npm run dev'
```

`npm ci` は明示的な checkout setup です。Nix package と CI derivation は
committed lock から network-independent に frontend を build します。

## Corpus pipeline

公開構成は JNLP と日本語版 Wikipedia です。TED content は permission が明確に
なるまで production artifact に入りません。source provenance と再取得条件は
`docs/corpus-sources.lock.json` と `docs/corpus-recoverability.md` にあります。

JNLP archive をローカルで変換します。`nkf` と `pandoc` は builder closure から
供給されます。

```bash
nix run .#build-corpus -- prepare-jnlp \
  data/NLP_LATEX_CORPUS.zip data/prepared-jnlp
```

既にローカルにある JNLP directory、checksummed Wikipedia Parquet shards、
wtpsplit model、content notices から immutable artifact を build します。

```bash
nix run .#build-corpus -- build \
  --artifacts-directory artifacts \
  --jnlp-root data/prepared-jnlp/NLP_LATEX_CORPUS \
  --wikipedia-parquet data/wikipedia/train-00000-of-00015.parquet \
  --splitter-model data/models/wtpsplit \
  --content-license LICENSE-CONTENT.txt \
  --attribution ATTRIBUTION.md
```

`--wikipedia-parquet` は shard ごとに繰り返します。builder は input を取得せず、
渡されたローカル source と model identity を記録します。完成した artifact を
atomic pointer で選択します。

```bash
nix run .#build-corpus -- publish artifacts/<instance-id> deploy
nix run .#build-corpus -- current deploy
```

小さな fixture artifact の build・validation・HTTP walkthrough は次で実行します。

```bash
nix build .#checks.x86_64-linux.server-smoke
```

## Checks and dependency locks

通常の gate は model-free です。

```bash
nix fmt
nix flake check --print-build-logs
nix build .#frontend .#server .#corpus-builder-cpu
nix build .#container                    # x86_64-linux
nix build .#nlp-model-integration        # release/scheduled NLP gate
```

Formatting changes files; checks do not. Dependency locks are updated only by explicit
commands, followed by the full gate:

```bash
uv lock --upgrade
(cd natsume-frontend && npm install)
nix flake update
nix flake check
```

CPU and CUDA Python extras are mutually exclusive. The published server and image are
CPU-only and contain no Torch, spaCy, GiNZA, wtpsplit, Polars, Node, Jupyter, or CUDA
closure.

## OCI image

The image is derived from the same `server` package; it is not a second deployment
definition.

```bash
image_path=$(nix build .#container --no-link --print-out-paths)
podman load --input "$image_path"
podman images natsume-simple

podman run --rm --read-only --user 65532:65532 \
  -p 8000:8000 \
  -v "$(readlink -f deploy/current):/var/lib/natsume/artifact:ro" \
  natsume-simple:<tag-shown-by-podman-load>
```

Production must put the service behind a reverse proxy (or equivalent edge) with an
initial limit of 2 API requests/second per source IP, burst 5, at most 4 concurrent API
requests per source IP, and a 3-second upstream timeout. The application independently
rejects work beyond its global 16-query capacity. The artifact mount and image root stay
read-only. `deploy/nginx.conf` is the checked reference configuration: public traffic
enters port 8080, while host-local probes use port 8081 (`/live` and `/ready`).

## API

- `GET /api/health/live`
- `GET /api/health/ready`
- `GET /api/corpora`
- `GET /api/suggestions?q=...&pos=noun`
- `GET /api/collocations?term=...&pos=noun&rankBy=raw`
- `GET /api/examples?noun=...&particle=を&verb=...`

例:

```bash
curl http://127.0.0.1:8000/api/corpora
curl 'http://127.0.0.1:8000/api/collocations?term=本&pos=noun&rankBy=meanPerMillion'
```

## Code walkthrough

- `src/natsume_simple/data.py`: Japanese text and source preparation primitives
- `src/natsume_simple/pattern_extraction.py`: normalization and NPV extraction
- `src/natsume_simple/corpus_pipeline.py`: adaptation → segmentation → extraction
- `src/natsume_simple/artifact_builder.py`: canonical schema-v1 persistence
- `src/natsume_simple/artifact_registry.py`: validate and atomically select artifacts
- `src/natsume_simple/api.py`: read-only public service
- `natsume-frontend/src/`: Svelte search interface
- `tests/test_teaching_examples.py`: fixture-heavy executable boundary examples

The notebooks import the extraction implementation rather than maintaining a second copy.
