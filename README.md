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

公開構成は JNLP、日本語版 Wikipedia、IWSLT 2017 の TED Talks です。TED は
permission が未解決のまま owner decision により含まれます。source provenance、
mixed terms、再取得条件は `docs/corpus-sources.lock.json` と
`docs/corpus-recoverability.md` にあります。WIT³ は release input ではありません。

Corpus source archives and generated databases are release assets, not Git content.
`data/`, `artifacts/`, `*.db`, and `*.duckdb` are ignored. Git records the source lock,
the frozen Wikipedia identity subset, notices, checksums, and release evidence needed to
recreate and audit them.

Acquire the three locked source files, then convert the JNLP archive. `nkf` and `pandoc`
come from the builder closure.

```bash
nix run .#build-corpus -- acquire-release-inputs \
  --source-lock docs/corpus-sources.lock.json \
  --output-directory data/release-inputs
nix run .#build-corpus -- prepare-jnlp \
  data/release-inputs/NLP_LATEX_CORPUS-2026-06-15.zip \
  data/prepared-jnlp-2026
nix run .#build-corpus -- inspect-inputs \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json \
  --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS \
  --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet \
  --ted-iwslt-archive data/release-inputs/ja-en.zip
```

既にローカルにある JNLP directory、checksummed Wikipedia Parquet shard と TED archive、
wtpsplit model、content notices から immutable artifact を build します。

```bash
nix run .#build-corpus -- build \
  --artifacts-directory artifacts \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json \
  --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS \
  --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet \
  --ted-iwslt-archive data/release-inputs/ja-en.zip \
  --splitter-model data/models/wtpsplit \
  --content-license corpus-notices/LICENSE-CONTENT.txt \
  --attribution corpus-notices/ATTRIBUTION.md \
  --max-rejections 200 \
  --max-rejection-fraction 0.15
```

The measured JNLP `missing_source_path` entries are metadata rows marked `*NA*`;
`missing_plain_text` records papers whose declared source could not be converted. Both
remain bounded and visible rather than being silently discarded.

Validate the immutable result before selecting it:

```bash
nix run .#build-corpus -- release-check artifacts/<instance-id> \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json
```

The builder records the verified local sources and model identity. `publish` changes only
the atomic pointer; a running process continues serving the artifact it validated at
startup. Restart or redeploy after publishing. Rollback is publishing the former instance
and restarting again.

```bash
nix run .#build-corpus -- publish artifacts/<instance-id> deploy
nix run .#build-corpus -- current deploy
```

Record host-specific service latency with the fixed release request family after the
server is running. Repeat `--corpus-id` for the selection being measured:

```bash
nix develop .#test --command python -m natsume_simple.benchmark_service \
  --base-url http://127.0.0.1:8000 \
  --corpus-id jnlp --corpus-id wiki --corpus-id ted \
  --requests 500 --concurrency 10 \
  --output /tmp/natsume-benchmark.json
```

This benchmark is diagnostic evidence tied to the named host and artifact. It is not a
CI latency gate.

小さな fixture artifact の build・validation・HTTP walkthrough は次で実行します。

```bash
nix build .#checks.x86_64-linux.server-smoke
```

## Checks and dependency locks

通常の gate は model-free です。

```bash
nix fmt flake.nix
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

The real corpus build, `release-check`, performance measurement, and live-service smoke
are operator release evidence because they require large external inputs and NLP models;
they are intentionally not default CI inputs. The network-independent builder smoke only
checks the packaged command surface with tiny fixtures.

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

The resolved bind mount pins the selected artifact for the container lifetime. After
`publish`, recreate the container to serve the new resolved directory.

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
- `GET /api/collocations?term=...&pos=noun`
- `GET /api/examples?noun=...&particle=を&verb=...`

例:

```bash
curl http://127.0.0.1:8000/api/corpora
curl 'http://127.0.0.1:8000/api/collocations?term=本&pos=noun'
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
