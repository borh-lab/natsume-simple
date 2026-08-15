# Lightweight Server Dependencies

Status: Approved for implementation on 2026-08-15.

## Purpose

Keep the public read-only server runnable without the corpus builder, spaCy, GiNZA,
Torch, accelerator libraries, or NLP models, and make both the Nix and uv workflows
prove that boundary.

## Current evidence

- `natsume_simple.api` imports DuckDB, Pydantic, AnyIO, FastAPI, and the local
  DuckDB-only artifact validator. It does not import the corpus pipeline or NLP code.
- The Nix `serverPython` environment selects only the `backend` extra.
- The Nix server closure check rejects several builder and accelerator dependencies,
  and the server smoke check starts the packaged service against a fixture artifact.
- Pydantic is currently a base dependency even though only the API uses it.
- AnyIO is imported directly by the API but currently arrives only as a transitive
  FastAPI dependency.
- The README documents the Nix server path but not the equivalent uv-only backend
  launch.

## Considered approaches

1. **Keep the runtime as-is and only document it.** Lowest churn, but leaves dependency
   ownership inaccurate and AnyIO implicit.
2. **Recommended: correct dependency ownership and strengthen existing checks.** Move
   Pydantic into `backend`, declare AnyIO there, document the existing uvicorn entry
   point, extend the Nix closure denylist, and add a uv backend-only import/startup
   smoke test. This reuses the existing API and launch path.
3. **Add a second `natsume-serve` Python CLI.** Gives uv a console script, but duplicates
   argument and environment handling already owned by the Nix wrapper and Uvicorn. No
   present consumer requires that extra interface.

Approach 2 is selected.

## Design

### Dependency ownership

The base project dependencies contain DuckDB, which is shared by artifact validation,
registry operations, building, and serving. The `backend` extra contains Pydantic,
AnyIO, FastAPI, and Uvicorn because the server directly uses those packages.

The builder, CPU, CUDA, ROCm, and model extras remain independent. Selecting
`backend` must not select any of them.

### uv launch path

The README documents this existing primitive rather than introducing another CLI:

```bash
NATSUME_ARTIFACT_DIR=deploy/current \
uv run --locked --no-dev --extra backend \
  uvicorn natsume_simple.api:app
```

The artifact must contain a valid `manifest.json` and `corpus.duckdb`. Missing NLP
dependencies are not an artifact-availability error.

### Verification

The existing Nix server closure check additionally rejects CuPy, ROCm, Triton,
Transformers, and Tokenizers. Its present consumer is the production server package:
it prevents accelerator or model dependency regressions from entering the closure.

A uv backend-only smoke command runs with `--locked --no-dev --extra backend` and
imports or starts the API against the existing fixture artifact without any builder
extra. It reuses an existing test or script where possible; no new server entry point
or general smoke framework is introduced.

Dependency-configuration tests pin the ownership contract so a later edit cannot move
Pydantic back to the base set or leave AnyIO implicit.

## Deliberate omissions

- Do not split `api.py`; dependency separation already occurs at its import boundary.
- Do not add a Python server CLI while Uvicorn and the Nix wrapper serve all current
  launch consumers. Revisit only if a non-Nix installed-package workflow needs stable
  server arguments beyond Uvicorn's interface.
- Do not make the default test suite dependency-free. This change protects the runtime
  server closure and the explicit backend-only uv path, not every developer test tool.

## Acceptance criteria

1. Base dependencies contain DuckDB but not Pydantic or AnyIO.
2. The `backend` extra directly declares Pydantic and AnyIO alongside FastAPI and
   Uvicorn.
3. The documented uv backend-only command imports or starts the server against a valid
   fixture artifact.
4. The Nix server closure contains none of the named NLP, builder, model, notebook, or
   accelerator dependency families.
5. Existing server smoke, backend tests, source-quality checks, and dependency lock
   validation pass.
