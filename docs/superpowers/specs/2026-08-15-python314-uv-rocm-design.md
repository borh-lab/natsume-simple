# Python 3.14 and uv-managed ROCm Design

**Status:** Approved for implementation

## Purpose

Make CPython 3.14 the project's single Python version and expose two independent optional choices through uv:

- the existing standard `ja_ginza` model or the transformer-based `ja_ginza_electra` model;
- CPU, CUDA, or ROCm execution.

The change keeps the standard builder environment lean, makes ROCm opt-in, and does not create a second dependency definition in Nix.

## Evidence

The 2026-08-15 host experiment established the following on an AMD Radeon RX 7900 XTX with ROCm 7.2.3:

- `cupy-rocm-7-0==14.1.1` executes Thinc kernels and the standard `ja_ginza` pipeline through HIP;
- `torch==2.13.0+rocm7.2` and CuPy can coexist in one process;
- `ja_ginza_electra==5.2.0` executes its transformer and parser through ROCm;
- Python 3.14.6 can load and execute both models when ELECTRA uses a current Python-3.14 Tokenizers build rather than `tokenizers==0.13.3`;
- five representative sentences had identical token text, lemmas, POS tags, dependencies, and heads under the working Python 3.12 and experimental Python 3.14 ELECTRA environments.

A disposable uv project then resolved and installed the proposed published-package-only stack on Python 3.14.6. `spacy==3.8.11`, `ginza-transformers==0.4.2`, `spacy-transformers==1.1.9`, `transformers==4.57.6`, and `tokenizers==0.22.2` loaded both `ja_ginza` and `ja_ginza_electra` and parsed the representative sentences without modifying site-packages. The pinned Nixpkgs input exposes Python 3.14.7.

The experiment also established two constraints:

- `tokenizers==0.13.3` can be forced to compile on Python 3.14 but segfaults while loading the SudachiTra normalizer, so it must never enter the Python 3.14 lock;
- current `spacy-alignments==0.9.2` metadata declares Python `<3.14`, so ELECTRA on 3.14 is project-validated but not yet upstream-supported.

uv supports optional-dependency-specific package indexes, explicit conflicts between extras, dependency overrides, and Python 3.14 interpreters. The PyTorch ROCm 7.2 index publishes CPython 3.14 wheels for Torch 2.13.

## Dependency Shape

### Python

- Add `.python-version` containing `3.14` so uv selects a managed CPython 3.14 interpreter by default.
- Set `project.requires-python = ">=3.14,<3.15"`. The upper bound prevents an untested Python minor from silently entering a native-extension-heavy lock.
- Update the uv2nix interpreter in `flake.nix` from Python 3.12 to Python 3.14. Nix continues to consume `uv.lock`; it does not define another Python package graph.

### Model extras

- `builder` retains `ginza==5.2.0` and `ja-ginza==5.2.0` and remains the normal corpus-builder dependency set.
- Add `electra` containing `ja-ginza-electra==5.2.0`, `transformers==4.57.6`, and `tokenizers==0.22.2`.
- Add `transformers==4.57.6` to `tool.uv.override-dependencies`. This replaces only the stale `<4.26` requirement published by `spacy-transformers==1.1.9`; Transformers is requested only by `electra`, so the override does not add it to standard builder environments.
- Add `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` for `spacy-alignments` under `tool.uv.extra-build-variables`. Python 3.14 has no published wheel for `spacy-alignments==0.9.2`; uv must carry the build input that made its source distribution compile successfully.
- Do not add a model-selection argument to `natsume-corpus`. The production builder continues to load `ja_ginza`; the `electra` extra serves direct model experiments and hardware validation. Add a builder model selector only when a corpus build is intentionally compared or switched.

### Accelerator extras

- Keep `cpu` with `torch==2.13.0+cpu`.
- Keep `cuda` with `torch==2.13.0+cu126` and `cupy-cuda12x>=14.1.1,<15`.
- Add `rocm` with `torch==2.13.0+rocm7.2` and `cupy-rocm-7-0==14.1.1` on Linux x86-64.
- Declare every pair among `cpu`, `cuda`, and `rocm` as conflicting. A uv environment has exactly one accelerator policy.
- Add an explicit `pytorch-rocm` index at `https://download.pytorch.org/whl/rocm7.2` and select it only from the `rocm` extra.

The supported compositions are:

```console
uv sync --extra builder --extra cpu
uv sync --extra builder --extra electra --extra cpu
uv sync --extra builder --extra rocm
uv sync --extra builder --extra electra --extra rocm
```

`electra` describes the model; `cpu`/`cuda`/`rocm` describes its execution backend. They remain orthogonal rather than creating combined extras such as `electra-rocm`.

## Python 3.14 ELECTRA Compatibility Contract

The implementation must use published packages resolved by uv. It must not patch files in `.venv`, commit third-party wheels, add a post-install hook, or weaken Transformers' runtime version check.

The lock deliberately carries two upstream metadata discrepancies:

- `spacy-transformers==1.1.9` declares `transformers<4.26`, while the project uses the verified `transformers==4.57.6`;
- `spacy-alignments==0.9.2` declares Python `<3.14`, while the project builds it on Python 3.14 with PyO3's forward-compatibility mode.

`uv pip check` therefore reports those two known discrepancies and is not a green acceptance gate for an ELECTRA environment. The stronger gate is a fresh locked build followed by model load and representative parsing. Any third discrepancy is a failure.

Do not add a Tokenizers override: `tokenizers==0.22.2` satisfies Transformers 4.57.6 directly, and current SudachiTra metadata has no conflicting upper bound. Do not widen to Transformers 5. Revisit both exceptions when `spacy-transformers` and `spacy-alignments` publish a Python 3.14-compatible path; the trigger is the next update of either dependency.

## Runtime Boundary

uv owns Python, both model dependency sets, Torch, and CuPy. It does not own the kernel driver, `/dev/kfd`, or the ROCm runtime exposed at `/opt/rocm`.

On a matching host, ROCm execution requires the host runtime paths already established by the experiment:

```console
PATH=/opt/rocm/bin:$PATH
LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64
HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
ROCM_HOME=/opt/rocm
uv run --extra builder --extra electra --extra rocm python ...
```

Multi-GPU hosts select the intended device with the existing `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL` controls. Installation does not imply that a usable GPU exists: `spacy.require_gpu(0)` remains the runtime gate and must fail rather than silently falling back when ROCm is unavailable.

## Verification

### Configuration contract

A model-free test reads `pyproject.toml` and asserts:

- the Python range is `>=3.14,<3.15`;
- `builder`, `electra`, `cpu`, `cuda`, and `rocm` exist;
- the three accelerator extras conflict pairwise;
- each Torch requirement selects its matching explicit index;
- the ROCm extra contains both ROCm Torch and CuPy.
- ELECTRA pins Transformers 4.57.6 and Tokenizers 0.22.2;
- the single dependency override is Transformers 4.57.6;
- the `spacy-alignments` build receives `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.

This test protects the relationship whose failure would otherwise make every uv sync select the wrong binary family.

### Resolution and model smoke tests

- Regenerate `uv.lock` with Python 3.14 and run `uv lock --check`.
- Sync and smoke `builder+cpu` and `builder+electra+cpu` from the lock.
- Load each model and parse the representative Japanese fixture set.
- Confirm `uv pip check` reports exactly the two recorded ELECTRA metadata discrepancies and no others.
- On the matching AMD host, sync `builder+rocm` and `builder+electra+rocm` from the same lock.
- Require CuPy HIP execution for standard GiNZA and both Torch ROCm allocation and successful parsing for ELECTRA.
- Compare the observable parses between CPU and ROCm. Do not add a performance threshold; this gate asks whether the backend works.
- Run the existing Python and Nix checks under Python 3.14.

## Deliberate Omissions

- No automatic CPU fallback. Add one only if a named runtime consumer requires degraded operation instead of a startup failure.
- No benchmark or performance gate. Add one only when accelerator choice depends on measured throughput.
- No new Nix ROCm package set. Add one only when a Nix-built ROCm closure, rather than a uv environment on a ROCm host, becomes a deployment requirement.
- No ELECTRA builder flag. Add one when an actual corpus build uses ELECTRA.
- No vendored dependency wheels or project-owned upstream fork. Reconsider only if upstream Python 3.14 support remains blocked and ELECTRA becomes a production requirement.

## Documentation

README setup examples name the four supported uv compositions and state the host ROCm prerequisite. The existing Nix CPU delivery commands remain documented separately.

## Success Criteria

- A fresh `uv sync --extra builder --extra cpu` uses Python 3.14 and loads `ja_ginza`.
- A fresh `uv sync --extra builder --extra electra --extra cpu` builds `spacy-alignments`, loads, and parses with `ja_ginza_electra` without ambient build variables or site-packages edits.
- A fresh ROCm composition selects only ROCm Torch/CuPy wheels and both models execute on the RX 7900 XTX.
- Selecting two accelerator extras fails during uv resolution.
- `uv lock --check`, the existing checks, and the model smoke tests pass.
- No model, database, virtual environment, uv cache, or other large generated artifact enters Git.
