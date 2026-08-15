# Python 3.14 and uv-managed ROCm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Python 3.14 the project's single Python version and add independent uv extras for the standard and ELECTRA GiNZA models on CPU, CUDA, or ROCm.

**Architecture:** `pyproject.toml` and `uv.lock` remain the only Python dependency graph; Nix consumes that lock with Python 3.14 for the existing CPU closures. Fast configuration tests protect extra/index/conflict wiring, CPU smoke tests prove both model choices, and an opt-in ROCm test proves real CuPy HIP and Torch HIP execution on the matching host without creating a Nix ROCm package set.

**Tech Stack:** CPython 3.14, uv, uv2nix, Nix flakes, pytest, spaCy/GiNZA, Transformers/Tokenizers, PyTorch 2.13, CuPy 14.1.1, ROCm 7.2.

## Global Constraints

- `.python-version` contains `3.14`; `project.requires-python` is exactly `>=3.14,<3.15`.
- `builder` keeps `ginza==5.2.0` and `ja-ginza==5.2.0`; `electra` is separate and contains `ja-ginza-electra==5.2.0`, `transformers==4.57.6`, and `tokenizers==0.22.2`.
- `cpu`, `cuda`, and `rocm` conflict pairwise; each Torch requirement uses only its matching explicit uv index.
- `rocm` contains `torch==2.13.0+rocm7.2` and `cupy-rocm-7-0==14.1.1` only on Linux x86-64.
- The only dependency override is `transformers==4.57.6`; do not add a Tokenizers override or widen to Transformers 5.
- `spacy-alignments` builds with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`; do not patch `.venv`, vendor wheels, or add post-install hooks.
- uv owns Python packages. The host owns `/dev/kfd`, the kernel driver, and the ROCm runtime under `/opt/rocm`.
- Nix remains CPU-only and consumes `uv.lock`; do not add a Nix ROCm package set.
- The corpus builder continues to load standard `ja_ginza`; do not add a model-selection CLI argument.
- ROCm validation is functional: require GPU execution and CPU/ROCm parse parity, but add no performance threshold.
- Do not track models, virtual environments, uv caches, databases, corpus archives, or other large generated artifacts.
- Preserve the owner's untracked `AGENDA.md`.

---

## File Responsibility Map

- `.python-version`: uv's default interpreter selection.
- `pyproject.toml`: Python range, model/accelerator extras, uv conflicts, package indexes, dependency override, build variable, and pytest marker declarations.
- `uv.lock`: the resolved Python 3.14 graph for every supported extra composition.
- `tests/test_dependency_configuration.py`: model-free contract for the cross-field uv configuration and Nix interpreter selection.
- `tests/model_smoke.py`: shared representative Japanese inputs and observable parse projection used by CPU and ROCm smoke tests.
- `tests/test_models.py`: standard `ja_ginza` CPU availability and parse smoke.
- `tests/test_electra_model.py`: opt-in ELECTRA CPU availability and parse smoke.
- `tests/test_rocm_models.py`: opt-in host validation for CuPy HIP, Torch HIP, and CPU/ROCm parse parity.
- `flake.nix`: existing uv2nix CPU closures and checks, all moved to Python 3.14; no ROCm closure.
- `README.md`: the four supported uv compositions, known ELECTRA metadata exceptions, ROCm host boundary, and verification commands.

---

### Task 1: Lock the Python, Model, and Accelerator Dependency Contract

**Files:**
- Create: `.python-version`
- Create: `tests/test_dependency_configuration.py`
- Modify: `pyproject.toml:8-84`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: the approved package versions and uv index URLs from the design.
- Produces: a Python 3.14 lock and extras named `builder`, `electra`, `cpu`, `cuda`, `rocm`, `backend`, and `test`; later tasks consume these names verbatim.

- [ ] **Step 1: Add the failing configuration contract**

Create `tests/test_dependency_configuration.py` with:

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def project_configuration() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_uv_selects_only_python_314() -> None:
    configuration = project_configuration()

    assert configuration["project"]["requires-python"] == ">=3.14,<3.15"
    assert (ROOT / ".python-version").read_text(encoding="utf-8") == "3.14\n"


def test_model_and_accelerator_extras_are_orthogonal() -> None:
    configuration = project_configuration()
    extras = configuration["project"]["optional-dependencies"]

    assert extras["builder"] == [
        "click>=8.4.2,<9",
        "polars[pyarrow,excel]>=1.43.2,<2",
        "ginza==5.2.0",
        "ja-ginza==5.2.0",
        "confection>=0.1.5,<1",
        "numpy>=2,<3",
        "spacy>=3.8.11,<3.8.12",
        "wtpsplit>=2.2.1,<3",
    ]
    assert extras["electra"] == [
        "ja-ginza-electra==5.2.0",
        "transformers==4.57.6",
        "tokenizers==0.22.2",
    ]
    assert extras["cpu"] == ["torch==2.13.0+cpu"]
    assert extras["cuda"] == [
        "torch==2.13.0+cu126",
        "cupy-cuda12x>=14.1.1,<15",
    ]
    platform_marker = "sys_platform == 'linux' and platform_machine == 'x86_64'"
    assert extras["rocm"] == [
        f"torch==2.13.0+rocm7.2; {platform_marker}",
        f"cupy-rocm-7-0==14.1.1; {platform_marker}",
    ]


def test_accelerator_extras_select_one_matching_torch_index() -> None:
    configuration = project_configuration()
    uv = configuration["tool"]["uv"]

    conflicts = {
        frozenset(member["extra"] for member in conflict)
        for conflict in uv["conflicts"]
    }
    assert conflicts == {
        frozenset(("cpu", "cuda")),
        frozenset(("cpu", "rocm")),
        frozenset(("cuda", "rocm")),
    }
    assert uv["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu"},
        {"index": "pytorch-cuda", "extra": "cuda"},
        {"index": "pytorch-rocm", "extra": "rocm"},
    ]
    indexes = {entry["name"]: entry for entry in uv["index"]}
    assert indexes == {
        "pytorch-cpu": {
            "name": "pytorch-cpu",
            "url": "https://download.pytorch.org/whl/cpu",
            "explicit": True,
        },
        "pytorch-cuda": {
            "name": "pytorch-cuda",
            "url": "https://download.pytorch.org/whl/cu126",
            "explicit": True,
        },
        "pytorch-rocm": {
            "name": "pytorch-rocm",
            "url": "https://download.pytorch.org/whl/rocm7.2",
            "explicit": True,
        },
    }


def test_python_314_compatibility_exceptions_are_narrow() -> None:
    uv = project_configuration()["tool"]["uv"]

    assert uv["override-dependencies"] == ["transformers==4.57.6"]
    assert uv["extra-build-variables"] == {
        "spacy-alignments": {"PYO3_USE_ABI3_FORWARD_COMPATIBILITY": "1"}
    }
```

- [ ] **Step 2: Run the contract and verify the current configuration fails**

Run:

```bash
nix develop .#test --command pytest tests/test_dependency_configuration.py -v
```

Expected: failures report the current `>=3.12` range, missing `.python-version`, missing `electra`/`rocm` extras, incomplete conflicts and indexes, and missing compatibility exceptions.

- [ ] **Step 3: Apply the minimal dependency configuration**

Create `.python-version` with exactly:

```text
3.14
```

In `pyproject.toml`:

1. Change `requires-python` to `">=3.14,<3.15"`.
2. Delete the two commented speculative dependency lines from `builder`.
3. Add:

```toml
electra = [
    "ja-ginza-electra==5.2.0",
    "transformers==4.57.6",
    "tokenizers==0.22.2",
]
rocm = [
    "torch==2.13.0+rocm7.2; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "cupy-rocm-7-0==14.1.1; sys_platform == 'linux' and platform_machine == 'x86_64'",
]
```

4. Replace `[tool.uv]` with the three pairwise conflicts and narrow override:

```toml
[tool.uv]
conflicts = [
    [
        { extra = "cpu" },
        { extra = "cuda" },
    ],
    [
        { extra = "cpu" },
        { extra = "rocm" },
    ],
    [
        { extra = "cuda" },
        { extra = "rocm" },
    ],
]
override-dependencies = ["transformers==4.57.6"]

[tool.uv.extra-build-variables]
spacy-alignments = { PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1" }
```

5. Add `{ index = "pytorch-rocm", extra = "rocm" }` to `tool.uv.sources.torch` and add:

```toml
[[tool.uv.index]]
name = "pytorch-rocm"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true
```

- [ ] **Step 4: Regenerate and validate the Python 3.14 lock**

Run:

```bash
uv lock --python 3.14
uv lock --check
```

Expected: resolution succeeds; `uv.lock` declares `requires-python = ">=3.14, <3.15"`, contains the standard and ELECTRA models, contains CPU/CUDA/ROCm Torch variants, pins `transformers==4.57.6` and `tokenizers==0.22.2`, and does not contain `tokenizers==0.13.3`.

- [ ] **Step 5: Run the configuration contract and lock assertions**

Run:

```bash
uv run --locked --extra test pytest tests/test_dependency_configuration.py -v
uv lock --check
rg -n 'name = "(ja-ginza|ja-ginza-electra|transformers|tokenizers|cupy-rocm-7-0)"|2\.13\.0\+(cpu|cu126|rocm7\.2)' uv.lock
if rg -n 'version = "0\.13\.3"' uv.lock; then exit 1; fi
```

Expected: pytest and lock check pass; the requested packages/variants are present; the forbidden Tokenizers version produces no match.

- [ ] **Step 6: Prove conflicting accelerator choices fail at resolution**

Run each command and require a non-zero status with an incompatibility message:

```bash
if uv sync --locked --extra cpu --extra cuda 2>conflict-cpu-cuda.log; then exit 1; fi
if uv sync --locked --extra cpu --extra rocm 2>conflict-cpu-rocm.log; then exit 1; fi
if uv sync --locked --extra cuda --extra rocm 2>conflict-cuda-rocm.log; then exit 1; fi
rg -n 'conflict|incompatible' conflict-cpu-cuda.log conflict-cpu-rocm.log conflict-cuda-rocm.log
rm conflict-cpu-cuda.log conflict-cpu-rocm.log conflict-cuda-rocm.log
```

Expected: all three compositions are rejected before installation and the small diagnostic logs are removed afterward.

- [ ] **Step 7: Commit the dependency contract**

```bash
git add .python-version pyproject.toml uv.lock tests/test_dependency_configuration.py
git commit -m "build: adopt python 3.14 uv dependency matrix"
```

---

### Task 2: Prove Both Models in Separate CPU Environments

**Files:**
- Create: `tests/model_smoke.py`
- Create: `tests/test_electra_model.py`
- Modify: `tests/test_models.py:1-13`
- Modify: `pyproject.toml:96-101`

**Interfaces:**
- Consumes: Task 1's `builder`, `electra`, `cpu`, and `test` extras.
- Produces: `representative_parse(nlp) -> tuple[tuple[tuple[str, str, str, str, int], ...], ...]`, reused by the ROCm parity test in Task 4.

- [ ] **Step 1: Add the shared observable parse projection**

Create `tests/model_smoke.py` with:

```python
from collections.abc import Iterable

from spacy.language import Language


REPRESENTATIVE_SENTENCES = (
    "情報を集めて判断する。",
    "研究者が日本語の文章を詳しく分析した。",
    "時間について考えながら結果を説明する。",
    "新しい方法で課題を解決できるか検証した。",
    "利用者は複数の資料から必要な事実を探す。",
    "自然言語処理の技術が社会で広く使われている。",
    "性能だけでなく解析結果の一致も確認する。",
    "明日の会議までに報告書を作成してください。",
)

TokenParse = tuple[str, str, str, str, int]
DocumentParse = tuple[TokenParse, ...]


def representative_parse(nlp: Language) -> tuple[DocumentParse, ...]:
    documents: Iterable = nlp.pipe(REPRESENTATIVE_SENTENCES, batch_size=8)
    return tuple(
        tuple(
            (token.text, token.lemma_, token.pos_, token.dep_, token.head.i)
            for token in document
        )
        for document in documents
    )


def assert_representative_parse(parses: tuple[DocumentParse, ...]) -> None:
    assert len(parses) == len(REPRESENTATIVE_SENTENCES)
    assert tuple(token[0] for token in parses[0]) == (
        "情報",
        "を",
        "集め",
        "て",
        "判断",
        "する",
        "。",
    )
    for document in parses:
        assert document
        assert sum(token[3] == "ROOT" for token in document) == 1
        assert all(token[1] and token[2] and token[3] for token in document)
```

- [ ] **Step 2: Replace fallback loading with explicit model contracts**

Replace `tests/test_models.py` with:

```python
import pytest
import spacy

from tests.model_smoke import assert_representative_parse, representative_parse


@pytest.mark.nlp_model
def test_standard_model_loading() -> None:
    nlp = spacy.load("ja_ginza")

    assert_representative_parse(representative_parse(nlp))
```

Create `tests/test_electra_model.py` with:

```python
import pytest
import spacy

from tests.model_smoke import assert_representative_parse, representative_parse


@pytest.mark.nlp_model
@pytest.mark.electra_model
def test_electra_model_loading() -> None:
    nlp = spacy.load("ja_ginza_electra")

    assert_representative_parse(representative_parse(nlp))
```

Add this marker to `tool.pytest.ini_options.markers`:

```toml
"electra_model: requires the optional transformer-based Japanese model",
```

- [ ] **Step 3: Run the ELECTRA test before syncing its extra**

Run:

```bash
uv sync --locked --extra builder --extra cpu --extra test
uv run --locked --extra builder --extra cpu --extra test \
  pytest tests/test_electra_model.py -v
```

Expected: FAIL because `ja_ginza_electra` is not installed in the standard builder composition. This proves the model extras are independent rather than the standard environment receiving ELECTRA transitively.

- [ ] **Step 4: Smoke the standard model in the standard composition**

Run:

```bash
uv run --locked --extra builder --extra cpu --extra test \
  pytest tests/test_models.py -v
```

Expected: PASS on Python 3.14 with `ja_ginza`; `uv run --extra electra` is not needed.

- [ ] **Step 5: Sync and smoke the ELECTRA composition**

Run:

```bash
uv sync --locked --extra builder --extra electra --extra cpu --extra test
uv run --locked --extra builder --extra electra --extra cpu --extra test \
  pytest tests/test_electra_model.py -v
```

Expected: uv builds `spacy-alignments` from source using the committed build variable; the test passes on Python 3.14 without edits under `.venv`.

- [ ] **Step 6: Confirm only the two accepted metadata discrepancies remain**

Run:

```bash
set +e
pip_check_output="$(uv pip check 2>&1)"
pip_check_status=$?
set -e
printf '%s\n' "$pip_check_output"
python - "$pip_check_status" "$pip_check_output" <<'PY'
import sys

status = int(sys.argv[1])
lines = [line for line in sys.argv[2].splitlines() if line.strip()]
assert status != 0
assert len(lines) == 2, lines
assert any("spacy-alignments" in line and "3.14" in line for line in lines), lines
assert any("spacy-transformers" in line and "transformers" in line for line in lines), lines
PY
```

Expected: the assertions pass. Any third non-empty diagnostic fails this step.

- [ ] **Step 7: Consolidate and commit the CPU model claims**

Run both model tests once more in the ELECTRA composition, which contains both models:

```bash
uv run --locked --extra builder --extra electra --extra cpu --extra test \
  pytest tests/test_models.py tests/test_electra_model.py -v
git add pyproject.toml tests/model_smoke.py tests/test_models.py tests/test_electra_model.py
git commit -m "test: prove both ginza model choices"
```

Expected: two tests pass and no fallback can hide a missing model.

---

### Task 3: Move Every Nix Python Consumer to 3.14

**Files:**
- Modify: `tests/test_dependency_configuration.py`
- Modify: `flake.nix:53-55,108,200-207`

**Interfaces:**
- Consumes: Task 1's Python 3.14 `uv.lock` and Task 2's `electra_model` marker.
- Produces: the existing server, builder, test, fixture, and model-integration CPU closures on `pkgs.python314`; later verification uses the unchanged flake output names.

- [ ] **Step 1: Add a failing Nix interpreter ownership test**

Append to `tests/test_dependency_configuration.py`:

```python
def test_nix_consumers_use_python_314() -> None:
    flake = (ROOT / "flake.nix").read_text(encoding="utf-8")

    assert "python = pkgs.python314;" in flake
    assert "smokeFixturePython = pkgs.python314.withPackages" in flake
    assert "pkgs.python312" not in flake
```

- [ ] **Step 2: Run the focused test and verify it fails on both 3.12 references**

Run:

```bash
uv run --locked --extra test pytest \
  tests/test_dependency_configuration.py::test_nix_consumers_use_python_314 -v
```

Expected: FAIL because `flake.nix` still contains `pkgs.python312`.

- [ ] **Step 3: Change only the two Nix interpreter owners**

In `flake.nix`, change:

```nix
python = pkgs.python312;
```

to:

```nix
python = pkgs.python314;
```

and change:

```nix
smokeFixturePython = pkgs.python312.withPackages (pythonPackages: [ pythonPackages.xlwt ]);
```

to:

```nix
smokeFixturePython = pkgs.python314.withPackages (pythonPackages: [ pythonPackages.xlwt ]);
```

Also change the model integration command to exclude the optional ELECTRA and ROCm tests:

```nix
pytest -m "nlp_model and not electra_model and not rocm"
```

The Nix package graph remains CPU-only and does not add `electra` or `rocm` to any dependency selector.

- [ ] **Step 4: Format and run the focused contract**

Run:

```bash
nix fmt flake.nix
nix develop .#test --command pytest tests/test_dependency_configuration.py -v
```

Expected: the configuration tests pass and `nix fmt` changes no unrelated file.

- [ ] **Step 5: Build the Python 3.14 CPU seams**

Run:

```bash
nix build --print-build-logs \
  .#checks.x86_64-linux.backend \
  .#checks.x86_64-linux.builder-smoke \
  .#nlp-model-integration
```

Expected: the model-free backend, packaged builder smoke, and standard GiNZA model gate all build with Python 3.14.

- [ ] **Step 6: Commit the Nix interpreter migration**

```bash
git add flake.nix tests/test_dependency_configuration.py
git commit -m "build: move nix python closures to 3.14"
```

---

### Task 4: Add an Opt-in ROCm Functional Contract

**Files:**
- Create: `tests/test_rocm_models.py`
- Modify: `pyproject.toml:96-103`

**Interfaces:**
- Consumes: Task 1's `rocm` extra and Task 2's `representative_parse` projection.
- Produces: two pytest cases named `test_model_matches_cpu_on_rocm[ja_ginza]` and `[ja_ginza_electra]`; they require a matching ROCm host and are excluded from default/Nix gates.

- [ ] **Step 1: Add the opt-in ROCm test**

Create `tests/test_rocm_models.py` with:

```python
import pytest
import spacy

from tests.model_smoke import representative_parse


@pytest.mark.nlp_model
@pytest.mark.rocm
@pytest.mark.parametrize("model_name", ["ja_ginza", "ja_ginza_electra"])
def test_model_matches_cpu_on_rocm(model_name: str) -> None:
    import cupy
    import torch
    from thinc.api import get_current_ops

    spacy.require_cpu()
    cpu_parse = representative_parse(spacy.load(model_name))

    assert cupy.cuda.runtime.is_hip
    assert torch.version.hip is not None
    assert torch.cuda.is_available()
    assert spacy.require_gpu(0)
    assert get_current_ops().xp is cupy

    torch.cuda.reset_peak_memory_stats(0)
    gpu_parse = representative_parse(spacy.load(model_name))
    cupy.cuda.runtime.deviceSynchronize()
    torch.cuda.synchronize(0)

    assert gpu_parse == cpu_parse
    assert cupy.get_default_memory_pool().total_bytes() > 0
    if model_name == "ja_ginza_electra":
        assert torch.cuda.max_memory_allocated(0) > 0
```

Add this marker to `tool.pytest.ini_options.markers`:

```toml
"rocm: requires a matching AMD GPU and host ROCm runtime",
```

- [ ] **Step 2: Verify the ROCm test is absent from the CPU composition**

Run:

```bash
uv sync --locked --extra builder --extra electra --extra cpu --extra test
uv run --locked --extra builder --extra electra --extra cpu --extra test \
  pytest tests/test_rocm_models.py -v
```

Expected: collection fails on missing `cupy`; the CPU composition does not receive either CUDA or ROCm CuPy transitively.

- [ ] **Step 3: Sync the standard ROCm composition and prove the standard model**

Run on the matching AMD host:

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export ROCM_HOME=/opt/rocm
export HIP_VISIBLE_DEVICES=0
uv sync --locked --extra builder --extra rocm --extra test
uv run --locked --extra builder --extra rocm --extra test \
  pytest 'tests/test_rocm_models.py::test_model_matches_cpu_on_rocm[ja_ginza]' -v
```

Expected: CuPy reports HIP, Thinc's active array module is CuPy, the standard model parses on the GPU, and its observable parse equals the CPU parse.

- [ ] **Step 4: Sync the ELECTRA ROCm composition and prove both models**

Run with the same host environment:

```bash
uv sync --locked --extra builder --extra electra --extra rocm --extra test
uv run --locked --extra builder --extra electra --extra rocm --extra test \
  pytest tests/test_rocm_models.py -v
```

Expected: both parameter rows pass; ELECTRA allocates Torch HIP memory; standard GiNZA allocates CuPy HIP memory; both GPU parses exactly match their CPU parses. Do not record or gate on elapsed time.

- [ ] **Step 5: Prove the default gate still excludes hardware tests**

Run:

```bash
uv run --locked --extra builder --extra electra --extra rocm --extra test \
  pytest -m 'not nlp_model' --collect-only -q
```

Expected: neither `test_rocm_models.py` case is selected because both carry `nlp_model`.

- [ ] **Step 6: Commit the hardware contract**

```bash
git add pyproject.toml tests/test_rocm_models.py
git commit -m "test: add opt-in rocm model contract"
```

---

### Task 5: Document uv Compositions and Verify the Complete Change

**Files:**
- Modify: `README.md:29-48,164-188`

**Interfaces:**
- Consumes: all commands and extra names proven by Tasks 1-4.
- Produces: the user-facing setup contract; no new scripts, Nix outputs, or runtime configuration objects.

- [ ] **Step 1: Add the four uv setup compositions**

After the Development shells introduction in `README.md`, add:

~~~markdown
### uv model and accelerator environments

`.python-version` selects Python 3.14. The model and accelerator choices are independent:

```bash
uv sync --locked --extra builder --extra cpu
uv sync --locked --extra builder --extra electra --extra cpu
uv sync --locked --extra builder --extra rocm
uv sync --locked --extra builder --extra electra --extra rocm
```

`builder` contains standard `ja_ginza`; `electra` adds `ja_ginza_electra` without
changing the corpus builder's selected model. Exactly one of `cpu`, `cuda`, and `rocm`
may be selected. The Nix packages and service remain the reproducible CPU delivery path;
the uv ROCm environments are opt-in host environments.
~~~

- [ ] **Step 2: Document the ROCm host boundary and functional command**

Continue the same README section with:

~~~markdown
ROCm requires a matching Linux x86-64 host with `/dev/kfd` and the runtime under
`/opt/rocm`; uv installs Torch and CuPy but not the kernel driver or host runtime.

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64
export HIP_DEVICE_LIB_PATH=/opt/rocm/amdgcn/bitcode
export ROCM_HOME=/opt/rocm
uv run --locked --extra builder --extra electra --extra rocm --extra test \
  pytest tests/test_rocm_models.py -v
```

The test requires GPU execution and CPU/ROCm parse parity for both models. It is a
compatibility check, not a performance benchmark. On multi-GPU hosts, select a device
with `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL`.
~~~

- [ ] **Step 3: Record the narrow Python 3.14 ELECTRA exception**

Add immediately after the ROCm command:

```markdown
The ELECTRA environment deliberately overrides `spacy-transformers`' stale
Transformers `<4.26` metadata with the tested `transformers==4.57.6`, and builds
`spacy-alignments==0.9.2` on Python 3.14 with PyO3 forward compatibility. Consequently,
`uv pip check` reports exactly those two known metadata discrepancies; model loading and
representative parsing are the acceptance gate. Revisit the override and build variable
when either upstream dependency publishes a Python 3.14-compatible release.
```

- [ ] **Step 4: Make dependency-update commands Python-version explicit**

Change the dependency lock command in `README.md` from:

```bash
uv lock --upgrade
```

to:

```bash
uv lock --upgrade --python 3.14
uv lock --check
```

Keep the existing Nix and npm commands unchanged.

- [ ] **Step 5: Run the model-free and source-quality gates**

Run:

```bash
uv lock --check
nix flake check --print-build-logs
```

Expected: every existing flake check passes under the Python 3.14 lock. The ROCm tests remain opt-in and are not evaluated by Nix.

- [ ] **Step 6: Re-run both CPU model choices from the committed lock**

Run:

```bash
uv sync --locked --extra builder --extra cpu --extra test
uv run --locked --extra builder --extra cpu --extra test \
  pytest tests/test_models.py -v
uv sync --locked --extra builder --extra electra --extra cpu --extra test
uv run --locked --extra builder --extra electra --extra cpu --extra test \
  pytest tests/test_models.py tests/test_electra_model.py -v
```

Expected: standard GiNZA passes without `electra`; both models pass together with `electra`.

- [ ] **Step 7: Audit repository hygiene**

Run:

```bash
git status --short
git diff --check
git ls-files -z | xargs -0 -r du -h | sort -h | tail -20
```

Expected: only intended source/config/docs changes plus the pre-existing untracked `AGENDA.md`; no `.venv`, model, database, cache, archive, or other large generated artifact is tracked.

- [ ] **Step 8: Commit the supported workflow documentation**

```bash
git add README.md
git commit -m "docs: explain python 3.14 uv accelerator setup"
```

- [ ] **Step 9: Review the complete implementation range**

Run:

```bash
git log --oneline --decorate -5
git diff --stat e3e89e8..HEAD
git diff --check e3e89e8..HEAD
git status --short
```

Expected: four focused implementation commits after the approved design, no whitespace errors, and only `AGENDA.md` untracked.
