# ROCm CuPy GiNZA Experiment Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the current spaCy/Thinc/GiNZA stack runs correctly through CuPy on the host's ROCm GPU, with only a lightweight throughput smoke check after correctness passes.

**Architecture:** Try CuPy's official ROCm 7.0 wheel first in a disposable environment on persistent btrfs. Only a wheel failure attributable to the host's ROCm 7.2 runtime licenses a source-build fallback; compatibility then advances from HIP arrays through Thinc custom kernels to GiNZA before exact CPU/GPU output comparison.

**Tech Stack:** Nix builder shell, Python 3.12, CuPy 14.1.1, ROCm 7.2.3/HIP, Thinc 8.3.11, spaCy 3.8.11, GiNZA 5.2.0.

## Global Constraints

- Put the virtual environment, caches, temporary files, scripts, logs, and results under one `mktemp -d -p /home/bor/Projects natsume-rocm-cupy-XXXXXX` directory on persistent btrfs; `/tmp` and `/home/bor` are tmpfs.
- Reuse the project builder environment with `--system-site-packages`. Install CuPy with `--no-deps` so dependency resolution cannot replace the inherited NumPy or NLP stack.
- Try `cupy-rocm-7-0==14.1.1` first. Use the source distribution only after an observed wheel failure identifies the ROCm 7.0-wheel/7.2-host boundary.
- Stop at the first failed compatibility gate. Test no speculative build variations.
- Do not modify project dependencies, Nix outputs, application code, tests, or the owner's untracked `AGENDA.md`.
- Correctness is the decision. Performance is only three batched smoke runs over 256 documents, with no threshold or tuning.
- Report the evidence before deleting the validated exact scratch directory.

---

### Task 1: Create Persistent Scratch and Preflight the Runtime

**Files:**
- Create outside Git: `/home/bor/Projects/natsume-rocm-cupy-XXXXXX/{logs,results,tmp,cupy-cache,uv-cache}`

**Interfaces:**
- Consumes: the project `.#builder` shell and host ROCm runtime.
- Produces: one exact experiment path and a runtime preflight log.

- [ ] **Step 1: Record the repository baseline**

Run:

```bash
git status --short
git rev-parse HEAD
```

Expected: only the pre-existing untracked `AGENDA.md` is present. Record the commit ID.

- [ ] **Step 2: Create and validate persistent scratch**

Run:

```bash
mktemp -d -p /home/bor/Projects natsume-rocm-cupy-XXXXXX
findmnt -T <experiment-dir> -no TARGET,FSTYPE,SOURCE
realpath <experiment-dir>
mkdir -p <experiment-dir>/{logs,results,tmp,cupy-cache,uv-cache}
df -h <experiment-dir>
```

Expected: retain the exact generated path; its resolved parent is `/home/bor/Projects` and filesystem type is `btrfs`.

- [ ] **Step 3: Capture the runtime preflight without a SIGPIPE-prone pipeline**

Run:

```bash
set -o pipefail
nix develop .#builder --command bash -lc '
set -euo pipefail
test -e /opt/rocm/lib/libamdhip64.so
rocminfo > <experiment-dir>/logs/rocminfo.txt
grep -q "Name:.*gfx1100" <experiment-dir>/logs/rocminfo.txt
python - <<"PY"
import importlib.metadata as metadata
import platform

expected = {
    "spacy": "3.8.11",
    "thinc": "8.3.11",
    "ja-ginza": "5.2.0",
    "numpy": "2.5.2",
}
print("python", platform.python_version())
for package, wanted in expected.items():
    actual = metadata.version(package)
    print(package, actual)
    if actual != wanted:
        raise SystemExit(f"{package}: expected {wanted}, got {actual}")
PY
rocm-smi --showproductname --showuniqueid --showmeminfo vram --json
' 2>&1 | tee <experiment-dir>/logs/preflight.log
```

Expected: Python 3.12.13, the expected NLP/NumPy versions, and the discrete `gfx1100` GPU are recorded. Stop before installation on failure.

---

### Task 2: Install CuPy by the Shallowest Working Route

**Files:**
- Create outside Git: `<experiment-dir>/venv/`
- Create outside Git: `<experiment-dir>/logs/cupy-wheel-install.log`
- Create only on fallback: `<experiment-dir>/logs/cupy-source-build.log`

**Interfaces:**
- Consumes: Task 1's runtime preflight.
- Produces: `<experiment-dir>/venv/bin/python` importing CuPy 14.1.1 while retaining the inherited NLP and NumPy versions.

- [ ] **Step 1: Create the isolated overlay environment**

Run:

```bash
nix develop .#builder --command uv venv \
  --system-site-packages \
  --python python3.12 \
  <experiment-dir>/venv
```

- [ ] **Step 2: Install the official ROCm 7.0 wheel without dependencies**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  UV_CACHE_DIR=<experiment-dir>/uv-cache \
  uv pip install \
    --python <experiment-dir>/venv/bin/python \
    --no-deps \
    'cupy-rocm-7-0==14.1.1' \
  2>&1 | tee <experiment-dir>/logs/cupy-wheel-install.log
```

Expected: the CPython 3.12 wheel installs without compiling or changing inherited packages.

- [ ] **Step 3: Test the wheel at its actual compatibility boundary**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  ROCM_HOME=/opt/rocm \
  HIP_VISIBLE_DEVICES=0 \
  <experiment-dir>/venv/bin/python - \
  2>&1 <<'PY' | tee <experiment-dir>/logs/cupy-wheel-import.log
import cupy

assert cupy.__version__ == "14.1.1"
assert cupy.cuda.runtime.is_hip
cupy.cuda.Device(0).use()
cupy.show_config()
print(cupy.cuda.runtime.getDeviceProperties(0)["name"])
PY
```

Expected: the official wheel imports through the ROCm 7.2 host runtime. If it passes, skip Step 4.

- [ ] **Step 4: Use one bounded source-build fallback only for a runtime/ABI failure**

Perform this step only if Step 3 failed with evidence that the ROCm 7.0 wheel cannot load or execute against the 7.2 host. Do not use it for an unrelated Python, device-permission, or missing-host-library failure.

First validate and recreate only the scratch venv:

```bash
test "$(dirname "$(realpath <experiment-dir>/venv)")" = "$(realpath <experiment-dir>)"
rm -rf -- <experiment-dir>/venv
nix develop .#builder --command uv venv \
  --system-site-packages \
  --python python3.12 \
  <experiment-dir>/venv
```

Then verify source-build prerequisites and install only the declared non-NumPy helper plus CuPy:

```bash
test -x /opt/rocm/bin/hipcc
test -f /opt/rocm/include/hip/hip_runtime.h

nix develop .#builder --command env \
  UV_CACHE_DIR=<experiment-dir>/uv-cache \
  uv pip install \
    --python <experiment-dir>/venv/bin/python \
    --no-deps \
    'cuda-pathfinder>=1.3.4,<2'

set -o pipefail
nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  UV_CACHE_DIR=<experiment-dir>/uv-cache \
  CUPY_CACHE_DIR=<experiment-dir>/cupy-cache \
  CUPY_INSTALL_USE_HIP=1 \
  ROCM_HOME=/opt/rocm \
  HCC_AMDGPU_TARGET=gfx1100 \
  CUPY_NUM_BUILD_JOBS=8 \
  uv pip install \
    --python <experiment-dir>/venv/bin/python \
    --no-deps \
    --no-binary cupy \
    'cupy==14.1.1' \
  2>&1 | tee <experiment-dir>/logs/cupy-source-build.log
```

Expected: the fallback aligns CuPy with ROCm 7.2.3. Stop and report the compiler/configuration failure if this one build fails.

- [ ] **Step 5: Verify the environment was not silently replaced**

Run:

```bash
nix develop .#builder --command <experiment-dir>/venv/bin/python - <<'PY'
import cupy
import importlib.metadata as metadata

print("cupy", cupy.__version__)
for package, wanted in {
    "spacy": "3.8.11",
    "thinc": "8.3.11",
    "ja-ginza": "5.2.0",
    "numpy": "2.5.2",
}.items():
    actual = metadata.version(package)
    print(package, actual)
    assert actual == wanted
PY
```

---

### Task 3: Test CuPy, Thinc Custom Kernels, and GiNZA in Order

**Files:**
- Create outside Git: `<experiment-dir>/compatibility_probe.py`
- Create outside Git: `<experiment-dir>/logs/compatibility.log`
- Create outside Git: `<experiment-dir>/results/compatibility.json`

**Interfaces:**
- Consumes: Task 2's working CuPy environment.
- Produces: success evidence for every compatibility layer, or the traceback from the first failed layer.

- [ ] **Step 1: Create the compatibility probe with `apply_patch`**

Create `<experiment-dir>/compatibility_probe.py` with exactly:

```python
from __future__ import annotations

import json
from pathlib import Path

import cupy
import numpy
import spacy
from thinc.api import get_current_ops, set_current_ops
from thinc.backends import CupyOps, NumpyOps


RESULT_PATH = Path(__file__).parent / "results" / "compatibility.json"


def host(value):
    if isinstance(value, tuple):
        return tuple(host(item) for item in value)
    return cupy.asnumpy(value)


def assert_equal(actual, expected) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            numpy.testing.assert_allclose(host(actual_item), expected_item)
    else:
        numpy.testing.assert_allclose(host(actual), expected)


result: dict[str, object] = {"gates": []}

assert cupy.cuda.runtime.is_hip
cupy.cuda.Device(0).use()
left = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
product = left @ cupy.eye(4, dtype=cupy.float32)
cupy.cuda.runtime.deviceSynchronize()
numpy.testing.assert_allclose(cupy.asnumpy(product), numpy.arange(16, dtype="f").reshape(4, 4))
result["gates"].append("cupy_hip")

gpu_ops = CupyOps()
cpu_ops = NumpyOps()
set_current_ops(gpu_ops)
assert get_current_ops().name == "cupy"

maxout_input = numpy.arange(24, dtype="f").reshape(2, 3, 4)
assert_equal(gpu_ops.maxout(cupy.asarray(maxout_input)), cpu_ops.maxout(maxout_input))

sequence = numpy.arange(12, dtype="f").reshape(4, 3)
assert_equal(gpu_ops.seq2col(cupy.asarray(sequence), 1), cpu_ops.seq2col(sequence, 1))

ids = numpy.array([1, 2, 3], dtype="uint64")
assert_equal(gpu_ops.hash(cupy.asarray(ids), 7), cpu_ops.hash(ids, 7))
cupy.cuda.runtime.deviceSynchronize()
result["gates"].append("thinc_custom_kernels")

assert spacy.require_gpu(0)
assert get_current_ops().name == "cupy"
result["gates"].append("spacy_require_gpu")

nlp = spacy.load("ja_ginza")
doc = nlp("情報を集めて判断する。")
cupy.cuda.runtime.deviceSynchronize()
assert [token.text for token in doc]

parameter_modules: set[str] = set()
for _name, component in nlp.pipeline:
    model = getattr(component, "model", None)
    if model is None:
        continue
    for node in model.walk():
        for parameter_name in node.param_names:
            parameter_modules.add(type(node.get_param(parameter_name)).__module__.split(".", 1)[0])

assert "cupy" in parameter_modules
result["parameter_modules"] = sorted(parameter_modules)
result["gates"].append("ginza_inference")
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

- [ ] **Step 2: Run through the gates and stop at the first failure**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  CUPY_CACHE_DIR=<experiment-dir>/cupy-cache \
  ROCM_HOME=/opt/rocm \
  HIP_VISIBLE_DEVICES=0 \
  <experiment-dir>/venv/bin/python \
  <experiment-dir>/compatibility_probe.py \
  2>&1 | tee <experiment-dir>/logs/compatibility.log
```

Expected: `compatibility.json` ends with `ginza_inference`. A Thinc kernel failure is already a decisive incompatibility result; do not debug it as a GiNZA problem.

---

### Task 4: Verify Exact Parse Parity and Smoke-Test Throughput

**Files:**
- Create outside Git: `<experiment-dir>/smoke_probe.py`
- Create outside Git: `<experiment-dir>/results/{cpu,gpu,summary}.json`
- Create outside Git: `<experiment-dir>/logs/{cpu,gpu}-smoke.log`

**Interfaces:**
- Consumes: a passing Task 3 compatibility result.
- Produces: exact CPU/GPU representative parses and three batched throughput observations per backend.

- [ ] **Step 1: Create the shared smoke probe with `apply_patch`**

Create `<experiment-dir>/smoke_probe.py` with exactly:

```python
from __future__ import annotations

import argparse
import json
import statistics
import time
from itertools import cycle, islice
from pathlib import Path

import spacy


SENTENCES = (
    "情報を集めて判断する。",
    "研究者が日本語の文章を詳しく分析した。",
    "時間について考えながら結果を説明する。",
    "新しい方法で課題を解決できるか検証した。",
    "利用者は複数の資料から必要な事実を探す。",
    "自然言語処理の技術が社会で広く使われている。",
    "性能だけでなく解析結果の一致も確認する。",
    "明日の会議までに報告書を作成してください。",
)
BATCH = tuple(islice(cycle(SENTENCES), 256))


def parsed(doc):
    return [
        {
            "text": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.i,
        }
        for token in doc
    ]


parser = argparse.ArgumentParser()
parser.add_argument("backend", choices=("cpu", "gpu"))
parser.add_argument("output", type=Path)
args = parser.parse_args()

cupy = None
if args.backend == "gpu":
    import cupy as cupy_module

    cupy = cupy_module
    assert spacy.require_gpu(0)
else:
    spacy.require_cpu()


def synchronize() -> None:
    if cupy is not None:
        cupy.cuda.runtime.deviceSynchronize()


nlp = spacy.load("ja_ginza")
representative = list(nlp.pipe(SENTENCES, batch_size=8))
list(nlp.pipe(BATCH[:128], batch_size=128))
synchronize()

times: list[float] = []
for _ in range(3):
    synchronize()
    started = time.perf_counter()
    list(nlp.pipe(BATCH, batch_size=128))
    synchronize()
    times.append(time.perf_counter() - started)

result = {
    "backend": args.backend,
    "representative_parse": [parsed(doc) for doc in representative],
    "documents": len(BATCH),
    "batch_size": 128,
    "seconds": times,
    "median_documents_per_second": len(BATCH) / statistics.median(times),
}
args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

- [ ] **Step 2: Run one CPU and one GPU process**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  <experiment-dir>/venv/bin/python \
  <experiment-dir>/smoke_probe.py cpu \
  <experiment-dir>/results/cpu.json \
  2>&1 | tee <experiment-dir>/logs/cpu-smoke.log

nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  CUPY_CACHE_DIR=<experiment-dir>/cupy-cache \
  ROCM_HOME=/opt/rocm \
  HIP_VISIBLE_DEVICES=0 \
  <experiment-dir>/venv/bin/python \
  <experiment-dir>/smoke_probe.py gpu \
  <experiment-dir>/results/gpu.json \
  2>&1 | tee <experiment-dir>/logs/gpu-smoke.log
```

Expected: both processes complete three 256-document runs. These are smoke measurements, not benchmark evidence.

- [ ] **Step 3: Require exact linguistic parity and calculate the observed ratio**

Run:

```bash
nix develop .#builder --command <experiment-dir>/venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("<experiment-dir>/results")
cpu = json.loads((root / "cpu.json").read_text())
gpu = json.loads((root / "gpu.json").read_text())
assert cpu["representative_parse"] == gpu["representative_parse"]

summary = {
    "correctness": "exact_match",
    "label": "throughput_smoke_check",
    "documents_per_run": 256,
    "cpu_documents_per_second": cpu["median_documents_per_second"],
    "gpu_documents_per_second": gpu["median_documents_per_second"],
    "observed_ratio": (
        gpu["median_documents_per_second"] / cpu["median_documents_per_second"]
    ),
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
```

Expected: representative token text, lemma, POS, dependency, and head index match exactly. Report the ratio without an acceptance threshold or performance generalization.

---

### Task 5: Report the Result and Remove Scratch

**Files:**
- Read outside Git: `<experiment-dir>/logs/*`
- Read outside Git: `<experiment-dir>/results/*`
- Delete: the exact validated `<experiment-dir>/`

**Interfaces:**
- Consumes: the first failed compatibility gate or all passing results.
- Produces: a concise final report and no retained scratch state.

- [ ] **Step 1: Capture final facts before cleanup**

Run:

```bash
du -sh <experiment-dir>
git status --short
git rev-parse HEAD
```

Report the chosen installation route, exact versions, passing gates or first failure, exact parity result, throughput smoke ratio if reached, scratch size, and unchanged repository state.

- [ ] **Step 2: Validate the exact destructive target**

Run:

```bash
test "$(dirname "$(realpath <experiment-dir>)")" = /home/bor/Projects
case "$(basename "$(realpath <experiment-dir>)")" in
  natsume-rocm-cupy-*) ;;
  *) exit 1 ;;
esac
```

- [ ] **Step 3: Delete only the validated literal path**

Substitute the exact generated path, not a glob or unresolved variable:

```bash
rm -rf -- /home/bor/Projects/natsume-rocm-cupy-XXXXXX
test ! -e /home/bor/Projects/natsume-rocm-cupy-XXXXXX
git status --short
```

Expected: scratch files are irrecoverably removed; only the pre-existing untracked `AGENDA.md` remains.

## Self-Review

- Present consumer: every scratch file either installs CuPy, proves one compatibility layer, checks linguistic parity, or produces the requested lightweight timing.
- Sufficiency rung: official wheel first; one source-build fallback only for the observed version-boundary risk.
- Performance claim: explicitly a three-run smoke check, not a benchmark or adoption gate.
- Protected boundaries: inherited NLP/NumPy versions, exact parse fields, persistent storage, repository cleanliness, and validated cleanup target.
- Deliberate omission trigger: design reproducible Nix packaging and accelerator CI only after this experiment proves compatibility and the owner requests a supported ROCm path.
