# ROCm CuPy GiNZA Experiment Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the current spaCy/Thinc/GiNZA stack runs correctly through CuPy on the host's ROCm GPU and record a directional CPU/GPU comparison.

**Architecture:** Build only CuPy in a disposable virtual environment under persistent btrfs, then advance through fail-fast probes from HIP arrays to Thinc custom kernels to GiNZA. CPU and GPU measurements use the same scratch-resident probe program; no dependency, Nix, application, or test-suite change is made in the repository.

**Tech Stack:** Nix builder shell, Python 3.12, CuPy 14.1.1 source build, ROCm 7.2.3/HIP, Thinc 8.3.11, spaCy 3.8.11, GiNZA 5.2.0.

## Global Constraints

- Use `/home/bor/Projects`, the persistent btrfs mount, for every virtual-environment, cache, temporary, log, and output file; `/tmp` and `/home/bor` are tmpfs.
- Create exactly one sibling directory named by `mktemp -d -p /home/bor/Projects natsume-rocm-cupy-XXXXXX` and record its exact path.
- Reuse the project builder environment with a `--system-site-packages` virtual environment; install only `cupy==14.1.1` into it.
- Set `CUPY_INSTALL_USE_HIP=1`, `ROCM_HOME=/opt/rocm`, and `HCC_AMDGPU_TARGET=gfx1100` for the source build and probes.
- Stop at the first failed compatibility gate. Test only one evidence-backed remediation at a time; do not stack speculative build flags.
- Do not modify `pyproject.toml`, `uv.lock`, `flake.nix`, application code, tests, or the untracked owner file `AGENDA.md`.
- Do not use AMD's ROCm 6.4 CuPy wheel on this ROCm 7.2.3 host.
- Do not tune batch size, model configuration, or kernels during this experiment.
- Preserve concise results in the final report, then remove the validated exact scratch path and report that it is unrecoverable.

---

### Task 1: Establish the Disposable Workspace and Preflight the Host

**Files:**
- Create outside Git: `/home/bor/Projects/natsume-rocm-cupy-XXXXXX/`
- Create outside Git: `<experiment-dir>/logs/`
- Create outside Git: `<experiment-dir>/results/`
- Create outside Git: `<experiment-dir>/tmp/`
- Create outside Git: `<experiment-dir>/cupy-cache/`
- Create outside Git: `<experiment-dir>/uv-cache/`

**Interfaces:**
- Consumes: project `.#builder` development shell and host `/opt/rocm` installation.
- Produces: one exact `EXPERIMENT_DIR` path and a passing preflight log used by every later task.

- [ ] **Step 1: Record the clean repository baseline**

Run:

```bash
git status --short
git rev-parse HEAD
```

Expected: only the pre-existing untracked `AGENDA.md` is present; record the commit ID.

- [ ] **Step 2: Create the experiment directory on persistent storage**

Run and retain the printed absolute path for every later command:

```bash
mktemp -d -p /home/bor/Projects natsume-rocm-cupy-XXXXXX
```

Verify the returned path before using it:

```bash
findmnt -T <experiment-dir> -no TARGET,FSTYPE,SOURCE
realpath <experiment-dir>
```

Expected: the filesystem is `btrfs`, the resolved parent is `/home/bor/Projects`, and the basename starts with `natsume-rocm-cupy-`.

- [ ] **Step 3: Create the bounded directory layout**

Run:

```bash
mkdir -p <experiment-dir>/{logs,results,tmp,cupy-cache,uv-cache}
df -h <experiment-dir>
```

Expected: all paths are beneath the one experiment directory and sufficient free space remains.

- [ ] **Step 4: Run and capture the complete preflight**

Run:

```bash
nix develop .#builder --command bash -lc '
set -euo pipefail
test -x /opt/rocm/bin/hipcc
test -f /opt/rocm/include/hip/hip_runtime.h
test -e /opt/rocm/lib/libamdhip64.so
rocminfo | grep -q "Name:.*gfx1100"
python - <<"PY"
import importlib.metadata as metadata
import platform

expected = {
    "spacy": "3.8.11",
    "thinc": "8.3.11",
    "ja-ginza": "5.2.0",
}
print("python", platform.python_version())
for package, version in expected.items():
    actual = metadata.version(package)
    print(package, actual)
    if actual != version:
        raise SystemExit(f"{package}: expected {version}, got {actual}")
PY
/opt/rocm/bin/hipcc --version
rocm-smi --showproductname --showuniqueid --showmeminfo vram --json
' 2>&1 | tee <experiment-dir>/logs/preflight.log
```

Expected: Python 3.12.13, spaCy 3.8.11, Thinc 8.3.11, ja-ginza 5.2.0, and the discrete `gfx1100` device are recorded. Stop here if any check fails.

---

### Task 2: Build CuPy Against ROCm

**Files:**
- Create outside Git: `<experiment-dir>/venv/`
- Create outside Git: `<experiment-dir>/logs/cupy-build.log`

**Interfaces:**
- Consumes: passing Task 1 preflight and exact experiment directory.
- Produces: `<experiment-dir>/venv/bin/python` with CuPy 14.1.1, while spaCy/Thinc/GiNZA continue to come from the builder environment.

- [ ] **Step 1: Create the system-site-packages virtual environment**

Run:

```bash
nix develop .#builder --command uv venv \
  --system-site-packages \
  --python python3.12 \
  <experiment-dir>/venv
```

Expected: the venv is created beneath persistent scratch and no `.venv` appears in the repository.

- [ ] **Step 2: Build and install only CuPy from source**

Run:

```bash
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
    --no-binary cupy \
    'cupy==14.1.1' \
  2>&1 | tee <experiment-dir>/logs/cupy-build.log
```

Expected: the source wheel builds and installs successfully. On failure, retain the full log, identify the first compiler/configuration failure, and stop unless one direct evidence-backed correction is available.

- [ ] **Step 3: Prove only CuPy was added**

Run:

```bash
nix develop .#builder --command <experiment-dir>/venv/bin/python - <<'PY'
import importlib.metadata as metadata

for package in ("cupy", "spacy", "thinc", "ja-ginza"):
    print(package, metadata.version(package))
PY
```

Expected: CuPy is 14.1.1 and the NLP versions still match Task 1.

---

### Task 3: Probe CuPy, Thinc Custom Kernels, and GiNZA

**Files:**
- Create outside Git: `<experiment-dir>/compatibility_probe.py`
- Create outside Git: `<experiment-dir>/logs/compatibility.log`
- Create outside Git: `<experiment-dir>/results/compatibility.json`

**Interfaces:**
- Consumes: `<experiment-dir>/venv/bin/python` from Task 2.
- Produces: a gate-by-gate JSON result on success; on failure, the compatibility log identifies the first decisive gate and preserves its traceback.

- [ ] **Step 1: Create the compatibility probe with `apply_patch`**

Create `<experiment-dir>/compatibility_probe.py` with exactly:

```python
from __future__ import annotations

import json
import time
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
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            numpy.testing.assert_allclose(host(actual_item), expected_item)
        return
    numpy.testing.assert_allclose(host(actual), expected)


result: dict[str, object] = {"gates": []}

assert cupy.cuda.runtime.is_hip
cupy.cuda.Device(0).use()
result["cupy_version"] = cupy.__version__
result["device_count"] = cupy.cuda.runtime.getDeviceCount()
result["device_name"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
result["gates"].append("cupy_import")
cupy.show_config()

left = cupy.arange(16, dtype=cupy.float32).reshape(4, 4)
right = cupy.eye(4, dtype=cupy.float32)
product = left @ right
cupy.cuda.runtime.deviceSynchronize()
numpy.testing.assert_allclose(cupy.asnumpy(product), numpy.arange(16, dtype="f").reshape(4, 4))
result["gates"].append("cupy_execution")

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

load_started = time.perf_counter()
nlp = spacy.load("ja_ginza")
result["model_load_seconds"] = time.perf_counter() - load_started
doc = nlp("情報を集めて判断する。")
cupy.cuda.runtime.deviceSynchronize()
assert [token.text for token in doc]

parameter_modules: set[str] = set()
parameter_count = 0
for _name, component in nlp.pipeline:
    model = getattr(component, "model", None)
    if model is None:
        continue
    for node in model.walk():
        for parameter_name in node.param_names:
            parameter = node.get_param(parameter_name)
            parameter_modules.add(type(parameter).__module__.split(".", 1)[0])
            parameter_count += 1

assert "cupy" in parameter_modules
result["parameter_modules"] = sorted(parameter_modules)
result["parameter_count"] = parameter_count
result["gates"].append("ginza_inference")

RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

- [ ] **Step 2: Run the probe with persistent compiler caches**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  CUPY_CACHE_DIR=<experiment-dir>/cupy-cache \
  CUPY_INSTALL_USE_HIP=1 \
  ROCM_HOME=/opt/rocm \
  HCC_AMDGPU_TARGET=gfx1100 \
  HIP_VISIBLE_DEVICES=0 \
  <experiment-dir>/venv/bin/python \
  <experiment-dir>/compatibility_probe.py \
  2>&1 | tee <experiment-dir>/logs/compatibility.log
```

Expected: all five recorded gates pass. If Thinc's runtime kernel compilation fails, record that as the decisive result and do not proceed to benchmarking.

- [ ] **Step 3: Inspect the structured result**

Run:

```bash
jq . <experiment-dir>/results/compatibility.json
```

Expected: `gates` ends with `ginza_inference`, `device_name` identifies the discrete AMD GPU, and `parameter_modules` contains `cupy`.

---

### Task 4: Compare Correctness and Directional Performance

**Files:**
- Create outside Git: `<experiment-dir>/benchmark_probe.py`
- Create outside Git: `<experiment-dir>/results/cpu.json`
- Create outside Git: `<experiment-dir>/results/gpu.json`
- Create outside Git: `<experiment-dir>/logs/cpu-benchmark.log`
- Create outside Git: `<experiment-dir>/logs/gpu-benchmark.log`

**Interfaces:**
- Consumes: passing Task 3 compatibility result.
- Produces: identical CPU/GPU parse records plus seven-run latency and throughput distributions.

- [ ] **Step 1: Create one backend-neutral benchmark probe with `apply_patch`**

Create `<experiment-dir>/benchmark_probe.py` with exactly:

```python
from __future__ import annotations

import argparse
import hashlib
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
SINGLE = SENTENCES[0]
BATCH = tuple(islice(cycle(SENTENCES), 1024))


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


def digest(documents) -> str:
    payload = json.dumps(
        [parsed(doc) for doc in documents],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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


load_started = time.perf_counter()
nlp = spacy.load("ja_ginza")
synchronize()
model_load_seconds = time.perf_counter() - load_started

list(nlp.pipe(SENTENCES, batch_size=8))
synchronize()

single_times: list[float] = []
single_digest = ""
for _ in range(7):
    synchronize()
    started = time.perf_counter()
    document = nlp(SINGLE)
    synchronize()
    single_times.append(time.perf_counter() - started)
    single_digest = digest([document])

batch_times: list[float] = []
batch_digest = ""
for _ in range(7):
    synchronize()
    started = time.perf_counter()
    documents = list(nlp.pipe(BATCH, batch_size=128))
    synchronize()
    batch_times.append(time.perf_counter() - started)
    current_digest = digest(documents)
    if batch_digest and current_digest != batch_digest:
        raise AssertionError("parse output changed between timed blocks")
    batch_digest = current_digest

result = {
    "backend": args.backend,
    "model_load_seconds": model_load_seconds,
    "representative_parse": [parsed(doc) for doc in nlp.pipe(SENTENCES, batch_size=8)],
    "single": {
        "documents": 1,
        "digest": single_digest,
        "seconds": single_times,
        "min_seconds": min(single_times),
        "median_seconds": statistics.median(single_times),
        "max_seconds": max(single_times),
    },
    "batch": {
        "documents": len(BATCH),
        "batch_size": 128,
        "digest": batch_digest,
        "seconds": batch_times,
        "min_seconds": min(batch_times),
        "median_seconds": statistics.median(batch_times),
        "max_seconds": max(batch_times),
        "median_documents_per_second": len(BATCH) / statistics.median(batch_times),
    },
}
if cupy is not None:
    pool = cupy.get_default_memory_pool()
    result["cupy_memory_pool_total_bytes"] = pool.total_bytes()
    result["cupy_memory_pool_used_bytes"] = pool.used_bytes()

args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
```

- [ ] **Step 2: Run the CPU process**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  <experiment-dir>/venv/bin/python \
  <experiment-dir>/benchmark_probe.py cpu \
  <experiment-dir>/results/cpu.json \
  2>&1 | tee <experiment-dir>/logs/cpu-benchmark.log
```

Expected: seven single and seven batch timings are recorded.

- [ ] **Step 3: Run the GPU process**

Run:

```bash
set -o pipefail
nix develop .#builder --command env \
  TMPDIR=<experiment-dir>/tmp \
  CUPY_CACHE_DIR=<experiment-dir>/cupy-cache \
  CUPY_INSTALL_USE_HIP=1 \
  ROCM_HOME=/opt/rocm \
  HCC_AMDGPU_TARGET=gfx1100 \
  HIP_VISIBLE_DEVICES=0 \
  <experiment-dir>/venv/bin/python \
  <experiment-dir>/benchmark_probe.py gpu \
  <experiment-dir>/results/gpu.json \
  2>&1 | tee <experiment-dir>/logs/gpu-benchmark.log
```

Expected: the same workload and seven-run timing shape are recorded through CuPy.

- [ ] **Step 4: Compare exact correctness fields and performance summaries**

Run:

```bash
nix develop .#builder --command <experiment-dir>/venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("<experiment-dir>/results")
cpu = json.loads((root / "cpu.json").read_text())
gpu = json.loads((root / "gpu.json").read_text())

assert cpu["representative_parse"] == gpu["representative_parse"]
assert cpu["single"]["digest"] == gpu["single"]["digest"]
assert cpu["batch"]["digest"] == gpu["batch"]["digest"]

summary = {
    "correctness": "exact_match",
    "cpu_model_load_seconds": cpu["model_load_seconds"],
    "gpu_model_load_seconds": gpu["model_load_seconds"],
    "cpu_single_median_seconds": cpu["single"]["median_seconds"],
    "gpu_single_median_seconds": gpu["single"]["median_seconds"],
    "cpu_single_range_seconds": [cpu["single"]["min_seconds"], cpu["single"]["max_seconds"]],
    "gpu_single_range_seconds": [gpu["single"]["min_seconds"], gpu["single"]["max_seconds"]],
    "cpu_batch_documents_per_second": cpu["batch"]["median_documents_per_second"],
    "gpu_batch_documents_per_second": gpu["batch"]["median_documents_per_second"],
    "cpu_batch_range_seconds": [cpu["batch"]["min_seconds"], cpu["batch"]["max_seconds"]],
    "gpu_batch_range_seconds": [gpu["batch"]["min_seconds"], gpu["batch"]["max_seconds"]],
    "batch_speedup": (
        gpu["batch"]["median_documents_per_second"]
        / cpu["batch"]["median_documents_per_second"]
    ),
    "gpu_memory_pool_total_bytes": gpu["cupy_memory_pool_total_bytes"],
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
```

Expected: correctness is an exact match; performance is reported without an adoption threshold.

---

### Task 5: Capture the Disposition and Remove Scratch State

**Files:**
- Read outside Git: `<experiment-dir>/results/*.json`
- Read outside Git: `<experiment-dir>/logs/*.log`
- Delete after reporting: exact validated `<experiment-dir>/`

**Interfaces:**
- Consumes: the first failing gate or all results from Tasks 3–4.
- Produces: a concise user-facing compatibility/performance report and no retained experiment files.

- [ ] **Step 1: Record the final environment and scratch size**

Run:

```bash
du -sh <experiment-dir>
git status --short
git rev-parse HEAD
```

Expected: repository status and commit match the Task 1 baseline; no dependency or application files changed.

- [ ] **Step 2: Summarize evidence before deletion**

Report:

- exact host GPU, ROCm, Python, CuPy, spaCy, Thinc, and GiNZA versions;
- first failing gate, or all passing gates;
- exact CPU/GPU correctness disposition;
- model-load times, median single-sentence latency, median batch throughput, and batch speedup;
- CuPy memory-pool use and device-memory observation when available;
- scratch-directory size and repository cleanliness;
- whether the result establishes incompatibility, technical feasibility only, or enough evidence to propose a separately designed supported ROCm path.

- [ ] **Step 3: Validate the destructive target**

Run with the exact path, not a glob or unresolved environment variable:

```bash
test "$(dirname "$(realpath <experiment-dir>)")" = /home/bor/Projects
case "$(basename "$(realpath <experiment-dir>)")" in
  natsume-rocm-cupy-*) ;;
  *) exit 1 ;;
esac
```

Expected: both checks pass. Stop without deleting if either fails.

- [ ] **Step 4: Remove the exact experiment directory**

Run only after substituting the verified absolute path literally:

```bash
rm -rf -- /home/bor/Projects/natsume-rocm-cupy-XXXXXX
test ! -e /home/bor/Projects/natsume-rocm-cupy-XXXXXX
```

Expected: the disposable environment, caches, logs, and results are deleted and cannot be recovered from the experiment directory. Nix store paths remain under Nix garbage-collection ownership.

- [ ] **Step 5: Verify final repository state**

Run:

```bash
git status --short
```

Expected: only the owner's pre-existing untracked `AGENDA.md` remains.

## Self-Review

- Spec coverage: persistent storage, source build, fail-fast gates, Thinc custom kernels, GiNZA inference, exact parity, directional performance, evidence, and cleanup each have an owning task.
- Placeholder scan: `<experiment-dir>` and the final generated suffix are deliberate execution substitutions for the exact path returned by `mktemp`; no behavioral or implementation decision is deferred.
- Interface consistency: Tasks 2–4 consistently consume `<experiment-dir>/venv/bin/python`; structured results are written beneath `<experiment-dir>/results`; cleanup targets only the exact Task 1 directory.
- Scope: no production file or dependency changes are planned. This is one disposable experiment, not a supported ROCm feature.
