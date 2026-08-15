# ROCm CuPy GiNZA Compatibility Experiment

Status: Approved for execution

## Purpose

Determine whether the current Natsume NLP stack can execute GiNZA/spaCy inference through Thinc's CuPy backend on the host's AMD GPU. This is a disposable compatibility experiment, not a dependency or production-backend change.

The host evidence at design time is:

- Radeon Navi 31 (`gfx1100`) with approximately 24 GiB VRAM;
- ROCm runtime and tools 7.2.3, with `/opt/rocm` resolving to the Nix-managed combined ROCm closure;
- Python 3.12.13, spaCy 3.8.11, Thinc 8.3.11, and GiNZA 5.2.0 in the project builder environment.

## Isolation and Storage

The experiment creates one uniquely named sibling directory with:

```text
mktemp -d -p /home/bor/Projects natsume-rocm-cupy-XXXXXX
```

`/home/bor/Projects` is persistent btrfs with 68 GiB available at design time. `/tmp` and the home directory are not used because they are tmpfs. The virtual environment, source build, and package cache all live under the one explicit sibling directory. The Git repository, lockfile, Nix flake, and configured environments remain unchanged.

After evidence is captured, the exact experiment directory is removed. Nix store paths remain under ordinary Nix garbage-collection ownership.

## Installation Approach

Use the project builder environment for the exact Python NLP stack and create a virtual environment with access to those installed packages. Build CuPy 14.1.1 from its source distribution against the host ROCm installation with:

- `CUPY_INSTALL_USE_HIP=1`;
- `ROCM_HOME=/opt/rocm`;
- `HCC_AMDGPU_TARGET=gfx1100`.

Do not use AMD's ROCm 6.4 wheel on the ROCm 7.2.3 host. Do not add CuPy to `pyproject.toml`, `uv.lock`, or `flake.nix` during the experiment.

If the build fails, capture the complete failing command, compiler error, and detected ROCm/CuPy configuration. Do not stack speculative build changes: form and test one new hypothesis at a time.

## Compatibility Gates

Run the following gates in order and stop at the first failure:

1. **ROCm device:** `rocminfo` exposes `gfx1100` and the discrete GPU is visible.
2. **CuPy/HIP:** import CuPy, confirm `runtime.is_hip`, select device 0, allocate arrays, perform matrix multiplication, synchronize, and verify the numeric result against NumPy.
3. **Thinc backend:** select `CupyOps`, confirm `get_current_ops().name == "cupy"`, allocate an array, and confirm its type/module is CuPy.
4. **spaCy activation:** call `spacy.require_gpu(0)` before loading any pipeline and require a true result.
5. **GiNZA inference:** load `ja_ginza`, parse representative Japanese sentences, and confirm that pipeline model arrays use the GPU backend.
6. **Correctness parity:** run the same fixed inputs through a fresh CPU process and compare token text, lemma, POS, dependency label, and head index exactly.

Passing an early gate is not evidence that a later layer works. In particular, a functioning CuPy array does not prove that GiNZA's complete component graph can execute on ROCm.

## Directional Performance Check

Measure CPU and GPU in separate fresh processes using the same installed model and fixed Japanese input set.

- Warm each process before timing.
- Synchronize the GPU before and after timed regions.
- Record median elapsed time over repeated runs for one sentence and for a repeated batch.
- Report parsing throughput and workload size, not only one elapsed duration.
- Exclude model-load time from steady-state inference timing, but report model-load time separately.

Single-sentence GPU latency may reasonably be worse due to transfer and kernel-launch overhead. The batch result is the meaningful throughput signal. These timings are diagnostic, not a production benchmark or acceptance threshold.

## Evidence and Disposition

The final report records:

- exact CuPy version and build configuration;
- ROCm runtime, GPU architecture, Python, spaCy, Thinc, and GiNZA versions;
- the first failing gate or all passing gates;
- CPU/GPU correctness comparison;
- directional timings and peak observed VRAM where available;
- experiment directory size before cleanup;
- confirmation that the repository stayed unchanged and the explicit scratch directory was removed.

A passing experiment only establishes technical feasibility on this host. Adding a supported ROCm build path would require a separate design covering reproducible Nix packaging, accelerator CI evidence, deterministic artifact identity, performance, and operational support.

## Deliberate Omissions

- No repository dependency, lockfile, Nix output, devcontainer, or production CLI change.
- No AMD CuPy binary wheel with a mismatched ROCm ABI.
- No container experiment unless the native source build is blocked specifically by host packaging.
- No corpus rebuild or multi-hour extraction run.
- No claim that experimental CuPy ROCm support is suitable for production.
