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

Use the project builder environment for the exact Python NLP stack and create a virtual environment with `--system-site-packages`. This reuses the project's spaCy, Thinc, GiNZA, NumPy, and Torch packages; CuPy is the only package added to the disposable environment.

Build CuPy 14.1.1 from its source distribution against the host ROCm installation with:

- `CUPY_INSTALL_USE_HIP=1`;
- `ROCM_HOME=/opt/rocm`;
- `HCC_AMDGPU_TARGET=gfx1100`.

Put `TMPDIR`, `UV_CACHE_DIR`, `CUPY_CACHE_DIR`, the virtual environment, probe scripts, and captured output beneath the experiment directory. Limit parallel compilation to a reasonable fixed job count rather than consuming every host core.

Do not use AMD's ROCm 6.4 wheel on the ROCm 7.2.3 host. Do not add CuPy to `pyproject.toml`, `uv.lock`, or `flake.nix` during the experiment.

The current Nixpkgs `python312Packages.cupy` derivation is not an efficient shortcut: it overrides the build environment with `cudaPackages.backendStdenv`, injects CUDA compiler and library paths, and depends on CUDA libraries. Adapting that package would be a separate packaging task rather than a smaller compatibility experiment. Use the upstream source build once; do not create a project-local Nix override unless the experiment passes and a supported ROCm path is later approved.

If the build fails, capture the complete failing command, compiler error, and detected ROCm/CuPy configuration. Do not stack speculative build changes: form and test one new hypothesis at a time.

## Efficient Execution Ladder

Run the following gates in order and stop at the first failure:

1. **Build preflight:** require the discrete `gfx1100` device, `/opt/rocm/bin/hipcc`, the HIP runtime header and library, writable persistent scratch space, and the expected Python/spaCy/Thinc/GiNZA versions. A missing prerequisite stops the experiment before downloading or compiling CuPy.
2. **CuPy build and import:** build only CuPy, import it, require `runtime.is_hip`, select the discrete device explicitly, and record `cupy.show_config()`.
3. **CuPy/HIP execution:** allocate arrays, perform matrix multiplication, synchronize, and verify the numeric result against NumPy.
4. **Thinc custom kernels:** select `CupyOps`, require `get_current_ops().name == "cupy"`, and compare representative `maxout`, `seq2col`, and `hash` results with `NumpyOps`. These calls exercise Thinc's runtime-compiled `RawModule` and `RawKernel` paths rather than only CuPy allocation.
5. **spaCy activation:** call `spacy.require_gpu(0)` before loading any pipeline and require a true result.
6. **GiNZA inference:** load `ja_ginza`, parse representative Japanese sentences, and confirm the active Thinc backend and pipeline arrays remain on CuPy.
7. **Correctness parity:** run the same fixed inputs through a fresh CPU process and compare token text, lemma, POS, dependency label, and head index exactly.

Passing an early gate is not evidence that a later layer works. In particular, a functioning CuPy array does not prove that GiNZA's complete component graph can execute on ROCm.

Gate 4 is the decisive compatibility boundary before model loading. Thinc's `CupyOps` compiles bundled CUDA-language custom kernels at runtime; ROCm support therefore depends on more than CuPy exposing a HIP-backed ndarray. If those kernels do not compile or agree numerically, stop there rather than debugging GiNZA symptoms downstream.

## Directional Performance Check

Measure CPU and GPU in separate fresh processes using one probe script stored in the scratch directory. The script accepts the backend as an argument so workload construction, output serialization, and timing logic cannot drift between the two runs.

- Use one fixed representative Japanese sentence for latency and a fixed sentence suite repeated to 1,024 documents for throughput.
- Run throughput through `nlp.pipe(..., batch_size=128)` on both backends. This measures the batched path where GPU execution can plausibly help rather than an artificial loop of single-document calls.
- Warm each process before timing, then run seven timed blocks and report median plus minimum and maximum.
- Synchronize the GPU before and after timed regions.
- Report parsing throughput and workload size, not only one elapsed duration.
- Exclude model-load time from steady-state inference timing, but report model-load time separately.
- Compare a stable digest of parsed outputs on every timed backend before accepting timing results.
- Record peak CuPy memory-pool use and the observed device-memory high-water mark when available.

Single-sentence GPU latency may reasonably be worse due to transfer and kernel-launch overhead. The batch result is the meaningful throughput signal. These timings are diagnostic, not a production benchmark or acceptance threshold.

Do not tune batch size, kernel settings, or model configuration during this experiment. A passing run answers whether the unmodified current stack works and gives a directional performance result. Optimization and packaging have no present consumer unless that result justifies a production ROCm proposal.

## Evidence and Disposition

The final report records:

- exact CuPy version and build configuration;
- ROCm runtime, GPU architecture, Python, spaCy, Thinc, and GiNZA versions;
- the first failing gate or all passing gates;
- CPU/GPU correctness comparison;
- directional timings and peak observed VRAM where available;
- experiment directory size before cleanup;
- confirmation that the repository stayed unchanged and the explicit scratch directory was removed.

Capture one machine-readable result file plus the build log inside the scratch directory, copy only the concise findings into the final report, and then remove the exact directory after verifying that its resolved parent is `/home/bor/Projects` and its basename begins with `natsume-rocm-cupy-`. No build cache or virtual environment is retained.

A passing experiment only establishes technical feasibility on this host. Adding a supported ROCm build path would require a separate design covering reproducible Nix packaging, accelerator CI evidence, deterministic artifact identity, performance, and operational support.

## Deliberate Omissions

- No repository dependency, lockfile, Nix output, devcontainer, or production CLI change.
- No AMD CuPy binary wheel with a mismatched ROCm ABI.
- No container experiment unless the native source build is blocked specifically by host packaging.
- No corpus rebuild or multi-hour extraction run.
- No claim that experimental CuPy ROCm support is suitable for production.
