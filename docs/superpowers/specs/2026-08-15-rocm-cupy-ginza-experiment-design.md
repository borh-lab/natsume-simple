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

`/home/bor/Projects` is persistent btrfs with 68 GiB available at design time. `/tmp` and the home directory are not used because they are tmpfs. The virtual environment, package download, optional source-build fallback, and caches all live under the one explicit sibling directory. The Git repository, lockfile, Nix flake, and configured environments remain unchanged.

After evidence is captured, the exact experiment directory is removed. Nix store paths remain under ordinary Nix garbage-collection ownership.

## Installation Approach

Use the project builder environment for the exact Python NLP stack and create a virtual environment with `--system-site-packages`. This reuses the project's spaCy, Thinc, GiNZA, NumPy, and Torch packages. Install CuPy with `--no-deps` so dependency resolution cannot shadow that inherited stack; the existing NumPy 2.5.2 satisfies CuPy 14.1.1's declared `numpy>=2.0,<2.6` requirement.

Try the official `cupy-rocm-7-0==14.1.1` CPython 3.12 wheel first. CuPy's [current repository](https://github.com/cupy/cupy) and [PyPI release](https://pypi.org/project/cupy-rocm-7-0/14.1.1/) publish that wheel, although the [stable installation page](https://docs.cupy.dev/en/stable/install.html#installing-binary-packages) still contains stale text saying recent ROCm wheels are unavailable. The wheel targets ROCm 7.0 while this host has ROCm 7.2.3, so importing and executing it is a compatibility gate rather than an assumed success.

Only if the wheel fails specifically because of the ROCm runtime/ABI boundary, discard and recreate the virtual environment, then build CuPy 14.1.1 from its source distribution against the host ROCm installation with:

- `CUPY_INSTALL_USE_HIP=1`;
- `ROCM_HOME=/opt/rocm`;
- `HCC_AMDGPU_TARGET=gfx1100`.

Put `TMPDIR`, `UV_CACHE_DIR`, `CUPY_CACHE_DIR`, the virtual environment, probe scripts, and captured output beneath the experiment directory. If the fallback is needed, limit parallel compilation to a reasonable fixed job count rather than consuming every host core.

Do not use AMD's ROCm 6.4 wheel on the ROCm 7.2.3 host. Do not keep both the binary and source distributions in one virtual environment. Do not add CuPy to `pyproject.toml`, `uv.lock`, or `flake.nix` during the experiment.

The current Nixpkgs `python312Packages.cupy` derivation is not an efficient shortcut: it overrides the build environment with `cudaPackages.backendStdenv`, injects CUDA compiler and library paths, and depends on CUDA libraries. Adapting that package would be a separate packaging task rather than a smaller compatibility experiment. If the official wheel exposes the known runtime-version boundary, use the upstream source build once; do not create a project-local Nix override unless the experiment passes and a supported ROCm path is later approved.

If the wheel fails, capture the complete import/runtime error before deciding whether it licenses the source-build fallback. If the fallback build fails, capture the complete failing command, compiler error, and detected ROCm/CuPy configuration. Do not stack speculative build changes: form and test one new hypothesis at a time.

## Efficient Execution Ladder

Run the following gates in order and stop at the first failure:

1. **Runtime preflight:** require the discrete `gfx1100` device, the HIP runtime library, writable persistent scratch space, and the expected Python/spaCy/Thinc/GiNZA versions. A missing prerequisite stops the experiment before downloading CuPy.
2. **CuPy installation and import:** install the official wheel without dependencies and try to import it. If and only if the failure identifies the host ROCm version/runtime boundary, require `/opt/rocm/bin/hipcc` and the HIP headers and rebuild from source in a fresh virtual environment. Require `runtime.is_hip`, select the discrete device explicitly, and record `cupy.show_config()`.
3. **CuPy/HIP execution:** allocate arrays, perform matrix multiplication, synchronize, and verify the numeric result against NumPy.
4. **Thinc custom kernels:** select `CupyOps`, require `get_current_ops().name == "cupy"`, and compare representative `maxout`, `seq2col`, and `hash` results with `NumpyOps`. These calls exercise Thinc's runtime-compiled `RawModule` and `RawKernel` paths rather than only CuPy allocation.
5. **spaCy activation:** call `spacy.require_gpu(0)` before loading any pipeline and require a true result.
6. **GiNZA inference:** load `ja_ginza`, parse representative Japanese sentences, and confirm the active Thinc backend and pipeline arrays remain on CuPy.
7. **Correctness parity:** run the same fixed inputs through a fresh CPU process and compare token text, lemma, POS, dependency label, and head index exactly.

Passing an early gate is not evidence that a later layer works. In particular, a functioning CuPy array does not prove that GiNZA's complete component graph can execute on ROCm.

Gate 4 is the decisive compatibility boundary before model loading. Thinc's `CupyOps` compiles bundled CUDA-language custom kernels at runtime; ROCm support therefore depends on more than CuPy exposing a HIP-backed ndarray. If those kernels do not compile or agree numerically, stop there rather than debugging GiNZA symptoms downstream.

## Performance Smoke Check

Compatibility and correctness are the decision. Performance is only a lightweight sanity check: if the current stack genuinely offloads GiNZA's batched model work, the difference should be large enough to see without building a statistically rigorous benchmark.

Measure CPU and GPU in separate fresh processes using one probe script stored in the scratch directory. The script accepts the backend as an argument so workload construction, output serialization, and timing logic cannot drift between the two runs.

- Use a fixed sentence suite repeated to 256 documents.
- Run throughput through `nlp.pipe(..., batch_size=128)` on both backends. This measures the batched path where GPU execution can plausibly help rather than an artificial loop of single-document calls.
- Warm each process before timing, then run three timed blocks and report their median.
- Synchronize the GPU before and after timed regions.
- Report parsing throughput and workload size.
- Keep model loading outside the timed region.
- Require exact CPU/GPU parse parity on the representative correctness inputs before reporting timing.

Do not treat these timings as a production benchmark, attach an acceptance threshold, repeat process-order permutations, or tune batch size, kernel settings, or model configuration. Report the observed ratio with an explicit smoke-check label. A passing run answers whether the unmodified current stack works; optimization and packaging have no present consumer unless that result justifies a production ROCm proposal.

## Evidence and Disposition

The final report records:

- exact CuPy version and build configuration;
- ROCm runtime, GPU architecture, Python, spaCy, Thinc, and GiNZA versions;
- the first failing gate or all passing gates;
- CPU/GPU correctness comparison;
- lightweight CPU/GPU throughput smoke result;
- experiment directory size before cleanup;
- confirmation that the repository stayed unchanged and the explicit scratch directory was removed.

Capture one machine-readable result file plus the build log inside the scratch directory, copy only the concise findings into the final report, and then remove the exact directory after verifying that its resolved parent is `/home/bor/Projects` and its basename begins with `natsume-rocm-cupy-`. No build cache or virtual environment is retained.

A passing experiment only establishes technical feasibility on this host. Adding a supported ROCm build path would require a separate design covering reproducible Nix packaging, accelerator CI evidence, deterministic artifact identity, performance, and operational support.

## Deliberate Omissions

- No repository dependency, lockfile, Nix output, devcontainer, or production CLI change.
- No AMD CuPy binary wheel with a mismatched ROCm ABI.
- No container experiment unless both the official wheel and native source fallback are blocked specifically by host packaging.
- No corpus rebuild or multi-hour extraction run.
- No claim that experimental CuPy ROCm support is suitable for production.
