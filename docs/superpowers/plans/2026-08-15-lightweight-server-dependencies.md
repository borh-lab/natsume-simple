# Lightweight Server Dependencies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the server's Python dependency ownership explicit and prove that both the uv backend composition and packaged Nix server exclude the NLP/model/accelerator stack.

**Architecture:** DuckDB remains the only base dependency because artifact tooling and serving share it. Pydantic, AnyIO, FastAPI, and Uvicorn belong to the `backend` extra; uv verifies that composition in a clean isolated environment, while the existing Nix closure and end-to-end server smoke remain the maintained production proofs.

**Tech Stack:** Python 3.14, uv, pytest, Nix flakes, FastAPI, AnyIO, Pydantic, Uvicorn, DuckDB.

## Global Constraints

- The server must run without `builder`, `cpu`, `cuda`, `rocm`, `electra`, spaCy, GiNZA, Torch, Polars, wtpsplit, or NLP model packages.
- Do not add a second server CLI, uv-specific fixture builder, smoke script, or CI workflow.
- The ordinary uv launch command is for development; only `uv run --isolated` proves absence from a clean composition.
- The existing Nix `server-smoke` check remains the sole maintained fixture/startup test.
- Keep `AGENDA.md` and all unrelated working-tree files untouched.

---

## File map

- `pyproject.toml`: owns direct Python dependency membership and version ranges.
- `uv.lock`: records the resolved dependency graph after ownership changes.
- `tests/test_dependency_configuration.py`: pins the project-level dependency boundary.
- `README.md`: documents the supported uv backend launch and the isolated diagnostic proof.
- `flake.nix`: owns the production server environment, closure exclusion gate, and end-to-end smoke.
- `docs/superpowers/specs/2026-08-15-lightweight-server-dependencies-design.md`: records final implementation evidence.

### Task 1: Correct Python dependency ownership and document the uv path

**Files:**
- Modify: `tests/test_dependency_configuration.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

**Interfaces:**
- Consumes: the existing `[project].dependencies` and `[project.optional-dependencies].backend` metadata.
- Produces: a base dependency list containing only `duckdb>=1.5.5,<2`, and a `backend` extra containing direct AnyIO, FastAPI, Pydantic, and Uvicorn requirements.

- [ ] **Step 1: Write the failing dependency-ownership test**

Append this test to `tests/test_dependency_configuration.py`:

```python
def test_backend_extra_owns_every_direct_server_dependency() -> None:
    configuration = project_configuration()

    assert configuration["project"]["dependencies"] == ["duckdb>=1.5.5,<2"]
    assert configuration["project"]["optional-dependencies"]["backend"] == [
        "anyio>=4.14.2,<5",
        "fastapi>=0.141.1,<1",
        "pydantic>=2.13.4,<3",
        "uvicorn>=0.52.1,<1",
    ]
```

This catches either server-only Pydantic returning to the shared base set or the API's direct AnyIO import becoming dependent on FastAPI's transitive metadata.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
nix develop .#test --command \
  pytest tests/test_dependency_configuration.py::test_backend_extra_owns_every_direct_server_dependency -v
```

Expected: FAIL because base dependencies still include Pydantic and the backend extra does not directly include AnyIO or Pydantic.

- [ ] **Step 3: Move the direct dependencies to their owning extra**

Change the relevant `pyproject.toml` sections to:

```toml
dependencies = [
    "duckdb>=1.5.5,<2",
]

[project.optional-dependencies]
backend = [
    "anyio>=4.14.2,<5",
    "fastapi>=0.141.1,<1",
    "pydantic>=2.13.4,<3",
    "uvicorn>=0.52.1,<1",
]
```

Leave every builder, model, and accelerator extra unchanged.

- [ ] **Step 4: Regenerate the committed uv lock without upgrading unrelated packages**

Run:

```bash
uv lock --offline
uv lock --check
```

Expected: the editable `natsume-simple` package metadata moves Pydantic from its base dependencies to `backend`, adds AnyIO to `backend`, and all resolved package versions remain unchanged.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run:

```bash
nix develop .#test --command \
  pytest tests/test_dependency_configuration.py::test_backend_extra_owns_every_direct_server_dependency -v
```

Expected: PASS.

- [ ] **Step 6: Document the ordinary uv server launch**

In `README.md`, after the existing Nix backend launch example, add:

````markdown
The equivalent uv environment selects only the backend extra:

```bash
NATSUME_ARTIFACT_DIR=deploy/current \
uv run --locked --no-dev --extra backend \
  uvicorn natsume_simple.api:app --host 127.0.0.1 --port 8000
```

This ordinary command may reuse the checkout's existing `.venv`. To prove the
backend composition itself contains no NLP or accelerator modules, use an isolated
environment:

```bash
uv run --isolated --locked --no-dev --extra backend python - <<'PY'
from importlib.util import find_spec

import natsume_simple.api

forbidden = (
    "cupy",
    "ginza",
    "polars",
    "spacy",
    "tokenizers",
    "torch",
    "transformers",
    "triton",
    "wtpsplit",
)
present = [name for name in forbidden if find_spec(name) is not None]
assert not present, present
print("backend-only import passed")
PY
```

The isolated command verifies dependency composition; `nix build
.#checks.x86_64-linux.server-smoke` remains the end-to-end packaged-server test.
````

- [ ] **Step 7: Execute the documented isolated proof**

Run the exact `uv run --isolated ...` block added to `README.md`.

Expected: `backend-only import passed`; none of the nine forbidden modules is discoverable.

- [ ] **Step 8: Run the complete dependency-configuration test file**

Run:

```bash
nix develop .#test --command pytest tests/test_dependency_configuration.py -v
```

Expected: all dependency configuration tests PASS.

- [ ] **Step 9: Commit the Python/uv boundary**

```bash
git add pyproject.toml uv.lock tests/test_dependency_configuration.py README.md
git commit -m "build: isolate backend python dependencies"
```

### Task 2: Strengthen the production Nix closure gate

**Files:**
- Modify: `flake.nix`

**Interfaces:**
- Consumes: the existing `server` derivation and `exportReferencesGraph` output named `server-closure`.
- Produces: the same `checks.<system>.server-closure` contract with a broader forbidden-family set; no new flake output.

This is a configuration-only hardening step. A new test abstraction would merely duplicate Nix's reference graph. The behavioral verification is the real `server-closure` derivation, followed by the existing packaged-server smoke.

- [ ] **Step 1: Expand the forbidden package-family expression**

Replace the existing closure grep with this alphabetized family list:

```nix
if grep -E -- '-(cuda|cupy|ginza|jupyter|nodejs|notebook|polars|rocm|spacy|tokenizers|torch|transformers|triton|wtpsplit)(-|$)' \
  server-closure; then
  echo "server closure contains a forbidden build or accelerator dependency" >&2
  exit 1
fi
```

Keep the existing `exportReferencesGraph`, check name, error behavior, and output unchanged.

- [ ] **Step 2: Format the edited Nix file**

Run:

```bash
nix fmt flake.nix
```

Expected: exit 0 and no unrelated formatting changes.

- [ ] **Step 3: Build the closure gate**

Run:

```bash
nix build .#checks.x86_64-linux.server-closure --no-link --print-build-logs
```

Expected: PASS; the production server reference graph contains none of the forbidden families.

- [ ] **Step 4: Build the packaged-server smoke**

Run:

```bash
nix build .#checks.x86_64-linux.server-smoke --no-link --print-build-logs
```

Expected: PASS; the packaged server reaches readiness, serves the frontend, and returns the fixture collocation response.

- [ ] **Step 5: Verify the Nix diff is limited to the closure expression**

Run:

```bash
git diff --check
git diff -- flake.nix
```

Expected: no whitespace errors; only the forbidden-family expression changes.

- [ ] **Step 6: Commit the closure hardening**

```bash
git add flake.nix
git commit -m "build: harden server dependency closure"
```

### Task 3: Run final gates and record implementation evidence

**Files:**
- Modify: `docs/superpowers/specs/2026-08-15-lightweight-server-dependencies-design.md`

**Interfaces:**
- Consumes: the completed uv metadata, isolated import proof, and existing Nix checks.
- Produces: a durable implementation status and exact evidence in the design record.

- [ ] **Step 1: Run the source and backend gates together**

Run:

```bash
nix build \
  .#checks.x86_64-linux.source-quality \
  .#checks.x86_64-linux.backend \
  .#checks.x86_64-linux.package-server \
  .#checks.x86_64-linux.server-closure \
  .#checks.x86_64-linux.server-smoke \
  --no-link --print-build-logs
```

Expected: all five derivations PASS.

- [ ] **Step 2: Re-run the isolated uv dependency proof**

Run the exact isolated command documented in Task 1 Step 6.

Expected: `backend-only import passed`.

- [ ] **Step 3: Record only evidence actually observed**

Change the design document status to:

```markdown
Status: Implemented on 2026-08-15.

Implementation evidence: the isolated locked uv backend import reports no builder,
NLP, model, or accelerator modules, and the Nix source-quality, backend,
package-server, server-closure, and server-smoke checks pass on the implemented tree.
```

Do not record a check as passing unless its command in Steps 1–2 exited zero in this execution.

- [ ] **Step 4: Review the complete diff and large-file hygiene**

Run:

```bash
git diff --check
git status --short
git diff --stat a5cd964..HEAD
git ls-files -z | xargs -0 -r du -h | sort -h | tail -20
```

Expected: only the planned dependency metadata, lock, README, test, Nix expression, design status, and this plan changed; `AGENDA.md` remains untracked and untouched; no database, archive, model, environment, or other large artifact is tracked.

- [ ] **Step 5: Commit the implementation evidence**

```bash
git add docs/superpowers/specs/2026-08-15-lightweight-server-dependencies-design.md
git commit -m "docs: record lightweight server verification"
```

- [ ] **Step 6: Inspect final history and worktree**

Run:

```bash
git log -5 --oneline
git status --short --branch
```

Expected: this plan and three implementation commits follow the two design commits; only the pre-existing untracked `AGENDA.md` remains.
