# Nix Packages and OCI Image Design

**Status:** Revised draft after written review

**Date:** 2026-08-12
**Boundary:** Authoritative Nix build/check interfaces, focused development
shells, production server closure, and derived OCI image

## Problem

The current flake primarily wraps mutable development commands. Entering the
shell runs setup, `uv sync`, and virtualenv activation; frontend/server commands
run package-manager installation and rebuild at execution time. This is useful
as convenience scripting but is not an authoritative reproducible production
artifact.

An untracked `container.nix` is an early sketch with placeholder `services.foo`
options and unresolved `nix2container`, `self'`, `runtime-packages`, and server
output assumptions. Its intent—derive a container from Nix—is valid, but its
interface is not established and must not become a second competing design.

## Goals

- Make the flake the source of truth for building, checking, developing, and
  running the project.
- Produce real immutable frontend, server, corpus-builder, and OCI outputs.
- Preserve the non-mutating dev-shell entry and minimal checks established by
  Spec 1 while replacing wrappers with real derivations.
- Ensure the production closure excludes build, notebook, NLP, and accelerator
  machinery.
- Derive the OCI image from exactly the same server package used outside a
  container.

## Non-Goals

- Kubernetes, Nomad/systemd deployment modules, registry publishing, signing,
  multi-instance orchestration, embedded production corpus images, or non-Linux
  containers.
- Requiring an OCI image for ordinary local Nix execution.
- Maintaining both a hand-written Docker production image and a Nix image.

## Flake Outputs

For every supported system where dependencies exist:

### Packages

- `frontend`: static Svelte output built from committed npm manifest/lock using
  `npm ci` semantics and no network during the derivation build.
- `server`: minimal Python environment/application with the `frontend` assets and
  one executable entry point.
- `corpus-builder-cpu`: offline builder with CPU data/NLP dependencies.
- `corpus-builder-cuda`: supported Linux systems only when Cohort 7 resolves and
  builds.
- `corpus-builder-rocm`: supported Linux systems only when Cohort 7 resolves and
  builds.
- `container`: x86_64-linux OCI image derived from `server`.
- `default`: `server`, unless the owner later chooses another explicit operator
  experience.

Packages do not invoke `uv sync`, `npm install`, or external downloads at
runtime. Their dependencies are realized during Nix builds from pinned inputs.

### Apps

- `serve`: run the server package with explicit artifact-path/host/port
  configuration.
- `build-corpus`: run the CPU builder by default.
- `check`: optional convenience app delegating to check-only commands; CI still
  targets flake checks directly.

Apps are thin launch interfaces over packages, not mutable setup scripts.

### Checks

- Nix format and flake evaluation.
- Python format, lint, type, unit/property/integration tests.
- Frontend format, lint, `svelte-check`, unit/component tests, and production
  build.
- OpenAPI-to-TypeScript drift.
- Model-free backend walkthrough doctests and their inventory.
- Fixture Playwright flow.
- npm high/critical advisory policy.
- Explicit builds of frontend, server, CPU builder, and Linux container.
- Container smoke test.

Check derivations are non-mutating and network-independent after fixed inputs are
available. Large corpus/model/hardware jobs are separate scheduled workflows,
not ordinary `nix flake check`.

The model-free normalization policy fixture is part of `nix flake check`.
`packages.<system>.nlp-model-integration` is a separate declared test derivation
whose locked closure includes `ja-ginza`; release and scheduled workflows build
it explicitly. It has no network access during execution and is not silently
skipped when an NLP cohort changes.

### Development shells

- `server`: Python/API development and fixture tests.
- `builder`: CPU corpus/NLP development.
- `frontend`: Node/npm/Svelte development.
- `default`: convenience union for contributors.

Shell entry sets tools/environment and prints concise help only. It does not
install Python versions, pin files, run dependency synchronization, activate a
checkout-owned virtualenv, update locks, build assets, or start processes.
Explicit documented commands perform setup and development servers.

## Server Package Contract

The server package contains:

- Natsume Python server code.
- FastAPI/Uvicorn, Pydantic, DuckDB and required runtime libraries.
- Static frontend output from `frontend`.
- CA certificates if outbound TLS remains necessary; otherwise omit them.
- An executable whose defaults bind safely and whose production host/port and
  artifact directory are explicit environment/CLI configuration.

It does not contain:

- Node/npm, compilers, notebooks, datasets, Polars builder tools, spaCy, GiNZA,
  wtpsplit, Torch, CUDA, or ROCm.
- A writable application data directory.
- The production corpus artifact.

The package can run directly with `nix run .#serve` against a read-only artifact
directory and passes the same smoke requests as the image.

## OCI Image Contract

Use a flake-native builder from the pinned Nix package set first. Add an external
image-builder input only if measured output size/layering/copy performance
requires it.

The initial image:

- Is x86_64 Linux.
- Contains the `server` closure and minimal `/etc` runtime identity files.
- Runs as a fixed non-root numeric UID/GID with no login shell.
- Uses the server executable as entrypoint.
- Exposes/document port 8000 while allowing explicit override.
- Reads the artifact directory from a documented read-only mount such as
  `/var/lib/natsume/artifact`.
- Has no declared writable volume; temporary filesystem needs are explicit and
  compatible with a read-only root filesystem.
- Includes OCI labels for application revision, supported schema version, source
  repository, and creation provenance where reproducibly available.
- Has no healthcheck shell dependency; deployment probes call the HTTP readiness
  endpoint.

The image is a packaging of `server`, not another deployment implementation.
Its smoke test compares direct-package and container behavior for readiness,
static page, and a representative search against the same fixture artifact.

## Public Edge Prerequisite

The server package and image are not exposed directly to the public internet.
The operator terminates public traffic at a reverse proxy or equivalent edge
that, for `/api` routes, starts with these conservative limits:

- 2 requests/second per source IP with a burst of 5;
- at most 4 concurrent API requests per source IP; and
- a 3-second upstream response timeout, longer than the application's 2-second
  DuckDB deadline.

The edge rejects excess work with `429` and does not forward unbounded request
bodies or headers. Health probes use a separately bounded internal path. These
values may be loosened only after the Spec 2 capacity benchmark demonstrates
headroom and the accepted-risk entry in the index is reviewed. The application
additionally owns its global 16-query admission bound; the edge is not its only
overload protection.

## Artifact Mount and Startup

Required configuration identifies an artifact directory containing both
`manifest.json` and `corpus.duckdb`. The mount is read-only. Startup fails and
readiness remains false for:

- Missing mount/directory/files.
- Manifest/database checksum or artifact-instance-ID mismatch.
- Unsupported schema.
- Unreadable DuckDB file.

Database refresh changes the host-side version pointer and restarts the
single-instance service. The running process does not watch or reopen an artifact
in place.

## CI and Cache Behavior

CI targets explicit outputs:

```text
nix flake check
nix build .#frontend
nix build .#server
nix build .#corpus-builder-cpu
nix build .#container
```

It may use a trusted binary cache pinned/configured by workflow policy. Flake
inputs and package sources remain locked; floating GitHub action branches are
replaced by pinned revisions where practical. CI permissions are least-privilege;
OIDC write is retained only when the selected cache actually requires it.

Checks run on x86_64 Linux as mandatory. Flake evaluation covers all declared
systems. Native builds for macOS/aarch64 are added only where runners and Python
package support exist; unsupported outputs are not declared as if verified.

## Developer Commands

The README documents:

- Entering each focused shell.
- Explicit dependency-lock refresh commands.
- Running backend/frontend development servers.
- Building and validating a fixture corpus artifact.
- Running `nix flake check` and explicit packages.
- Running the explicit `nlp-model-integration` release gate.
- Loading/running the OCI image with a read-only fixture/production artifact
  mount.
- Formatting separately from checking.

The generated help output derives from real apps/packages rather than maintaining
a second hand-written list of mutable wrappers.

## Superseding Existing Machinery

- Spec 1 has already removed runtime setup calls from check paths, stopped the
  dev-shell `shellHook` from mutating/activating `.venv`, and exposed minimal
  checks. This spec removes the remaining setup behavior from runtime/build
  paths and replaces command wrappers with real package/check derivations.
- Process-compose may remain a development-only coordinator that launches the
  already available server/frontend development commands; it does not define
  production ownership.
- The untracked `container.nix` is either replaced by the accepted package module
  or rewritten to expose only the concrete `container` output. Placeholder
  `services.foo` and commented speculative service-flake layers are removed.
- `.devcontainer` remains a development entry point into the flake and does not
  become a production image definition.

## Test Strategy

- Flake output evaluation on all declared systems.
- Repeated server/frontend builds from unchanged locks produce the same Nix
  output path.
- Closure inspection asserts forbidden builder/notebook/GPU packages are absent
  from `server` and `container` references.
- Dev-shell smoke checks prove entry does not change tracked files or start
  package-manager synchronization.
- Direct server smoke: liveness, readiness, static page, one query.
- Container smoke: non-root UID, read-only root filesystem, read-only artifact
  mount, same HTTP assertions, clean termination.
- Missing/incompatible artifact tests fail startup/readiness as specified.
- Build logs are scanned/isolated so derivations cannot fetch undeclared npm,
  PyPI, dataset, or model inputs during build.

## Acceptance Criteria

- All declared packages and checks evaluate; mandatory x86_64 Linux outputs build.
- `nix flake check` is green and non-mutating.
- Frontend/server/builder packages run without `npm install` or `uv sync`.
- Server/container closures exclude all forbidden dependency classes.
- The container runs non-root with a read-only root and artifact mount and passes
  HTTP smoke tests.
- Deployment documentation/configuration proves the public endpoint is behind
  the stated edge limits; a direct unbounded public bind is not a supported
  production topology.
- Direct server and image expose identical application behavior for the fixture.
- Entering a development shell causes no checkout mutation or automatic setup.
- README/help accurately describe the real flake interface.

## Rollback

Application rollback selects a prior locked flake revision/package. Database
rollback independently selects a prior artifact pointer compatible with that
server. Container rollback uses the image produced from the prior server
revision. Because image and direct package share one server derivation, there is
no separate Dockerfile rollback path.

## Decision Log

| Decision                                            | Status   | Reason                                                                        | Revisit trigger                                                                   |
| --------------------------------------------------- | -------- | ----------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Nix is authoritative                                | Accepted | Owner-selected production interface and reproducible closures                 | Target platform cannot consume Nix artifacts                                      |
| OCI derives from server                             | Accepted | One runtime definition and behavior                                           | Measured image constraints require specialized layering                           |
| Database is a mount, not default image content      | Accepted | Data refresh independent of app build                                         | Deployment strongly prefers self-contained immutable images                       |
| Focused non-mutating shells                         | Accepted | Keeps dependency ownership clear and entry predictable                        | Contributor evidence shows union shell is sufficient                              |
| x86_64 Linux image initially                        | Accepted | Current public-host target                                                    | Deployment requires another architecture                                          |
| Require bounded public edge                         | Accepted | Anonymous expensive queries need overload protection even on one instance     | Authentication, multi-instance limiting, or measured capacity changes the control |
| Keep large-model integration outside default checks | Accepted | Default checks remain small while release evidence owns the real pinned model | Model closure becomes appropriate for every check                                 |
