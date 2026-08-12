# Major-Version Dependency Migration Design

**Status:** Revised draft after written review

**Date:** 2026-08-12
**Boundary:** Direct dependency ownership, compatibility cohorts, lockfiles,
security policy, and accelerator support evidence

## Problem

Python, npm, and Nix inputs have aged together without a complete behavioral
gate. The installed npm graph reports high/critical advisories, and major
updates are available across Vite, Vitest, Tailwind, ESLint and related plugins.
The Python graph spans serving, corpus acquisition, NLP, notebooks, CPU, and
incompatible CUDA/ROCm variants in one project environment. A global update would make it
impossible to attribute failures and could silently change Japanese extraction
behavior.

The repository already contains uncommitted owner work updating Nix inputs and
moving accelerator packages toward Torch 2.6. The owner has rejected the ROCm
path after review; implementation removes only ROCm-specific files, dependency
hunks, and lock entries while preserving unrelated in-flight updates.

## Goals

- Upgrade to current supported stable majors where ecosystem compatibility is
  demonstrated.
- Remove dependencies without a present consumer before upgrading them.
- Separate server, builder, accelerator, test, and notebook closures.
- Group upgrades by compatibility and behavioral blast radius.
- Make every supported CPU/CUDA claim evidence-backed.
- Enforce lockfile and high/critical advisory policy.

## Non-Goals

- Automatically select the numerically latest release when peer ranges or
  platform wheels do not support it.
- Replace working libraries solely because alternatives are newer.
- Mix feature work, architecture restructuring, formatting churn, and upgrade
  changes.
- Run a full production corpus build for every dependency PR.

## Dependency Ownership

Python dependency sets:

| Set                | Present consumers                                          | Excluded from                      |
| ------------------ | ---------------------------------------------------------- | ---------------------------------- |
| `server`           | FastAPI app and DuckDB queries                             | Builder/NLP/notebook-only packages |
| `builder`          | Acquisition, segmentation, extraction, artifact validation | FastAPI serving extras             |
| `accelerator-cuda` | CUDA additions for the optional accelerated builder        | Server and default CPU closure     |
| `test`             | Static analysis and automated tests                        | Production closures                |
| `notebook`         | Interactive research                                       | CI and production closures         |

`python-fasthtml` is removed because no code consumes it. Deprecated
`tool.uv.dev-dependencies` moves to `[dependency-groups]`. The builder's CPU
requirements explicitly own any inference dependency that is currently
accidentally present only through accelerator extras.

Frontend removals before upgrades:

- `@sveltejs/adapter-auto`, because configuration uses `adapter-static`.
- Lodash, replacing its only debounce use in Cohort 1 with a minimal local
  lifecycle-aware helper; Spec 5 may later move timing into the controller.
- Biome and `biome.json`, removed by Spec 1. Prettier owns formatting and ESLint
  owns Svelte-aware linting. Biome 2.5.5 is present in the Nix environment, but
  its [Svelte support remains experimental](https://biomejs.dev/internals/language-support/)
  and would duplicate file ownership rather than replace both tools here.

Every retained direct dependency is documented by at least one production,
build, or test consumer.

## Compatibility Cohorts

### Cohort 1: security updates within the current architecture

Update to the newest compatible releases in the current major lines for Svelte
5, SvelteKit 2, adapter-static 3, Playwright 1, PostCSS 8, and affected direct
tooling. Regenerate the npm lock and prove the existing full gate before major
build-tool migration.

### Cohort 2: frontend build and unit-test majors

Upgrade together:

- Vite 8.
- `@sveltejs/vite-plugin-svelte` 7.
- Vitest 4.
- Latest Svelte 5 and SvelteKit 2 versions compatible with those tools.

Vite 8 uses Rolldown and requires Node 20.19+, 22.12+, or 24+; the Nix-selected
Node version must fall in the upstream-supported range. The Vite 5-to-8 and
Vitest 2-to-4 jumps cross multiple major boundaries. Implementation inventories
and applies the breaking changes for every intermediate major in order, using
each official migration/release guide; the latest guide alone is not treated as
cumulative evidence. A successful bundle without the full gate is insufficient.

TypeScript is pinned to the newest stable release explicitly supported by the
selected SvelteKit peer range. At design time, that means 6.x rather than
TypeScript 7. The exact version is recorded when implementation resolves peers.

Registry evidence on 2026-08-12 confirms Vite 8.2.1, Vitest 4.1.10,
`@sveltejs/vite-plugin-svelte` 7.3.0, ESLint 10.8.1, TypeScript 7.0.2,
eslint-plugin-svelte 3.22.0, globals 17.10.0, prettier-plugin-svelte 4.1.1,
Tailwind CSS 4.3.3, datasets 5.0.1, pytest 9.1.1, and DuckDB 1.5.5 are published.
The selected SvelteKit peer range supports TypeScript 5.3 or 6, and current
typescript-eslint requires TypeScript below 6.1, so TypeScript 7 is intentionally
deferred rather than assumed unavailable. Versions are re-resolved at cohort
implementation time.

### Cohort 3: frontend lint and format majors

Upgrade ESLint 10, eslint-plugin-svelte 3, globals 17, and
prettier-plugin-svelte 4 as a configuration cohort. Rule/config changes are
reviewed. Existing violations are fixed or consciously documented; broad rule
disabling is not an upgrade success condition.

ESLint 9-to-10, TypeScript 5-to-6, and each plugin major are likewise reviewed
against every crossed major's official migration notes and peer ranges.

### Cohort 4: Tailwind CSS 4

Tailwind 4 migrates separately because its CSS entry/configuration and Vite or
PostCSS integration differ from Tailwind 3. Targeted layout behavior and browser
screenshots/assertions land before the upgrade. Unused configuration is removed
only after the new generated CSS proves it unnecessary.

### Cohort 5: Python serving stack

Upgrade FastAPI, Pydantic 2, Uvicorn, HTTPX test support, and DuckDB 1 to their
newest mutually compatible stable releases. Spec 2 contract, readiness,
connection-concurrency, error-envelope, response-bound, and query-plan tests run
before and after.

### Cohort 6: Python data and NLP stack

Upgrade datasets 3 to 5, NumPy 1 to 2, pytest 8 to 9, and supported current
releases of Polars, spaCy, GiNZA/ja-ginza, wtpsplit, and their direct runtime
requirements.

This cohort cannot begin until Spec 3 Gate 3A proves every planned public corpus
can be reacquired without dataset remote code and records an explicit exclusion
outcome for every inventoried remainder.
Datasets 4 removed dataset-script/remote-code loading; the target 5.x release is
therefore a deliberate adapter migration, not a compatible parameter update.
See the [official datasets releases](https://github.com/huggingface/datasets/releases).

Acceptance requires:

- All source adapters load pinned local fixtures without remote code.
- Sentence segmentation fixtures retain intended boundaries or record an
  explicit approved behavior change.
- Japanese normalization/extraction fixtures match the domain contract.
- Fixture artifact relational contents and API responses reconcile.
- Any intentional NLP behavior change is reviewed as a semantic patch before
  updating golden results.

### Cohort 7: CUDA accelerator profile

Select mutually compatible spaCy, Thinc, CuPy, transformer/PyTorch, and model
versions for:

- CPU on required host platforms.
- One explicitly named CUDA runtime on supported Linux systems.

spaCy documents CUDA GPU support through CuPy, and Thinc's documented backends
are NumPy, CuPy/CUDA, AppleOps, and MPS—not ROCm. A ROCm PyTorch wheel therefore
does not establish a supported spaCy/GiNZA extraction path. The ROCm extra,
lockfile entries, Nix output, devcontainer, and support claims are removed.
Apple acceleration is not advertised by this Linux-focused project; supported
Apple hosts use CPU unless a future owner supplies a separate end-to-end case.

- <https://spacy.io/usage>
- <https://thinc.ai/docs/api-backends>

CUDA index URLs, exact builds, companion packages, environment markers, and
supported architectures are locked. CPU is the mandatory full CI baseline. The
CUDA profile receives:

- Resolver/lock validation.
- Nix derivation evaluation/build where hardware is not required.
- Import and device-discovery smoke test on scheduled matching hardware.
- A tiny extraction fixture on scheduled matching hardware before release.
- Repeated ordered-relational comparison required by Spec 3 before that
  accelerator profile is allowed to publish a corpus.

An accelerator is not advertised as supported solely because CPU resolves.

## Version Policy

- Direct dependencies declare intentional compatible major ranges; lockfiles
  select exact artifacts.
- Pre-1.0 packages use a reviewed compatible minor range where SemVer does not
  provide a stable major contract.
- Python itself remains at the Nix-pinned supported version until builder/server
  packages and NLP wheels support a newer version as a separate cohort.
- No prerelease dependencies enter production unless a recorded decision names
  the missing stable capability, owner, expiry trigger, and rollback.
- The incumbent manifest range `@sveltejs/vite-plugin-svelte@^4.0.0-next.6` is
  explicitly transitional even though the lock resolved stable `4.0.0`; Cohort 1
  replaces the prerelease range with a stable constraint before other upgrades.
- Package engines and peer dependencies are treated as constraints, not warnings.

## Lockfile Contract

- Commit `uv.lock`, `natsume-frontend/package-lock.json`, and `flake.lock`.
- Generate them only through `uv`, `npm`, and Nix lock commands from their
  owning roots.
- Use `npm ci` for reproducible install/build checks.
- CI fails when a dependency manifest changes without the corresponding lock.
- A cohort commit contains its manifest/lock/config/code compatibility changes
  and nothing unrelated.
- Nix inputs are updated with targeted `nix flake lock --update-input` or an
  intentional full `nix flake update`, never manual JSON editing.

## Security Advisory Policy

`npm audit` (or the selected machine-readable equivalent) is enforced for high
and critical severities after the initial security cohort. A temporary exception
must record:

- Advisory identifier and affected package/path.
- Whether vulnerable functionality is present in build or runtime.
- Exposure analysis for this repository.
- Compensating control.
- Owner.
- Expiration date or dependency/event trigger.

Expired or ownerless exceptions fail CI. Lower severities remain visible in CI
but do not block unless their exposure warrants escalation. Python and Nix
advisory mechanisms are added only if their databases can be pinned and produce
actionable direct/transitive paths; an unreliable scanner is not treated as a
green badge.

## Automated Update Grouping

Dependabot/Renovate groups mirror the cohorts. Patch/minor updates may group
within a cohort. Major updates always receive a dedicated review. NLP model/data
updates never auto-merge. Accelerator changes never auto-merge. Lockfile-only
refreshes identify why resolution changed.

## Verification Per Cohort

Every cohort runs:

- Clean lock/install or Nix dependency materialization.
- All Spec 1 checks.
- Relevant Spec 2, 3, and 5 integration/browser contracts available at that
  cohort boundary.
- `nix flake check` and affected explicit package builds.
- Direct dependency/peer/engine inspection.
- Security audit comparison.
- Diff review for generated lock and build artifacts.
- Walkthrough boundary mapping and behavior; a dependency cohort cannot erase a
  taught seam merely because the total example collection remains green.

The data/NLP and accelerator cohorts additionally run their specialized fixture
and scheduled hardware gates. The next cohort does not begin until the current
one is green and committed.

### Execution order relative to capability specs

- Cohorts 1–4 run after the Spec 1 baseline and before frontend Spec 5, using
  Spec 2's generated wire contract once available. The component refactor is
  therefore implemented and verified once on the target frontend toolchain.
- Cohort 5 runs after Spec 2's serving characterization.
- Cohort 6 runs only after Spec 3 Gate 3A and fixture artifact contracts.
- Cohort 7 follows Cohort 6 and the CPU builder package.

Spec 1 uses narrow transitional types only; it does not perfect the wire and
component models that Specs 2 and 5 replace.

## Acceptance Criteria

- Every retained direct dependency has a named current consumer.
- Server closure excludes builder, notebook, Torch, and CUDA dependencies.
- Notebook closure is absent from ordinary CI and production builds.
- Frontend uses supported Node/Vite/Svelte/TypeScript peer combinations.
- Data/NLP fixture behavior is reviewed rather than blindly re-recorded.
- The executable backend walkthrough remains readable and every boundary-mapped
  behavior target is preserved across every cohort.
- CPU/CUDA support statements match resolver, build, and scheduled hardware
  evidence.
- Final npm graph has no unaccepted high/critical advisory.
- All three lockfiles are current, reproducible, and generated by owning tools.
- Every cohort has an independently green commit and rollback point.

## Rollback

Revert one cohort, including its manifest, lockfile, compatibility source, and
configuration changes. Do not partially restore an old lock against a new
manifest. Security cohort rollback is permitted only if it does not reintroduce
an exposed unmitigated advisory; otherwise roll forward with a focused fix.

## Decision Log

| Decision                                    | Status   | Reason                                                                                         | Revisit trigger                                                             |
| ------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Upgrade in seven cohorts                    | Accepted | Isolates compatible ecosystems and behavioral risk                                             | A cohort proves internally too broad                                        |
| Prune before upgrading                      | Accepted | No value in modernizing unused machinery                                                       | A removed dependency gains a present consumer                               |
| Hold TypeScript to supported peer range     | Accepted | Latest unsupported is not modernization                                                        | SvelteKit supports the next major                                           |
| Treat NLP goldens as domain behavior        | Accepted | Model changes can silently alter corpus facts                                                  | Domain owner approves a changed policy                                      |
| CPU full CI; scheduled GPU evidence         | Accepted | Practical baseline without pretending GPU support                                              | Hosted matching GPU CI becomes economical                                   |
| Remove ROCm support                         | Accepted | spaCy/Thinc expose no supported ROCm backend; a ROCm Torch wheel cannot prove GiNZA extraction | spaCy/Thinc document a supported ROCm backend and the full fixture passes   |
| Prettier and ESLint replace Biome           | Accepted | Clear ownership and mature Svelte-specific behavior                                            | Biome's Svelte support is stable and can replace both with equivalent rules |
| Frontend cohorts precede component refactor | Accepted | Avoids rebuilding the new component architecture on obsolete tooling                           | A cohort cannot pass without the refactor                                   |
