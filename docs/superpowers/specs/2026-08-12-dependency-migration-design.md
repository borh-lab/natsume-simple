# Major-Version Dependency Migration Design

**Status:** Draft for written review

**Date:** 2026-08-12
**Boundary:** Direct dependency ownership, compatibility cohorts, lockfiles,
security policy, and accelerator support evidence

## Problem

Python, npm, and Nix inputs have aged together without a complete behavioral
gate. The installed npm graph reports high/critical advisories, and major
updates are available across Vite, Vitest, Tailwind, ESLint and related plugins.
The Python graph spans serving, corpus acquisition, NLP, notebooks, and three
accelerator variants in one project environment. A global update would make it
impossible to attribute failures and could silently change Japanese extraction
behavior.

The repository already contains uncommitted owner work updating Nix inputs and
moving accelerator packages toward Torch 2.6. This spec does not discard or
overwrite that work; implementation first reconciles it with the accepted
compatibility matrix.

## Goals

- Upgrade to current supported stable majors where ecosystem compatibility is
  demonstrated.
- Remove dependencies without a present consumer before upgrading them.
- Separate server, builder, accelerator, test, and notebook closures.
- Group upgrades by compatibility and behavioral blast radius.
- Make every supported CPU/CUDA/ROCm claim evidence-backed.
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

| Set | Present consumers | Excluded from |
|---|---|---|
| `server` | FastAPI app and DuckDB queries | Builder/NLP/notebook-only packages |
| `builder` | Acquisition, segmentation, extraction, artifact validation | FastAPI serving extras |
| `accelerator-cuda` | Optional CUDA builder | CPU/server/ROCm |
| `accelerator-rocm` | Optional ROCm builder | CPU/server/CUDA |
| `test` | Static analysis and automated tests | Production closures |
| `notebook` | Interactive research | CI and production closures |

`python-fasthtml` is removed because no code consumes it. Deprecated
`tool.uv.dev-dependencies` moves to `[dependency-groups]`. The builder's CPU
requirements explicitly own any inference dependency that is currently
accidentally present only through accelerator extras.

Frontend removals before upgrades:

- `@sveltejs/adapter-auto`, because configuration uses `adapter-static`.
- Lodash, replacing its only debounce use with a small local lifecycle-aware
  helper or controller timing owned by Spec 4.
- Any duplicate formatter/linter whose file ownership is superseded by the
  selected check stack.

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
Node version must fall in the upstream-supported range. Migration follows the
official Vite guide rather than relying only on a successful bundle:
<https://vite.dev/guide/migration.html>.

TypeScript is pinned to the newest stable release explicitly supported by the
selected SvelteKit peer range. At design time, that means 6.x rather than
TypeScript 7. The exact version is recorded when implementation resolves peers.

### Cohort 3: frontend lint and format majors

Upgrade ESLint 10, eslint-plugin-svelte 3, globals 17, and
prettier-plugin-svelte 4 as a configuration cohort. Rule/config changes are
reviewed. Existing violations are fixed or consciously documented; broad rule
disabling is not an upgrade success condition.

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

Acceptance requires:

- All source adapters load pinned local fixtures without remote code.
- Sentence segmentation fixtures retain intended boundaries or record an
  explicit approved behavior change.
- Japanese normalization/extraction fixtures match the domain contract.
- Fixture artifact relational contents and API responses reconcile.
- Any intentional NLP behavior change is reviewed as a semantic patch before
  updating golden results.

### Cohort 7: accelerator matrix

Select a PyTorch 2.x version whose official indices provide supported artifacts
for:

- CPU on required host platforms.
- One explicitly named CUDA runtime on supported Linux systems.
- One explicitly named ROCm runtime on supported Linux systems.

CUDA and ROCm are mutually exclusive extras. Index URLs, exact Torch build
versions, companion packages, environment markers, and supported architectures
are locked. CPU is the mandatory full CI baseline. CUDA and ROCm each receive:

- Resolver/lock validation.
- Nix derivation evaluation/build where hardware is not required.
- Import and device-discovery smoke test on scheduled matching hardware.
- A tiny extraction fixture on scheduled matching hardware before release.

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
- Relevant Spec 2–4 integration/browser contracts.
- `nix flake check` and affected explicit package builds.
- Direct dependency/peer/engine inspection.
- Security audit comparison.
- Diff review for generated lock and build artifacts.

The data/NLP and accelerator cohorts additionally run their specialized fixture
and scheduled hardware gates. The next cohort does not begin until the current
one is green and committed.

## Acceptance Criteria

- Every retained direct dependency has a named current consumer.
- Server closure excludes builder, notebook, Torch, CUDA, and ROCm dependencies.
- Notebook closure is absent from ordinary CI and production builds.
- Frontend uses supported Node/Vite/Svelte/TypeScript peer combinations.
- Data/NLP fixture behavior is reviewed rather than blindly re-recorded.
- CPU/CUDA/ROCm support statements match resolver, build, and scheduled hardware
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

| Decision | Status | Reason | Revisit trigger |
|---|---|---|---|
| Upgrade in seven cohorts | Accepted | Isolates compatible ecosystems and behavioral risk | A cohort proves internally too broad |
| Prune before upgrading | Accepted | No value in modernizing unused machinery | A removed dependency gains a present consumer |
| Hold TypeScript to supported peer range | Accepted | Latest unsupported is not modernization | SvelteKit supports the next major |
| Treat NLP goldens as domain behavior | Accepted | Model changes can silently alter corpus facts | Domain owner approves a changed policy |
| CPU full CI; scheduled GPU evidence | Accepted | Practical baseline without pretending GPU support | Hosted matching GPU CI becomes economical |
