# Changelog

All notable changes to `juniper-data-client` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`create_dataset`'s `persist` parameter and everything after it are now keyword-only**, on both
  `JuniperDataClient` and `FakeDataClient` (defect-register `APD-DCLIENT-008`). Nine
  positional-or-keyword parameters made `create_dataset("spiral", p, False)` legal and unreadable,
  and any future signature reordering would have silently rebound arguments at every call site.
  Only `generator` / `params` — the universal pair — remain positional. **Breaking only for
  positional calls beyond the second argument**: an ecosystem-wide AST census at change time
  (223 calls across the 7 consuming repos) found zero such calls — every caller already passes
  `persist` onward by keyword. A new signature-pin test holds the real and fake clients to the
  identical convention so neither can drift.

### Fixed

- **A hostless `base_url` now fails at construction with `JuniperDataConfigurationError`** (defect-register `APD-DCLIENT-004`). `_normalize_url` normalised the scheme, trailing slash and `/v1` suffix but never checked that a host survived the parse, so an empty string, a bare `http://`, or a path-only value produced a broken hostless URL that failed opaquely on the *first request* deep inside `requests`. The guard is the one juniper-recurrence-client already carries (the reference implementation in the register's §2.3 sibling table): a missing host after scheme-defaulting raises the typed configuration error naming the offending value. Valid forms — schemeless hosts, trailing slashes, `/v1` suffixes, surrounding whitespace — are untouched, pinned by the pre-existing normalization tests. **Hardened after a confirmed review finding on the cascor-client port of this same guard**: scheme matching is **case-insensitive** (RFC 3986 — a case-sensitive check re-prefixed `HTTPS://host` into `http://HTTPS://host`, a silent TLS downgrade sending the API key over HTTP to hostname `https`), and the guard reads `parsed.hostname` rather than `netloc` (netloc accepts a userinfo-only `http://user:secret@` as truthy; hostname is `None` for it).
- **Exceptions now carry `status_code`, `detail` and `response`** (defect-register `APD-DCLIENT-001` / `APD-DCLIENT-003`). Every exception in the hierarchy previously subclassed `Exception` with a bare `pass`, so a 400 and a 422 raised the *same type with the same text* and the only way to tell them apart was substring-matching the message. The base `JuniperDataClientError.__init__` now accepts keyword-only `status_code` / `detail` / `response`, and `_request` populates them on every response-derived branch (404, 400/422, and the generic fallback). **Backward compatible**: the new parameters are keyword-only, so the single-positional-message construction used everywhere — including this package's own `FakeDataClient` — is unchanged, and locally raised errors (configuration, connection, timeout, retry-exhausted) simply report `status_code=None`.
- **A FastAPI 422 `detail` list is no longer flattened into an unparseable repr** (`APD-DCLIENT-003`). FastAPI answers a validation failure with a *list* of error objects; that list was interpolated straight into an f-string, producing `Validation error: [{'type': 'missing', 'loc': [...], ...}]` — text a caller could not parse and a human could not read. The structure is now attached to `exc.detail` **unmodified**, while the message renders it as `body.seed: Field required; body.n_spirals: Input should be a valid integer` via a new `_render_error_detail` helper. Non-list details and unexpected shapes pass through `str` unchanged.
- **`FakeDataClient` mirrors the real status codes.** The fake is documented as a drop-in replacement, so a double that raised the right type with `status_code=None` would let a consumer's test pass against behaviour production does not have. It now reports 404 for missing resources, 400 for an unknown generator (juniper-data raises an explicit `HTTPException(400)`), and 422 for a `ttl_seconds` violation (a pydantic `Field(ge=1)`, which FastAPI answers 422).
- **Exception context survives `pickle` and `copy`.** `BaseException.__reduce__` returns `(cls, args, self.__dict__)` whenever the instance dict is non-empty, so the keyword-only context is restored automatically — but only while `args` holds exactly the constructor's positional message, which is the invariant `test_context_survives_pickle_and_copy` pins (the failure mode flake8-bugbear's `B042` warns about). An interim `__reduce__` override that reproduced this default byte-for-byte was removed once the same override was mutation-proven redundant in `juniper-service-core` (its stated rationale — that the default rebuilds from `args` alone — was wrong there and here); exceptions cross process boundaries here whenever a worker returns a failure to its parent.
- **`validate_npz_contract` violations now raise `JuniperDataContractError`** (defect-register `APD-DCLIENT-002`). All **nine** raise sites in `contract.py` — the register row anchors only the first — threw bare `ValueError`, so the one error class this package detects *itself* was the one escaping the hierarchy `APD-DCLIENT-001` established: `except JuniperDataClientError` did not cover the client's own contract gate. The new class subclasses **both** `JuniperDataClientError` and `ValueError`, so the validator's original `Raises: ValueError` contract still holds — juniper-recurrence's routers, which catch `(JuniperDataClientError, ValueError)` and re-document the `ValueError`, are unaffected — and every message is byte-identical. Contract violations are local (detected after download), so `status_code` / `detail` / `response` stay `None` per the base-class convention. The existing `pytest.raises(ValueError)` tests are deliberately unchanged as the back-compat pin; a new arm per raise site asserts the concrete type, adding coverage for the `dt`-shape site that previously had none.
- **All nine file-header `Version:` fields now agree with the package version, and a test keeps them agreeing** (defect-register `APD-DCLIENT-005`). The register filed six drifted decorative headers with three distinct values; re-derivation found **nine with four values** — `tests/conftest.py`, `tests/test_fake_client.py` and `tests/test_fake_client_batch.py` sat at `0.3.1`, below the register's own floor. All nine now read `0.4.2`, and `tests/test_file_header_versions.py` scans every `*.py` under the package and `tests/` and fails naming any header that disagrees with `__version__` — with an anti-vacuous floor so a broken scan cannot pass by matching nothing. Deliberate consequence: a release version bump now requires touching the headers, converting silent drift into a loud diff at bump time.

### Added

- **7 missing generator-name constants + live-registry parity cross-check (W-9).** `constants.py` gains `GENERATOR_EQUITIES`, `GENERATOR_EQUITIES_SEQ`, `GENERATOR_MULTI_SINE`, `GENERATOR_MACKEY_GLASS`, `GENERATOR_AR_P`, `GENERATOR_IRREGULAR_SINE`, and `GENERATOR_DELAY_PRODUCT` (each with a paired `GENERATOR_DESCRIPTION_*`), covering the equities pair and the five WS-1 sequence generators the client lacked since juniper-data 0.8/0.9. Critically, `tests/test_generator_parity.py`'s `EXPECTED_SERVER_GENERATORS` — previously a stale hand-kept mirror that let the reverse parity assertion pass vacuously — is now cross-checked against the **live** `juniper_data.api.routes.generators.GENERATOR_REGISTRY` whenever juniper-data is importable (skips when it is not), so the mirror can no longer drift silently. CLI experimentation plan §11 register item W-9 (juniper-ml `notes/JUNIPER_2026-07-29_JUNIPER-ECOSYSTEM_CASCOR-RECURRENCE-CLI-TEST-VALIDATION-EXPERIMENTATION-PLAN.md`).
- **Blocking per-file coverage gate — Phase C / C-2 of the ecosystem per-file-coverage rollout.** The `unit-tests` CI job (a required check) now emits `coverage.json` and runs `juniper-coverage-gap-map --coverage-json coverage.json --enforce` (from `juniper-ci-tools>=0.6.0,<0.7.0`), failing the build when any source file drops below 90% statement coverage or any packaged sub-module below 95% pooled coverage. To clear the bar, `client.py` was lifted from 89.56% to 100% with 8 targeted `responses`-based tests in `tests/test_client_coverage_gaps.py` covering the previously-uncovered paths: the `_resolve_api_key_from_env` `*_FILE` failure branches (unreadable file, empty secret), the `_request` non-JSON error-body fallback, the `wait_for_ready` polling loop (ready + timeout), and the optional-argument branches of `create_dataset` / `create_spiral_dataset` / `batch_update_tags`. See juniper-ml `notes/JUNIPER_ECOSYSTEM_PER_FILE_COVERAGE_ROLLOUT_SCOPING_2026-06-30.md`.

### Changed

- **Removed the redundant `pass` statements from the exception subclasses** (defect-register `APD-DCLIENT-007`). The register filed six; `APD-DCLIENT-001`'s fix had already given the base class a real body, so five remained — a docstring is a complete class body on its own, and the trailing `pass` was dead weight in every subclass. No behavioural change; the whole test suite passes untouched.
- **`validate_npz_contract` returns `ContractKind`, a `Literal["tabular", "sequence"]`, instead of bare `str`** (defect-register `APD-DCLIENT-006`). A caller can now exhaustively match on the result and a type checker will flag a misspelled comparison; the constants `CONTRACT_KIND_TABULAR` / `CONTRACT_KIND_SEQUENCE` are annotated `Final[ContractKind]` so their values and the Literal's members cannot drift apart — a mismatch fails type checking. The alias is exported from the package root for consumer annotations. Source-compatible: `Literal[...]` values still assign to `str`.

## [0.4.2] - 2026-06-17

### Added

- **`validate_npz_contract` NPZ data-contract validator** — public helper (`from juniper_data_client import validate_npz_contract`) that classifies an artifact's array bundle as `"tabular"` (2-D `X`) or `"sequence"` (3-D `X` with the WS-1 irregular-Δt keys) and validates the contract invariants (`dt >= 0`, `dt[:, 0] == 0`, mask/shape consistency) with a configurable `dt_atol`. Lets consumers (e.g. the juniper-recurrence app) gate 3-D Δt artifacts up front instead of relying on model-side shape checks. Shipped in the source tree since the WS-1 data foundation but absent from the published 0.4.1 wheel; 12 unit tests in `tests/test_contract.py`.
- **`JUNIPER_DATA_API_KEY_FILE` Docker-secret indirection** (defense-in-depth follow-up to cascor#331): `JuniperDataClient` now resolves its API key from a `JUNIPER_DATA_API_KEY_FILE` env var (a path to a file whose stripped contents are the key, e.g. `/run/secrets/juniper_data_api_keys`) before falling back to the plain `JUNIPER_DATA_API_KEY` env var. An explicit `api_key=` constructor argument still wins over both. This mirrors how the Juniper services resolve their own secrets (cascor `api.secrets.get_secret`) and means a consumer that mounts the key as a Docker secret — and sets only the `_FILE` form — authenticates without an extra wrapper. New `API_KEY_FILE_ENV_VAR` constant and a module-private `_resolve_api_key_from_env()` helper; 3 unit tests pin file-resolution, `_FILE`-over-plain-env precedence, and `api_key=`-over-`_FILE` precedence, plus the existing no-key test hardened to clear both vars.
- **CI lints now run via the `juniper-ci-tools` PyPI package** — the AGENTS.md version-drift lint (`juniper-lint-agents-md-version`) and the workflow script-path lint (`juniper-lint-workflow-paths`) run from the shared `juniper-ci-tools>=0.2.0` console scripts; the former inline `util/test_agents_md_version_drift.py` / `util/test_workflow_script_paths.py` copies were removed. Includes the one-line `AGENTS.md` bump (0.3.2 → 0.4.1) that cleared the drift the version lint surfaces.
- **AGENTS.md header-schema lint + auto-bump workflow** — adopts `juniper-ci-tools` v0.4.0's AGENTS.md header-schema lint (pins the canonical six-field header) plus the companion `agents-md-touch-up` workflow that auto-refreshes the `Last Updated` field on PRs touching `AGENTS.md`.

### Changed

- **README compatibility matrix sync** — refreshed the README compatibility matrix for the cascor / canopy `0.5.0` and cascor-worker `0.4.0` releases.

## [0.4.1] - 2026-05-02

**Summary**: METRICS-MON R4.3 + R4.6 ship the optional `on_request` instrumentation hook and outbound `X-Request-ID` propagation. Also rolls in the previously-accumulated `[Unreleased]` work (constants module, DC-01/03 / XREPO-01/01c/09/11, retry-policy change, GENERATOR_CIRCLE alias).

### Added

- **METRICS-MON R4.3 / seed-13**: optional `on_request(method, url, status, duration_ms, error)` instrumentation hook on `JuniperDataClient.__init__`. Default is a no-op so standalone use pays nothing; consumers (canopy, cascor) supply a closure that emits to their preferred surface (Prometheus, OpenTelemetry, structured logs). Hook fires once per HTTP call on every outcome — success, transport error, 404, 422, 500, timeout. Hook exceptions are caught and logged at WARNING so instrumentation never crashes a production HTTP path. New `RequestHook` type alias exported from the package's public surface so consumers can type their hook closures (mypy in CI catches drift). 11 unit tests pin all 6 outcome paths plus resilience.
- **METRICS-MON R4.6 / R3.6 sweep follow-up**: outbound `X-Request-ID` propagation in `_request()`. Reads `juniper_observability.request_id_var` at call time; when the calling thread has a non-empty value, copies it into the outbound HTTP header so juniper-data can correlate the inbound request back to the caller's chain (canopy/cascor → data-client → data). Best-effort: `ImportError` (lib not installed) and `LookupError` (ContextVar unset) silently no-op. Caller-supplied `X-Request-ID` headers always win. New `[observability]` extra (`pip install juniper-data-client[observability]`) for standalone users; canopy/cascor get propagation transparently via their existing `juniper-observability` deps. 4 unit tests pin set / unset / caller-wins / standalone paths.
- New `juniper_data_client/constants.py` module centralizing every previously inline literal: `API_KEY_*` and `API_VERSION_*` wire-protocol identifiers, the full set of `ENDPOINT_*` paths (including f-string templates for parameterized routes), `DEFAULT_*` constructor defaults, `RETRY_*` tuning, and per-generator parameter defaults (spiral, xor, circles, gaussian, checkerboard) used by `testing/generators.py`.
- **DC-03 / XREPO-01c**: constants for the five server-side generators the client previously lacked -- `GENERATOR_GAUSSIAN`, `GENERATOR_CHECKERBOARD`, `GENERATOR_CSV_IMPORT`, `GENERATOR_MNIST`, `GENERATOR_ARC_AGI` -- with matching `GENERATOR_DESCRIPTION_*` entries. Downstream code should now import these instead of hardcoding string literals.
- `tests/test_generator_parity.py`: parity suite that prevents future drift between client generator constants and the server `GENERATOR_REGISTRY`, and exercises the legacy `"circle"` -> `"circles"` alias through the fake client.
- **XREPO-09 (Phase 4B)**: `create_dataset()` on both `JuniperDataClient` and `FakeDataClient` now accepts `tags: Optional[List[str]]` and `ttl_seconds: Optional[int]`. Both are forwarded to the server's `CreateDatasetRequest` (the client previously dropped them even though the server has accepted them since juniper-data v0.6.0). The fake mirrors the server's `ge=1` Pydantic bound on `ttl_seconds`.
- `tests/test_create_dataset_tags_ttl.py`: regression suite covering POST-body shape (via mocked `_request`), fake-client metadata round-trip, validation of non-positive TTL, and JSON serializability.
- `tests/test_retry_policy.py`: new suite guarding `RETRY_ALLOWED_METHODS` and `RETRYABLE_STATUS_CODES` against regression; asserts the `Retry` adapter mounted on the session reflects these constants end-to-end.

### Changed

- `client.py`, `testing/fake_client.py`, and `testing/generators.py` now import from `juniper_data_client.constants` instead of embedding inline literals (~87 replacements total).
- `API_KEY_HEADER_NAME` and `API_VERSION_PATH_SUFFIX` are bit-identical to the corresponding values exposed by the `juniper-data` server, eliminating literal duplication across the client/server boundary.
- `AGENTS.md` gained a new "Constants" section documenting the categories, server alignment, and contribution rules for the constants module.
- **DC-01 / XREPO-01 (BREAKING, with deprecation alias)**: `GENERATOR_CIRCLE` now resolves to `"circles"` to match the server registry key; the previous value `"circle"` was silently rejected by the server with HTTP 400. Callers passing the legacy string to `FakeDataClient.create_dataset()` or `get_generator_schema()` are transparently routed to the new name and emit a `DeprecationWarning`. A new `GENERATOR_CIRCLE_LEGACY` constant exposes the old value for one release cycle.
- Existing fake-client tests updated to use the canonical `"circles"` name; a dedicated legacy-alias regression lives in `tests/test_generator_parity.py`.
- **XREPO-11 (Phase 4B, BEHAVIOR CHANGE)**: `RETRY_ALLOWED_METHODS` is now `["HEAD", "GET", "PUT"]`. POST, PATCH, and DELETE were previously included, which could cause duplicate dataset creation (on POST) or repeated side-effects (on DELETE) when a transient 5xx retried a request that had already been applied server-side. Callers that need retry for mutations must layer their own idempotency (e.g., use client-supplied dataset names so POST collapses via the existing server-side dedupe path).

### Deprecated

- The legacy generator name `"circle"` (and the `GENERATOR_CIRCLE_LEGACY` constant). Both will be removed after the next release; migrate callers to `GENERATOR_CIRCLE` / `"circles"` now.

### Notes

- No public method signatures change; only the value of `GENERATOR_CIRCLE` and the set of available generator constants.
- Server counterpart (`juniper-data`) is gaining a `MoonGenerator` to match `GENERATOR_MOON` in the same release cycle (XREPO-01b / DC-02).

## [0.4.0] - 2026-04-08

**Summary**: New public API surface -- batch operations, dataset versioning, and extended create_dataset parameters. Includes performance benchmarks and CI dependency updates.

### Added: [0.4.0]

- Batch operation client methods: `batch_delete`, `batch_create`, `batch_update_tags`, `batch_export` on both `JuniperDataClient` and `FakeDataClient` (CAN-DEF-006)
- Dataset versioning methods: `list_versions(name)` and `get_latest(name)` for named dataset version management (CAN-DEF-005 Phase 2)
- Extended `create_dataset()` with optional `name`, `description`, `created_by`, and `parent_dataset_id` parameters for versioning support
- `FakeDataClient` versioning support with auto-incrementing version counters
- Performance benchmark test suite: 14 FakeDataClient benchmarks (always run) + 9 live service benchmarks gated behind `JUNIPER_DATA_BENCHMARK=1` (CAN-DEF-007)
- `util/run_all_tests.bash` test runner script
- Developer cheatsheet documentation (`docs/DEVELOPER_CHEATSHEET.md`)

### Changed: [0.4.0]

- Bumped `github/codeql-action` from 3.28.0 to 4.35.1 (Dependabot)
- Bumped `actions/setup-python` from 5.6.0 to 6.2.0 (Dependabot)
- Bumped `actions/checkout` from 4.2.2 to 6.0.2 (Dependabot)
- Bumped `actions/upload-artifact` from 4.6.0 to 7.0.0 (Dependabot)
- Bumped `actions/cache` from 4.2.3 to 5.0.4 (Dependabot)
- Comprehensive AGENTS.md audit and update to reflect current codebase
- Propagated V2 worktree cleanup procedure (fixes CWD-trap bug)
- Added markdownlint and pre-commit configuration files
- Updated documentation overview with ecosystem links and index fixes

### Fixed: [0.4.0]

- Aligned test assertions with actual service responses
- Fixed test failures in test suite
- Fixed markdown linting issues across documentation files

### Technical Notes: [0.4.0]

- **SemVer impact**: MINOR -- 6 new public API methods (batch_delete, batch_create, batch_update_tags, batch_export, list_versions, get_latest) plus 4 new create_dataset parameters
- **Test count**: 88+ passed (expanded with 17 versioning tests, batch operation tests, and 14+ performance benchmarks)

## [0.3.2] - 2026-03-03

**Summary**: Security hardening — build attestations enabled and scheduled security scanning. Also includes previously unreleased CI/CD improvements.

### Security: [0.3.2]

- Enabled build attestations in publish workflow (`attestations: true`)

### Added: [0.3.2]

- `.github/workflows/security-scan.yml` — Weekly scheduled security scanning (Bandit, pip-audit)
- Dependabot configuration for automated dependency updates
- CODEOWNERS file for PR review routing
- This CHANGELOG

### Changed: [0.3.2]

- Hardened CI pipeline: added security scans (Bandit, pip-audit), build verification, quality gate, Python 3.12/3.13/3.14 matrix
- SHA-pinned all GitHub Actions to immutable commit hashes

### Technical Notes: [0.3.2]

- **SemVer impact**: PATCH — CI/CD and supply chain improvements only; no API changes
- **Test count**: 88 passed, 0 failed
- **Part of**: Cross-ecosystem security audit (7 repos, 24 findings)

## [0.3.1] - 2026-02-23

### Changed

- Bumped Python requirement to `>=3.12` (dropped 3.11)
- Added ecosystem compatibility matrix to README
- Added documentation link validation and dependency docs generation to CI
- Added worktree setup/cleanup procedures

### Fixed

- Enabled verbose logging and disabled attestations in publish workflow

## [0.3.0] - 2026-02-18

### Added

- Initial release of `juniper-data-client`
- `JuniperDataClient` class with full JuniperData API coverage
- Health check, dataset creation, artifact download methods
- `JuniperDataError` exception hierarchy
- Type annotations with `py.typed` marker
- Unit test suite with 80%+ coverage
- CI/CD pipeline with GitHub Actions
- PyPI and TestPyPI trusted publishing
- README with API documentation and examples

[Unreleased]: https://github.com/pcalnon/juniper-data-client/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/pcalnon/juniper-data-client/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/pcalnon/juniper-data-client/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/pcalnon/juniper-data-client/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/pcalnon/juniper-data-client/releases/tag/v0.3.0
