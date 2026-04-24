# Changelog

All notable changes to `juniper-data-client` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
