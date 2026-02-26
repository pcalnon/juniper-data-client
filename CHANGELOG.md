# Changelog

All notable changes to `juniper-data-client` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Hardened CI pipeline: added security scans (Bandit, pip-audit), build verification
  (`python -m build` + `twine check`), quality gate aggregator, `workflow_dispatch` trigger,
  and expanded Python matrix to 3.12/3.13/3.14
- SHA-pinned all GitHub Actions to immutable commit hashes

### Added

- Dependabot configuration for automated dependency updates
- CODEOWNERS file for PR review routing
- This CHANGELOG

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

[Unreleased]: https://github.com/pcalnon/juniper-data-client/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/pcalnon/juniper-data-client/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/pcalnon/juniper-data-client/releases/tag/v0.3.0
