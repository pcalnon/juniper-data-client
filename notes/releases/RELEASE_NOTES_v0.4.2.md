# juniper-data-client v0.4.2 Release Notes

**Release Date:** 2026-06-17
**Version:** 0.4.2
**Release Type:** PATCH (additive)

---

## Overview

Ships the accumulated unreleased work, headlined by the **`validate_npz_contract`** NPZ data-contract validator — which had lived in the source tree since the WS-1 data foundation but never reached a published wheel, so PyPI consumers (the juniper-recurrence app) silently skipped the Δt contract gate. 0.4.2 closes that gap, and also delivers Docker-secret API-key indirection and the CI-lint migration to `juniper-ci-tools`. Additive only; existing consumers pinning `<0.5.0` pick it up automatically.

> **Status:** STABLE — additive, backward-compatible.

---

## Release Summary

- **Release type:** PATCH (additive)
- **Primary focus:** Publish the Δt contract validator; secrets hardening; CI tooling
- **Breaking changes:** NO
- **Priority summary:** Unblocks the juniper-recurrence app's mandatory Δt gate (roadmap I1 / D-2)

---

## What's New

### Data contract validation

#### `validate_npz_contract` (WS-1, #87)

Public helper `from juniper_data_client import validate_npz_contract` that classifies an artifact's array bundle as `"tabular"` (2-D `X`) or `"sequence"` (3-D `X` with the WS-1 irregular-Δt keys) and validates the contract invariants — `dt >= 0`, `dt[:, 0] == 0`, mask/shape consistency — with a configurable `dt_atol`. Lets consumers gate 3-D Δt artifacts up front instead of relying on model-side shape checks. 12 unit tests in `tests/test_contract.py`.

> Present in source since WS-1 but **absent from the published 0.4.1 wheel** — this is the release that ships it.

### Secrets hardening

#### `JUNIPER_DATA_API_KEY_FILE` Docker-secret indirection (#90)

`JuniperDataClient` now resolves its API key from a `JUNIPER_DATA_API_KEY_FILE` env var (a path whose stripped file contents are the key, e.g. `/run/secrets/juniper_data_api_keys`) before falling back to plain `JUNIPER_DATA_API_KEY`; an explicit `api_key=` still wins over both. Mirrors the services' `get_secret` pattern. New `API_KEY_FILE_ENV_VAR` constant + `_resolve_api_key_from_env()` helper; 3 unit tests.

### CI tooling

#### Lints migrated to `juniper-ci-tools` (#75–#77)

The AGENTS.md version-drift lint (`juniper-lint-agents-md-version`) and workflow script-path lint (`juniper-lint-workflow-paths`) now run from the shared `juniper-ci-tools>=0.2.0` console scripts; the former inline `util/` copies were removed. Also adopts the v0.4.0 header-schema lint + AGENTS.md auto-bump workflow (#77) and syncs the README compatibility matrix (#78).

### Dependencies

- Routine `python-minor` group bumps and GitHub Actions updates (Dependabot, #79–#92).

---

## Upgrade Notes

Backward-compatible. Consumers pinning `juniper-data-client>=0.4.1,<0.5.0` resolve 0.4.2 automatically; the `validate_npz_contract` symbol becomes importable on a fresh install.

```bash
pip install --upgrade juniper-data-client
```

For the **juniper-recurrence app**: bump the pin to `>=0.4.2,<0.5.0` and drop the optional-import guard to make the Δt gate mandatory (queued follow-up).

---

## Known Issues

None known at time of release.

---

## What's Next

- Consumers that guarded the optional import (e.g. the juniper-recurrence app) can now pin `>=0.4.2` and call `validate_npz_contract` unconditionally.

---

## Links

- Changelog: `CHANGELOG.md`
- Roadmap (I1 / D-2): [JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md](https://github.com/pcalnon/juniper-ml/blob/main/notes/JUNIPER_RECURRENCE_STATE_ASSESSMENT_AND_ROADMAP_2026-06-17.md)
- **Full Changelog:** https://github.com/pcalnon/juniper-data-client/compare/v0.4.1...v0.4.2

---

## Contributors

- Paul Calnon (@pcalnon)
