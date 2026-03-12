# Pull Request: Security Hardening — Build Attestations and Security Scanning

**Date:** 2026-03-03
**Version(s):** 0.3.1 → 0.3.2
**Author:** Paul Calnon
**Status:** READY_FOR_MERGE

---

## Summary

Supply chain security improvements for juniper-data-client: enables PyPI build attestations and adds scheduled security scanning.

---

## Changes

### Security

- Enabled build attestations in publish workflow

### Added

- `.github/workflows/security-scan.yml` — Weekly Bandit and pip-audit scanning

---

## Impact & SemVer

- **SemVer impact:** PATCH (0.3.1 → 0.3.2)
- **Breaking changes:** NO
- **Security/privacy impact:** MEDIUM — Supply chain verification via attestations

---

## Testing & Results

| Test Type | Passed | Failed | Skipped | Notes             |
| --------- | ------ | ------ | ------- | ----------------- |
| Unit      | 88     | 0      | 0       | All tests passing |

---

## Files Changed

- `.github/workflows/publish.yml` — Enabled build attestations
- `.github/workflows/security-scan.yml` — New scanning workflow

---

## Related Issues / Tickets

- Phase Documentation: `juniper-ml/notes/SECURITY_AUDIT_PLAN.md`
