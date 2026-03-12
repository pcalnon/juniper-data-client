# Documentation Overview

## Navigation Guide to juniper-data-client Documentation

**Version:** 0.3.1
**Status:** Active
**Last Updated:** March 3, 2026
**Project:** Juniper - Dataset Service Client Library

---

## Table of Contents

- [Quick Navigation](#quick-navigation)
- [Document Index](#document-index)
- [Ecosystem Context](#ecosystem-context)
- [Related Documentation](#related-documentation)

---

## Quick Navigation

### I Want To

| Goal | Document | Location |
|------|----------|----------|
| **Install and use the client** | [QUICK_START.md](QUICK_START.md) | docs/ |
| **See the full API reference** | [REFERENCE.md](REFERENCE.md) | docs/ |
| **Understand the project** | [README.md](../README.md) | Root |
| **See development conventions** | [AGENTS.md](../AGENTS.md) | Root |
| **See version history** | [CHANGELOG.md](../CHANGELOG.md) | Root |
| **Run tests** | [AGENTS.md](../AGENTS.md) | Root |

---

## Document Index

### docs/ Directory

| File | Lines | Type | Purpose |
|------|-------|------|---------|
| **DOCUMENTATION_OVERVIEW.md** | ~120 | Overview | This file -- navigation index |
| **QUICK_START.md** | ~130 | Tutorial | Install, configure, and use in 5 minutes |
| **REFERENCE.md** | ~260 | Reference | Complete API, configuration, and exception reference |

### Root Directory

| File | Lines | Type | Purpose |
|------|-------|------|---------|
| **README.md** | ~220 | Overview | Project overview, features, quick examples |
| **AGENTS.md** | ~200 | Guide | Development conventions, commands, worktree setup |
| **CHANGELOG.md** | ~150 | History | Version history and release notes |

---

## Ecosystem Context

`juniper-data-client` is the official Python HTTP client for the juniper-data REST API. It is a shared dependency consumed by:

- **juniper-cascor** -- `SpiralDataProvider` uses the client to fetch training datasets
- **juniper-canopy** -- `DemoMode` and `CascorIntegration` use it for dataset operations

### Dependency Graph

```
juniper-data-client ──calls──> juniper-data (REST API, port 8100)
juniper-cascor ──uses──> juniper-data-client
juniper-canopy ──uses──> juniper-data-client
juniper-ml ──meta-package──> juniper-data-client
```

### Compatibility

| juniper-data-client | juniper-data | juniper-cascor | juniper-canopy |
|---------------------|-------------|----------------|----------------|
| 0.3.x | 0.4.x | 0.3.x | 0.2.x |

---

## Related Documentation

### Upstream Service

- **juniper-data** -- [API Reference](https://github.com/pcalnon/juniper-data) (service that this client calls)
- **Data contract**: NPZ artifacts with keys `X_train`, `y_train`, `X_test`, `y_test`, `X_full`, `y_full` (all `float32`)

### Downstream Consumers

- **juniper-cascor** -- [SpiralDataProvider integration](https://github.com/pcalnon/juniper-cascor)
- **juniper-canopy** -- [Dashboard dataset integration](https://github.com/pcalnon/juniper-canopy)

### Meta-Package

- **juniper-ml** -- `pip install juniper-ml[data]` installs this client automatically

---

**Last Updated:** March 3, 2026
**Version:** 0.3.1
**Maintainer:** Paul Calnon
