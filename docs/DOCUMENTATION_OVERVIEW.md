# Documentation Overview

## Navigation Guide to juniper-data-client Documentation

**Version:** 0.4.2
**Status:** Active
**Last Updated:** August 24, 2026
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

| Goal                            | Document                                           | Location |
|---------------------------------|----------------------------------------------------|----------|
| **Install and use the client**  | [QUICK_START.md](QUICK_START.md)                   | docs/    |
| **See the full API reference**  | [REFERENCE.md](REFERENCE.md)                       | docs/    |
| **Understand the project**      | [README.md](../README.md)                          | Root     |
| **See development conventions** | [AGENTS.md](../AGENTS.md)                          | Root     |
| **See version history**         | [CHANGELOG.md](../CHANGELOG.md)                    | Root     |
| **Quick-reference dev tasks**   | [DEVELOPER_CHEATSHEET.md](DEVELOPER_CHEATSHEET.md) | docs/    |
| **Run tests**                   | [AGENTS.md](../AGENTS.md)                          | Root     |

---

## Document Index

### docs/ Directory

| File                          | Lines | Type       | Purpose                                              |
|-------------------------------|-------|------------|------------------------------------------------------|
| **DOCUMENTATION_OVERVIEW.md** | ~110  | Overview   | This file -- navigation index                        |
| **QUICK_START.md**            | ~160  | Tutorial   | Install, configure, and use in 5 minutes             |
| **REFERENCE.md**              | ~440  | Reference  | Complete API, configuration, and exception reference |
| **DEVELOPER_CHEATSHEET.md**   | ~180  | Cheatsheet | Quick-reference card for common development tasks    |

### Root Directory

| File             | Lines | Type     | Purpose                                           |
|------------------|-------|----------|---------------------------------------------------|
| **README.md**    | ~220  | Overview | Project overview, features, quick examples        |
| **AGENTS.md**    | ~200  | Guide    | Development conventions, commands, worktree setup |
| **CHANGELOG.md** | ~150  | History  | Version history and release notes                 |

---

## Ecosystem Context

`juniper-data-client` is the official Python HTTP client for the juniper-data REST API. It is a shared dependency consumed by:

- **juniper-cascor** -- `SpiralDataProvider` uses the client to fetch training datasets
- **juniper-canopy** -- `DemoMode` and `CascorIntegration` use it for dataset operations

### Dependency Graph

```bash
juniper-data-client ──calls──> juniper-data (REST API, port 8100)
juniper-cascor ──uses──> juniper-data-client
juniper-canopy ──uses──> juniper-data-client
juniper-ml ──meta-package──> juniper-data-client
```

### Compatibility

This client is **0.4.x** (`pip install juniper-data-client`; pin `>=0.4.2,<0.5.0` or via `juniper-ml[data]`). Server and consumer versions live in those repos — do not copy a stale matrix from this index.

Construction-time URL guards, exception context (`status_code` / `detail` / `response`), `JuniperDataContractError`, and idempotent-only retries are documented in [REFERENCE.md](REFERENCE.md).

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

**Last Updated:** August 24, 2026
**Version:** 0.4.2
**Maintainer:** Paul Calnon

> See the [Juniper Ecosystem Guide](../CLAUDE.md) for the full project map and dependency graph.
