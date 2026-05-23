<!-- markdownlint-disable MD013 MD033 MD041 -->
<!--
  MD013 (line-length): README contains prose paragraphs that intentionally
                       exceed the 512-char ecosystem limit (canonical
                       §4 layout). Wrapping harms PyPI rendering.
  MD033 (no-inline-html): The right-aligned logo + spacing rely on HTML.
  MD041 (first-line-heading): The HTML logo is the first line by design.
  Mirrors juniper-ml #283, juniper-cascor #276, juniper-cascor-worker #71.
-->
<div align="right" width="150px" height="150px" align="right" valign="top"> <img src="images/Juniper_Logo_150px.png" alt="Juniper" align="right" valign="top" width="150px" /></div>
<br /> <br /> <br /> <br />

# Juniper: Dynamic Neural Network Research Platform

Juniper is an AI/ML research platform for investigating dynamic neural network architectures and novel learning paradigms.  The project emphasizes ground-up implementations from primary literature, enabling a more transparent exploration of fundamental algorithms.

## Juniper Data Client

`juniper-data-client` is the **Python HTTP client library** for the `juniper-data` dataset-generation service. The package exposes a single `JuniperDataClient` class whose methods correspond to the REST surface of `juniper-data`: generator schemas, dataset creation, tag-based filtering, batch operations, the named-version registry, and download of the resulting NPZ artifacts as ready-to-train `numpy` arrays. It is the canonical dataset-fetch interface used by both `juniper-cascor` (for training) and `juniper-canopy` (for visualisation), and it serializes the platform's shared `X_train` / `y_train` / `X_test` / `y_test` / `X_full` / `y_full` NPZ schema (all `float32`).

## Distribution

`juniper-data-client` is published on PyPI as **[`juniper-data-client`](https://pypi.org/project/juniper-data-client/)**.
The package is also surfaced through the platform meta-distribution
**[`juniper-ml`](https://pypi.org/project/juniper-ml/)**, which installs
the full client stack via `pip install juniper-ml[all]`.

```bash
pip install juniper-data-client
```

## Ecosystem Compatibility

This client library is part of the [Juniper](https://github.com/pcalnon/juniper-ml) ecosystem.
Verified compatible versions:

| juniper-data | juniper-cascor | juniper-canopy | data-client | cascor-client | cascor-worker |
|--------------|----------------|----------------|-------------|---------------|---------------|
| 0.6.x        | 0.5.x          | 0.5.x          | >=0.4.1     | >=0.4.0       | >=0.4.0       |

For full-stack Docker deployment and integration tests, see [`juniper-deploy`](https://github.com/pcalnon/juniper-deploy).

## Architecture

`juniper-data-client` is a thin synchronous HTTP client. It does not embed any dataset-generation logic of its own: every call resolves to an HTTP request against a `juniper-data` instance, and every NPZ artifact returned has been produced server-side.

```text
┌────────────────────────┐                     ┌──────────────────┐
│     Caller (e.g.       │  HTTP (requests +   │  juniper-data    │
│ juniper-cascor /       │  urllib3 Retry)     │  REST service    │
│ juniper-canopy /       │ ──────────────────► │  Port 8100       │
│ research notebook)     │ ◄────────────────── │  /v1/...         │
└──────────┬─────────────┘   JSON + NPZ        └──────────────────┘
           │ uses
           ▼
┌────────────────────────┐
│  juniper-data-client   │
│  JuniperDataClient     │
│  (this package)        │
└────────────────────────┘
```

The client retries idempotent verbs (GET) on transient failures via `urllib3.Retry` with exponential backoff; mutating verbs (POST, PATCH, DELETE) are **not** auto-retried, to avoid duplicate-side-effect bugs against the dataset registry. Optional `X-Request-ID` propagation and a `RequestHook` instrumentation surface are available for callers that integrate with the platform's observability stack.

## Related Services

| Service | Relationship | Notes |
|---------|-------------|-------|
| [juniper-data](https://github.com/pcalnon/juniper-data) | The HTTP service this client targets | Set `base_url` to the service's URL (default `http://localhost:8100`) |
| [juniper-cascor](https://github.com/pcalnon/juniper-cascor) | Primary consumer; uses this client to fetch training datasets | Reads `JUNIPER_DATA_URL` |
| [juniper-canopy](https://github.com/pcalnon/juniper-canopy) | Secondary consumer; uses this client to fetch visualisation data | Reads `JUNIPER_DATA_URL` |

## Active Research Components

`juniper-data-client` does not host research components of its own; it is the surface through which other components of the Juniper platform reach the research components hosted by `juniper-data`. Through this client, callers access the **ARC-AGI dataset families** (ARC-AGI-1 and ARC-AGI-2), the **named-version dataset registry** (`list_versions`, `get_latest`), the **batch dataset operations** (`batch_create`, `batch_delete`, `batch_update_tags`, `batch_export`), and the **NPZ artifact contract** that the rest of the platform consumes. Treating these as research artifacts is appropriate: each is a stable interface around which comparative experiments are composed.

## Quick Start Guide

### Prerequisites

- Python ≥ 3.12
- A running `juniper-data` instance reachable at the URL passed as `base_url` (typically `http://localhost:8100`)

### Installation

```bash
pip install juniper-data-client
```

Optional extras: `[observability]` enables `X-Request-ID` propagation through `juniper-observability`; `[test]` installs the testing dependencies; `[dev]` adds linting and type-checking tools.

### Verification

```python
from juniper_data_client import JuniperDataClient

with JuniperDataClient("http://localhost:8100") as client:
    health = client.health_check()
    print(f"Service status: {health['status']}")

    result = client.create_spiral_dataset(
        n_spirals=2,
        n_points_per_spiral=100,
        noise=0.1,
        seed=42,
    )
    arrays = client.download_artifact_npz(result["dataset_id"])

    print(f"Training samples: {len(arrays['X_train'])}")
    print(f"Test samples:     {len(arrays['X_test'])}")
```

For PyTorch consumers, the returned arrays convert directly with `torch.from_numpy`:

```python
import torch
X_train = torch.from_numpy(arrays["X_train"])  # torch.float32
y_train = torch.from_numpy(arrays["y_train"])  # torch.float32
```

### Next Steps

- [`docs/QUICK_START.md`](docs/QUICK_START.md) — complete installation and verification guide
- [`docs/REFERENCE.md`](docs/REFERENCE.md) — full API reference, configuration, and error model
- [`docs/DEVELOPER_CHEATSHEET.md`](docs/DEVELOPER_CHEATSHEET.md) — quick-reference card for development tasks
- [`juniper-data`](https://github.com/pcalnon/juniper-data) — the upstream dataset service
- [`juniper-ml`](https://pypi.org/project/juniper-ml/) — platform meta-package on PyPI

## Research Philosophy

The Juniper platform exists to study learning algorithms whose network architecture is not fixed in advance. Its initial anchor is the Cascade-Correlation algorithm of Fahlman and Lebiere (1990), implemented from the primary literature without recourse to higher-level abstractions that elide the algorithm's operational detail. The organising commitment is that algorithm implementations remain inspectable at the level at which they were originally specified: candidate units, correlation objectives, weight-freezing semantics, and the structural events that grow the network are first-class artifacts of the codebase rather than internal details of a library wrapper. This permits comparative work — across algorithms, datasets, and hyperparameter regimes — to be conducted on a known and reproducible substrate.

The current platform comprises a Cascade-Correlation training service exposing a REST and WebSocket interface, a dataset-generation service with a named-version registry that includes the ARC-AGI families, a real-time monitoring dashboard for inspecting training dynamics as they occur, and a distributed worker that parallelises candidate-unit training across hosts. Near-term work extends the architectural-growth catalogue beyond Cascade-Correlation, introduces multi-network orchestration for comparative experiments at the level of network populations rather than individual runs, and tightens the dataset–training–monitoring loop into a reproducible research workbench. The longer-term direction is the systematic empirical study of constructive and architecture-growing learning algorithms, with first-class infrastructure for the ablation, comparison, and replication that such a study requires.

Within this programme, `juniper-data-client` is the canonical integration boundary between the training and data services. Its contract is the dataset-fetch interface used by every training-side consumer; changes to its surface are therefore changes to the platform's shared dataset semantics.

## Documentation

| Document | Purpose |
|----------|---------|
| [`docs/DOCUMENTATION_OVERVIEW.md`](docs/DOCUMENTATION_OVERVIEW.md) | Navigation index for all `juniper-data-client` documentation |
| [`docs/QUICK_START.md`](docs/QUICK_START.md) | Complete installation and verification guide |
| [`docs/REFERENCE.md`](docs/REFERENCE.md) | Full API reference, configuration, error model, and NPZ schema |
| [`docs/DEVELOPER_CHEATSHEET.md`](docs/DEVELOPER_CHEATSHEET.md) | Quick-reference card for development tasks |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

## License

MIT License — see [`LICENSE`](LICENSE) for details.
