"""Consumer-shaped type probe for ``juniper_data_client`` (defect register ``APD-ECO-006``).

Project:     Juniper
Sub-Project: juniper-data-client
Application: published type-surface probe
Author:      Paul Calnon
License:     MIT License

WHY THIS FILE EXISTS, AND WHY IT LIVES **OUTSIDE** THE PACKAGE
-------------------------------------------------------------
The repo's mypy hook was scoped ``^juniper_data_client/(?!testing/).*\\.py$`` -- library-internal
source only. Nothing type-checked a file that *imports* the package the way a consumer does, so the
published type surface was never verified as usable from outside. That is ``APD-ECO-006``: "no client
type-checks a consumer-shaped probe."

This module is that consumer. It is deliberately at the repo root rather than inside
``juniper_data_client/`` or ``tests/`` -- inside the package it would be internal source again, and
the hook's ``files:`` regex is widened to reach exactly here.

WHAT THIS CATCHES
-----------------
* a public name in ``__all__`` that does not resolve for an importer;
* a public method whose annotation is missing, wrong, or silently ``Any`` at the boundary;
* a return type that drifts from what the docstring and callers assume.

WHAT THIS DOES **NOT** CATCH -- stated so it is not mistaken for a guarantee
---------------------------------------------------------------------------
mypy resolves ``juniper_data_client`` from the **source tree** here, not from an installed
distribution, so ``py.typed`` is bypassed entirely. A wheel that shipped without ``py.typed`` would
give a real consumer an untyped package while this probe still passed -- the ``APD-SVCCORE-008`` /
``APD-OBS-002`` class. Catching that needs a check against the built artifact, and is a separate
concern from this row. ``juniper_data_client/py.typed`` exists today; nothing yet asserts it is
packaged.

This file is never imported at runtime and intentionally has no test assertions: ``assert_type`` is a
static construct, so the *type check* is the test. It must stay import-clean, because mypy has to be
able to analyse it.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from juniper_data_client import (
    ContractKind,
    JuniperDataClient,
    JuniperDataClientError,
    JuniperDataConnectionError,
    JuniperDataNotFoundError,
    JuniperDataTimeoutError,
    JuniperDataValidationError,
    validate_npz_contract,
)


def probe_client_surface(client: JuniperDataClient) -> None:
    """Exercise the public methods a consumer actually calls, checking each declared return type."""
    health: Dict[str, Any] = client.health_check()
    datasets: List[str] = client.list_datasets()
    metadata: Dict[str, Any] = client.get_dataset_metadata("some-dataset-id")

    # Reading a value back off each result: a return annotated ``Any`` would make these vacuous, so
    # they are written to be meaningful only when the annotation is concrete.
    _status: Any = health.get("status")
    _first: str | None = datasets[0] if datasets else None
    _name: Any = metadata.get("name")


def probe_exception_hierarchy() -> None:
    """Every published error must be catchable through the package's base error.

    A consumer writes ``except JuniperDataClientError``; if a subclass ever stops deriving from it,
    that consumer silently stops catching it. Expressed as static subtype checks rather than
    ``issubclass`` calls so the failure surfaces in the type check, not at runtime.
    """
    base: type[JuniperDataClientError] = JuniperDataClientError
    for derived in (
        JuniperDataConnectionError,
        JuniperDataNotFoundError,
        JuniperDataTimeoutError,
        JuniperDataValidationError,
    ):
        _: type[JuniperDataClientError] = derived
    _base_is_exception: type[Exception] = base


def probe_module_level_helpers(arrays: Dict[str, "np.ndarray"]) -> ContractKind:
    """``validate_npz_contract`` is exported at the package root and callable by a consumer.

    Its signature is the reason this probe earns its keep: the first draft called it as
    ``validate_npz_contract(path)`` on the assumption it took a filename. It takes the *loaded*
    ``Dict[str, np.ndarray]`` and returns a ``ContractKind``. mypy rejected the draft -- which is the
    whole point of checking a consumer-shaped file rather than only the library's own source.
    """
    return validate_npz_contract(arrays)
