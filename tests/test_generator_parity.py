"""Parity tests for generator-name constants vs the server registry.

These tests protect the client from drifting away from the server
(XREPO-01 / DC-01 and XREPO-01c / DC-03). They do **not** require the
juniper-data service to be installed — the expected server-side
generator names are pinned here. If the server registry changes, this
list must be updated in lockstep with the client constants.

Tests covered:

* ``GENERATOR_CIRCLE`` matches the server's ``"circles"`` key (DC-01).
* Every server generator has a corresponding client constant (DC-03).
* The legacy ``"circle"`` alias still works through the fake client
  (so in-flight callers are not broken during the deprecation window).
"""

from __future__ import annotations

import warnings

import pytest

from juniper_data_client import constants
from juniper_data_client.testing import FakeDataClient

# Source of truth for the server-side generator registry. Must be kept
# aligned with ``juniper_data/api/routes/generators.py::GENERATOR_REGISTRY``.
EXPECTED_SERVER_GENERATORS: frozenset[str] = frozenset(
    {
        "spiral",
        "xor",
        "gaussian",
        "circles",
        "moon",
        "checkerboard",
        "csv_import",
        "mnist",
        "arc_agi",
    }
)


CLIENT_GENERATOR_CONSTANTS: dict[str, str] = {
    "GENERATOR_SPIRAL": constants.GENERATOR_SPIRAL,
    "GENERATOR_XOR": constants.GENERATOR_XOR,
    "GENERATOR_CIRCLE": constants.GENERATOR_CIRCLE,
    "GENERATOR_MOON": constants.GENERATOR_MOON,
    "GENERATOR_GAUSSIAN": constants.GENERATOR_GAUSSIAN,
    "GENERATOR_CHECKERBOARD": constants.GENERATOR_CHECKERBOARD,
    "GENERATOR_CSV_IMPORT": constants.GENERATOR_CSV_IMPORT,
    "GENERATOR_MNIST": constants.GENERATOR_MNIST,
    "GENERATOR_ARC_AGI": constants.GENERATOR_ARC_AGI,
}


class TestGeneratorConstantParity:
    """All client generator constants must match a server registry key."""

    def test_circle_matches_server_plural(self) -> None:
        """Regression for XREPO-01 / DC-01 (was ``"circle"``)."""
        assert constants.GENERATOR_CIRCLE == "circles"

    def test_legacy_circle_alias_preserved(self) -> None:
        """Deprecation alias should still expose the old singular name."""
        assert constants.GENERATOR_CIRCLE_LEGACY == "circle"

    def test_every_client_constant_maps_to_server(self) -> None:
        """No client constant should reference a non-existent generator."""
        unexpected = {name: value for name, value in CLIENT_GENERATOR_CONSTANTS.items() if value not in EXPECTED_SERVER_GENERATORS}
        assert not unexpected, f"Client constants point at unknown generators: {unexpected}"

    def test_every_server_generator_has_client_constant(self) -> None:
        """Regression for XREPO-01c / DC-03 (5 missing generators)."""
        client_values = set(CLIENT_GENERATOR_CONSTANTS.values())
        missing = EXPECTED_SERVER_GENERATORS - client_values
        assert not missing, f"Client missing constants for server generators: {sorted(missing)}"

    @pytest.mark.parametrize(
        "name,description",
        [
            (constants.GENERATOR_SPIRAL, constants.GENERATOR_DESCRIPTION_SPIRAL),
            (constants.GENERATOR_XOR, constants.GENERATOR_DESCRIPTION_XOR),
            (constants.GENERATOR_CIRCLE, constants.GENERATOR_DESCRIPTION_CIRCLE),
            (constants.GENERATOR_MOON, constants.GENERATOR_DESCRIPTION_MOON),
            (constants.GENERATOR_GAUSSIAN, constants.GENERATOR_DESCRIPTION_GAUSSIAN),
            (constants.GENERATOR_CHECKERBOARD, constants.GENERATOR_DESCRIPTION_CHECKERBOARD),
            (constants.GENERATOR_CSV_IMPORT, constants.GENERATOR_DESCRIPTION_CSV_IMPORT),
            (constants.GENERATOR_MNIST, constants.GENERATOR_DESCRIPTION_MNIST),
            (constants.GENERATOR_ARC_AGI, constants.GENERATOR_DESCRIPTION_ARC_AGI),
        ],
    )
    def test_description_exists_for_each_generator(self, name: str, description: str) -> None:
        """Every generator constant has a matching human-readable description."""
        assert isinstance(description, str)
        assert description.strip(), f"Description for {name!r} must be non-empty"


class TestFakeClientLegacyAlias:
    """The fake client must accept the legacy ``"circle"`` name transparently."""

    def test_create_dataset_with_legacy_circle_warns_and_succeeds(self) -> None:
        with FakeDataClient() as client:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                result = client.create_dataset("circle", {"n_points": 30, "seed": 1})
            # The fake returns the canonical name, not the legacy alias.
            assert result["generator"] == constants.GENERATOR_CIRCLE == "circles"
            assert any(issubclass(w.category, DeprecationWarning) and "circle" in str(w.message) for w in recorded), "Legacy generator name should emit a DeprecationWarning"

    def test_create_dataset_with_new_circles_is_clean(self) -> None:
        with FakeDataClient() as client:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                result = client.create_dataset("circles", {"n_points": 30, "seed": 1})
            assert result["generator"] == "circles"
            assert not any(issubclass(w.category, DeprecationWarning) for w in recorded), "Canonical generator name must not emit DeprecationWarning"

    def test_get_generator_schema_accepts_legacy_circle(self) -> None:
        with FakeDataClient() as client:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                legacy_schema = client.get_generator_schema("circle")
                canonical_schema = client.get_generator_schema("circles")
            assert legacy_schema == canonical_schema


class TestFakeClientCatalog:
    """The fake catalog should advertise every supported generator using canonical names."""

    def test_catalog_uses_canonical_circles_key(self) -> None:
        with FakeDataClient() as client:
            catalog = client.list_generators()
        names = {entry["name"] for entry in catalog}
        assert "circles" in names
        assert "circle" not in names  # legacy name must not leak into the catalog
