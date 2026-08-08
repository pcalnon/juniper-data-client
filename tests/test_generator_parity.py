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

# Offline mirror of the server-side generator registry, kept aligned with
# ``juniper_data/api/routes/generators.py::GENERATOR_REGISTRY``. W-9: this is
# no longer trusted blindly — ``TestPinnedMirrorMatchesLiveRegistry`` below
# cross-checks it against the LIVE registry whenever juniper-data is
# importable, so the reverse assertion cannot pass vacuously against a stale
# hand-kept list.
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
        # W-9 (2026-08-08): the equities pair + the five sequence generators.
        "equities",
        "equities_seq",
        "multi_sine",
        "mackey_glass",
        "ar_p",
        "irregular_sine",
        "delay_product",
    }
)


def _live_server_generators() -> "frozenset[str] | None":
    """The real server registry's keys when juniper-data is importable here, else None.

    Importing the routes module pulls FastAPI along; any import failure —
    juniper-data absent or its dependency stack unavailable — means "cannot
    cross-check in this environment", never a test failure.
    """
    try:
        from juniper_data.api.routes.generators import GENERATOR_REGISTRY
    except Exception:  # noqa: BLE001 — absence of the optional cross-check env, not an error
        return None
    return frozenset(GENERATOR_REGISTRY)


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
    # W-9 (2026-08-08): the 7 previously-missing generators.
    "GENERATOR_EQUITIES": constants.GENERATOR_EQUITIES,
    "GENERATOR_EQUITIES_SEQ": constants.GENERATOR_EQUITIES_SEQ,
    "GENERATOR_MULTI_SINE": constants.GENERATOR_MULTI_SINE,
    "GENERATOR_MACKEY_GLASS": constants.GENERATOR_MACKEY_GLASS,
    "GENERATOR_AR_P": constants.GENERATOR_AR_P,
    "GENERATOR_IRREGULAR_SINE": constants.GENERATOR_IRREGULAR_SINE,
    "GENERATOR_DELAY_PRODUCT": constants.GENERATOR_DELAY_PRODUCT,
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
            (constants.GENERATOR_EQUITIES, constants.GENERATOR_DESCRIPTION_EQUITIES),
            (constants.GENERATOR_EQUITIES_SEQ, constants.GENERATOR_DESCRIPTION_EQUITIES_SEQ),
            (constants.GENERATOR_MULTI_SINE, constants.GENERATOR_DESCRIPTION_MULTI_SINE),
            (constants.GENERATOR_MACKEY_GLASS, constants.GENERATOR_DESCRIPTION_MACKEY_GLASS),
            (constants.GENERATOR_AR_P, constants.GENERATOR_DESCRIPTION_AR_P),
            (constants.GENERATOR_IRREGULAR_SINE, constants.GENERATOR_DESCRIPTION_IRREGULAR_SINE),
            (constants.GENERATOR_DELAY_PRODUCT, constants.GENERATOR_DESCRIPTION_DELAY_PRODUCT),
        ],
    )
    def test_description_exists_for_each_generator(self, name: str, description: str) -> None:
        """Every generator constant has a matching human-readable description."""
        assert isinstance(description, str)
        assert description.strip(), f"Description for {name!r} must be non-empty"


class TestPinnedMirrorMatchesLiveRegistry:
    """W-9: validate the hand-kept mirror against the LIVE server registry.

    ``EXPECTED_SERVER_GENERATORS`` is only a mirror; before W-9 nothing checked
    it against juniper-data itself, so the reverse assertion above passed
    vacuously while the server grew 7 generators the client never heard of.
    When juniper-data is importable (dev envs; any CI lane that installs it),
    the mirror must equal the real ``GENERATOR_REGISTRY`` keys exactly; when it
    is not, this cross-check skips and the pinned mirror remains the gate.
    """

    def test_pinned_mirror_matches_live_registry(self) -> None:
        live = _live_server_generators()
        if live is None:
            pytest.skip("juniper-data not importable here — pinned mirror not cross-checked")
        assert EXPECTED_SERVER_GENERATORS == live, f"Pinned mirror drifted from the live server registry. Missing from mirror: {sorted(live - EXPECTED_SERVER_GENERATORS)}; stale in mirror: {sorted(EXPECTED_SERVER_GENERATORS - live)}"


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
