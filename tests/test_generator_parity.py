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

import os
import sys
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


#: W-9 enforcement switch, set by the CI lane that installs juniper-data.
#:
#: The cross-check below degrades to ``pytest.skip`` when juniper-data is absent, which is
#: correct for a laptop or a lane that never promised to run it — and useless as a gate,
#: because a skip is indistinguishable from a pass in a green run. Before this switch NO
#: lane installed juniper-data (all four ``pip install -e ".[test]"`` sites install pytest,
#: pytest-cov, pytest-timeout, responses and juniper-observability), so the cross-check
#: skipped in CI 100% of the time and the pinned mirror was exactly as stale-able as it was
#: before W-9 added it. The trap is that juniper-data IS importable on a dev workstation, so
#: the check runs and passes locally: local green proved nothing about the lane.
REQUIRE_LIVE_REGISTRY_ENV = "JUNIPER_DATA_CLIENT_REQUIRE_LIVE_REGISTRY"


def _requires_live_registry() -> bool:
    """Whether this environment promised to cross-check, making a skip a failure."""
    return os.environ.get(REQUIRE_LIVE_REGISTRY_ENV, "").strip().lower() in {"1", "true", "yes"}


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
            if _requires_live_registry():
                pytest.fail(f"{REQUIRE_LIVE_REGISTRY_ENV} is set, so this lane exists to cross-check the pinned " "mirror against the live registry — but juniper-data is not importable here, so the " "cross-check would have SKIPPED and the lane would have reported success having " "verified nothing. Install juniper-data in this lane, or unset the switch.")
            pytest.skip("juniper-data not importable here — pinned mirror not cross-checked")
        assert EXPECTED_SERVER_GENERATORS == live, f"Pinned mirror drifted from the live server registry. Missing from mirror: {sorted(live - EXPECTED_SERVER_GENERATORS)}; stale in mirror: {sorted(EXPECTED_SERVER_GENERATORS - live)}"


class TestLiveRegistryEnforcementSwitch:
    """W-9: the switch that turns a vacuous skip into a failure must itself be tested.

    A guard against vacuous passes is worth nothing if the guard is vacuous. These cover
    ``_requires_live_registry`` directly rather than the ``pytest.fail`` branch, which
    cannot be exercised in-process without failing the test that calls it.
    """

    def test_unset_means_skipping_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A laptop or a lane that never promised to cross-check may still skip."""
        monkeypatch.delenv(REQUIRE_LIVE_REGISTRY_ENV, raising=False)
        assert _requires_live_registry() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", " 1 "])
    def test_truthy_values_demand_the_cross_check(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """The enforcing lane sets this, and several spellings must all mean "enforce".

        A switch that silently reads as off when someone writes ``true`` instead of ``1``
        reintroduces the exact silent-skip failure it exists to prevent.
        """
        monkeypatch.setenv(REQUIRE_LIVE_REGISTRY_ENV, value)
        assert _requires_live_registry() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_falsy_values_leave_the_skip_in_place(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        """Explicitly disabling it is honoured, so the switch can be turned off deliberately."""
        monkeypatch.setenv(REQUIRE_LIVE_REGISTRY_ENV, value)
        assert _requires_live_registry() is False

    def test_a_missing_juniper_data_fails_rather_than_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The negative control: the exact regression this switch exists to catch.

        If the enforcing lane ever loses its ``pip install juniper-data`` — a dependency
        bump, a refactor of the install step — the cross-check silently reverts to skipping
        and the lane goes green having verified nothing. That is how W-9 shipped. Here the
        probe is forced to report "not importable" with the switch on, and the cross-check
        must raise rather than skip.
        """
        monkeypatch.setenv(REQUIRE_LIVE_REGISTRY_ENV, "1")
        monkeypatch.setattr(sys.modules[__name__], "_live_server_generators", lambda: None)

        with pytest.raises(pytest.fail.Exception, match="verified nothing"):
            TestPinnedMirrorMatchesLiveRegistry().test_pinned_mirror_matches_live_registry()

    def test_a_missing_juniper_data_still_skips_when_not_enforcing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off the enforcing lane the degradation is still a skip, not a failure.

        Contributors without juniper-data installed must not see a red suite.
        """
        monkeypatch.delenv(REQUIRE_LIVE_REGISTRY_ENV, raising=False)
        monkeypatch.setattr(sys.modules[__name__], "_live_server_generators", lambda: None)

        with pytest.raises(pytest.skip.Exception):
            TestPinnedMirrorMatchesLiveRegistry().test_pinned_mirror_matches_live_registry()

    def test_the_enforcing_lane_can_actually_import_juniper_data(self) -> None:
        """In the enforcing lane the probe must return a real registry, not None.

        This is the positive assertion the acceptance asks for: it is not enough that the
        cross-check *would* fail on a skip, the lane must demonstrably reach the live
        registry. Outside that lane this is a no-op.
        """
        if not _requires_live_registry():
            pytest.skip(f"{REQUIRE_LIVE_REGISTRY_ENV} not set — not the enforcing lane")
        live = _live_server_generators()
        assert live is not None, "the enforcing lane must be able to import juniper_data"
        assert live, "the live registry must not be empty"


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
