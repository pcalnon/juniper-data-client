"""Signature pins for ``create_dataset`` (defect-register ``APD-DCLIENT-008``).

Nine positional-or-keyword parameters made ``create_dataset("spiral", p, False)``
legal and unreadable, and any future signature reordering would have silently
rebound arguments at every call site. ``persist`` and everything after it are
now keyword-only; only ``generator`` / ``params`` — the universal pair — stay
positional. An ecosystem-wide AST census at fix time (223 calls across 7 repos)
found **zero** calls passing a third positional argument, so the boundary broke
no caller.

The fake must present the identical convention: a consumer test that calls the
fake positionally has to fail exactly as it would against the real client.
"""

import inspect

import pytest

from juniper_data_client import JuniperDataClient
from juniper_data_client.testing import FakeDataClient

KEYWORD_ONLY_EXPECTED = [
    "persist",
    "name",
    "description",
    "created_by",
    "parent_dataset_id",
    "tags",
    "ttl_seconds",
]


@pytest.mark.parametrize("cls", [JuniperDataClient, FakeDataClient], ids=["real", "fake"])
def test_persist_onward_is_keyword_only(cls):
    kinds = {name: p.kind for name, p in inspect.signature(cls.create_dataset).parameters.items() if name != "self"}
    assert [n for n, k in kinds.items() if k is inspect.Parameter.KEYWORD_ONLY] == KEYWORD_ONLY_EXPECTED
    assert [n for n, k in kinds.items() if k is inspect.Parameter.POSITIONAL_OR_KEYWORD] == ["generator", "params"]


def test_fake_signature_matches_real():
    real = inspect.signature(JuniperDataClient.create_dataset)
    fake = inspect.signature(FakeDataClient.create_dataset)
    assert [(n, p.kind, p.default) for n, p in real.parameters.items()] == [(n, p.kind, p.default) for n, p in fake.parameters.items()]


def test_third_positional_argument_raises_typeerror():
    with FakeDataClient() as client:
        with pytest.raises(TypeError):
            client.create_dataset("spiral", {"n_points": 10}, False)  # noqa: B026 — the positional call IS the assertion


def test_keyword_form_still_works():
    with FakeDataClient() as client:
        result = client.create_dataset("spiral", {"n_points": 12}, persist=False)
        assert "dataset_id" in result
