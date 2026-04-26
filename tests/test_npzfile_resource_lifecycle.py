"""Regression tests for CC-07 — NpzFile resource lifecycle in download_artifact_npz.

Phase 4E (CC-07) closed a file-handle leak in
``JuniperDataClient.download_artifact_npz``: the previous implementation
called ``np.load(io.BytesIO(content))`` and held the resulting NpzFile only
through a dict comprehension, leaving its underlying ZipFile open until
garbage collection. In long-running consumers (canopy, cascor) this
manifested as accumulated ``ResourceWarning`` records and eventual
fd-exhaustion under sustained download traffic.

These tests pin the new behaviour:

1. ``download_artifact_npz`` does not emit a ResourceWarning when its
   surrounding scope ends (i.e. the NpzFile is closed deterministically).
2. The returned dict still contains all arrays produced by the server's
   NPZ payload and the values are real ``numpy.ndarray`` instances (not
   views over a closed memmap).
3. Calling ``download_artifact_npz`` repeatedly does not steadily inflate
   the open-file-descriptor count.
"""

from __future__ import annotations

import io
import os
import warnings
from unittest.mock import patch

import numpy as np
import pytest

from juniper_data_client import JuniperDataClient


def _make_npz_bytes() -> bytes:
    rng = np.random.default_rng(seed=0)
    arrays = {
        "X_train": rng.random((4, 3), dtype=np.float32),
        "y_train": rng.random((4, 2), dtype=np.float32),
        "X_test": rng.random((2, 3), dtype=np.float32),
        "y_test": rng.random((2, 2), dtype=np.float32),
        "X_full": rng.random((6, 3), dtype=np.float32),
        "y_full": rng.random((6, 2), dtype=np.float32),
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


class _FakeResponse:
    status_code = 200
    ok = True
    headers = {"content-type": "application/octet-stream"}

    def __init__(self, body: bytes) -> None:
        self.content = body

    def raise_for_status(self) -> None:
        return None


def _patch_download_bytes(client: JuniperDataClient, content: bytes):
    """Patch ``download_artifact_bytes`` so it returns ``content`` directly.

    This lets us exercise ``download_artifact_npz`` without going through
    the request/auth/retry machinery, isolating the test to the NpzFile
    lifecycle that CC-07 actually fixes.
    """
    return patch.object(JuniperDataClient, "download_artifact_bytes", lambda self, dataset_id: content)


def test_download_artifact_npz_does_not_emit_resource_warning() -> None:
    """CC-07: closing the NpzFile prevents a ResourceWarning at gc time."""
    client = JuniperDataClient(base_url="http://localhost:8100")
    npz_bytes = _make_npz_bytes()

    with _patch_download_bytes(client, npz_bytes):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            arrays = client.download_artifact_npz("dataset-cc07")
            del arrays

    resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not resource_warnings, f"unexpected ResourceWarning(s): {[str(w.message) for w in resource_warnings]}"


def test_download_artifact_npz_returns_complete_array_dict() -> None:
    """CC-07: the returned dict still contains every array from the server payload."""
    client = JuniperDataClient(base_url="http://localhost:8100")
    npz_bytes = _make_npz_bytes()

    with _patch_download_bytes(client, npz_bytes):
        arrays = client.download_artifact_npz("dataset-cc07")

    assert set(arrays.keys()) == {"X_train", "y_train", "X_test", "y_test", "X_full", "y_full"}
    for value in arrays.values():
        assert isinstance(value, np.ndarray)
        # The arrays must remain readable AFTER download_artifact_npz returns —
        # i.e. the NpzFile must have copied/loaded the data, not handed back
        # views into a now-closed BytesIO. Touching every element via
        # np.asarray + .sum() forces materialisation; if the underlying file
        # were closed this would raise.
        _ = np.asarray(value).sum()


@pytest.mark.skipif(not hasattr(os, "listdir") or not os.path.isdir("/proc/self/fd"), reason="fd accounting only available on /proc-style platforms")
def test_download_artifact_npz_does_not_leak_fds() -> None:
    """CC-07: a tight loop of downloads must not inflate the open-fd count."""
    client = JuniperDataClient(base_url="http://localhost:8100")
    npz_bytes = _make_npz_bytes()

    with _patch_download_bytes(client, npz_bytes):
        # Warm up — first call may open additional caches.
        _ = client.download_artifact_npz("dataset-cc07")
        baseline = len(os.listdir("/proc/self/fd"))
        for _ in range(50):
            _ = client.download_artifact_npz("dataset-cc07")
        final = len(os.listdir("/proc/self/fd"))

    # Allow some slack for unrelated allocator activity but no steady growth.
    assert final - baseline <= 5, f"fd count grew from {baseline} to {final} over 50 downloads"


def test_no_uncontext_managed_np_load_in_client_module() -> None:
    """CC-07 (style): catch any future regression that re-introduces a leak."""
    import inspect

    from juniper_data_client import client as client_module

    src = inspect.getsource(client_module)
    # Quick textual proxy: every np.load(...) appears within 80 chars of "with " or
    # the immediate enclosing "with np.load(" form. Anything else needs review.
    for idx, line in enumerate(src.splitlines(), start=1):
        if "np.load(" not in line:
            continue
        if line.lstrip().startswith("#"):
            continue  # comments are fine
        assert "with " in line, f"juniper_data_client/client.py line {idx} uses np.load() outside a 'with' block: {line.strip()!r}"
