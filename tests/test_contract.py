"""Tests for validate_npz_contract (WS-1 / juniper-data#168 NPZ contract).

Covers the X.ndim dispatch (2-D tabular vs 3-D sequence), each sequence rule
(missing t/dt, bad dt sign/first-column, inconsistent t/dt, non-binary or
mis-shaped masks, observed-on-padded), and a save -> load -> validate round-trip
on a crafted irregular-Δt artifact (the §6.4 reference consumer end to end).

Also pins APD-DCLIENT-002: every violation raises ``JuniperDataContractError``
(a ``ValueError`` inside the package hierarchy). The original
``pytest.raises(ValueError)`` assertions are deliberately kept unchanged -- they
are the back-compat pin that the pre-0.5 documented contract still holds.
"""

import importlib
import io

import numpy as np
import pytest

from juniper_data_client import JuniperDataClientError, JuniperDataContractError, validate_npz_contract
from juniper_data_client.constants import CONTRACT_KIND_SEQUENCE, CONTRACT_KIND_TABULAR


def _tabular():
    return {
        "X_train": np.zeros((4, 3), np.float32),
        "X_test": np.zeros((1, 3), np.float32),
        "X_full": np.zeros((5, 3), np.float32),
    }


def _sequence(n_train=4, lookback=3, n_features=2, with_t=False):
    """A valid 3-D sequence artifact with an irregular dt (weekend-style gaps)."""
    arrays = {}
    gap_row = np.array(([0.0] + [1.0, 3.0, 1.0, 2.0][: lookback - 1]), dtype=np.float32)
    for split, n in (("train", n_train), ("test", 2), ("full", n_train + 2)):
        arrays[f"X_{split}"] = np.zeros((n, lookback, n_features), np.float32)
        arrays[f"dt_{split}"] = np.tile(gap_row, (n, 1))
        arrays[f"observed_mask_{split}"] = np.ones((n, lookback), np.uint8)
        if with_t:
            arrays[f"t_{split}"] = np.cumsum(arrays[f"dt_{split}"], axis=1).astype(np.float64)
    return arrays


def test_tabular_2d_returns_tabular():
    assert validate_npz_contract(_tabular()) == CONTRACT_KIND_TABULAR


def test_sequence_3d_returns_sequence():
    assert validate_npz_contract(_sequence()) == CONTRACT_KIND_SEQUENCE


def test_sequence_with_consistent_t_and_dt():
    assert validate_npz_contract(_sequence(with_t=True)) == CONTRACT_KIND_SEQUENCE


def test_rejects_4d_x():
    with pytest.raises(ValueError, match="2-D"):
        validate_npz_contract({"X_train": np.zeros((2, 3, 4, 5), np.float32)})


def test_sequence_missing_t_and_dt_raises():
    with pytest.raises(ValueError, match="at least one"):
        validate_npz_contract({"X_train": np.zeros((3, 4, 2), np.float32)})


def test_negative_dt_raises():
    arrays = _sequence()
    arrays["dt_train"][0, 1] = -1.0
    with pytest.raises(ValueError, match="negative"):
        validate_npz_contract(arrays)


def test_nonzero_first_dt_raises():
    arrays = _sequence()
    arrays["dt_train"][0, 0] = 1.0
    with pytest.raises(ValueError, match="must be 0"):
        validate_npz_contract(arrays)


def test_inconsistent_t_and_dt_raises():
    arrays = _sequence(with_t=True)
    arrays["dt_train"][:, 1] = 99.0  # no longer matches diff(t)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_npz_contract(arrays)


def test_non_binary_mask_raises():
    arrays = _sequence()
    arrays["observed_mask_train"][0, 0] = 2
    with pytest.raises(ValueError, match="binary"):
        validate_npz_contract(arrays)


def test_mis_shaped_mask_raises():
    arrays = _sequence()
    arrays["observed_mask_train"] = np.ones((arrays["X_train"].shape[0], 99), np.uint8)
    with pytest.raises(ValueError, match="shape"):
        validate_npz_contract(arrays)


def test_observed_mask_on_padded_step_raises():
    arrays = _sequence()
    padding = np.ones_like(arrays["observed_mask_train"])
    padding[0, -1] = 0  # this step is structural padding...
    arrays["padding_mask_train"] = padding
    arrays["observed_mask_train"][0, -1] = 1  # ...but marked as a real observation
    with pytest.raises(ValueError, match="padded"):
        validate_npz_contract(arrays)


def test_round_trip_through_npz_bytes():
    # Build a crafted irregular-Δt artifact, save to NPZ bytes, reload, validate.
    arrays = _sequence(n_train=4, lookback=3, n_features=2)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    buf.seek(0)
    with np.load(buf) as npz:
        loaded = {key: npz[key] for key in npz.files}
    assert validate_npz_contract(loaded) == CONTRACT_KIND_SEQUENCE
    # The irregular gap (3-day weekend-style jump) survives the round-trip.
    assert loaded["dt_full"][0, 2] == 3.0


# ---------------------------------------------------------------------------
# APD-DCLIENT-002: violations raise JuniperDataContractError, not bare
# ValueError. One builder per raise site in contract.py, so reverting any
# single site back to ``raise ValueError`` fails exactly that arm.
# ---------------------------------------------------------------------------


def _v_x_neither_2d_nor_3d():
    return {"X_train": np.zeros((2, 3, 4, 5), np.float32)}, "2-D"


def _v_missing_t_and_dt():
    return {"X_train": np.zeros((3, 4, 2), np.float32)}, "at least one"


def _v_dt_wrong_shape():
    arrays = _sequence()
    arrays["dt_train"] = np.zeros((arrays["X_train"].shape[0], 99), np.float32)
    return arrays, "dt_train shape"


def _v_negative_dt():
    arrays = _sequence()
    arrays["dt_train"][0, 1] = -1.0
    return arrays, "negative"


def _v_nonzero_first_dt():
    arrays = _sequence()
    arrays["dt_train"][0, 0] = 1.0
    return arrays, "must be 0"


def _v_inconsistent_t_dt():
    arrays = _sequence(with_t=True)
    arrays["dt_train"][:, 1] = 99.0
    return arrays, "inconsistent"


def _v_mask_wrong_shape():
    arrays = _sequence()
    arrays["observed_mask_train"] = np.ones((arrays["X_train"].shape[0], 99), np.uint8)
    return arrays, "observed_mask_train shape"


def _v_non_binary_mask():
    arrays = _sequence()
    arrays["observed_mask_train"][0, 0] = 2
    return arrays, "binary"


def _v_observed_on_padded():
    arrays = _sequence()
    padding = np.ones_like(arrays["observed_mask_train"])
    padding[0, -1] = 0
    arrays["padding_mask_train"] = padding
    arrays["observed_mask_train"][0, -1] = 1
    return arrays, "padded"


@pytest.mark.parametrize(
    "build",
    [
        _v_x_neither_2d_nor_3d,
        _v_missing_t_and_dt,
        _v_dt_wrong_shape,
        _v_negative_dt,
        _v_nonzero_first_dt,
        _v_inconsistent_t_dt,
        _v_mask_wrong_shape,
        _v_non_binary_mask,
        _v_observed_on_padded,
    ],
    ids=lambda fn: fn.__name__,
)
def test_each_violation_raises_the_contract_error_type(build):
    arrays, match = build()
    with pytest.raises(JuniperDataContractError, match=match):
        validate_npz_contract(arrays)


def test_contract_error_joins_hierarchy_and_stays_a_valueerror():
    assert issubclass(JuniperDataContractError, JuniperDataClientError)
    assert issubclass(JuniperDataContractError, ValueError)
    # Catchable under the package base -- the point of APD-DCLIENT-002...
    with pytest.raises(JuniperDataClientError):
        validate_npz_contract({"X_train": np.zeros((5,), np.float32)})
    # ...and still under ValueError, the originally documented contract.
    with pytest.raises(ValueError):
        validate_npz_contract({"X_train": np.zeros((5,), np.float32)})


def test_contract_error_is_exported():
    # importlib rather than a module-level ``import juniper_data_client``:
    # mixing that with the ``from juniper_data_client import ...`` above trips
    # CodeQL's py/import-and-import-from (an unresolved review thread blocks
    # the merge while every check reads green).
    mod = importlib.import_module("juniper_data_client")
    assert "JuniperDataContractError" in mod.__all__
    assert mod.JuniperDataContractError is JuniperDataContractError


def test_contract_error_carries_no_http_context_and_survives_pickle():
    """A contract violation is local -- no HTTP response behind it -- and the
    subclass must round-trip through pickle/copy as itself (the base
    ``__reduce__`` rebuilds via ``self.__class__``, not a hard-coded type).
    """
    import copy as copy_module

    # Nothing untrusted: the payload is produced by ``pickle.dumps`` below, in
    # this process, from the exception this test just caught. Same suppressions
    # and reasoning as test_client.py's pickle round-trip.
    import pickle  # nosec B403

    with pytest.raises(JuniperDataContractError) as excinfo:
        validate_npz_contract({"X_train": np.zeros((3, 4, 2), np.float32)})
    original = excinfo.value
    assert original.status_code is None
    assert original.detail is None
    assert original.response is None

    round_tripped = pickle.loads(pickle.dumps(original))  # nosec B301
    for rebuilt in (round_tripped, copy_module.copy(original), copy_module.deepcopy(original)):
        assert type(rebuilt) is JuniperDataContractError
        assert isinstance(rebuilt, ValueError)
        assert str(rebuilt) == str(original)
