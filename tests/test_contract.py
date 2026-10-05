"""Tests for validate_npz_contract (WS-1 / juniper-data#168 NPZ contract).

Covers the X.ndim dispatch (2-D tabular vs 3-D sequence), each sequence rule
(missing t/dt, bad dt sign/first-column, inconsistent t/dt, non-binary or
mis-shaped masks, observed-on-padded), and a save -> load -> validate round-trip
on a crafted irregular-Δt artifact (the §6.4 reference consumer end to end).

Also pins APD-DCLIENT-002: every violation raises ``JuniperDataContractError``
(a ``ValueError`` inside the package hierarchy). The original
``pytest.raises(ValueError)`` assertions are deliberately kept unchanged -- they
are the back-compat pin that the pre-0.5 documented contract still holds.

W1.4 (findings F-P3 / F-S3; ruling R2) adds four rules, each rejection with its
own test: ``dt`` must be finite; ``target_dt`` (if present) must be ``(W,)``,
finite and ``>= 0``; ``seq_lengths`` (if present) must be ``(W,)``, integer and
in ``[1, L]``; and ``X`` / ``y`` / ``y_reg`` must be ``float32`` in every
partition, on the 2-D path as well as the 3-D one. A conforming three-partition
sequence artifact carrying every optional key, and a conforming flat artifact,
still validate. Plan: juniper-ml
``notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``.
"""

import importlib
import io
import re

import numpy as np
import pytest

from juniper_data_client import JuniperDataClientError, JuniperDataContractError, validate_npz_contract
from juniper_data_client.constants import CONTRACT_KIND_SEQUENCE, CONTRACT_KIND_TABULAR

# Window counts for the W1.4 builders: every partition non-empty and of a different
# size, so a rule that read one partition's shape for another would be caught.
_SPLIT_WINDOWS = (("train", 6), ("val", 2), ("test", 3))


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


def _full_sequence(windows=_SPLIT_WINDOWS, lookback=4, n_features=3):
    """A conforming three-partition sequence artifact carrying every optional key.

    Mirrors the ``equities_seq`` key inventory (plan §3.4): ``X`` / ``y`` / ``y_reg``
    float32, ``dt`` / ``target_dt`` float32, ``date`` / ``window_end_date`` /
    ``ticker_code`` int32, ``observed_mask`` uint8 and a unicode ``ticker_vocab``. It
    adds the optional keys that producer does not emit: ``t`` (float64, consistent
    with ``dt``), and left-padded windows whose ``seq_lengths`` agree with
    ``padding_mask``. The train lengths run ``1 .. L`` so both bounds are present.
    No ``*_full`` key: decision 11 retired the family.
    """
    rng = np.random.default_rng(11)
    gap_row = np.array([0.0, 1.0, 3.0, 1.0, 2.0, 1.0][:lookback], dtype=np.float32)
    arrays = {"ticker_vocab": np.array(["AAPL", "MSFT"], dtype=np.str_)}
    for split, w in windows:
        dt = np.tile(gap_row, (w, 1))
        seq_lengths = (np.arange(w) % lookback + 1).astype(np.int64)
        padding = (np.arange(lookback)[None, :] >= (lookback - seq_lengths)[:, None]).astype(np.uint8)
        arrays[f"X_{split}"] = rng.standard_normal((w, lookback, n_features)).astype(np.float32)
        arrays[f"y_{split}"] = np.eye(2, dtype=np.float32)[rng.integers(0, 2, size=w)]
        arrays[f"y_reg_{split}"] = rng.standard_normal((w, 1)).astype(np.float32)
        arrays[f"dt_{split}"] = dt
        arrays[f"t_{split}"] = np.cumsum(dt, axis=1).astype(np.float64)
        arrays[f"target_dt_{split}"] = np.full(w, 1.0, dtype=np.float32)
        arrays[f"seq_lengths_{split}"] = seq_lengths
        arrays[f"padding_mask_{split}"] = padding
        arrays[f"observed_mask_{split}"] = padding.copy()  # observed only where not padded
        arrays[f"date_{split}"] = np.full((w, lookback), 20260105, dtype=np.int32)
        arrays[f"window_end_date_{split}"] = np.full(w, 20260105, dtype=np.int32)
        arrays[f"ticker_code_{split}"] = (np.arange(w) % 2).astype(np.int32)
    return arrays


def _full_tabular(n_features=15):
    """A conforming flat (2-D) three-partition artifact, with the flat ``equities`` keys."""
    rng = np.random.default_rng(12)
    arrays = {"ticker_vocab": np.array(["AAPL"], dtype=np.str_)}
    for split, n in (("train", 8), ("val", 2), ("test", 3)):
        arrays[f"X_{split}"] = rng.standard_normal((n, n_features)).astype(np.float32)
        arrays[f"y_{split}"] = np.eye(2, dtype=np.float32)[rng.integers(0, 2, size=n)]
        arrays[f"y_reg_{split}"] = rng.standard_normal((n, 1)).astype(np.float32)
        arrays[f"date_{split}"] = np.full(n, 20260105, dtype=np.int32)
        arrays[f"ticker_code_{split}"] = np.zeros(n, dtype=np.int32)
    return arrays


def test_tabular_2d_returns_tabular():
    assert validate_npz_contract(_tabular()) == CONTRACT_KIND_TABULAR


def test_sequence_3d_returns_sequence():
    assert validate_npz_contract(_sequence()) == CONTRACT_KIND_SEQUENCE


def test_sequence_with_consistent_t_and_dt():
    assert validate_npz_contract(_sequence(with_t=True)) == CONTRACT_KIND_SEQUENCE


def test_sequence_with_t_only_returns_sequence():
    """WS-1: a 3-D artifact needs at least one of t/dt — t alone is valid.

    Requiring both would reject recurrence artifacts that store absolute time
    without a precomputed ``dt`` channel.
    """
    arrays = _sequence(with_t=True)
    for split in ("train", "test", "full"):
        del arrays[f"dt_{split}"]
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE


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


def test_non_binary_padding_mask_raises():
    """The mask loop must visit ``padding_mask``, not only ``observed_mask``."""
    arrays = _sequence()
    arrays["padding_mask_train"] = np.ones_like(arrays["observed_mask_train"])
    arrays["padding_mask_train"][0, 0] = 2
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


def _v_non_binary_padding_mask():
    arrays = _sequence()
    arrays["padding_mask_train"] = np.ones_like(arrays["observed_mask_train"])
    arrays["padding_mask_train"][0, 0] = 2
    return arrays, "binary"


def _v_observed_on_padded():
    arrays = _sequence()
    padding = np.ones_like(arrays["observed_mask_train"])
    padding[0, -1] = 0
    arrays["padding_mask_train"] = padding
    arrays["observed_mask_train"][0, -1] = 1
    return arrays, "padded"


def _v_not_float32():
    arrays = _tabular()
    arrays["X_train"] = arrays["X_train"].astype(np.float64)
    return arrays, "X_train must be float32"


def _v_non_finite_dt():
    arrays = _sequence()
    arrays["dt_train"][0, 1] = np.nan
    return arrays, "dt_train has non-finite values"


def _v_target_dt_wrong_shape():
    arrays = _full_sequence()
    arrays["target_dt_train"] = arrays["target_dt_train"].reshape(-1, 1)
    return arrays, "target_dt_train shape"


def _v_non_finite_target_dt():
    arrays = _full_sequence()
    arrays["target_dt_train"][0] = np.inf
    return arrays, "target_dt_train has non-finite values"


def _v_negative_target_dt():
    arrays = _full_sequence()
    arrays["target_dt_train"][0] = -1.0
    return arrays, "target_dt_train has negative horizons"


def _v_seq_lengths_wrong_shape():
    arrays = _full_sequence()
    arrays["seq_lengths_train"] = arrays["seq_lengths_train"][:-1]
    return arrays, "seq_lengths_train shape"


def _v_seq_lengths_not_integer():
    arrays = _full_sequence()
    arrays["seq_lengths_train"] = arrays["seq_lengths_train"].astype(np.float32)
    return arrays, "seq_lengths_train must be an integer dtype"


def _v_seq_lengths_below_one():
    arrays = _full_sequence()
    arrays["seq_lengths_train"][0] = 0
    return arrays, "seq_lengths_train has values below 1"


def _v_seq_lengths_above_window_length():
    arrays = _full_sequence()
    arrays["seq_lengths_train"][0] = arrays["X_train"].shape[1] + 1
    return arrays, "seq_lengths_train has values above the window length"


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
        _v_non_binary_padding_mask,
        _v_observed_on_padded,
        _v_not_float32,
        _v_non_finite_dt,
        _v_target_dt_wrong_shape,
        _v_non_finite_target_dt,
        _v_negative_target_dt,
        _v_seq_lengths_wrong_shape,
        _v_seq_lengths_not_integer,
        _v_seq_lengths_below_one,
        _v_seq_lengths_above_window_length,
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


# ---------------------------------------------------------------------------
# W1.4 (F-P3 / F-S3; ruling R2). Conforming artifacts first, so a rule that
# refused a valid artifact fails here before any rejection test can mask it.
# ---------------------------------------------------------------------------


def test_conforming_three_partition_sequence_artifact_with_every_optional_key_validates():
    assert validate_npz_contract(_full_sequence()) == CONTRACT_KIND_SEQUENCE


def test_conforming_flat_artifact_validates():
    assert validate_npz_contract(_full_tabular()) == CONTRACT_KIND_TABULAR


def test_conforming_sequence_artifact_validates_from_npz_bytes_and_as_an_npzfile():
    """The dtypes the rules check survive an NPZ round-trip, and an open ``NpzFile`` works as-is."""
    buf = io.BytesIO()
    np.savez(buf, **_full_sequence())
    buf.seek(0)
    with np.load(buf) as npz:
        assert validate_npz_contract(npz) == CONTRACT_KIND_SEQUENCE
        assert validate_npz_contract({key: npz[key] for key in npz.files}) == CONTRACT_KIND_SEQUENCE


def test_an_empty_partition_passes_the_per_window_rules():
    """A zero-window partition (small-n truncation) has nothing to violate.

    Also pins ``np.any`` over ``min()`` / ``max()`` in the ``seq_lengths`` range
    check: those raise on an empty array.
    """
    arrays = _full_sequence(windows=(("train", 6), ("val", 0), ("test", 3)))
    assert arrays["seq_lengths_val"].shape == (0,)
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE


# --- dt must be finite -----------------------------------------------------


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "posinf", "neginf"])
@pytest.mark.parametrize("column", [0, 2], ids=["first-column", "later-column"])
def test_non_finite_dt_raises(bad, column):
    """Finiteness is checked before the sign and first-column rules, and named as such.

    Checked after them, ``-inf`` would read as a negative gap and a non-finite first
    column as a convention breach, while ``+inf`` or ``NaN`` past the first column
    would pass outright -- the gap F-P3 recorded.
    """
    arrays = _full_sequence()
    arrays["dt_val"][1, column] = bad
    with pytest.raises(JuniperDataContractError, match=r"^dt_val has non-finite values$"):
        validate_npz_contract(arrays)


# --- target_dt: (W,), finite, >= 0 -----------------------------------------


def test_target_dt_wrong_ndim_raises():
    """A ``(W, 1)`` column -- the ``y_reg`` shape -- is not one horizon per window."""
    arrays = _full_sequence()
    w = arrays["X_test"].shape[0]
    arrays["target_dt_test"] = arrays["target_dt_test"].reshape(w, 1)
    with pytest.raises(JuniperDataContractError, match=re.escape(f"target_dt_test shape {(w, 1)} != {(w,)}")):
        validate_npz_contract(arrays)


def test_target_dt_wrong_length_raises():
    arrays = _full_sequence()
    w = arrays["X_test"].shape[0]
    arrays["target_dt_test"] = np.ones(w + 1, dtype=np.float32)
    with pytest.raises(JuniperDataContractError, match=re.escape(f"target_dt_test shape {(w + 1,)} != {(w,)}")):
        validate_npz_contract(arrays)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "posinf", "neginf"])
def test_non_finite_target_dt_raises(bad):
    arrays = _full_sequence()
    arrays["target_dt_test"][0] = bad
    with pytest.raises(JuniperDataContractError, match=r"^target_dt_test has non-finite values$"):
        validate_npz_contract(arrays)


def test_negative_target_dt_raises():
    arrays = _full_sequence()
    arrays["target_dt_test"][-1] = -1.0
    with pytest.raises(JuniperDataContractError, match=r"^target_dt_test has negative horizons$"):
        validate_npz_contract(arrays)


def test_zero_target_dt_is_allowed():
    """The rule is ``>= 0``: a zero horizon is degenerate, not a contract breach."""
    arrays = _full_sequence()
    arrays["target_dt_test"][:] = 0.0
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE


# --- seq_lengths: (W,), integer, in [1, L] ---------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.bool_], ids=["float32", "float64", "bool"])
def test_seq_lengths_non_integer_dtype_raises(dtype):
    """Whole-number floats are refused too: the rule is the dtype, not the values."""
    arrays = _full_sequence()
    arrays["seq_lengths_train"] = arrays["seq_lengths_train"].astype(dtype)
    with pytest.raises(JuniperDataContractError, match=r"^seq_lengths_train must be an integer dtype, got "):
        validate_npz_contract(arrays)


@pytest.mark.parametrize("value", [0, -3])
def test_seq_lengths_below_one_raises(value):
    arrays = _full_sequence()
    lookback = arrays["X_train"].shape[1]
    arrays["seq_lengths_train"][0] = value
    with pytest.raises(JuniperDataContractError, match=re.escape(f"seq_lengths_train has values below 1 (must be in [1, {lookback}])")):
        validate_npz_contract(arrays)


def test_seq_lengths_above_window_length_raises():
    """``L`` is the matching ``X``'s window length, so ``L + 1`` steps cannot fit."""
    arrays = _full_sequence()
    lookback = arrays["X_train"].shape[1]
    arrays["seq_lengths_train"][0] = lookback + 1
    with pytest.raises(JuniperDataContractError, match=re.escape(f"seq_lengths_train has values above the window length {lookback} (must be in [1, {lookback}])")):
        validate_npz_contract(arrays)


def test_seq_lengths_wrong_shape_raises():
    arrays = _full_sequence()
    w = arrays["X_train"].shape[0]
    arrays["seq_lengths_train"] = arrays["seq_lengths_train"].reshape(w, 1)
    with pytest.raises(JuniperDataContractError, match=re.escape(f"seq_lengths_train shape {(w, 1)} != {(w,)}")):
        validate_npz_contract(arrays)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8, np.uint64])
def test_seq_lengths_of_any_integer_dtype_validates_at_both_bounds(dtype):
    arrays = _full_sequence()
    lookback = arrays["X_train"].shape[1]
    lengths = arrays["seq_lengths_train"]
    assert lengths.min() == 1 and lengths.max() == lookback  # the builder hits both bounds
    arrays["seq_lengths_train"] = lengths.astype(dtype)
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE


# --- float32 on X / y / y_reg (R2) -----------------------------------------


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("stem", ["X", "y", "y_reg"])
def test_float64_features_or_targets_raise_on_a_sequence_artifact(stem, split):
    """Every partition present is checked, not only ``train``."""
    arrays = _full_sequence()
    key = f"{stem}_{split}"
    arrays[key] = arrays[key].astype(np.float64)
    with pytest.raises(JuniperDataContractError, match=rf"^{key} must be float32, got float64$"):
        validate_npz_contract(arrays)


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("stem", ["X", "y", "y_reg"])
def test_float64_features_or_targets_raise_on_a_flat_artifact(stem, split):
    """``float32`` is the Data Contract's dtype, not a sequence rule: the 2-D path enforces it too."""
    arrays = _full_tabular()
    key = f"{stem}_{split}"
    arrays[key] = arrays[key].astype(np.float64)
    with pytest.raises(JuniperDataContractError, match=rf"^{key} must be float32, got float64$"):
        validate_npz_contract(arrays)


@pytest.mark.parametrize("dtype", [np.float16, np.int64, np.uint8], ids=["float16", "int64", "uint8"])
def test_any_non_float32_dtype_raises(dtype):
    """The rule is "is float32", not "is not float64"."""
    arrays = _full_tabular()
    arrays["X_train"] = arrays["X_train"].astype(dtype)
    with pytest.raises(JuniperDataContractError, match=r"^X_train must be float32, got "):
        validate_npz_contract(arrays)


def test_big_endian_float32_is_still_float32():
    """The rule compares the scalar type, so a non-native byte order is not refused."""
    arrays = _full_sequence()
    arrays["X_train"] = arrays["X_train"].astype(">f4")
    arrays["y_reg_test"] = arrays["y_reg_test"].astype(">f4")
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE


def test_the_dtype_rule_leaves_the_auxiliary_keys_alone():
    """R2 covers ``X`` / ``y`` / ``y_reg`` only.

    ``dt`` / ``target_dt`` / ``t`` in float64, a bool mask and int64 calendar keys
    all still validate: the producer emits those keys in their own dtypes.
    """
    arrays = _full_sequence()
    for split, _ in _SPLIT_WINDOWS:
        for stem in ("dt", "target_dt"):
            arrays[f"{stem}_{split}"] = arrays[f"{stem}_{split}"].astype(np.float64)
        arrays[f"observed_mask_{split}"] = arrays[f"observed_mask_{split}"].astype(np.bool_)
        for stem in ("date", "window_end_date", "ticker_code"):
            arrays[f"{stem}_{split}"] = arrays[f"{stem}_{split}"].astype(np.int64)
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE


def test_a_legacy_full_pair_is_tolerated_and_not_checked():
    """Decision 11: tolerate ``*_full``, never require it, never forbid it.

    ``full`` is not a partition (``NPZ_SPLITS`` omits it), so no W1.4 rule reaches
    it: a float64 legacy pair passes on either path, as does a ``dt_full`` the
    sequence rules would refuse.
    """
    flat = _full_tabular()
    flat["X_full"] = np.concatenate([flat[f"X_{s}"] for s in ("train", "val", "test")]).astype(np.float64)
    flat["y_full"] = np.concatenate([flat[f"y_{s}"] for s in ("train", "val", "test")]).astype(np.float64)
    assert validate_npz_contract(flat) == CONTRACT_KIND_TABULAR

    seq = _full_sequence()
    seq["X_full"] = np.concatenate([seq[f"X_{s}"] for s, _ in _SPLIT_WINDOWS]).astype(np.float64)
    seq["y_full"] = np.concatenate([seq[f"y_{s}"] for s, _ in _SPLIT_WINDOWS]).astype(np.float64)
    seq["dt_full"] = np.full(seq["X_full"].shape[:2], np.nan, dtype=np.float32)
    assert validate_npz_contract(seq) == CONTRACT_KIND_SEQUENCE
