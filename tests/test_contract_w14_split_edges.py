"""Complementary pins for the W1.4 contract rules (PR #222).

``tests/test_contract.py`` already rejects non-finite ``dt``, a bad ``target_dt``,
a bad ``seq_lengths``, and a non-float32 ``X`` / ``y`` / ``y_reg``. Those
artifacts share one lookback, and every new rejection carries both ``t`` and
``dt`` on every split, so they cannot see:

- ``seq_lengths``, ``dt`` and the masks are bounded by **that split's** window
  length. A validator that reused ``X_train``'s length would still pass.
- ``target_dt`` and ``seq_lengths`` are not gated on which time channel is
  present. Nesting them under ``if has_dt`` (or ``if has_t``) would still pass.
- An optional key is presence-conditional per split. Requiring it on every
  split once any split carries it would still pass.
- A zero-window partition is not exempt from the float32 rule. Skipping empty
  arrays would still pass the non-empty dtype tests.
- The dtype sweep is reported before sequence rules. A non-finite ``dt`` on an
  earlier split must not hide a wrong dtype on a later one.
"""

import re

import numpy as np
import pytest

from juniper_data_client import JuniperDataContractError, validate_npz_contract
from juniper_data_client.constants import CONTRACT_KIND_SEQUENCE

# (split, n_windows, lookback). The three lookbacks differ on purpose: a rule
# that read one split's window length for another cannot satisfy all three.
_SPLITS = (("train", 3, 4), ("val", 2, 6), ("test", 2, 3))


def _artifact(splits=_SPLITS):
    """A conforming sequence artifact whose splits do not share a window length."""
    arrays = {}
    for split, windows, lookback in splits:
        gap = np.zeros(lookback, np.float32)
        if lookback > 1:
            gap[1:] = 1.0
        dt = np.tile(gap, (windows, 1))
        padding = np.ones((windows, lookback), np.uint8)
        arrays[f"X_{split}"] = np.zeros((windows, lookback, 2), np.float32)
        arrays[f"y_{split}"] = np.zeros((windows, 2), np.float32)
        arrays[f"y_reg_{split}"] = np.zeros((windows, 1), np.float32)
        arrays[f"dt_{split}"] = dt
        arrays[f"t_{split}"] = np.cumsum(dt, axis=1).astype(np.float64)
        arrays[f"target_dt_{split}"] = np.ones(windows, np.float32)
        arrays[f"seq_lengths_{split}"] = np.full(windows, lookback, np.int64)
        arrays[f"padding_mask_{split}"] = padding
        arrays[f"observed_mask_{split}"] = padding.copy()
    return arrays


def test_each_split_is_bounded_by_its_own_window_length():
    """Val may be longer than train, and test shorter, in the same artifact.

    ``seq_lengths_val == 6`` is legal only against val's own ``X``. ``dt_val``
    and both masks are wider than train's, so a shared train lookback fails
    this artifact before the rejection below is reached. The rejection is a
    length that train would allow (4) and test must not: taking the longest
    window in the file (val's 6) as every split's cap would let it through.
    """
    arrays = _artifact()
    assert arrays["X_train"].shape[1] == 4
    assert arrays["dt_val"].shape == (2, 6)
    assert arrays["padding_mask_val"].shape[1] == 6
    assert arrays["observed_mask_test"].shape[1] == 3
    assert int(arrays["seq_lengths_val"].max()) == 6
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE

    arrays["seq_lengths_test"][0] = 4
    with pytest.raises(
        JuniperDataContractError,
        match=re.escape("seq_lengths_test has values above the window length 3 (must be in [1, 3])"),
    ):
        validate_npz_contract(arrays)


def test_horizon_and_length_rules_apply_when_dt_is_absent():
    """``t`` alone still validates, and a present ``target_dt`` is still enforced."""
    arrays = _artifact()
    for split, _, _ in _SPLITS:
        del arrays[f"dt_{split}"]
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE

    arrays["target_dt_val"][0] = -1.0
    with pytest.raises(JuniperDataContractError, match=r"^target_dt_val has negative horizons$"):
        validate_npz_contract(arrays)


def test_horizon_and_length_rules_apply_when_t_is_absent():
    """``dt`` alone still validates, and a present ``seq_lengths`` is still enforced."""
    arrays = _artifact()
    for split, _, _ in _SPLITS:
        del arrays[f"t_{split}"]
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE

    arrays["seq_lengths_val"][0] = 0
    with pytest.raises(
        JuniperDataContractError,
        match=re.escape("seq_lengths_val has values below 1 (must be in [1, 6])"),
    ):
        validate_npz_contract(arrays)


def test_optional_keys_are_independent_per_split():
    """Missing on one split is legal; the same key on another split is still checked.

    Val has no ``target_dt`` while train and test do, and test has no
    ``seq_lengths`` while val does. Requiring a key on every split once any
    split carries it rejects this artifact. Gating ``seq_lengths`` on the
    sibling ``target_dt`` being present would then ignore the zero length.
    """
    arrays = _artifact()
    del arrays["target_dt_val"]
    del arrays["seq_lengths_test"]
    assert validate_npz_contract(arrays) == CONTRACT_KIND_SEQUENCE

    arrays["seq_lengths_val"][0] = 0
    with pytest.raises(
        JuniperDataContractError,
        match=re.escape("seq_lengths_val has values below 1 (must be in [1, 6])"),
    ):
        validate_npz_contract(arrays)


@pytest.mark.parametrize("stem", ["X", "y", "y_reg"])
def test_empty_partition_is_not_exempt_from_float32(stem):
    """A zero-window partition has no values and still has a dtype."""
    splits = (("train", 3, 4), ("val", 0, 6), ("test", 2, 3))
    arrays = _artifact(splits)
    key = f"{stem}_val"
    assert arrays[key].shape[0] == 0
    arrays[key] = arrays[key].astype(np.float64)
    with pytest.raises(JuniperDataContractError, match=rf"^{key} must be float32, got float64$"):
        validate_npz_contract(arrays)


def test_a_later_splits_dtype_is_reported_ahead_of_an_earlier_sequence_violation():
    """Float32 is checked for every partition before any sequence rule runs.

    ``dt_train`` is non-finite and ``y_test`` is float64. Sequence-rules-first
    would report the train gap and never reach the test dtype.
    """
    arrays = _artifact()
    arrays["dt_train"][0, 1] = np.nan
    arrays["y_test"] = arrays["y_test"].astype(np.float64)
    with pytest.raises(JuniperDataContractError, match=r"^y_test must be float32, got float64$"):
        validate_npz_contract(arrays)
