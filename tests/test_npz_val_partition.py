"""Complementary pins for the three-way train/val/test partition (PR #187).

#187 already updates ``tests/test_fake_client.py`` and
``tests/test_fake_client_batch.py`` to require ``X_val`` / ``y_val`` keys,
``n_val`` in FakeDataClient metadata, and
``n_full == n_train + n_val + n_test`` with an ``n_val > 0`` guard. Those
cannot see:

- Contiguous, index-disjoint reconstruction: a count-sum still passes if
  ``X_val`` is a copy of the first ``n_val`` train rows.
- The exact default carve ``int(n * train_ratio)`` / ``int(n * 0.1)`` /
  remainder. The existing ~80% train band still passes a 5/15 val/test split.
- ``y_val`` one-hot encoding (the existing loop lists ``y_train`` /
  ``y_test`` / ``y_full`` only).
- ``validate_npz_contract`` now iterates ``\"val\"``: a sequence artifact
  that carries ``X_val`` is subject to the same ``t`` / ``dt`` / mask rules.
  Dropping ``\"val\"`` from ``NPZ_SPLITS`` silently skips a negative
  ``dt_val``.
- ``_split_dataset`` edges the public generators never hit: small-n val
  truncation to empty, ``train_ratio + val_ratio > 1`` emptying test, and a
  caller-supplied ``val_ratio``.
"""

from typing import Callable, Dict

import numpy as np
import pytest

from juniper_data_client import JuniperDataContractError, validate_npz_contract
from juniper_data_client.constants import CONTRACT_KIND_SEQUENCE, FAKE_VAL_RATIO_DEFAULT, NPZ_SPLITS
from juniper_data_client.testing.generators import (
    _split_dataset,
    generate_circle,
    generate_moon,
    generate_spiral,
    generate_xor,
)


def _toy_split(n: int, train_ratio: float, val_ratio: float = FAKE_VAL_RATIO_DEFAULT, seed: int = 0) -> Dict[str, np.ndarray]:
    """Unique-row features so reconstruction / emptiness can be asserted exactly."""
    x = np.arange(n, dtype=np.float32).reshape(n, 1)
    y = np.zeros((n, 2), dtype=np.float32)
    y[:, 0] = 1.0
    return _split_dataset(x, y, train_ratio, np.random.default_rng(seed), val_ratio)


def _sequence_with_val() -> Dict[str, np.ndarray]:
    """Valid 3-D sequence artifact that *does* carry the new val split."""
    arrays: Dict[str, np.ndarray] = {}
    gap_row = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    for split, n in (("train", 4), ("val", 2), ("test", 2), ("full", 8)):
        arrays[f"X_{split}"] = np.zeros((n, 3, 2), np.float32)
        arrays[f"dt_{split}"] = np.tile(gap_row, (n, 1))
    return arrays


@pytest.mark.unit
class TestPublishedSplitConstants:
    """``NPZ_SPLITS`` / ``FAKE_VAL_RATIO_DEFAULT`` are the published contract."""

    def test_npz_splits_is_train_val_test_full(self) -> None:
        assert NPZ_SPLITS == ("train", "val", "test", "full")

    def test_fake_val_ratio_default_is_one_tenth(self) -> None:
        assert FAKE_VAL_RATIO_DEFAULT == 0.1


@pytest.mark.unit
class TestPartitionReconstruction:
    """The three partitions are contiguous blocks of the shuffled full array.

    A length identity cannot catch a val slice that overlaps train or is a
    copy of test. ``vstack`` equality is the property the real generators
    rely on and the claim #187's changelog makes.
    """

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: generate_spiral(n_spirals=2, n_points_per_spiral=100, seed=42),
            lambda: generate_xor(n_points=100, seed=42),
            lambda: generate_circle(n_points=200, seed=42),
            lambda: generate_moon(n_points=200, seed=42),
        ],
        ids=("spiral", "xor", "circle", "moon"),
    )
    def test_vstack_of_three_partitions_equals_full(self, factory: Callable[[], Dict[str, np.ndarray]]) -> None:
        arrays = factory()
        np.testing.assert_array_equal(
            np.vstack((arrays["X_train"], arrays["X_val"], arrays["X_test"])),
            arrays["X_full"],
        )
        np.testing.assert_array_equal(
            np.vstack((arrays["y_train"], arrays["y_val"], arrays["y_test"])),
            arrays["y_full"],
        )

    def test_unique_rows_are_index_disjoint(self) -> None:
        """With unique feature rows, the three partitions share no sample."""
        arrays = _toy_split(n=40, train_ratio=0.8)
        train = set(arrays["X_train"].ravel().tolist())
        val = set(arrays["X_val"].ravel().tolist())
        test = set(arrays["X_test"].ravel().tolist())
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)
        assert train | val | test == set(arrays["X_full"].ravel().tolist())


@pytest.mark.unit
class TestDefaultCarve:
    """Default 0.8 / 0.1 / remainder sizes — not just 'train is about 80%'."""

    def test_spiral_200_is_160_20_20(self) -> None:
        arrays = generate_spiral(n_spirals=2, n_points_per_spiral=100, seed=7)
        assert arrays["X_train"].shape[0] == 160
        assert arrays["X_val"].shape[0] == 20
        assert arrays["X_test"].shape[0] == 20

    def test_xor_100_is_80_10_10(self) -> None:
        arrays = generate_xor(n_points=100, seed=7)
        assert arrays["X_train"].shape[0] == 80
        assert arrays["X_val"].shape[0] == 10
        assert arrays["X_test"].shape[0] == 10

    def test_y_val_is_one_hot(self) -> None:
        """The existing one-hot loop never visits ``y_val``."""
        arrays = generate_spiral(n_spirals=2, n_points_per_spiral=50, seed=3)
        y_val = arrays["y_val"]
        assert y_val.size > 0
        np.testing.assert_allclose(y_val.sum(axis=1), np.ones(y_val.shape[0], dtype=np.float32), atol=1e-6)
        assert set(np.unique(y_val).tolist()) <= {0.0, 1.0}


@pytest.mark.unit
class TestSplitDatasetEdges:
    """Ratio edges the public generators do not exercise."""

    def test_small_n_truncates_val_to_empty(self) -> None:
        """``int(9 * 0.1) == 0`` — val vanishes; test takes the remainder."""
        arrays = _toy_split(n=9, train_ratio=0.8)
        assert arrays["X_train"].shape[0] == 7
        assert arrays["X_val"].shape[0] == 0
        assert arrays["X_test"].shape[0] == 2
        assert arrays["y_val"].shape[0] == 0
        np.testing.assert_array_equal(
            np.vstack((arrays["X_train"], arrays["X_test"])),
            arrays["X_full"],
        )

    def test_train_plus_val_exceeding_one_empties_test(self) -> None:
        """No clamp: ``0.95 + 0.1`` on n=20 cuts test to empty, not an error."""
        arrays = _toy_split(n=20, train_ratio=0.95)
        assert arrays["X_train"].shape[0] == 19
        assert arrays["X_val"].shape[0] == 1
        assert arrays["X_test"].shape[0] == 0
        assert arrays["y_test"].shape[0] == 0
        np.testing.assert_array_equal(
            np.vstack((arrays["X_train"], arrays["X_val"])),
            arrays["X_full"],
        )

    def test_custom_val_ratio_is_honoured(self) -> None:
        arrays = _toy_split(n=200, train_ratio=0.5, val_ratio=0.25)
        assert arrays["X_train"].shape[0] == 100
        assert arrays["X_val"].shape[0] == 50
        assert arrays["X_test"].shape[0] == 50


@pytest.mark.unit
class TestContractIteratesVal:
    """Adding ``\"val\"`` to ``NPZ_SPLITS`` extends the sequence validator."""

    def test_sequence_with_val_split_validates(self) -> None:
        assert validate_npz_contract(_sequence_with_val()) == CONTRACT_KIND_SEQUENCE

    def test_x_val_without_t_or_dt_raises(self) -> None:
        arrays = _sequence_with_val()
        del arrays["dt_val"]
        with pytest.raises(JuniperDataContractError, match="at least one"):
            validate_npz_contract(arrays)

    def test_negative_dt_val_raises(self) -> None:
        """If ``\"val\"`` is dropped from ``NPZ_SPLITS`` this invalid channel is skipped."""
        arrays = _sequence_with_val()
        arrays["dt_val"][0, 1] = -1.0
        with pytest.raises(JuniperDataContractError, match="negative"):
            validate_npz_contract(arrays)
