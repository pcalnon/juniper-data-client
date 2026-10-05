"""NPZ contract validation for tabular and 3-D sequence dataset artifacts.

WS-1 (juniper-data#168) adds an additive, ``X.ndim``-dispatched NPZ contract: a
2-D ``X`` is the legacy tabular artifact, while a 3-D ``X``
``(W, L, F)`` is a time-series / irregular-Δt sequence artifact carrying a
per-step ``dt`` (or absolute ``t``) channel plus optional ``observed_mask`` /
``padding_mask``.

``validate_npz_contract`` classifies a loaded artifact as ``"tabular"`` or
``"sequence"``. Either kind must carry ``float32`` ``X`` / ``y`` / ``y_reg`` in
every partition it has, the Data Contract's dtype; past that check the 2-D path
returns immediately. A 3-D artifact must also satisfy the sequence rules: at
least one of ``t`` / ``dt``; ``dt`` finite and ``>= 0`` with ``dt[:, 0] == 0``;
consistent ``t`` / ``dt``; ``target_dt`` (if present) ``(W,)``, finite and
``>= 0``; ``seq_lengths`` (if present) ``(W,)``, integer and within ``[1, L]``;
binary masks of the right shape; and ``observed_mask`` only meaningful where
``padding_mask == 1``. Violations raise
:class:`~juniper_data_client.exceptions.JuniperDataContractError`, a
``ValueError`` subclass inside the package hierarchy (``APD-DCLIENT-002``).

References: ``juniper-ml/notes/JUNIPER_2026-06-05_JUNIPER-RECURRENCE_RECURSE-DELTA-T-HANDLING.md``
§6 (the sequence rules); ``juniper-ml/notes/JUNIPER_2026-10-03_JUNIPER-RECURRENCE_EQUITIES-END-TO-END-AUDIT-AND-DEVELOPMENT-PLAN.md``
W1.4 (findings F-P3 / F-S3: finite ``dt``, ``target_dt``, ``seq_lengths``, and
the ``float32`` rule of ruling R2).

Project: Juniper
Sub-Project: juniper-data-client
Application: JuniperDataClient
Author: Paul Calnon
Version: 0.5.0
License: MIT License
"""

from typing import Dict, Tuple

import numpy as np

from juniper_data_client.constants import (
    CONTRACT_KIND_SEQUENCE,
    CONTRACT_KIND_TABULAR,
    DEFAULT_ARRAY_DTYPE,
    NPZ_KEY_DT,
    NPZ_KEY_OBSERVED_MASK,
    NPZ_KEY_PADDING_MASK,
    NPZ_KEY_SEQ_LENGTHS,
    NPZ_KEY_T,
    NPZ_KEY_TARGET_DT,
    NPZ_KEY_X,
    NPZ_KEY_Y,
    NPZ_KEY_Y_REG,
    NPZ_SPLITS,
    ContractKind,
)
from juniper_data_client.exceptions import JuniperDataContractError

# The Data Contract's dtype for features and targets ("Dtype: float32"). Compared by
# scalar type, not dtype equality, so a big-endian ``>f4`` array -- float32 all the
# same -- is not refused for its byte order.
_CONTRACT_FLOAT_TYPE = np.dtype(DEFAULT_ARRAY_DTYPE).type

# The per-partition stems the dtype rule covers, and only these. ``dt`` / ``t`` /
# ``target_dt``, the masks, ``date`` / ``window_end_date`` / ``ticker_code`` and
# ``ticker_vocab`` carry time, flags, calendar integers or labels, and the producer
# emits each in the dtype that suits it (``observed_mask`` is ``uint8``, ``date`` is
# ``int32``), so a float32 rule there would refuse conforming artifacts.
_FLOAT32_STEMS: Tuple[str, ...] = (NPZ_KEY_X, NPZ_KEY_Y, NPZ_KEY_Y_REG)


def validate_npz_contract(arrays: Dict[str, np.ndarray], *, dt_atol: float = 1e-6) -> ContractKind:
    """Classify and validate a loaded NPZ artifact's contract.

    Args:
        arrays: NPZ array mapping (e.g. the dict returned by
            :meth:`JuniperDataClient.download_artifact_npz`). An
            ``np.lib.npyio.NpzFile`` also works (it supports ``in`` / ``[]``).
        dt_atol: absolute tolerance for the ``t`` / ``dt`` consistency check.

    Returns:
        ``"tabular"`` for a 2-D ``X`` (legacy path; only the dtype rule applies),
        or ``"sequence"`` for a validated 3-D artifact.

    Raises:
        JuniperDataContractError: if ``X`` is neither 2-D nor 3-D; if an
            ``X`` / ``y`` / ``y_reg`` partition is not ``float32``; or if any
            3-D sequence rule is violated (missing ``t`` / ``dt``; bad ``dt``
            shape, finiteness, sign, or ``dt[:, 0]``; inconsistent ``t`` /
            ``dt``; a mis-shaped, non-finite or negative ``target_dt``; a
            mis-shaped, non-integer or out-of-range ``seq_lengths``; a
            non-binary or mis-shaped mask; or ``observed_mask`` set on a
            padded step). The class subclasses both ``JuniperDataClientError``
            and ``ValueError``, so call sites written against the original
            ``Raises: ValueError`` contract are unaffected.
    """
    # Dispatch on X_train, not X_full. X_full was the rank probe until decision 11
    # removed it from the contract; X_train is present in every artifact by definition,
    # so this needs no fallback and does not consult a key the producer has stopped
    # emitting. A legacy artifact carrying X_full still classifies identically -- the
    # two never disagreed about rank.
    x = arrays[f"{NPZ_KEY_X}_train"]
    if x.ndim not in (2, 3):
        raise JuniperDataContractError(f"X must be 2-D (tabular) or 3-D (sequence), got {x.ndim}-D")
    _validate_float32(arrays)
    if x.ndim == 2:
        return CONTRACT_KIND_TABULAR

    for split in NPZ_SPLITS:
        x_key = f"{NPZ_KEY_X}_{split}"
        if x_key in arrays:
            xs = arrays[x_key]
            _validate_sequence_split(arrays, split, int(xs.shape[0]), int(xs.shape[1]), dt_atol)
    return CONTRACT_KIND_SEQUENCE


def _validate_float32(arrays: Dict[str, np.ndarray]) -> None:
    """``X`` / ``y`` / ``y_reg`` must be ``float32`` in every partition present.

    Ruling R2 of the W1.4 plan: this applies the plan's recommended R2 pending the
    owner's ruling. The alternative is documented dtype tolerance, which would remove
    this call and document the tolerance instead.

    Presence-conditional like every rule here: a partition or target the artifact
    does not carry is skipped, never required. A legacy ``*_full`` pair is not a
    partition (``NPZ_SPLITS`` omits it), so it is neither checked nor required --
    decision 11's "tolerate, never require".
    """
    for split in NPZ_SPLITS:
        for stem in _FLOAT32_STEMS:
            key = f"{stem}_{split}"
            if key in arrays:
                dtype = arrays[key].dtype
                if dtype.type is not _CONTRACT_FLOAT_TYPE:
                    raise JuniperDataContractError(f"{key} must be {DEFAULT_ARRAY_DTYPE}, got {dtype}")


def _validate_sequence_split(arrays: Dict[str, np.ndarray], split: str, n_windows: int, lookback: int, dt_atol: float) -> None:
    """Enforce the 3-D sequence rules for one split's keys."""
    t_key = f"{NPZ_KEY_T}_{split}"
    dt_key = f"{NPZ_KEY_DT}_{split}"
    has_t = t_key in arrays
    has_dt = dt_key in arrays
    if not (has_t or has_dt):
        raise JuniperDataContractError(f"{split}: a 3-D artifact needs at least one of {t_key!r} / {dt_key!r}")
    if has_dt:
        _validate_dt(arrays[dt_key], n_windows, lookback, dt_key)
    if has_t and has_dt:
        _validate_t_dt_consistency(arrays[t_key], arrays[dt_key], split, dt_atol)
    target_dt_key = f"{NPZ_KEY_TARGET_DT}_{split}"
    if target_dt_key in arrays:
        _validate_target_dt(arrays[target_dt_key], n_windows, target_dt_key)
    seq_lengths_key = f"{NPZ_KEY_SEQ_LENGTHS}_{split}"
    if seq_lengths_key in arrays:
        _validate_seq_lengths(arrays[seq_lengths_key], n_windows, lookback, seq_lengths_key)
    _validate_masks(arrays, split, n_windows, lookback)


def _validate_dt(dt: np.ndarray, n_windows: int, lookback: int, dt_key: str) -> None:
    """``dt`` must be ``(W, L)``, finite, non-negative, with a zero first column."""
    if dt.shape != (n_windows, lookback):
        raise JuniperDataContractError(f"{dt_key} shape {dt.shape} != {(n_windows, lookback)}")
    # Finiteness before sign: NaN compares False against 0 and +inf is not negative, so
    # neither trips the sign test, while -inf would be misreported as a negative gap.
    if not np.isfinite(dt).all():
        raise JuniperDataContractError(f"{dt_key} has non-finite values")
    if np.any(dt < 0):
        raise JuniperDataContractError(f"{dt_key} has negative gaps")
    if n_windows and np.any(dt[:, 0] != 0):
        raise JuniperDataContractError(f"{dt_key}[:, 0] must be 0 by convention")


def _validate_t_dt_consistency(t: np.ndarray, dt: np.ndarray, split: str, dt_atol: float) -> None:
    """When both ``t`` and ``dt`` are present they must agree to tolerance."""
    recon = np.zeros_like(t)
    recon[:, 1:] = np.diff(t, axis=1)
    if not np.allclose(recon, dt, atol=dt_atol):
        raise JuniperDataContractError(f"{split}: t_ and dt_ are inconsistent")


def _validate_target_dt(target_dt: np.ndarray, n_windows: int, target_dt_key: str) -> None:
    """``target_dt`` must be ``(W,)`` (one forecast horizon per window), finite and non-negative."""
    if target_dt.shape != (n_windows,):
        raise JuniperDataContractError(f"{target_dt_key} shape {target_dt.shape} != {(n_windows,)}")
    # Finiteness before sign, as for ``dt``: -inf is non-finite, not merely negative.
    if not np.isfinite(target_dt).all():
        raise JuniperDataContractError(f"{target_dt_key} has non-finite values")
    if np.any(target_dt < 0):
        raise JuniperDataContractError(f"{target_dt_key} has negative horizons")


def _validate_seq_lengths(seq_lengths: np.ndarray, n_windows: int, lookback: int, seq_lengths_key: str) -> None:
    """``seq_lengths`` must be ``(W,)``, an integer dtype, and every value within ``[1, L]``."""
    if seq_lengths.shape != (n_windows,):
        raise JuniperDataContractError(f"{seq_lengths_key} shape {seq_lengths.shape} != {(n_windows,)}")
    # ``np.integer`` covers signed and unsigned ints and excludes ``bool``, which numpy
    # does not class as an integer: a True/False "length" is a mask, not a step count.
    if not np.issubdtype(seq_lengths.dtype, np.integer):
        raise JuniperDataContractError(f"{seq_lengths_key} must be an integer dtype, got {seq_lengths.dtype}")
    # ``np.any`` rather than ``min()`` / ``max()``, which raise on an empty partition.
    if np.any(seq_lengths < 1):
        raise JuniperDataContractError(f"{seq_lengths_key} has values below 1 (must be in [1, {lookback}])")
    if np.any(seq_lengths > lookback):
        raise JuniperDataContractError(f"{seq_lengths_key} has values above the window length {lookback} (must be in [1, {lookback}])")


def _validate_masks(arrays: Dict[str, np.ndarray], split: str, n_windows: int, lookback: int) -> None:
    """Masks (if present) must be binary, correctly shaped, and consistent."""
    for mask_key in (f"{NPZ_KEY_OBSERVED_MASK}_{split}", f"{NPZ_KEY_PADDING_MASK}_{split}"):
        if mask_key in arrays:
            mask = arrays[mask_key]
            if mask.shape != (n_windows, lookback):
                raise JuniperDataContractError(f"{mask_key} shape {mask.shape} != {(n_windows, lookback)}")
            if not np.isin(mask, (0, 1)).all():
                raise JuniperDataContractError(f"{mask_key} must be binary (0/1)")

    observed_key = f"{NPZ_KEY_OBSERVED_MASK}_{split}"
    padding_key = f"{NPZ_KEY_PADDING_MASK}_{split}"
    if observed_key in arrays and padding_key in arrays:
        observed = arrays[observed_key]
        padding = arrays[padding_key]
        if np.any((padding == 0) & (observed == 1)):
            raise JuniperDataContractError(f"{split}: observed_mask=1 on a padded (padding_mask=0) step")
