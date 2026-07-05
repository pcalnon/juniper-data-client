"""NPZ contract validation for tabular and 3-D sequence dataset artifacts.

WS-1 (juniper-data#168) adds an additive, ``X.ndim``-dispatched NPZ contract: a
2-D ``X`` is the legacy tabular artifact (unchanged), while a 3-D ``X``
``(W, L, F)`` is a time-series / irregular-Δt sequence artifact carrying a
per-step ``dt`` (or absolute ``t``) channel plus optional ``observed_mask`` /
``padding_mask``.

``validate_npz_contract`` classifies a loaded artifact as ``"tabular"`` or
``"sequence"`` and enforces the sequence rules: at least one of ``t`` / ``dt``;
``dt >= 0`` with ``dt[:, 0] == 0``; consistent ``t`` / ``dt``; binary masks of the
right shape; and ``observed_mask`` only meaningful where ``padding_mask == 1``.
The 2-D path is untouched and returns immediately.

Reference: ``juniper-ml/notes/JUNIPER_2026-06-05_JUNIPER-RECURRENCE_RECURSE-DELTA-T-HANDLING.md`` §6.

Project: Juniper
Sub-Project: juniper-data-client
Application: JuniperDataClient
Author: Paul Calnon
Version: 0.4.1
License: MIT License
"""

from typing import Dict

import numpy as np

from juniper_data_client.constants import (
    CONTRACT_KIND_SEQUENCE,
    CONTRACT_KIND_TABULAR,
    NPZ_KEY_DT,
    NPZ_KEY_OBSERVED_MASK,
    NPZ_KEY_PADDING_MASK,
    NPZ_KEY_T,
    NPZ_KEY_X,
    NPZ_SPLITS,
)


def validate_npz_contract(arrays: Dict[str, np.ndarray], *, dt_atol: float = 1e-6) -> str:
    """Classify and validate a loaded NPZ artifact's contract.

    Args:
        arrays: NPZ array mapping (e.g. the dict returned by
            :meth:`JuniperDataClient.download_artifact_npz`). An
            ``np.lib.npyio.NpzFile`` also works (it supports ``in`` / ``[]``).
        dt_atol: absolute tolerance for the ``t`` / ``dt`` consistency check.

    Returns:
        ``"tabular"`` for a 2-D ``X`` (legacy path, no further checks), or
        ``"sequence"`` for a validated 3-D artifact.

    Raises:
        ValueError: if ``X`` is neither 2-D nor 3-D, or any 3-D sequence rule is
            violated (missing ``t`` / ``dt``; bad ``dt`` shape, sign, or
            ``dt[:, 0]``; inconsistent ``t`` / ``dt``; a non-binary or mis-shaped
            mask; or ``observed_mask`` set on a padded step).
    """
    x_full_key = f"{NPZ_KEY_X}_full"
    x_train_key = f"{NPZ_KEY_X}_train"
    x = arrays[x_full_key] if x_full_key in arrays else arrays[x_train_key]
    if x.ndim == 2:
        return CONTRACT_KIND_TABULAR
    if x.ndim != 3:
        raise ValueError(f"X must be 2-D (tabular) or 3-D (sequence), got {x.ndim}-D")

    for split in NPZ_SPLITS:
        x_key = f"{NPZ_KEY_X}_{split}"
        if x_key in arrays:
            xs = arrays[x_key]
            _validate_sequence_split(arrays, split, int(xs.shape[0]), int(xs.shape[1]), dt_atol)
    return CONTRACT_KIND_SEQUENCE


def _validate_sequence_split(arrays: Dict[str, np.ndarray], split: str, n_windows: int, lookback: int, dt_atol: float) -> None:
    """Enforce the 3-D sequence rules for one split's keys."""
    t_key = f"{NPZ_KEY_T}_{split}"
    dt_key = f"{NPZ_KEY_DT}_{split}"
    has_t = t_key in arrays
    has_dt = dt_key in arrays
    if not (has_t or has_dt):
        raise ValueError(f"{split}: a 3-D artifact needs at least one of {t_key!r} / {dt_key!r}")
    if has_dt:
        _validate_dt(arrays[dt_key], n_windows, lookback, dt_key)
    if has_t and has_dt:
        _validate_t_dt_consistency(arrays[t_key], arrays[dt_key], split, dt_atol)
    _validate_masks(arrays, split, n_windows, lookback)


def _validate_dt(dt: np.ndarray, n_windows: int, lookback: int, dt_key: str) -> None:
    """``dt`` must be ``(W, L)``, non-negative, with a zero first column."""
    if dt.shape != (n_windows, lookback):
        raise ValueError(f"{dt_key} shape {dt.shape} != {(n_windows, lookback)}")
    if np.any(dt < 0):
        raise ValueError(f"{dt_key} has negative gaps")
    if n_windows and np.any(dt[:, 0] != 0):
        raise ValueError(f"{dt_key}[:, 0] must be 0 by convention")


def _validate_t_dt_consistency(t: np.ndarray, dt: np.ndarray, split: str, dt_atol: float) -> None:
    """When both ``t`` and ``dt`` are present they must agree to tolerance."""
    recon = np.zeros_like(t)
    recon[:, 1:] = np.diff(t, axis=1)
    if not np.allclose(recon, dt, atol=dt_atol):
        raise ValueError(f"{split}: t_ and dt_ are inconsistent")


def _validate_masks(arrays: Dict[str, np.ndarray], split: str, n_windows: int, lookback: int) -> None:
    """Masks (if present) must be binary, correctly shaped, and consistent."""
    for mask_key in (f"{NPZ_KEY_OBSERVED_MASK}_{split}", f"{NPZ_KEY_PADDING_MASK}_{split}"):
        if mask_key in arrays:
            mask = arrays[mask_key]
            if mask.shape != (n_windows, lookback):
                raise ValueError(f"{mask_key} shape {mask.shape} != {(n_windows, lookback)}")
            if not np.isin(mask, (0, 1)).all():
                raise ValueError(f"{mask_key} must be binary (0/1)")

    observed_key = f"{NPZ_KEY_OBSERVED_MASK}_{split}"
    padding_key = f"{NPZ_KEY_PADDING_MASK}_{split}"
    if observed_key in arrays and padding_key in arrays:
        observed = arrays[observed_key]
        padding = arrays[padding_key]
        if np.any((padding == 0) & (observed == 1)):
            raise ValueError(f"{split}: observed_mask=1 on a padded (padding_mask=0) step")
