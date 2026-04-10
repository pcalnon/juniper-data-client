#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   juniper-data-client
# Application:   juniper_data_client
# File Name:     generators.py
# Author:        Paul Calnon
# Version:       0.4.0
#
# Date Created:  2026-02-27
# Last Modified: 2026-02-27
#
# License:       MIT License
# Copyright:     Copyright (c) 2024-2026 Paul Calnon
#
# Description:
#    Synthetic dataset generators for FakeDataClient testing.
#    Produces numpy arrays matching the NPZ data contract used by
#    the real JuniperData service: X_train, y_train, X_test, y_test,
#    X_full, y_full — all float32, with one-hot encoded labels.
#####################################################################################################################################################################################################

"""Synthetic dataset generators for testing without a live JuniperData service.

Each generator produces a Dict[str, np.ndarray] with the standard NPZ keys:
X_train, y_train, X_test, y_test, X_full, y_full (all float32, one-hot labels).
"""

from typing import Dict, Optional

import numpy as np

from juniper_data_client.constants import (
    CIRCLE_FACTOR_DEFAULT,
    CIRCLE_N_POINTS_DEFAULT,
    CIRCLE_NOISE_DEFAULT,
    CIRCLE_NUM_CLASSES,
    CIRCLE_TRAIN_RATIO_DEFAULT,
    MOON_LOWER_X_OFFSET,
    MOON_LOWER_Y_OFFSET,
    MOON_LOWER_Y_SHIFT,
    MOON_N_POINTS_DEFAULT,
    MOON_NOISE_DEFAULT,
    MOON_NUM_CLASSES,
    MOON_TRAIN_RATIO_DEFAULT,
    SPIRAL_ANGLE_TURNS,
    SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT,
    SPIRAL_N_SPIRALS_DEFAULT,
    SPIRAL_NOISE_DEFAULT,
    SPIRAL_RADIUS_SCALE,
    SPIRAL_TRAIN_RATIO_DEFAULT,
    XOR_CORNER_LABELS,
    XOR_CORNERS,
    XOR_N_POINTS_DEFAULT,
    XOR_NOISE_DEFAULT,
    XOR_NUM_CLASSES,
    XOR_NUM_CORNERS,
    XOR_TRAIN_RATIO_DEFAULT,
)


def _one_hot_encode(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Convert integer class labels to one-hot encoded float32 arrays.

    Args:
        labels: 1-D array of integer class labels.
        n_classes: Total number of classes.

    Returns:
        2-D float32 array of shape (n_samples, n_classes).
    """
    one_hot = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return one_hot


def _split_dataset(
    X: np.ndarray,
    y_one_hot: np.ndarray,
    train_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Shuffle and split features/labels into train/test/full splits.

    Args:
        X: Feature array of shape (n_samples, n_features), float32.
        y_one_hot: One-hot label array of shape (n_samples, n_classes), float32.
        train_ratio: Fraction of samples used for training.
        rng: Numpy random generator for reproducible shuffling.

    Returns:
        Dictionary with keys X_train, y_train, X_test, y_test, X_full, y_full.
    """
    n_samples = X.shape[0]
    indices = rng.permutation(n_samples)
    X = X[indices]
    y_one_hot = y_one_hot[indices]

    split_idx = int(n_samples * train_ratio)

    return {
        "X_train": X[:split_idx].astype(np.float32),
        "y_train": y_one_hot[:split_idx].astype(np.float32),
        "X_test": X[split_idx:].astype(np.float32),
        "y_test": y_one_hot[split_idx:].astype(np.float32),
        "X_full": X.astype(np.float32),
        "y_full": y_one_hot.astype(np.float32),
    }


def generate_spiral(
    n_spirals: int = SPIRAL_N_SPIRALS_DEFAULT,
    n_points_per_spiral: int = SPIRAL_N_POINTS_PER_SPIRAL_DEFAULT,
    noise: float = SPIRAL_NOISE_DEFAULT,
    seed: Optional[int] = None,
    train_ratio: float = SPIRAL_TRAIN_RATIO_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Generate a multi-arm spiral dataset.

    Each spiral arm is a separate class. Points are distributed along Archimedean
    spirals with configurable Gaussian noise applied to both x and y coordinates.

    Args:
        n_spirals: Number of spiral arms / classes (default: 2).
        n_points_per_spiral: Number of points per spiral arm (default: 100).
        noise: Standard deviation of Gaussian noise added to coordinates (default: 0.1).
        seed: Random seed for reproducibility (optional).
        train_ratio: Fraction of data used for training (default: 0.8).

    Returns:
        Dictionary with keys X_train, y_train, X_test, y_test, X_full, y_full.
        All arrays are float32. Labels are one-hot encoded.
    """
    rng = np.random.default_rng(seed)
    n_total = n_spirals * n_points_per_spiral

    X = np.zeros((n_total, 2), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)

    for spiral_idx in range(n_spirals):
        start = spiral_idx * n_points_per_spiral
        end = start + n_points_per_spiral

        # Evenly spaced parameter along the spiral arm
        t = np.linspace(0, 1, n_points_per_spiral)
        # Radius grows linearly; angle spans multiple turns
        radius = t * SPIRAL_RADIUS_SCALE
        angle = t * SPIRAL_ANGLE_TURNS * np.pi + (2.0 * np.pi * spiral_idx / n_spirals)

        X[start:end, 0] = radius * np.cos(angle) + rng.normal(0, noise, n_points_per_spiral).astype(np.float32)
        X[start:end, 1] = radius * np.sin(angle) + rng.normal(0, noise, n_points_per_spiral).astype(np.float32)
        labels[start:end] = spiral_idx

    y_one_hot = _one_hot_encode(labels, n_spirals)
    return _split_dataset(X, y_one_hot, train_ratio, rng)


def generate_xor(
    n_points: int = XOR_N_POINTS_DEFAULT,
    noise: float = XOR_NOISE_DEFAULT,
    seed: Optional[int] = None,
    train_ratio: float = XOR_TRAIN_RATIO_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Generate an XOR dataset.

    The four canonical XOR points (0,0), (0,1), (1,0), (1,1) are each surrounded
    by a cluster of normally-distributed samples. Class 0 covers (0,0) and (1,1);
    class 1 covers (0,1) and (1,0).

    Args:
        n_points: Total number of points (distributed equally among 4 corners) (default: 100).
        noise: Standard deviation of Gaussian noise around each corner (default: 0.1).
        seed: Random seed for reproducibility (optional).
        train_ratio: Fraction of data used for training (default: 0.8).

    Returns:
        Dictionary with keys X_train, y_train, X_test, y_test, X_full, y_full.
        All arrays are float32. Labels are one-hot encoded.
    """
    rng = np.random.default_rng(seed)

    # Four XOR corners with their class labels (sourced from constants)
    corners = np.array(XOR_CORNERS, dtype=np.float32)
    corner_labels = np.array(XOR_CORNER_LABELS, dtype=np.int64)

    points_per_corner = n_points // XOR_NUM_CORNERS
    remainder = n_points - (points_per_corner * XOR_NUM_CORNERS)

    all_X = []
    all_labels = []

    for i, (corner, label) in enumerate(zip(corners, corner_labels)):
        # Distribute remainder points to the first corners
        count = points_per_corner + (1 if i < remainder else 0)
        cluster = corner + rng.normal(0, noise, (count, 2)).astype(np.float32)
        all_X.append(cluster)
        all_labels.append(np.full(count, label, dtype=np.int64))

    X = np.vstack(all_X).astype(np.float32)
    labels = np.concatenate(all_labels)

    y_one_hot = _one_hot_encode(labels, XOR_NUM_CLASSES)
    return _split_dataset(X, y_one_hot, train_ratio, rng)


def generate_circle(
    n_points: int = CIRCLE_N_POINTS_DEFAULT,
    noise: float = CIRCLE_NOISE_DEFAULT,
    factor: float = CIRCLE_FACTOR_DEFAULT,
    seed: Optional[int] = None,
    train_ratio: float = CIRCLE_TRAIN_RATIO_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Generate a concentric circles dataset.

    Two concentric circles of different radii, with Gaussian noise. The inner
    circle is class 0 and the outer circle is class 1.

    Args:
        n_points: Total number of points (split equally between circles) (default: 200).
        noise: Standard deviation of Gaussian noise (default: 0.1).
        factor: Ratio of inner circle radius to outer circle radius (default: 0.5).
        seed: Random seed for reproducibility (optional).
        train_ratio: Fraction of data used for training (default: 0.8).

    Returns:
        Dictionary with keys X_train, y_train, X_test, y_test, X_full, y_full.
        All arrays are float32. Labels are one-hot encoded.
    """
    rng = np.random.default_rng(seed)

    n_outer = n_points // 2
    n_inner = n_points - n_outer

    # Outer circle (class 1)
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    outer_x = np.cos(outer_angles) + rng.normal(0, noise, n_outer)
    outer_y = np.sin(outer_angles) + rng.normal(0, noise, n_outer)

    # Inner circle (class 0)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    inner_x = factor * np.cos(inner_angles) + rng.normal(0, noise, n_inner)
    inner_y = factor * np.sin(inner_angles) + rng.normal(0, noise, n_inner)

    X = np.vstack(
        [
            np.column_stack([inner_x, inner_y]),
            np.column_stack([outer_x, outer_y]),
        ]
    ).astype(np.float32)

    labels = np.concatenate(
        [
            np.zeros(n_inner, dtype=np.int64),
            np.ones(n_outer, dtype=np.int64),
        ]
    )

    y_one_hot = _one_hot_encode(labels, CIRCLE_NUM_CLASSES)
    return _split_dataset(X, y_one_hot, train_ratio, rng)


def generate_moon(
    n_points: int = MOON_N_POINTS_DEFAULT,
    noise: float = MOON_NOISE_DEFAULT,
    seed: Optional[int] = None,
    train_ratio: float = MOON_TRAIN_RATIO_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Generate a two-moons dataset.

    Two interleaving half-circles (moons), each representing a different class.
    The upper moon is class 0 and the lower (shifted) moon is class 1.

    Args:
        n_points: Total number of points (split equally between moons) (default: 200).
        noise: Standard deviation of Gaussian noise (default: 0.1).
        seed: Random seed for reproducibility (optional).
        train_ratio: Fraction of data used for training (default: 0.8).

    Returns:
        Dictionary with keys X_train, y_train, X_test, y_test, X_full, y_full.
        All arrays are float32. Labels are one-hot encoded.
    """
    rng = np.random.default_rng(seed)

    n_upper = n_points // 2
    n_lower = n_points - n_upper

    # Upper moon (class 0) — semicircle from 0 to pi
    upper_angles = np.linspace(0, np.pi, n_upper)
    upper_x = np.cos(upper_angles) + rng.normal(0, noise, n_upper)
    upper_y = np.sin(upper_angles) + rng.normal(0, noise, n_upper)

    # Lower moon (class 1) — semicircle from 0 to pi, shifted right and down
    lower_angles = np.linspace(0, np.pi, n_lower)
    lower_x = MOON_LOWER_X_OFFSET - np.cos(lower_angles) + rng.normal(0, noise, n_lower)
    lower_y = MOON_LOWER_Y_OFFSET - np.sin(lower_angles) - MOON_LOWER_Y_SHIFT + rng.normal(0, noise, n_lower)

    X = np.vstack(
        [
            np.column_stack([upper_x, upper_y]),
            np.column_stack([lower_x, lower_y]),
        ]
    ).astype(np.float32)

    labels = np.concatenate(
        [
            np.zeros(n_upper, dtype=np.int64),
            np.ones(n_lower, dtype=np.int64),
        ]
    )

    y_one_hot = _one_hot_encode(labels, MOON_NUM_CLASSES)
    return _split_dataset(X, y_one_hot, train_ratio, rng)
