"""Eight-dimensional tactile-code extraction."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "total_intensity",
    "peak_pressure",
    "effective_area",
    "centroid_x",
    "centroid_y",
    "spatial_dispersion",
    "left_right_asymmetry",
    "temporal_variation",
)


def extract_tactile_codes(
    adc_counts: np.ndarray,
    contact_fraction: float = 0.20,
) -> np.ndarray:
    """Extract the predefined tactile code from trial-major frame arrays.

    Parameters
    ----------
    adc_counts
        Array with shape ``(trial, frame, row, column)``.
    contact_fraction
        Fraction of the frame peak used only for the effective-area count.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(trial, frame, 8)`` in the order in ``FEATURE_NAMES``.
    """

    frames = np.asarray(adc_counts, dtype=np.float64)
    if frames.ndim != 4:
        raise ValueError("adc_counts must have shape (trial, frame, row, column)")
    if not 0.0 < contact_fraction < 1.0:
        raise ValueError("contact_fraction must be between zero and one")

    _, _, height, width = frames.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float64)
    total = frames.sum(axis=(2, 3))
    peak = frames.max(axis=(2, 3))
    effective_area = (
        frames > contact_fraction * peak[:, :, None, None]
    ).sum(axis=(2, 3)).astype(np.float64)

    safe_total = total + np.finfo(np.float64).eps
    centroid_x = (frames * xx).sum(axis=(2, 3)) / safe_total
    centroid_y = (frames * yy).sum(axis=(2, 3)) / safe_total
    spatial_dispersion = (
        frames
        * (
            (xx[None, None] - centroid_x[:, :, None, None]) ** 2
            + (yy[None, None] - centroid_y[:, :, None, None]) ** 2
        )
    ).sum(axis=(2, 3)) / safe_total

    split_column = width // 2
    left = frames[:, :, :, :split_column].sum(axis=(2, 3))
    right = frames[:, :, :, split_column:].sum(axis=(2, 3))
    left_right_asymmetry = (right - left) / safe_total

    temporal_variation = np.abs(
        np.diff(frames, axis=1, prepend=frames[:, :1])
    ).mean(axis=(2, 3))

    return np.stack(
        (
            total,
            peak,
            effective_area,
            centroid_x,
            centroid_y,
            spatial_dispersion,
            left_right_asymmetry,
            temporal_variation,
        ),
        axis=2,
    )

