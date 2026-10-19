"""Predictive mismatch and comparison scores."""

from __future__ import annotations

import numpy as np

from .config import AnalysisConfig
from .data import TactileDataset
from .model import FittedPredictor


METHOD_ORDER = (
    "Predictive mismatch",
    "Unweighted predictive",
    "Raw frame difference",
    "Raw-difference deviation",
    "Static tactile-code template",
    "Tactile-code difference",
    "Peak pressure",
)


def compute_scores(
    dataset: TactileDataset,
    predictor: FittedPredictor,
    config: AnalysisConfig,
) -> dict[str, np.ndarray]:
    """Return aligned score arrays with shape ``(trial, frame - 1)``."""

    residual = predictor.residuals
    predictive = np.sqrt(
        np.mean((residual / predictor.residual_scale[None, None, :]) ** 2, axis=2)
    )
    unweighted = np.sqrt(np.mean(residual**2, axis=2))

    frames = dataset.adc_counts.astype(np.float64)
    frame_delta = np.diff(frames, axis=1)
    raw_difference = np.sqrt(np.mean(frame_delta**2, axis=(2, 3)))
    training_indices = dataset.indices(condition="normal_zip", split="train")
    raw_reference = raw_difference[
        training_indices, config.start_frame:
    ].reshape(-1)
    raw_difference_deviation = np.abs(raw_difference - raw_reference.mean())
    raw_scale = raw_reference.std()
    if raw_scale == 0.0:
        raw_scale = 1.0
    raw_difference_deviation /= raw_scale

    static_template = np.sqrt(np.mean(predictor.codes[:, 1:] ** 2, axis=2))
    code_delta = np.diff(predictor.codes, axis=1)
    code_delta_magnitude = np.sqrt(np.mean(code_delta**2, axis=2))
    delta_reference = code_delta_magnitude[
        training_indices, config.start_frame:
    ].reshape(-1)
    delta_scale = delta_reference.std()
    if delta_scale == 0.0:
        delta_scale = 1.0
    code_difference_deviation = (
        np.abs(code_delta_magnitude - delta_reference.mean()) / delta_scale
    )

    peak_pressure = frames[:, 1:].max(axis=(2, 3))
    return {
        "Predictive mismatch": predictive,
        "Unweighted predictive": unweighted,
        "Raw frame difference": raw_difference,
        "Raw-difference deviation": raw_difference_deviation,
        "Static tactile-code template": static_template,
        "Tactile-code difference": code_difference_deviation,
        "Peak pressure": peak_pressure,
    }
