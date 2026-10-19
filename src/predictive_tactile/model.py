"""Training-only normalization and one-step transition fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AnalysisConfig
from .data import TactileDataset


@dataclass(frozen=True)
class FittedPredictor:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    transition_matrix: np.ndarray
    bias: np.ndarray
    residual_scale: np.ndarray
    codes: np.ndarray
    predicted_next: np.ndarray

    @property
    def residuals(self) -> np.ndarray:
        return self.codes[:, 1:] - self.predicted_next


def standardize_from_training(
    features: np.ndarray,
    training_indices: np.ndarray,
    start_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = features[training_indices, start_frame:].reshape(-1, features.shape[-1])
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (features - mean) / scale, mean, scale


def _ridge_fit(design: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(
        design.T @ design + ridge * penalty,
        design.T @ target,
    )


def fit_predictor(
    features: np.ndarray,
    dataset: TactileDataset,
    config: AnalysisConfig,
) -> FittedPredictor:
    training_indices = dataset.indices(condition="normal_zip", split="train")
    if len(training_indices) != 6:
        raise ValueError("The frozen protocol requires six normal training trials")

    codes, mean, scale = standardize_from_training(
        features,
        training_indices,
        config.start_frame,
    )
    current = codes[training_indices, config.start_frame:-1].reshape(-1, 8)
    measured_next = codes[training_indices, config.start_frame + 1 :].reshape(-1, 8)
    design = np.column_stack((current, np.ones(len(current), dtype=np.float64)))
    beta = _ridge_fit(design, measured_next, config.ridge)
    transition_matrix = beta[:-1]
    bias = beta[-1]
    predicted_next = codes[:, :-1] @ transition_matrix + bias
    residual = codes[:, 1:] - predicted_next
    residual_scale = residual[
        training_indices, config.start_frame:
    ].reshape(-1, 8).std(axis=0)
    residual_scale[residual_scale == 0.0] = 1.0

    return FittedPredictor(
        feature_mean=mean,
        feature_scale=scale,
        transition_matrix=transition_matrix,
        bias=bias,
        residual_scale=residual_scale,
        codes=codes,
        predicted_next=predicted_next,
    )

