"""Held-out metrics and the validation-fixed event decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import AnalysisConfig
from .data import TactileDataset
from .features import FEATURE_NAMES
from .model import FittedPredictor
from .scores import METHOD_ORDER


CONDITION_ORDER = (
    "weak_jam",
    "strong_jam",
    "unilateral_slip",
    "bilateral_slip",
)


@dataclass(frozen=True)
class OperatingEvaluation:
    threshold: float
    anomaly_trials: pd.DataFrame
    normal_trials: pd.DataFrame
    control_trials: pd.DataFrame
    summary: pd.DataFrame


def _aligned_labels(
    dataset: TactileDataset,
    indices: np.ndarray,
    start_frame: int,
) -> np.ndarray:
    return dataset.event_mask[:, start_frame + 1 :][indices].reshape(-1).astype(int)


def _aligned_values(
    score: np.ndarray,
    indices: np.ndarray,
    start_frame: int,
) -> np.ndarray:
    return score[indices, start_frame:].reshape(-1)


def overall_metrics(
    dataset: TactileDataset,
    scores: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method in METHOD_ORDER:
        row: dict[str, float | str] = {"method": method}
        for split_name, prefix in (("validation", "validation"), ("test", "test")):
            indices = dataset.indices(split=split_name)
            labels = _aligned_labels(dataset, indices, config.start_frame)
            values = _aligned_values(scores[method], indices, config.start_frame)
            row[f"{prefix}_AUC"] = float(roc_auc_score(labels, values))
            if split_name == "test":
                row["test_average_precision"] = float(
                    average_precision_score(labels, values)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def per_condition_auc(
    dataset: TactileDataset,
    scores: dict[str, np.ndarray],
    config: AnalysisConfig,
) -> pd.DataFrame:
    normal_test = dataset.indices(condition="normal_zip", split="test")
    rows: list[dict[str, float | str]] = []
    for condition in CONDITION_ORDER:
        anomaly_test = dataset.indices(condition=condition, split="test")
        indices = np.concatenate((normal_test, anomaly_test))
        labels = _aligned_labels(dataset, indices, config.start_frame)
        for method in METHOD_ORDER:
            values = _aligned_values(scores[method], indices, config.start_frame)
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "test_AUC": float(roc_auc_score(labels, values)),
                }
            )
    return pd.DataFrame(rows)


def normal_prediction_quality(
    dataset: TactileDataset,
    predictor: FittedPredictor,
    config: AnalysisConfig,
) -> pd.DataFrame:
    normal_test = dataset.indices(condition="normal_zip", split="test")
    measured = predictor.codes[
        normal_test, config.start_frame + 1 :
    ].reshape(-1, 8)
    predicted = predictor.predicted_next[
        normal_test, config.start_frame:
    ].reshape(-1, 8)
    rows = []
    for column, feature in enumerate(FEATURE_NAMES):
        correlation = float(np.corrcoef(measured[:, column], predicted[:, column])[0, 1])
        normalized_rmse = float(
            np.sqrt(np.mean((measured[:, column] - predicted[:, column]) ** 2))
        )
        rows.append(
            {
                "feature": feature,
                "Pearson_r": correlation,
                "normalized_RMSE": normalized_rmse,
                "training_residual_SD": float(predictor.residual_scale[column]),
            }
        )
    return pd.DataFrame(rows)


def consecutive_run_starts(
    score: np.ndarray,
    threshold: float,
    persistence: int,
    start_score_index: int,
) -> np.ndarray:
    exceeds = np.asarray(score > threshold, dtype=bool).copy()
    exceeds[:start_score_index] = False
    if len(exceeds) < persistence:
        return np.empty(0, dtype=int)
    counts = np.convolve(
        exceeds.astype(np.int8),
        np.ones(persistence, dtype=np.int8),
        mode="valid",
    )
    return np.flatnonzero(counts == persistence)


def _trial_event_detection(
    dataset: TactileDataset,
    trial_index: int,
    score: np.ndarray,
    threshold: float,
    config: AnalysisConfig,
) -> dict[str, object]:
    starts = consecutive_run_starts(
        score,
        threshold,
        config.persistence_frames,
        config.start_frame,
    )
    first_frames = starts + 1
    confirmation_frames = starts + config.persistence_frames
    onset = int(dataset.event_onset_frame[trial_index])
    end = int(dataset.event_end_frame[trial_index])

    qualifying: list[tuple[int, int]] = []
    for first, confirmation in zip(first_frames, confirmation_frames):
        if onset <= first <= confirmation < end:
            qualifying.append((int(first), int(confirmation)))

    detected = bool(qualifying)
    first_crossing = qualifying[0][0] if detected else None
    confirmation = qualifying[0][1] if detected else None
    first_monitored_crossing = int(first_frames[0]) if len(first_frames) else None
    first_distance = (
        (first_crossing - onset) / dataset.fs_hz * dataset.speed_mm_s[trial_index]
        if detected
        else np.nan
    )
    confirmed_distance = (
        (confirmation - onset) / dataset.fs_hz * dataset.speed_mm_s[trial_index]
        if detected
        else np.nan
    )
    pre_annotation_crossing = (
        bool(np.any(confirmation_frames < onset)) if onset >= 0 else False
    )
    return {
        "detected": detected,
        "detection_frame": first_crossing,
        "confirmation_frame": confirmation,
        "first_monitored_crossing_frame": first_monitored_crossing,
        "distance_to_detection_mm": float(first_distance),
        "confirmed_detection_distance_mm": float(confirmed_distance),
        "pre_annotation_persistent_crossing": pre_annotation_crossing,
        "all_run_starts": starts,
    }


def evaluate_operating_point(
    dataset: TactileDataset,
    predictive_score: np.ndarray,
    config: AnalysisConfig,
) -> OperatingEvaluation:
    validation_normal = dataset.indices(condition="normal_zip", split="validation")
    threshold = float(
        np.percentile(
            predictive_score[validation_normal, config.start_frame:].reshape(-1),
            config.threshold_percentile,
        )
    )

    anomaly_rows = []
    for index in dataset.indices(split="test"):
        if dataset.condition[index] == "normal_zip":
            continue
        result = _trial_event_detection(
            dataset,
            index,
            predictive_score[index],
            threshold,
            config,
        )
        anomaly_rows.append(
            {
                "trial_id": dataset.trial_id[index],
                "condition": dataset.condition[index],
                "detected": result["detected"],
                "detection_frame": result["detection_frame"],
                "confirmation_frame": result["confirmation_frame"],
                "first_monitored_crossing_frame": result[
                    "first_monitored_crossing_frame"
                ],
                "distance_to_detection_mm": result["distance_to_detection_mm"],
                "confirmed_detection_distance_mm": result[
                    "confirmed_detection_distance_mm"
                ],
                "pre_annotation_persistent_crossing": result[
                    "pre_annotation_persistent_crossing"
                ],
            }
        )
    anomaly_table = pd.DataFrame(anomaly_rows)

    normal_rows = []
    for index in dataset.indices(condition="normal_zip", split="test"):
        starts = consecutive_run_starts(
            predictive_score[index],
            threshold,
            config.persistence_frames,
            config.start_frame,
        )
        normal_rows.append(
            {
                "trial_id": dataset.trial_id[index],
                "condition": dataset.condition[index],
                "flagged": bool(len(starts)),
                "first_crossing_frame": int(starts[0] + 1) if len(starts) else None,
            }
        )
    normal_table = pd.DataFrame(normal_rows)

    control_rows = []
    for index in dataset.indices(condition="end_stop", split="control"):
        result = _trial_event_detection(
            dataset,
            index,
            predictive_score[index],
            threshold,
            config,
        )
        control_rows.append(
            {
                "trial_id": dataset.trial_id[index],
                "condition": dataset.condition[index],
                "flagged": result["detected"],
                "first_crossing_frame": result["detection_frame"],
                "confirmation_frame": result["confirmation_frame"],
            }
        )
    control_table = pd.DataFrame(control_rows)

    detected = int(anomaly_table["detected"].sum())
    normal_flagged = int(normal_table["flagged"].sum())
    controls_flagged = int(control_table["flagged"].sum())
    summary = pd.DataFrame(
        [
            {
                "threshold_rule": (
                    f"{config.threshold_percentile:g}th percentile of normal validation predictive mismatch"
                ),
                "threshold": threshold,
                "persistence_frames": config.persistence_frames,
                "persistence_time_ms": (
                    1000.0 * config.persistence_frames / dataset.fs_hz
                ),
                "anomaly_trials_detected": detected,
                "anomaly_trials_total": len(anomaly_table),
                "normal_test_trials_flagged": normal_flagged,
                "normal_test_trials_total": len(normal_table),
                "mean_distance_to_detection_mm": float(
                    anomaly_table.loc[
                        anomaly_table["detected"], "distance_to_detection_mm"
                    ].mean()
                ),
                "mean_confirmed_detection_distance_mm": float(
                    anomaly_table.loc[
                        anomaly_table["detected"],
                        "confirmed_detection_distance_mm",
                    ].mean()
                ),
                "end_stop_controls_flagged": controls_flagged,
                "end_stop_controls_total": len(control_table),
            }
        ]
    )
    return OperatingEvaluation(
        threshold=threshold,
        anomaly_trials=anomaly_table,
        normal_trials=normal_table,
        control_trials=control_table,
        summary=summary,
    )
