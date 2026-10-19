"""End-to-end reproduction and archived-result verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig, load_config
from .data import TactileDataset, load_dataset
from .evaluation import (
    OperatingEvaluation,
    normal_prediction_quality,
    overall_metrics,
    per_condition_auc,
    evaluate_operating_point,
)
from .features import FEATURE_NAMES, extract_tactile_codes
from .model import FittedPredictor, fit_predictor
from .plots import (
    save_condition_auc_plot,
    save_representative_mismatch_plot,
    save_roc_plot,
)
from .scores import compute_scores


@dataclass(frozen=True)
class ReproductionArtifacts:
    """Objects and paths produced by one complete reproduction run."""

    dataset: TactileDataset
    config: AnalysisConfig
    predictor: FittedPredictor
    scores: dict[str, np.ndarray]
    output_dir: Path
    tables_dir: Path
    figures_dir: Path
    model_dir: Path
    overall_table: pd.DataFrame
    condition_table: pd.DataFrame
    prediction_quality_table: pd.DataFrame
    operating_evaluation: OperatingEvaluation


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _write_tables(
    output_dir: Path,
    overall: pd.DataFrame,
    condition: pd.DataFrame,
    prediction_quality: pd.DataFrame,
    operating: OperatingEvaluation,
) -> Path:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    overall.to_csv(tables_dir / "overall_method_comparison.csv", index=False)
    condition.to_csv(tables_dir / "per_condition_auc.csv", index=False)
    prediction_quality.to_csv(
        tables_dir / "normal_prediction_quality.csv", index=False
    )
    operating.summary.to_csv(
        tables_dir / "recommended_operating_point.csv", index=False
    )

    manuscript_columns = (
        "trial_id",
        "condition",
        "detected",
        "detection_frame",
        "distance_to_detection_mm",
    )
    operating.anomaly_trials.loc[:, manuscript_columns].to_csv(
        tables_dir / "recommended_operating_point_per_trial.csv", index=False
    )
    operating.anomaly_trials.to_csv(
        tables_dir / "anomaly_detection_detailed.csv", index=False
    )
    operating.normal_trials.to_csv(
        tables_dir / "normal_test_outcomes.csv", index=False
    )
    operating.control_trials.to_csv(
        tables_dir / "end_stop_control_outcomes.csv", index=False
    )
    return tables_dir


def _write_model(
    output_dir: Path,
    predictor: FittedPredictor,
) -> Path:
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "fitted_predictor.npz",
        feature_names=np.asarray(FEATURE_NAMES),
        feature_mean=predictor.feature_mean,
        feature_scale=predictor.feature_scale,
        transition_matrix=predictor.transition_matrix,
        bias=predictor.bias,
        residual_scale=predictor.residual_scale,
    )
    return model_dir


def _write_figures(
    output_dir: Path,
    dataset: TactileDataset,
    scores: dict[str, np.ndarray],
    condition_table: pd.DataFrame,
    operating: OperatingEvaluation,
    config: AnalysisConfig,
) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_roc_plot(dataset, scores, config, figures_dir / "roc_comparison.png")
    save_condition_auc_plot(
        condition_table, figures_dir / "per_condition_auc.png"
    )
    save_representative_mismatch_plot(
        dataset,
        scores["Predictive mismatch"],
        operating.threshold,
        config,
        figures_dir / "representative_mismatch.png",
    )
    return figures_dir


def run_reproduction(
    data_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
    *,
    make_figures: bool = True,
) -> ReproductionArtifacts:
    """Run the frozen analysis from measured tactile frames to final outputs."""

    config = load_config(config_path)
    dataset = load_dataset(data_path, config, manifest_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    features = extract_tactile_codes(dataset.adc_counts, config.contact_fraction)
    predictor = fit_predictor(features, dataset, config)
    scores = compute_scores(dataset, predictor, config)

    overall = overall_metrics(dataset, scores, config)
    condition = per_condition_auc(dataset, scores, config)
    prediction_quality = normal_prediction_quality(dataset, predictor, config)
    operating = evaluate_operating_point(
        dataset, scores["Predictive mismatch"], config
    )

    tables_dir = _write_tables(
        output, overall, condition, prediction_quality, operating
    )
    model_dir = _write_model(output, predictor)
    figures_dir = output / "figures"
    if make_figures:
        figures_dir = _write_figures(
            output, dataset, scores, condition, operating, config
        )

    predictive_row = overall.loc[
        overall["method"] == "Predictive mismatch"
    ].iloc[0]
    operating_row = operating.summary.iloc[0]
    summary = {
        "software_version": "1.0.0",
        "input_file": dataset.source_path.name,
        "input_sha256": sha256_file(dataset.source_path),
        "dataset": {
            "trials": dataset.n_trials,
            "frames_per_trial": dataset.n_frames,
            "array_shape": [
                int(dataset.adc_counts.shape[2]),
                int(dataset.adc_counts.shape[3]),
            ],
            "sampling_rate_hz": dataset.fs_hz,
        },
        "analysis_config": config.to_dict(),
        "headline_results": {
            "predictive_mismatch_validation_auc": float(
                predictive_row["validation_AUC"]
            ),
            "predictive_mismatch_test_auc": float(predictive_row["test_AUC"]),
            "predictive_mismatch_test_average_precision": float(
                predictive_row["test_average_precision"]
            ),
            "threshold": float(operating_row["threshold"]),
            "anomaly_trials_detected": int(
                operating_row["anomaly_trials_detected"]
            ),
            "anomaly_trials_total": int(operating_row["anomaly_trials_total"]),
            "normal_test_trials_flagged": int(
                operating_row["normal_test_trials_flagged"]
            ),
            "normal_test_trials_total": int(
                operating_row["normal_test_trials_total"]
            ),
            "mean_distance_to_detection_mm": float(
                operating_row["mean_distance_to_detection_mm"]
            ),
            "end_stop_controls_flagged": int(
                operating_row["end_stop_controls_flagged"]
            ),
            "end_stop_controls_total": int(
                operating_row["end_stop_controls_total"]
            ),
        },
    }
    (output / "reproduction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return ReproductionArtifacts(
        dataset=dataset,
        config=config,
        predictor=predictor,
        scores=scores,
        output_dir=output,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        model_dir=model_dir,
        overall_table=overall,
        condition_table=condition,
        prediction_quality_table=prediction_quality,
        operating_evaluation=operating,
    )


def _compare_tables(
    observed_path: Path,
    reference_path: Path,
    key_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    atol: float,
) -> dict[str, object]:
    observed = pd.read_csv(observed_path)
    reference = pd.read_csv(reference_path)

    missing_observed = set((*key_columns, *numeric_columns)).difference(observed.columns)
    missing_reference = set((*key_columns, *numeric_columns)).difference(reference.columns)
    if missing_observed or missing_reference:
        return {
            "passed": False,
            "error": (
                f"missing observed columns {sorted(missing_observed)}; "
                f"missing reference columns {sorted(missing_reference)}"
            ),
        }

    observed = observed.set_index(list(key_columns)).sort_index()
    reference = reference.set_index(list(key_columns)).sort_index()
    if not observed.index.equals(reference.index):
        return {
            "passed": False,
            "error": "row keys differ",
            "observed_keys": [str(value) for value in observed.index.tolist()],
            "reference_keys": [str(value) for value in reference.index.tolist()],
        }

    differences: dict[str, float] = {}
    column_pass: dict[str, bool] = {}
    for column in numeric_columns:
        left = observed[column].to_numpy(dtype=float)
        right = reference[column].to_numpy(dtype=float)
        finite = np.isfinite(left) & np.isfinite(right)
        same_missing = np.array_equal(np.isnan(left), np.isnan(right))
        max_difference = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        differences[column] = max_difference
        column_pass[column] = bool(same_missing and max_difference <= atol)
    return {
        "passed": bool(all(column_pass.values())),
        "max_absolute_difference": differences,
        "column_pass": column_pass,
    }


def verify_against_reference(
    output_dir: str | Path,
    reference_dir: str | Path,
    atol: float = 1e-5,
) -> dict[str, object]:
    """Compare recomputed tables with the archived, read-only result tables."""

    output = Path(output_dir).resolve()
    tables = output / "tables"
    reference = Path(reference_dir).resolve()
    specifications = {
        "overall_method_comparison": (
            "overall_method_comparison.csv",
            ("method",),
            ("validation_AUC", "test_AUC", "test_average_precision"),
        ),
        "per_condition_auc": (
            "per_condition_auc.csv",
            ("condition", "method"),
            ("test_AUC",),
        ),
        "normal_prediction_quality": (
            "normal_prediction_quality.csv",
            ("feature",),
            ("Pearson_r", "normalized_RMSE", "training_residual_SD"),
        ),
        "recommended_operating_point": (
            "recommended_operating_point.csv",
            ("threshold_rule",),
            (
                "threshold",
                "persistence_frames",
                "persistence_time_ms",
                "anomaly_trials_detected",
                "anomaly_trials_total",
                "normal_test_trials_flagged",
                "normal_test_trials_total",
                "mean_distance_to_detection_mm",
                "end_stop_controls_flagged",
                "end_stop_controls_total",
            ),
        ),
        "recommended_operating_point_per_trial": (
            "recommended_operating_point_per_trial.csv",
            ("trial_id", "condition"),
            ("detected", "detection_frame", "distance_to_detection_mm"),
        ),
    }

    checks: dict[str, object] = {}
    for name, (filename, keys, columns) in specifications.items():
        observed_path = tables / filename
        reference_path = reference / filename
        if not observed_path.exists() or not reference_path.exists():
            checks[name] = {
                "passed": False,
                "error": (
                    f"missing file: observed={observed_path.exists()}, "
                    f"reference={reference_path.exists()}"
                ),
            }
            continue
        checks[name] = _compare_tables(
            observed_path, reference_path, keys, columns, atol
        )

    passed = bool(all(bool(check.get("passed")) for check in checks.values()))
    report = {
        "passed": passed,
        "absolute_tolerance": atol,
        "checks": checks,
    }
    (output / "verification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "Predictive tactile coding reproduction verification",
        f"Overall status: {'PASS' if passed else 'FAIL'}",
        f"Absolute tolerance: {atol:g}",
        "",
    ]
    for name, check in checks.items():
        lines.append(f"{name}: {'PASS' if check.get('passed') else 'FAIL'}")
        if "max_absolute_difference" in check:
            for column, value in check["max_absolute_difference"].items():
                lines.append(f"  {column}: max |difference| = {value:.12g}")
        if "error" in check:
            lines.append(f"  {check['error']}")
    (output / "verification_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report
