"""Dataset loading and integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AnalysisConfig


REQUIRED_KEYS = (
    "adc_counts",
    "event_mask",
    "condition",
    "split",
    "trial_id",
    "speed_mm_s",
    "event_onset_frame",
    "event_end_frame",
    "fs_hz",
    "pressure_range_kpa",
    "active_area_mm",
)


@dataclass(frozen=True)
class TactileDataset:
    adc_counts: np.ndarray
    event_mask: np.ndarray
    condition: np.ndarray
    split: np.ndarray
    trial_id: np.ndarray
    speed_mm_s: np.ndarray
    event_onset_frame: np.ndarray
    event_end_frame: np.ndarray
    fs_hz: float
    pressure_range_kpa: float
    active_area_mm: float
    source_path: Path

    @property
    def n_trials(self) -> int:
        return int(self.adc_counts.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.adc_counts.shape[1])

    def indices(self, *, condition: str | None = None, split: str | None = None) -> np.ndarray:
        keep = np.ones(self.n_trials, dtype=bool)
        if condition is not None:
            keep &= self.condition == condition
        if split is not None:
            keep &= self.split == split
        return np.flatnonzero(keep)


def _validate_event_annotations(dataset: TactileDataset) -> None:
    for index in range(dataset.n_trials):
        onset = int(dataset.event_onset_frame[index])
        end = int(dataset.event_end_frame[index])
        mask = dataset.event_mask[index]
        if onset < 0 or end < 0:
            if mask.any():
                raise ValueError(f"{dataset.trial_id[index]} has an event mask but no event window")
            continue
        if not 0 <= onset < end <= dataset.n_frames:
            raise ValueError(f"Invalid event window for {dataset.trial_id[index]}: {onset}, {end}")
        if dataset.condition[index] == "end_stop" and dataset.split[index] == "control":
            if mask.any():
                raise ValueError(
                    f"{dataset.trial_id[index]} is a control but has positive ROC labels"
                )
            continue
        expected = np.zeros(dataset.n_frames, dtype=bool)
        expected[onset:end] = True
        if not np.array_equal(mask, expected):
            raise ValueError(f"event_mask and event window disagree for {dataset.trial_id[index]}")


def _validate_expected_layout(dataset: TactileDataset, config: AnalysisConfig) -> None:
    expected_shape = (
        config.expected_trials,
        config.expected_frames_per_trial,
        config.expected_rows,
        config.expected_columns,
    )
    if dataset.adc_counts.shape != expected_shape:
        raise ValueError(
            f"Expected adc_counts shape {expected_shape}, found {dataset.adc_counts.shape}"
        )
    if dataset.event_mask.shape != expected_shape[:2]:
        raise ValueError("event_mask must have one value per trial and frame")
    if len(np.unique(dataset.trial_id)) != dataset.n_trials:
        raise ValueError("trial_id values must be unique")
    if dataset.fs_hz <= 0:
        raise ValueError("fs_hz must be positive")

    expected_counts = {
        ("normal_zip", "train"): 6,
        ("normal_zip", "validation"): 2,
        ("normal_zip", "test"): 2,
        ("weak_jam", "validation"): 1,
        ("weak_jam", "test"): 3,
        ("strong_jam", "validation"): 1,
        ("strong_jam", "test"): 3,
        ("unilateral_slip", "validation"): 1,
        ("unilateral_slip", "test"): 3,
        ("bilateral_slip", "validation"): 1,
        ("bilateral_slip", "test"): 3,
        ("end_stop", "control"): 4,
    }
    observed = {
        key: int(np.sum((dataset.condition == key[0]) & (dataset.split == key[1])))
        for key in expected_counts
    }
    if observed != expected_counts:
        raise ValueError(f"Unexpected condition/split counts: {observed}")


def validate_manifest(dataset: TactileDataset, manifest_path: str | Path) -> None:
    manifest = pd.read_csv(manifest_path)
    required = {
        "trial_id",
        "condition",
        "split",
        "speed_mm_s",
        "event_onset_frame",
        "event_end_frame",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if len(manifest) != dataset.n_trials:
        raise ValueError("Manifest row count differs from the NPZ trial count")
    lookup = manifest.set_index("trial_id")
    for index, trial in enumerate(dataset.trial_id):
        if trial not in lookup.index:
            raise ValueError(f"Manifest is missing trial {trial}")
        row = lookup.loc[trial]
        checks = {
            "condition": (str(row["condition"]), str(dataset.condition[index])),
            "split": (str(row["split"]), str(dataset.split[index])),
            "event_onset_frame": (
                int(row["event_onset_frame"]),
                int(dataset.event_onset_frame[index]),
            ),
            "event_end_frame": (
                int(row["event_end_frame"]),
                int(dataset.event_end_frame[index]),
            ),
        }
        for field, (manifest_value, npz_value) in checks.items():
            if manifest_value != npz_value:
                raise ValueError(
                    f"Manifest mismatch for {trial}, {field}: {manifest_value} != {npz_value}"
                )
        if not np.isclose(float(row["speed_mm_s"]), float(dataset.speed_mm_s[index])):
            raise ValueError(f"Manifest speed mismatch for {trial}")


def load_dataset(
    path: str | Path,
    config: AnalysisConfig,
    manifest_path: str | Path | None = None,
) -> TactileDataset:
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise ValueError(f"NPZ is missing required keys: {missing}")
        dataset = TactileDataset(
            adc_counts=np.asarray(archive["adc_counts"]),
            event_mask=np.asarray(archive["event_mask"], dtype=bool),
            condition=np.asarray(archive["condition"]).astype(str),
            split=np.asarray(archive["split"]).astype(str),
            trial_id=np.asarray(archive["trial_id"]).astype(str),
            speed_mm_s=np.asarray(archive["speed_mm_s"], dtype=np.float64),
            event_onset_frame=np.asarray(archive["event_onset_frame"], dtype=int),
            event_end_frame=np.asarray(archive["event_end_frame"], dtype=int),
            fs_hz=float(archive["fs_hz"]),
            pressure_range_kpa=float(archive["pressure_range_kpa"]),
            active_area_mm=float(archive["active_area_mm"]),
            source_path=source,
        )
    _validate_expected_layout(dataset, config)
    _validate_event_annotations(dataset)
    if manifest_path is not None:
        validate_manifest(dataset, manifest_path)
    return dataset
