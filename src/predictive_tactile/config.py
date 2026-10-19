"""Frozen analysis configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    start_frame: int = 80
    contact_fraction: float = 0.20
    ridge: float = 1e-3
    threshold_percentile: float = 99.5
    persistence_frames: int = 3
    expected_rows: int = 44
    expected_columns: int = 44
    expected_frames_per_trial: int = 500
    expected_trials: int = 30
    verification_atol: float = 1e-5

    def validate(self) -> None:
        if self.start_frame < 0:
            raise ValueError("start_frame must be nonnegative")
        if not 0.0 < self.contact_fraction < 1.0:
            raise ValueError("contact_fraction must be between zero and one")
        if self.ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if not 0.0 < self.threshold_percentile < 100.0:
            raise ValueError("threshold_percentile must be between zero and 100")
        if self.persistence_frames < 1:
            raise ValueError("persistence_frames must be positive")
        if self.start_frame >= self.expected_frames_per_trial - 1:
            raise ValueError("start_frame leaves no transition for analysis")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def load_config(path: str | Path) -> AnalysisConfig:
    source = Path(path)
    values = json.loads(source.read_text(encoding="utf-8"))
    config = AnalysisConfig(**values)
    config.validate()
    return config

