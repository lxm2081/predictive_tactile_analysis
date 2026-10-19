"""Reproducible Python analysis for predictive tactile sensing."""

from .config import AnalysisConfig, load_config
from .data import TactileDataset, load_dataset
from .features import FEATURE_NAMES, extract_tactile_codes
from .model import FittedPredictor, fit_predictor
from .pipeline import run_reproduction, verify_against_reference

__all__ = [
    "AnalysisConfig",
    "FEATURE_NAMES",
    "FittedPredictor",
    "TactileDataset",
    "extract_tactile_codes",
    "fit_predictor",
    "load_config",
    "load_dataset",
    "run_reproduction",
    "verify_against_reference",
]

__version__ = "1.0.0"
