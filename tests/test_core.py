"""Deterministic unit tests for feature extraction and event decisions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from predictive_tactile.evaluation import consecutive_run_starts
from predictive_tactile.features import extract_tactile_codes


class FeatureExtractionTests(unittest.TestCase):
    def test_two_frame_hand_calculation(self) -> None:
        frames = np.asarray(
            [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 2.0]]]]
        )
        codes = extract_tactile_codes(frames, contact_fraction=0.20)
        self.assertEqual(codes.shape, (1, 2, 8))
        np.testing.assert_allclose(
            codes[0, 0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            codes[0, 1],
            [2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.75],
            atol=1e-12,
        )


class PersistenceTests(unittest.TestCase):
    def test_consecutive_run_start(self) -> None:
        score = np.asarray([0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 3.0, 3.0])
        starts = consecutive_run_starts(score, threshold=1.0, persistence=3, start_score_index=0)
        np.testing.assert_array_equal(starts, [1, 5])

    def test_initialization_is_excluded(self) -> None:
        score = np.asarray([4.0, 4.0, 4.0, 0.0, 2.0, 2.0, 2.0])
        starts = consecutive_run_starts(score, threshold=1.0, persistence=3, start_score_index=3)
        np.testing.assert_array_equal(starts, [4])


if __name__ == "__main__":
    unittest.main()

