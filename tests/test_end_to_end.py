"""End-to-end regression test using the separately hosted measured dataset."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from predictive_tactile.pipeline import run_reproduction, verify_against_reference

DATA_PATH = Path(
    os.environ.get(
        "TACTILE_DATA_PATH",
        PACKAGE_ROOT / "data" / "zipper_44x44_v3_predictive_clean.npz",
    )
)
MANIFEST_PATH = Path(
    os.environ.get(
        "TACTILE_MANIFEST_PATH",
        PACKAGE_ROOT / "data" / "trial_manifest.csv",
    )
)


class EndToEndRegressionTest(unittest.TestCase):
    @unittest.skipUnless(
        DATA_PATH.is_file() and MANIFEST_PATH.is_file(),
        "Download the Zenodo dataset and place the NPZ and CSV in data/, or set TACTILE_DATA_PATH and TACTILE_MANIFEST_PATH.",
    )
    def test_full_numerical_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = run_reproduction(
                DATA_PATH,
                PACKAGE_ROOT / "config" / "reproduce_v3.json",
                Path(temporary),
                MANIFEST_PATH,
                make_figures=False,
            )
            predictive = artifacts.overall_table.loc[
                artifacts.overall_table["method"] == "Predictive mismatch"
            ].iloc[0]
            self.assertAlmostEqual(float(predictive["test_AUC"]), 0.9161178, places=5)
            operating = artifacts.operating_evaluation.summary.iloc[0]
            self.assertEqual(int(operating["anomaly_trials_detected"]), 12)
            self.assertEqual(int(operating["normal_test_trials_flagged"]), 0)
            self.assertEqual(int(operating["end_stop_controls_flagged"]), 4)
            self.assertAlmostEqual(
                float(operating["mean_distance_to_detection_mm"]),
                0.05833333333333335,
                places=12,
            )
            self.assertEqual(
                int(
                    artifacts.operating_evaluation.anomaly_trials[
                        "pre_annotation_persistent_crossing"
                    ].sum()
                ),
                2,
            )
            report = verify_against_reference(
                temporary,
                PACKAGE_ROOT / "reference_results",
                artifacts.config.verification_atol,
            )
            self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
