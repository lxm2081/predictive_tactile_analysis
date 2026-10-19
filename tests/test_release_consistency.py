"""Check that the public release matches the code-only scope and frozen analysis."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from predictive_tactile.config import load_config


class ReleaseConsistencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PACKAGE_ROOT / "config" / "reproduce_v3.json")

    def test_manuscript_facing_configuration(self) -> None:
        self.assertEqual(self.config.start_frame, 80)
        self.assertEqual(self.config.contact_fraction, 0.20)
        self.assertEqual(self.config.ridge, 0.001)
        self.assertEqual(self.config.threshold_percentile, 99.5)
        self.assertEqual(self.config.persistence_frames, 3)
        self.assertEqual(self.config.expected_trials, 30)
        self.assertEqual(self.config.expected_frames_per_trial, 500)

    def test_dataset_is_referenced_separately(self) -> None:
        readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://doi.org/10.5281/zenodo.23420547", readme)
        self.assertFalse(
            (PACKAGE_ROOT / "data" / "zipper_44x44_v3_predictive_clean.npz").exists()
        )
        self.assertFalse((PACKAGE_ROOT / "data" / "trial_manifest.csv").exists())

    def test_no_fpga_rtl_is_distributed(self) -> None:
        self.assertFalse((PACKAGE_ROOT / "fpga").exists())
        self.assertFalse((PACKAGE_ROOT / "src" / "predictive_tactile" / "hardware.py").exists())
        readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("does not distribute FPGA/SystemVerilog RTL", readme)

    def test_public_text_has_no_internal_placeholders(self) -> None:
        prohibited = (
            "AUTHOR" + " CHECK",
            "ILLUSTRATIVE" + " VALUES",
            "WORKING" + " MANUSCRIPT",
            "finalized before" + " submission",
        )
        suffixes = {".md", ".txt", ".json", ".toml", ".yml", ".cff"}
        for path in PACKAGE_ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                text = path.read_text(encoding="utf-8")
                for marker in prohibited:
                    self.assertNotIn(marker, text, f"{marker!r} remains in {path}")


if __name__ == "__main__":
    unittest.main()
