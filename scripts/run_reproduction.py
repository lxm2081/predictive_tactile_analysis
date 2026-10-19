#!/usr/bin/env python3
"""Run the complete frozen analysis and optionally verify archived values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from predictive_tactile.pipeline import run_reproduction, verify_against_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce predictive tactile coding results from measured 44 x 44 frames."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PACKAGE_ROOT / "data" / "zipper_44x44_v3_predictive_clean.npz",
        help="Input NPZ dataset.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PACKAGE_ROOT / "data" / "trial_manifest.csv",
        help="Trial manifest used for an independent metadata check.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config" / "reproduce_v3.json",
        help="Frozen analysis configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "reproduction",
        help="Directory for recomputed tables, figures and parameters.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PACKAGE_ROOT / "reference_results",
        help="Archived result tables used only after recomputation.",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Do not compare recomputed tables with archived results.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip PNG generation for a faster numerical run.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(
            f"Dataset file not found: {args.data}. Download the Zenodo dataset "
            "(https://doi.org/10.5281/zenodo.23420547), extract it, and pass "
            "the path to zipper_44x44_v3_predictive_clean.npz with --data."
        )
    if not args.manifest.is_file():
        parser.error(
            f"Trial manifest not found: {args.manifest}. Pass the path to "
            "trial_manifest.csv with --manifest."
        )
    artifacts = run_reproduction(
        data_path=args.data,
        config_path=args.config,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        make_figures=not args.no_figures,
    )
    predictive = artifacts.overall_table.loc[
        artifacts.overall_table["method"] == "Predictive mismatch"
    ].iloc[0]
    operating = artifacts.operating_evaluation.summary.iloc[0]

    print("Reproduction completed")
    print(f"  output: {artifacts.output_dir}")
    print(f"  predictive mismatch test AUC: {predictive['test_AUC']:.9f}")
    print(
        "  anomaly trials detected: "
        f"{int(operating['anomaly_trials_detected'])}/"
        f"{int(operating['anomaly_trials_total'])}"
    )
    print(
        "  normal test trials flagged: "
        f"{int(operating['normal_test_trials_flagged'])}/"
        f"{int(operating['normal_test_trials_total'])}"
    )

    if args.skip_verification:
        return 0
    report = verify_against_reference(
        artifacts.output_dir,
        args.reference_dir,
        artifacts.config.verification_atol,
    )
    print(f"  archived-result verification: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

