#!/usr/bin/env python3
"""Verify recomputed tables against the archived reference values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from predictive_tactile.config import load_config
from predictive_tactile.pipeline import verify_against_reference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "outputs" / "reproduction",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PACKAGE_ROOT / "reference_results",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config" / "reproduce_v3.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report = verify_against_reference(
        args.output_dir, args.reference_dir, config.verification_atol
    )
    print(f"Verification: {'PASS' if report['passed'] else 'FAIL'}")
    print(Path(args.output_dir).resolve() / "verification_report.txt")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

