#!/usr/bin/env python3
"""Validate the measured dataset and print its frozen split structure."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from predictive_tactile.config import load_config
from predictive_tactile.data import load_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PACKAGE_ROOT / "data" / "zipper_44x44_v3_predictive_clean.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PACKAGE_ROOT / "data" / "trial_manifest.csv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "config" / "reproduce_v3.json",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(
            f"Dataset file not found: {args.data}. Download the Zenodo dataset "
            "(https://doi.org/10.5281/zenodo.23420547) and pass the NPZ path with --data."
        )
    if not args.manifest.is_file():
        parser.error(
            f"Trial manifest not found: {args.manifest}. Pass its path with --manifest."
        )
    dataset = load_dataset(args.data, load_config(args.config), args.manifest)
    split_table = (
        pd.DataFrame({"condition": dataset.condition, "split": dataset.split})
        .value_counts(sort=False)
        .rename("trials")
        .reset_index()
        .sort_values(["condition", "split"])
    )
    print("Dataset validation: PASS")
    print(f"path: {dataset.source_path}")
    print(f"sha256: {sha256(dataset.source_path)}")
    print(f"adc_counts shape: {dataset.adc_counts.shape}")
    print(f"sampling rate: {dataset.fs_hz:g} Hz")
    print(split_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

