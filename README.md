# Predictive Tactile Analysis

This repository contains the Python analysis code for the study *Predictive tactile coding in a memristive neuromorphic tactile system*. It extracts eight-dimensional tactile states from recorded tactile sequences, fits a one-step transition model, calculates prediction mismatch, compares baseline methods, and regenerates the main quantitative results and figures.

## Data

The measured tactile dataset is available from [Zenodo](https://doi.org/10.5281/zenodo.23420547).

Download the dataset archive, extract it, and place these two files in the repository's `data/` directory:

- `zipper_44x44_v3_predictive_clean.npz` — recorded tactile sequences and frame-level annotations.
- `trial_manifest.csv` — trial conditions and train/validation/test assignments.

The dataset contains 30 trials, with 500 frames per trial recorded at 200 Hz using a 44 × 44 tactile array. All recorded frames are retained in the released dataset. The analysis settings are specified in `config/reproduce_v3.json`.

## Requirements

Python 3.10 or later is recommended.

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the analysis

After placing the two dataset files in `data/`, run the following commands from the repository root:

```bash
python scripts/inspect_dataset.py
python scripts/run_reproduction.py
```

The analysis outputs are written to `outputs/reproduction/`. They include regenerated result tables, figures, fitted model parameters, and a report comparing the recomputed tables with the reference results in `reference_results/`.

If the data files are stored elsewhere, provide their paths explicitly:

```bash
python scripts/run_reproduction.py --data /path/to/zipper_44x44_v3_predictive_clean.npz --manifest /path/to/trial_manifest.csv
```

## Tests

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

The end-to-end tests require the dataset files to be present in `data/`.

## Repository contents

- `src/predictive_tactile/` — feature extraction, transition modelling, scoring, and evaluation.
- `scripts/` — dataset inspection, analysis, and result verification.
- `config/` — fixed analysis settings.
- `reference_results/` — reference tables for checking regenerated results.
- `tests/` — unit tests and data-dependent regression tests.
- `docs/` — algorithm specification.
- `data/` — instructions for obtaining the dataset.

## Citation and license

Citation metadata for this software are provided in `CITATION.cff`. When using the code or dataset, please cite the associated study and the Zenodo dataset.

The code is released under the MIT License. The dataset is distributed under the license specified in its Zenodo record.
