# Dataset access

The measured tactile data are hosted separately on Zenodo:

https://doi.org/10.5281/zenodo.23420547

Download the archive named `predictive_tactile_dataset_v1.0.0.zip` from the Zenodo record, then extract it. The archive contains a `data/` folder. Copy the following files from that folder into this repository's `data/` directory before running the analysis:

- `zipper_44x44_v3_predictive_clean.npz` — the tactile measurements
- `trial_manifest.csv` — the trial metadata

The ZIP archive name is the name of the file deposited on Zenodo; the NPZ and CSV names above refer to files contained inside that archive.

The NPZ contains 30 trials, each with 500 frames from a 44 × 44 tactile array sampled at 200 Hz. The raw frames are retained in full. The analysis configuration applies a fixed exclusion of frames 0–79 during model fitting, residual-scale estimation, threshold selection and held-out scoring.

The NPZ includes `adc_counts`, `event_mask`, `condition`, `split`, `trial_id`, `speed_mm_s`, `event_onset_frame`, `event_end_frame`, `fs_hz`, `pressure_range_kpa` and `active_area_mm`. The CSV manifest duplicates trial metadata in human-readable form and is cross-checked by the loader.

The dataset and its metadata are distributed under the license specified in the Zenodo record. Dataset licensing does not change the MIT License that applies to the software in this repository.
