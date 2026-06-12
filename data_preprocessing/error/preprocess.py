"""Error-related potential (ErrP) dataset -> per-subject .pkl (stage 1 of 2).

For every subject under ``train_path`` and ``test_path``: load each BrainVision ``.vhdr``
under ``<subject>/data/``, filter + resample, epoch on the S96/S48 markers, and stack into
a dict ``{"data": (n_trials, n_channels, n_times), "df": <trial_idx, class, experiment>}``.
Train subjects are written as ``{subject}.pkl``, test subjects as ``{subject}_test.pkl``.

Run ``consolidate.py`` afterwards to merge each subject's train/test pkls into the single
``.h5`` (with a ``set`` column) that ``ErrorDataset`` consumes.

Usage:
    python preprocess.py --yaml ../yamls/error.yaml
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.preprocess_utils import (  # noqa: E402
    load_config, get_subfolders, load_single_file_brainvision,
    apply_filters, apply_downsampling, extract_epoched_data,
)


def get_args_parser():
    p = argparse.ArgumentParser("error_preprocessing", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "error.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def process_subjects(subjects, base_path, cfg, out_dir, suffix=""):
    for subject in subjects:
        subject_dir = Path(base_path) / subject / "data"
        vhdr_files = sorted(subject_dir.glob("*.vhdr"))
        sub_epochs, sub_dfs = [], []
        for vhdr_file in vhdr_files:
            raw = load_single_file_brainvision(vhdr_file, cfg)
            raw = apply_filters(raw, cfg)
            raw = apply_downsampling(raw, cfg)
            X, df = extract_epoched_data(raw, cfg, "error")
            sub_epochs.append(X)
            sub_dfs.append(df)
        all_epochs = np.vstack(sub_epochs)
        all_df = pd.concat(sub_dfs, ignore_index=True)
        all_df["trial_idx"] = all_df.index

        out_path = out_dir / f"{subject}{suffix}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"data": all_epochs, "df": all_df}, f)
        print(f"  -> {out_path.name}: {all_epochs.shape}")


def main():
    cfg = load_config(get_args_parser().parse_args().yaml)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Train subjects:")
    process_subjects(get_subfolders(cfg["data"]["train_path"]), cfg["data"]["train_path"],
                     cfg, out_dir, suffix="")
    print("Test subjects:")
    process_subjects(get_subfolders(cfg["data"]["test_path"]), cfg["data"]["test_path"],
                     cfg, out_dir, suffix="_test")


if __name__ == "__main__":
    main()
