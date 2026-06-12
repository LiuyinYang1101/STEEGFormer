"""Binocular-rivalry SSVEP -> per-subject .pkl.

For each subject CSV: reshape into a ``(n_targets, repeats, 64, trial_len)`` array, build
sliding-window metadata, and compute 5-fold leave-one-repeat-out splits. Each subject's file
is a dict matching the schema read by ``BinocularSSVEPDataset``::

    data   : float (40, 5, 64, trial_len)
    df     : DataFrame [example_idx, target, epoch, start_sample]
    splits : list of (train_idx, sync_idx, async_idx), one per fold
             sync  = windows at start_sample == 0 of the held-out repeat
             async = all windows of the held-out repeat

Usage:
    python preprocess.py --yaml ../yamls/binocular_ssvep.yaml

NOTE: this reproduces the original pipeline but fixes three bugs in the source notebook:
the metadata was built from an undefined ``trials`` (now ``eeg_data``), the loop had a
leftover ``break``, and the per-subject save was commented out.
"""
import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.preprocess_utils import load_config  # noqa: E402


def get_args_parser():
    p = argparse.ArgumentParser("binocular_ssvep_preprocessing", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "binocular_ssvep.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def load_trials_from_csv(csv_path, repeats=5, metadata_cols=("time", "condition", "epoch")):
    """Reshape a subject CSV into ``(n_targets, repeats, 64, trial_len)``."""
    df = pd.read_csv(csv_path, index_col=False)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    for m in metadata_cols:
        if m not in df.columns:
            raise ValueError(f"Expected metadata column {m!r} in CSV")

    channels = [c for c in df.columns if c not in metadata_cols]
    if len(channels) != 64:
        raise ValueError(f"Expected 64 channels, found {len(channels)}: {channels}")

    df["within_cond_idx"] = df.groupby("condition").cumcount()
    trial_length = (df.groupby("condition")["within_cond_idx"].max().iloc[0] + 1) // repeats
    df["trial_id"] = (df["within_cond_idx"] // trial_length).astype(int)

    targets = sorted(df["condition"].unique())
    T, R, C, L = len(targets), repeats, len(channels), trial_length
    trials = np.zeros((T, R, C, L), dtype=float)
    for ti, cond in enumerate(targets):
        cond_df = df[df["condition"] == cond]
        for ri in range(R):
            tr = cond_df[cond_df["trial_id"] == ri].sort_values("within_cond_idx")
            trials[ti, ri, :, :] = tr[channels].values.T
    return trials, df, channels


def create_sw_meta(trials, fs=250.0, window_size_s=1.0, step_size_s=0.1):
    """Sliding-window metadata: columns [example_idx, target, epoch, start_sample]."""
    T, R, C, L = trials.shape
    wlen = int(window_size_s * fs)
    step = int(step_size_s * fs)
    records = []
    ex_idx = 0
    for t in range(T):
        for r in range(R):
            for start in range(0, L - wlen + 1, step):
                records.append({"example_idx": ex_idx, "target": t, "epoch": r, "start_sample": start})
                ex_idx += 1
    return pd.DataFrame(records)


def epoch_stratified_splits(sw_meta) -> List[Tuple[list, list, list]]:
    """5-fold leave-one-repeat-out -> [(train_idx, sync_idx, async_idx), ...]."""
    splits = []
    for r in sorted(sw_meta["epoch"].unique()):
        test_mask = sw_meta["epoch"] == r
        async_idx = sw_meta.loc[test_mask, "example_idx"].tolist()
        sync_idx = sw_meta.loc[test_mask & (sw_meta["start_sample"] == 0), "example_idx"].tolist()
        train_idx = sw_meta.loc[~test_mask, "example_idx"].tolist()
        splits.append((train_idx, sync_idx, async_idx))
    return splits


def main():
    cfg = load_config(get_args_parser().parse_args().yaml)
    data_folder = Path(cfg["data"]["path"])
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = cfg["fs"]
    repeats = cfg["repeats"]
    wsz = cfg["sliding_window"]["window_size_s"]
    step = cfg["sliding_window"]["step_size_s"]

    csv_files = sorted(data_folder.glob("*.csv"))
    print(f"Found {len(csv_files)} subject CSVs.")
    for csv_path in csv_files:
        subject = csv_path.stem
        eeg_data, _, _ = load_trials_from_csv(csv_path, repeats=repeats)
        sw_meta = create_sw_meta(eeg_data, fs=fs, window_size_s=wsz, step_size_s=step)
        folds = epoch_stratified_splits(sw_meta)
        with open(out_dir / f"{subject}.pkl", "wb") as f:
            pickle.dump({"data": eeg_data, "df": sw_meta, "splits": folds}, f)
        print(f"  -> {subject}.pkl: data={eeg_data.shape}, windows={len(sw_meta)}")


if __name__ == "__main__":
    main()
