"""Upper-Limb movement -> per-subject .h5 with stratified folds (stage 2 of 2).

For each subject, concatenate that subject's per-recording ``.pkl`` files (both
``motorimagination`` and ``motorexecution``), compute stratified K-folds *per experiment*,
and write one ``.h5`` matching the schema read by ``UpperLimbDataset``::

    X                                  : float (n_trials, 61, n_times)
    df/{trial_idx, class, experiment}  : per-trial metadata
    folds/<experiment>/fold_<i>/{train,test} : int64 index arrays

Output goes to ``consolidate.out_dir``.

Usage:
    python consolidate.py --yaml ../yamls/upper_limb.yaml
"""
import argparse
import pickle
import re
import sys
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.preprocess_utils import load_config  # noqa: E402


def get_args_parser():
    p = argparse.ArgumentParser("upper_limb_consolidate", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "upper_limb.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def get_experiment_stratified_folds(df, n_splits=5, random_state=None):
    """Stratified K-fold on df['class'], computed independently per df['experiment']."""
    folds = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for exp in df["experiment"].unique():
        exp_idx = df.index[df["experiment"] == exp].to_numpy()
        labels = df.loc[exp_idx, "class"].to_numpy()
        folds[exp] = [(exp_idx[tr], exp_idx[te]) for tr, te in skf.split(exp_idx, labels)]
    return folds


def concatenate_subject_pkls(pkl_paths):
    """Stack the (data, df) pairs from multiple recordings into one (X, df)."""
    X_list, df_list = [], []
    for path in pkl_paths:
        with open(path, "rb") as f:
            d = pickle.load(f)
        X_list.append(d["data"])
        df_list.append(d["df"].copy())
    X_big = np.concatenate(X_list, axis=0)
    df_big = pd.concat(df_list, ignore_index=True)
    df_big["trial_idx"] = np.arange(len(df_big))
    return X_big, df_big


def save_X_df_folds_h5(X, df, folds, h5_path):
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as f5:
        f5.create_dataset("X", data=X, compression="lzf")
        grp_df = f5.create_group("df")
        grp_df.create_dataset("trial_idx", data=df["trial_idx"].to_numpy(), dtype="i8")
        grp_df.create_dataset("class", data=np.array(df["class"], dtype=object), dtype=str_dt)
        grp_df.create_dataset("experiment", data=np.array(df["experiment"], dtype=object), dtype=str_dt)
        grp_folds = f5.create_group("folds")
        for exp_label, exp_folds in folds.items():
            grp_exp = grp_folds.create_group(exp_label)
            for i, (train_idx, test_idx) in enumerate(exp_folds):
                grp_fold = grp_exp.create_group(f"fold_{i}")
                grp_fold.create_dataset("train", data=np.array(train_idx, dtype="i8"))
                grp_fold.create_dataset("test", data=np.array(test_idx, dtype="i8"))
    print(f"  -> {Path(h5_path).name}: X={X.shape}, experiments={list(folds)}")


def main():
    cfg = load_config(get_args_parser().parse_args().yaml)
    in_dir = Path(cfg["output"]["dir"])
    out_dir = Path(cfg["consolidate"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_splits = cfg["consolidate"]["n_splits"]
    random_state = cfg["consolidate"]["random_state"]

    # Discover subject numbers from the per-recording filenames (..._subject<N>_run<M>.pkl)
    subjects = sorted({int(m.group(1))
                       for p in in_dir.glob("*.pkl")
                       for m in [re.search(r"subject(\d+)_", p.name)] if m})
    print(f"Subjects found: {subjects}")
    for sub in subjects:
        files = sorted(glob(str(in_dir / f"*subject{sub}_*.pkl")))
        if not files:
            continue
        X, df = concatenate_subject_pkls(files)
        folds = get_experiment_stratified_folds(df, n_splits=n_splits, random_state=random_state)
        save_X_df_folds_h5(X, df, folds, out_dir / f"sub{sub}.h5")


if __name__ == "__main__":
    main()
