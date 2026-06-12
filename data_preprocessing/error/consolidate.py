"""Error-related potential (ErrP) dataset -> per-subject .h5 (stage 2 of 2).

Merge each subject's ``{subject}.pkl`` (train) and ``{subject}_test.pkl`` (test) produced by
``preprocess.py`` into one ``.h5`` with a ``set`` column tagging train vs test, matching the
schema read by ``ErrorDataset``::

    X           : float  (n_trials, n_channels, n_times)
    df/trial_idx: int64  (n_trials,)
    df/class    : str    (n_trials,)   # 'error' / 'no_error'
    df/set      : str    (n_trials,)   # 'train' / 'test'

Output goes to ``output.consolidate_dir`` (named ``consolidate`` to match
``util/dataset_specs.yaml``'s ``error.data_dir``).

Usage:
    python consolidate.py --yaml ../yamls/error.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.preprocess_utils import load_config  # noqa: E402


def get_args_parser():
    p = argparse.ArgumentParser("error_consolidate", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "error.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def convert_subject_pkl_to_h5(train_pkl_path, test_pkl_path, out_h5_path):
    with open(train_pkl_path, "rb") as f:
        tr = pickle.load(f)
    X_tr, df_tr = tr["data"], tr["df"].copy()
    df_tr["set"] = "train"

    with open(test_pkl_path, "rb") as f:
        te = pickle.load(f)
    X_te, df_te = te["data"], te["df"].copy()
    df_te["set"] = "test"

    X_all = np.concatenate([X_tr, X_te], axis=0)
    df_all = pd.concat([df_tr, df_te], ignore_index=True)
    df_all["trial_idx"] = np.arange(len(df_all))

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_h5_path, "w") as f5:
        f5.create_dataset("X", data=X_all, compression="lzf")
        grp = f5.create_group("df")
        grp.create_dataset("trial_idx", data=df_all["trial_idx"].to_numpy(), dtype="i8")
        grp.create_dataset("class", data=np.array(df_all["class"], dtype=object), dtype=str_dt)
        grp.create_dataset("set", data=np.array(df_all["set"], dtype=object), dtype=str_dt)
    print(f"  -> {out_h5_path.name}: {X_all.shape}")


def main():
    cfg = load_config(get_args_parser().parse_args().yaml)
    data_path = Path(cfg["output"]["dir"])
    out_dir = Path(cfg["output"]["consolidate_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for train_pkl in sorted(data_path.glob("*.pkl")):
        if train_pkl.stem.endswith("_test"):
            continue
        subject = train_pkl.stem
        test_pkl = data_path / f"{subject}_test.pkl"
        if not test_pkl.exists():
            print(f"  !! test file for '{subject}' not found, skipping.")
            continue
        convert_subject_pkl_to_h5(train_pkl, test_pkl, out_dir / f"{subject}.h5")


if __name__ == "__main__":
    main()
