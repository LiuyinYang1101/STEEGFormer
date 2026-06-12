"""Inner Speech (Nieto et al. 2022, OpenNeuro ds003626) -> per-subject .h5.

Consumes the dataset's pre-epoched *derivatives* (loaded by ``utils.load_mne``). For each
subject, concatenate the three sessions, keep only the four *imagined* directions, crop the
action interval ``[crop.tmin, crop.tmax]``, build stratified 5-fold splits, and write one
``.h5`` matching the schema read by ``InnerSpeechDataset``::

    X                       : float (n_trials, 128, n_times)
    df/{trial, label}       : per-trial metadata ('Arriba/Imagined', ...)
    folds/fold_<i>/{train,test} : int64 index arrays

Usage:
    python preprocess.py --yaml ../yamls/inner_speech.yaml
"""
import argparse
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))          # local utils.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # common/
import utils  # noqa: E402
from common.preprocess_utils import load_config  # noqa: E402

mne.set_log_level("ERROR")

# Task/mode integer codes used by the Nieto event files.
TASK_DCT = {"Arriba": 0, "Abajo": 1, "Derecha": 2, "Izquierda": 3}
MODE_DCT = {"Spoken": 0, "Imagined": 1, "Visual": 2}


def get_args_parser():
    p = argparse.ArgumentParser("inner_speech_preprocessing", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "inner_speech.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def save_subject_h5(X, df, splits, h5_path):
    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(h5_path, "w") as f5:
        f5.create_dataset("X", data=X, compression="lzf")
        grp_df = f5.create_group("df")
        grp_df.create_dataset("trial", data=df["trial"].to_numpy(), dtype="i8")
        grp_df.create_dataset("label", data=df["label"].to_numpy().astype(object), dtype=str_dt)
        grp_folds = f5.create_group("folds")
        for i, (train_idx, test_idx) in enumerate(splits):
            grp_fold = grp_folds.create_group(f"fold_{i}")
            grp_fold.create_dataset("train", data=np.array(train_idx, dtype="i8"))
            grp_fold.create_dataset("test", data=np.array(test_idx, dtype="i8"))
    print(f"  -> {Path(h5_path).name}: X={X.shape}")


def main():
    cfg = load_config(get_args_parser().parse_args().yaml)
    data_dir = cfg["data"]["path"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    fs = cfg["fs"]
    start_t, end_t = cfg["crop"]["tmin"], cfg["crop"]["tmax"]
    cl_labels = list(cfg["classes"])
    n_splits = cfg["folds"]["n_splits"]
    random_state = cfg["folds"]["random_state"]
    subjects = cfg.get("subjects", list(range(1, 11)))
    sessions = cfg.get("sessions", list(range(1, 4)))

    # joint (task, mode) condition dictionaries
    kdct, vdct = utils.merge_dicts(TASK_DCT, MODE_DCT)
    kdct = {"/".join(k): v for k, v in kdct.items()}
    reversed_dict = {v: k for k, v in kdct.items()}
    biosemi128 = mne.channels.make_standard_montage("biosemi128")

    for subject_idx in subjects:
        sub_x, sub_y = [], []
        for session_idx in sessions:
            epochs, _, _, events = utils.load_mne(data_dir, subject_idx, session_idx)
            epochs.set_montage(biosemi128)
            epochs.events = utils.update_events(events, vdct)
            epochs.event_id = kdct
            sub_x.append(epochs[cl_labels].get_data())
            sub_y.append(epochs[cl_labels].events[:, 2])

        all_epochs = np.vstack(sub_x)[:, :, int(fs * start_t):int(fs * end_t)]
        all_y = np.concatenate(sub_y, axis=0)
        df = pd.DataFrame({"trial": np.arange(len(all_y)),
                           "label": [reversed_dict[i] for i in all_y]})

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = [(tr, te) for tr, te in skf.split(X=df, y=df["label"])]
        save_subject_h5(all_epochs, df, splits, out_dir / f"sub{subject_idx}.h5")


if __name__ == "__main__":
    main()
