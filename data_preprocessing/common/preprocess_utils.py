"""Shared MNE preprocessing helpers for ST-EEGFormer downstream datasets.

This module consolidates the per-dataset ``preprocess_utils.py`` helpers that were
duplicated across the original HPC pipeline (``error/`` and ``upper_limb/``) into a
single shared utility. The behaviour is unchanged; only paths were de-hardcoded and
the two channel loaders (BrainVision vs GDF) were merged here.

Config schema (YAML), see ``data_preprocessing/yamls/`` for examples::

    data:
      path / train_path / test_path: <file-or-dir>
      pattern: <regex with named groups>   # optional, dir mode
    channels:
      n_channels: int
      names: [list of channel names in axis order]
    filter:
      l_freq: float | null
      h_freq: float | null
      notch:  [freqs] | null
    resample_freq: int | null              # target sampling rate (Hz)
    events:
      mapping: {raw_code: friendly_label}
    epochs:
      tmin: float
      tmax: float
      baseline: [start, end] | null
    output:
      dir: <output directory>
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import mne
import yaml

mne.set_log_level("ERROR")


# --------------------------------------------------------------------------- #
# Config / filesystem helpers
# --------------------------------------------------------------------------- #
def load_config(config_path):
    """Load a dataset preprocessing YAML into a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_subfolders(path):
    """Return the names of immediate sub-directories of ``path``."""
    return [name for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))]


def find_matching_files(base_dir, pattern, suffix="*.gdf"):
    """Recursively find files under ``base_dir`` whose *name* matches ``pattern``.

    Returns a list of absolute paths (as strings).
    """
    base_path = Path(base_dir)
    regex = re.compile(pattern)
    matched = []
    for filepath in base_path.rglob(suffix):
        if regex.match(filepath.name):
            matched.append(str(filepath.resolve()))
    return sorted(matched)


# --------------------------------------------------------------------------- #
# Raw loaders (one per file format)
# --------------------------------------------------------------------------- #
def load_single_file_brainvision(file_path, cfg=None):
    """Load a BrainVision ``.vhdr`` recording (used by the ErrP dataset).

    Renames the few upper-case 10-20 labels to MNE's mixed-case convention, sets the
    ``standard_1020`` montage, and zero-fills any NaNs.
    """
    raw = mne.io.read_raw_brainvision(file_path, preload=True)
    rename_map = {
        "FP1": "Fp1", "FP2": "Fp2",
        "FZ": "Fz", "CZ": "Cz",
        "PZ": "Pz", "OZ": "Oz",
        "CPZ": "CPz", "POZ": "POz",
    }
    # guard the map against absent keys (mne.rename_channels raises on unknown names)
    raw.rename_channels({k: v for k, v in rename_map.items() if k in raw.ch_names})
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    if np.isnan(raw._data).any():
        print("Warning: NaNs detected in the raw data! Zero-padding.")
        raw.apply_function(lambda x: np.nan_to_num(x), picks="eeg", channel_wise=True)
    return raw


def load_single_file_gdf(file_path, cfg):
    """Load a GDF recording (used by the Upper-Limb dataset).

    Keeps the first ``len(cfg['channels']['names'])`` channels and renames them, in
    order, to the names declared in the config. Zero-fills any NaNs.
    """
    raw = mne.io.read_raw_gdf(file_path, preload=True)
    target_names = cfg["channels"]["names"]
    old_names = raw.ch_names[: len(target_names)]
    raw_eeg = raw.copy().pick_channels(old_names)
    raw_eeg.rename_channels(dict(zip(old_names, target_names)))
    if np.isnan(raw_eeg._data).any():
        print("Warning: NaNs detected in the raw data! Zero-padding.")
        raw_eeg.apply_function(lambda x: np.nan_to_num(x), picks="eeg", channel_wise=True)
    return raw_eeg


# --------------------------------------------------------------------------- #
# Filtering / resampling
# --------------------------------------------------------------------------- #
def apply_filters(raw, cfg):
    """Apply band-pass and notch filters as configured. Returns a new Raw."""
    fcfg = cfg.get("filter", {}) or {}
    raw = raw.copy()
    l_freq = fcfg.get("l_freq")
    h_freq = fcfg.get("h_freq")
    if l_freq or h_freq:
        raw.filter(l_freq=l_freq, h_freq=h_freq)
    notch = fcfg.get("notch")
    if notch:
        raw.notch_filter(notch)
    return raw


def apply_downsampling(raw, cfg):
    """Resample to ``cfg['resample_freq']`` if set. Returns a new Raw."""
    dsf = cfg.get("resample_freq")
    if dsf:
        raw = raw.copy()
        raw.resample(dsf)
    return raw


# --------------------------------------------------------------------------- #
# Epoching
# --------------------------------------------------------------------------- #
def extract_events(raw, cfg):
    """Build an MNE ``event_id`` from the config's ``events.mapping``."""
    events, annot_map = mne.events_from_annotations(raw)
    event_id = {
        label: annot_map[str(raw_code)]
        for raw_code, label in cfg["events"]["mapping"].items()
        if str(raw_code) in annot_map
    }
    if not event_id:
        raise RuntimeError(
            "No matching annotation codes found!\n"
            f"  YAML keys: {list(cfg['events']['mapping'])}\n"
            f"  Annot map keys: {list(annot_map)}"
        )
    return events, event_id


def extract_epoched_data(raw, cfg, experiment_label):
    """Epoch ``raw`` per the config and flatten into ``(X, df)``.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_channels, n_times)
    df : pandas.DataFrame with columns ['trial_idx', 'class', 'experiment']
    """
    events, event_id = extract_events(raw, cfg)
    epochs = mne.Epochs(
        raw, events,
        picks="eeg",
        event_id=event_id,
        tmin=cfg["epochs"]["tmin"],
        tmax=cfg["epochs"]["tmax"],
        baseline=cfg["epochs"].get("baseline"),
        preload=True,
        on_missing="warn",
    )
    X = epochs.get_data()
    assert not np.isnan(X).any(), "ERROR: NaNs detected in the epoch data!"

    code2label = {v: k for k, v in epochs.event_id.items()}
    labels = [code2label[c] for c in epochs.events[:, 2]]
    df = pd.DataFrame({
        "trial_idx": np.arange(len(labels)),
        "class": labels,
        "experiment": experiment_label,
    })
    return X, df
