"""Alzheimer's resting-state EEG (OpenNeuro ds004504) -> per-subject .pkl.

For each ``sub-XXX`` folder: load the EEGLAB ``.set`` recording, verify channel order
against the config, notch + band-pass + resample, then cut the continuous recording into
non-overlapping windows of ``(tmax - tmin) * resample_freq`` samples. Each subject's file
is a dict ``{"eeg": (n_windows, n_channels, n_times), "group": <C|A|F>}`` -- the contract
consumed by ``AlzheimerDataset`` (it does its own internal 80/20 split).

Usage:
    python preprocess.py --yaml ../yamls/alzheimer.yaml
"""
import argparse
import os
import pickle
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yaml

mne.set_log_level("ERROR")


def get_args_parser():
    p = argparse.ArgumentParser("alzheimer_preprocessing", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "alzheimer.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def get_sub_data(data_dir, sub):
    """Load one subject's eyes-closed EEGLAB recording + its channel-name list."""
    raw_subject_folder = f"{data_dir}/{sub}/eeg"
    channel_name_file = f"{raw_subject_folder}/{sub}_task-eyesclosed_channels.tsv"
    raw_eeg_file = f"{raw_subject_folder}/{sub}_task-eyesclosed_eeg.set"
    raw = mne.io.read_raw_eeglab(raw_eeg_file, preload=True)
    channel_df = pd.read_csv(channel_name_file, sep="\t")
    return raw, channel_df["name"].tolist()


def process_eeg(raw, f_notch, f_l, f_h, fs):
    """Notch -> band-pass -> resample, returning the raw data array (channels, time)."""
    raw = raw.notch_filter(freqs=f_notch)
    raw.filter(f_l, f_h, picks=raw.info["ch_names"], fir_design="firwin")
    raw.resample(fs, npad="auto")
    return raw._data


def segment_eeg(data, segment_length):
    """Segment (channels, time) into non-overlapping (num_segments, channels, segment_length)."""
    channels, total_time = data.shape
    num_segments = total_time // segment_length
    data_trimmed = data[:, : num_segments * segment_length]
    segmented = data_trimmed.reshape(channels, num_segments, segment_length)
    return np.transpose(segmented, (1, 0, 2))


def main():
    cfg = yaml.safe_load(open(get_args_parser().parse_args().yaml))

    data_dir = cfg["data"]["path"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects_df = pd.read_csv(cfg["data"]["participants_tsv"], sep="\t")

    all_sub_folders = [s for s in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, s)) and s.startswith("sub-")]
    all_sub_folders = sorted(all_sub_folders, key=lambda x: int(re.search(r"\d+", x).group()))

    seg_length = int(cfg["resample_freq"] * (cfg["epochs"]["tmax"] - cfg["epochs"]["tmin"]))

    for subject in all_sub_folders:
        raw, sub_channel = get_sub_data(data_dir, subject)
        assert sub_channel == cfg["channels"]["names"], (
            f"Channel order for {subject} does not match config:\n"
            f"  file: {sub_channel}\n  yaml: {cfg['channels']['names']}"
        )
        preprocessed = process_eeg(raw, cfg["filter"]["notch"], cfg["filter"]["l_freq"],
                                   cfg["filter"]["h_freq"], cfg["resample_freq"])
        this_sub_data = segment_eeg(preprocessed, seg_length)
        this_sub_group = subjects_df.loc[subjects_df["participant_id"] == subject, "Group"].values[0]

        print(f"process {subject}: {this_sub_data.shape}, group={this_sub_group}")
        with open(out_dir / f"A{subject}.pkl", "wb") as f:
            pickle.dump({"eeg": this_sub_data, "group": this_sub_group}, f)


if __name__ == "__main__":
    main()
