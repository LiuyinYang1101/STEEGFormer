"""Upper-Limb movement (Ofner et al. 2017) -> per-recording .pkl (stage 1 of 2).

For each ``.gdf`` matching the config ``pattern``: keep + rename the first 61 channels,
filter + resample, epoch on the 1536-1542 movement markers, and write a dict
``{"data": (n_trials, 61, n_times), "df": <trial_idx, class, experiment>}`` per recording.
``experiment`` is the filename prefix (``motorimagination`` / ``motorexecution``).

Run ``consolidate.py`` afterwards to build the per-subject ``.h5`` with stratified folds
that ``UpperLimbDataset`` consumes.

NOTE: the original HPC ``preprocess.py`` called ``apply_downsampling(raw, cfg)`` on the
*unfiltered* raw (discarding the band-pass/notch). This clean version chains
filter -> downsample as intended. To reproduce the exact published upper_limb numbers,
swap the two lines marked below.

Usage:
    python preprocess.py --yaml ../yamls/upper_limb.yaml      # 128 Hz
    python preprocess.py --yaml ../yamls/upper_limb_256.yaml  # 256 Hz
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.preprocess_utils import (  # noqa: E402
    load_config, find_matching_files, load_single_file_gdf,
    apply_filters, apply_downsampling, extract_epoched_data,
)


def get_args_parser():
    p = argparse.ArgumentParser("upper_limb_preprocessing", add_help=True)
    p.add_argument("--yaml", default=str(Path(__file__).resolve().parent.parent / "yamls" / "upper_limb.yaml"),
                   type=str, help="preprocessing YAML config")
    return p


def main():
    cfg = load_config(get_args_parser().parse_args().yaml)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = find_matching_files(cfg["data"]["path"], cfg["data"]["pattern"], suffix="*.gdf")
    print(f"Found {len(recordings)} recordings.")
    for recording in recordings:
        name = Path(recording).stem
        experiment = name.split("_")[0]              # motorimagination / motorexecution
        raw = load_single_file_gdf(recording, cfg)
        raw = apply_filters(raw, cfg)                # filter first ...
        raw = apply_downsampling(raw, cfg)           # ... then resample (original passed the unfiltered raw here)
        X, df = extract_epoched_data(raw, cfg, experiment)

        out_path = out_dir / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"data": X, "df": df}, f)
        print(f"  -> {out_path.name}: {X.shape} ({experiment})")


if __name__ == "__main__":
    main()
