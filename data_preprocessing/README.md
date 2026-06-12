# Downstream dataset preprocessing

Reproducible preprocessing scripts that turn raw recordings into the per-subject files the
ST-EEGFormer benchmark consumes. They are the companion to the end-to-end onboarding
walkthrough in [`easy_start/bci_iv2a_dataset_tutorial.ipynb`](../easy_start/bci_iv2a_dataset_tutorial.ipynb),
which uses **BCI-IV-2a** as the worked example (preprocess → `Dataset` class → register in two
YAMLs → dispatcher → train). Read that notebook first; the scripts here apply the same pattern
to the other benchmark datasets.

## Layout

```
data_preprocessing/
├── common/preprocess_utils.py   # shared MNE helpers (config, loaders, filter/resample, epoching)
├── yamls/<dataset>.yaml         # per-dataset config (EDIT THE PATHS before running)
├── <dataset>/preprocess.py      # raw -> per-subject file(s)
├── <dataset>/consolidate.py     # (some datasets) per-subject files -> .h5 with CV folds
└── tools/verify_schemas.py      # offline schema smoke-test (no raw data needed)
```

Every script is config-driven: `python preprocess.py --yaml ../yamls/<dataset>.yaml`. The YAML
paths are placeholders (`/path/to/...`) — point `data.*` at your raw download and `output.*` at
`~/workspace/outputs/...`.

## Datasets

| Dataset | Raw source | Script(s) | Output (per subject) | `Dataset` class |
|---|---|---|---|---|
| **alzheimer** | OpenNeuro ds004504 (`.set`) | `preprocess.py` | `A{sub}.pkl` `{eeg:(N,19,768), group}` | `AlzheimerDataset` |
| **error** (ErrP) | BrainVision `.vhdr` | `preprocess.py` → `consolidate.py` | `consolidate/{sub}.h5` `X, df/{trial_idx,class,set}` | `ErrorDataset` |
| **inner_speech** | OpenNeuro ds003626 *derivatives* | `preprocess.py` (+ `utils.py`) | `sub{N}.h5` `X, df/{trial,label}, folds/fold_k` | `InnerSpeechDataset` |
| **upper_limb** | Ofner 2017 `.gdf` | `preprocess.py` → `consolidate.py` | `consolidate/sub{N}.h5` `X, df/{trial_idx,class,experiment}, folds/{exp}/fold_k` | `UpperLimbDataset` |
| **binocular_ssvep** | per-subject `.csv` | `preprocess.py` | `{subject}.pkl` `{data:(40,5,64,L), df, splits}` | `BinocularSSVEPDataset` |
| **bci_iv2a** | BNCI Horizon `.mat` | see the [tutorial notebook](../easy_start/bci_iv2a_dataset_tutorial.ipynb) | `A{sub}.pkl` `{trainX,trainY,testX,testY}` | `BCI2aDataset` |

The `Dataset` classes live in
[`benchmark/neural_networks/util/eeg_downstream_dataset.py`](../benchmark/neural_networks/util/eeg_downstream_dataset.py);
the matching benchmark registration is in `util/dataset_specs.yaml` + `util/downstream_task_specs.yaml`.

## Quick start (example: error / ErrP)

```bash
conda activate eeg311
cd data_preprocessing/error
# edit ../yamls/error.yaml: data.train_path / data.test_path / output.*
python preprocess.py   --yaml ../yamls/error.yaml     # -> {sub}.pkl, {sub}_test.pkl
python consolidate.py  --yaml ../yamls/error.yaml     # -> consolidate/{sub}.h5
```

`upper_limb` is identical (two stages). `alzheimer`, `inner_speech`, `binocular_ssvep` are a
single `preprocess.py` step. `upper_limb_256.yaml` is the 256 Hz variant of `upper_limb.yaml`.

## Verify the output contract (no raw data required)

`tools/verify_schemas.py` compiles every script and round-trips a synthetic output of each
dataset's format through the real `Dataset` class, confirming the file layout / keys / shapes /
label flow the trainer expects:

```bash
conda run -n eeg311 python data_preprocessing/tools/verify_schemas.py
```

This checks the *interface*, not the numerics — the actual signal processing must be validated
against the raw datasets.

## Caveats

- **`upper_limb` filtering.** The original HPC `preprocess.py` resampled the *unfiltered* raw
  (the filtered copy was discarded). The clean script chains filter → resample as intended;
  see the comment in `upper_limb/preprocess.py`. To reproduce the exact published upper_limb
  numbers, restore the original (unfiltered) ordering.
- **`inner_speech` derivatives.** The pipeline consumes the Nieto *derivatives*
  (`derivatives/sub-XX/ses-YY/*_eeg-epo.fif` + `*_events.dat`), not the raw `.bdf`. Download
  them with the raw data from OpenNeuro ds003626; `utils.py` is vendored from the Nieto
  "Thinking out loud" tutorial repo (see its header for attribution).
- **`dtu` (auditory attention) is not included here.** Its raw preprocessing is MATLAB
  (`preproc_data.m` + the COCOHA toolbox); only the Python `.h5` consolidation is portable.
  It is deferred to a later pass.
- **Sampling-rate variants.** The benchmark also uses re-resampled copies for some baselines
  (LaBraM/CBraMod at 200 Hz, EEGPT/BENDR at 256 Hz). The 256 Hz upper_limb variant is provided
  (`upper_limb_256.yaml`); other rates follow the same recipe with `resample_freq` changed.
