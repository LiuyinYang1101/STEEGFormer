# ST-EEGFormer
<p align="center">
  <img src="assets/steegformer_logo.png" alt="ST-EEGFormer Logo">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11.5-blue.svg?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/torch-2.4.1-ee4c2c.svg?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <a href="https://openreview.net/forum?id=5Xwm8e6vbh">
    <img src="https://img.shields.io/badge/OpenReview-ICLR'25-8c1b13.svg" alt="Paper">
  </a>
  <a href="https://eeg2025.github.io/leaderboard/">
    <img src="https://img.shields.io/badge/NeurIPS_Challenge-1st_Place-gold.svg?logo=google-scholar&logoColor=white" alt="NeurIPS Winner">
  </a>
</p>

*A fair EEG-BCI benchmark framework and a simple yet powerful ST-EEGFormer foundation model.*

**Authors:**  
Liuyin Yang ([liuyin.yang@kuleuven.be](mailto:liuyin.yang@kuleuven.be))  
Qiang Sun ([qiang.sun@kuleuven.be](mailto:qiang.sun@kuleuven.be))  
Ang Li ([ang_lii@outlook.com](mailto:ang_lii@outlook.com))  
Marc Van Hulle ([marc.vanhulle@kuleuven.be](mailto:marc.vanhulle@kuleuven.be))  

All rights reserved.

## Citation

If you use our model or find it useful, please cite the following paper:

> *Currently under double-blind review; the citation will be updated as soon as it is published. In the meantime, you are welcome to read our previous version on* [*OpenReview*](https://openreview.net/forum?id=V5Zn0VVvBE), which got rejected in ICLR 2025.

```bibtex
@misc{
  yang2025learning,
  title={Learning Robust {EEG} Representations with a Large Spatiotemporal Transformer as a Foundation Model},
  author={Liuyin Yang and Marc M. Van Hulle},
  year={2025},
  url={https://openreview.net/forum?id=V5Zn0VVvBE}
}
```

---

## 1. Environment

The models are implemented in **PyTorch** and can be used in standard Python environments.

> **Python version used for pre-training:** `3.11.5`

### 1.1 Core Dependencies (for loading & using the model)

| Package | Version | Note                                            |
|---------|:-------:|-------------------------------------------------|
| `timm`  | 1.0.10  | Basic implementations of transformer models     |
| `torch` | 2.4.1   | Deep learning framework                         |

### 1.2 Extra Dependencies (for training foundation & neural models)

| Package        | Version | Note                                      |
|----------------|:-------:|-------------------------------------------|
| `wandb`        | 0.22.2  | Training monitoring & experiment logging  |
| `mat73`        | 0.65    | Loading MATLAB v7.3 `.mat` data files     |
| `scikit-learn` | 1.3.2   | Evaluation metrics and utilities          |

### 1.3 Classic EEG Model Dependencies

If you want to run the training code for **classic EEG models**, you will also need:

#### For all downstream tasks except SSVEP
| Package | Version | Note |
| :--- | :---: | :--- |
| `scipy` | 1.16.0 | General scientific computing utilities |
| `numpy` | 1.25.2 | Core numerical computing library |
| `mne` | 1.9.0 | EEG preprocessing and data handling |
| `pyriemann` | 0.6 | Riemannian geometry-based EEG classification |
| `scikit-learn` | 1.4.2 | Machine learning toolkit |
| `lightgbm` | 4.6.0 | Gradient boosting models for tabular features |

#### Specifically for SSVEP task as meegkit toolbox has compatibility issue with others
| Package | Version | Note |
| :--- | :---: | :--- |
| `scipy` | 1.15.3 | General scientific computing utilities |
| `numpy` | 2.2.6 | Core numerical computing library |
| `mne` | 1.9.0 | EEG preprocessing and data handling |
| `scikit-learn` | 1.7.0 | Machine learning toolkit |
| `meegkit` | 0.1.9 | EEG/MEG signal processing utilities |

---

## 2. Model Specs

**ST-EEGFormer** is designed for **128 Hz EEG data**.

- Pre-trained to reconstruct **6-second EEG segments**  
- Supports up to **145 EEG channels**  
- Recommended input: **≤ 6-second segments**, sampled at **128 Hz**

The list of available/pretrained channels can be found in:

```text
pretrain/senloc_file
```

## 3. Quick Start

A Jupyter notebook containing a minimal tutorial on how to use the model can be found in:

```text
easy_start/simple_example.ipynb
```

## 4. Reproducibility

If you want to pre-train a model, use the script:

```text
pretrain/ddp_train_eeg.py
```

You will need to prepare your own custom dataset that provides EEG segments and the corresponding channel indices.

If you want to run benchmark experiments on downstream BCI tasks using neural networks, use:

```text
benchmark/neural_networks/wandb_downstream_evaluation.py
```

For dataset preparation and configuration details, please refer to the README file in:

```text
benchmark/neural_networks
```

For the EEG 2025 Foundation Challenge, the code is located in:

```text
eeg_foundation_2025
```

where the models are slightly modified (they include additional channel embeddings for the HBN dataset).


## 5. Pre-trained Models

We release small, base, and large ST-EEGFormer models in the GitHub releases.

[ST-EEGFormer-small release](https://github.com/LiuyinYang1101/STEEGFormer/releases/tag/ST-EEGFormer-small).

[ST-EEGFormer-base release](https://github.com/LiuyinYang1101/STEEGFormer/releases/tag/ST-EEGFormer-base).

[ST-EEGFormer-large release](https://github.com/LiuyinYang1101/STEEGFormer/releases/tag/ST-EEGFormer-large).

Additionally, we provide large-ST-EEGFormerV2, which has undergone further pre-training on the HBN datasets for the EEG 2025 Foundation Challenge.

[ST-EEGFormer-large release-HBN](https://github.com/LiuyinYang1101/STEEGFormer/releases/tag/ST-EEGFormer-largeV2).

