# \# ST-EEGFormer

# 

# \*A fair EEG-BCI benchmark framework and a simple ST-EEGFormer foundation model.\*

# 

# \*\*Authors\*\*  

# Liuyin Yang (liuyin.yang@kuleuven.be)  

# Qiang Sun (qiang.sun@kuleuven.be)

# 

# All rights reserved.

# 

# ---

# 

# \## 1. Environment

# 

# The models are implemented in \*\*PyTorch\*\* and can be used in standard Python environments.

# 

# > \*\*Python version used for pre-training:\*\* `Python 3.11.5`

# 

# \### 1.1 Core Dependencies (for loading \& using the model)

# 

# | Package | Version | Note                                             |

# |---------|:-------:|--------------------------------------------------|

# | `timm`  | 1.0.10  | Basic implementations of transformer models      |

# | `torch` | 2.4.1   | Deep learning framework                          |

# 

# \### 1.2 Additional Dependencies (for training foundation \& classic neural models)

# 

# | Package       | Version | Note                                      |

# |---------------|:-------:|-------------------------------------------|

# | `wandb`       | 0.22.2  | Training monitoring and experiment logging |

# | `mat73`       | 0.65    | Loading MATLAB v7.3 `.mat` data files     |

# | `scikit-learn`| 1.3.2   | Evaluation metrics and utilities          |

# 

# \### 1.3 Classic EEG Model Dependencies

# 

# If you want to run the training code for \*\*classic EEG models\*\* (e.g. Riemannian / traditional ML baselines), you will also need:

# 

# | Package     | Version | Note                                           |

# |-------------|:-------:|------------------------------------------------|

# | `mne`       | 0.22.2  | EEG preprocessing and data handling            |

# | `pyriemann` | 0.3     | Riemannian geometry-based EEG classification   |

# | `lightgbm`  | 3.3.0   | Gradient boosting models for tabular features  |

# | `meegkit`   | 0.1.0   | EEG/MEG signal processing utilities           |

# | `scipy`     | 1.11.4  | General scientific computing utilities        |

# 

# \*(Versions above are examples; match them to your actual `requirements.txt` if needed.)\*

# 

# ---

# 

# \## 2. Model Specs

# 

# \*\*ST-EEGFormer\*\* is designed for \*\*128 Hz EEG data\*\*.

# 

# \- During pre-training, the model was trained to \*\*reconstruct 6-second EEG segments\*\*.  

# \- It supports up to \*\*145 EEG channels\*\*.  

# \- We recommend using it with \*\*segments up to 6 seconds\*\* sampled at \*\*128 Hz\*\*.

# 

# The list of available/pretrained channels can be found in:

# 

# ```text

# pretrain/senloc\_file

# 

