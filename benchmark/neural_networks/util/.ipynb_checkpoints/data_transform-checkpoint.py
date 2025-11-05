import numpy as np
from scipy.signal import resample

def resample_eeg(data, original_sfreq, new_sfreq):
    """
    Resample EEG data to a new sampling frequency.
    
    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_times)
    - original_sfreq: original sampling frequency (Hz)
    - new_sfreq: desired sampling frequency (Hz)
    
    Returns:
    - resampled_data: np.ndarray of shape (n_trials, n_channels, new_n_times)
    """
    n_trials, n_channels, n_times = data.shape
    new_n_times = int(round(n_times * new_sfreq / original_sfreq))

    # Resample each trial and channel
    resampled_data = resample(data, new_n_times, axis=2)
    
    return resampled_data

def standardize_per_channel_per_trial(data):
    """
    Standardize EEG data per trial and per channel.

    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    # Compute mean and std along the time axis (axis=2)
    mean = np.mean(data, axis=2, keepdims=True)  # shape: (trial, channel, 1)
    std = np.std(data, axis=2, keepdims=True)    # shape: (trial, channel, 1)

    # Avoid division by zero
    std[std == 0] = 1.0

    standardized_data = (data - mean) / std
    return standardized_data

def normalize_by_channel_percentile(data, percentile=95, eps=1e-6):
    """
    Normalize each channel in each trial by its absolute-amplitude percentile.

    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_trials, n_channels, n_times).
    percentile : float
        Which percentile of |amplitude| to use (default 95).
    eps : float
        Small constant to avoid division by zero (default 1e-6).

    Returns
    -------
    normalized : np.ndarray
        The same shape as `data`, where data[t, c, :] has been
        divided by its own percentile value.
    """
    # compute absolute values
    abs_data = np.abs(data)

    # compute the percentile along the time axis → shape (n_trials, n_channels)
    pvals = np.percentile(abs_data, percentile, axis=-1)

    # avoid division by zero
    pvals_safe = pvals + eps

    # broadcast and normalize
    normalized = data / pvals_safe[..., None]

    return normalized


def normalize_to_pm1_numpy(data, axis=-1, eps=1e-6):
    """
    Min–max normalize `data` so that along `axis`,
    min→−1 and max→+1.

    Parameters
    ----------
    data : np.ndarray
        Any shape, e.g. (n_trials, n_channels, n_times).
    axis : int
        Axis along which to compute min/max.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    normalized : np.ndarray
        Same shape as `data`, scaled to [−1, +1] along `axis`.
    """
    d_min = data.min(axis=axis, keepdims=True)
    d_max = data.max(axis=axis, keepdims=True)
    scale = (d_max - d_min) + eps
    returned = 2 * (data - d_min) / scale - 1
    return returned
    
class LabramDataTransformerWithChannelSelection:
    """
    Resample EEG to 200Hz for Labram
    Setting the unit to 0.1mV
    only use the supported channels
    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    
    def __init__(self, divisor, channel_idx, original_sfreq=128, new_sfreq=200):
        self.divisor = divisor
        self.channel_idx = channel_idx
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq

    def __call__(self, data):
        # Resample the time axis (axis=2)
        if self.original_sfreq != self.new_sfreq:
            resampled = resample_eeg(data, original_sfreq=self.original_sfreq, new_sfreq=self.new_sfreq)
            #print(f"resample from {self.original_sfreq} to {self.new_sfreq}")
        else:
            resampled = data
        # scaling it to be mostly (-1,1)
        standardized_data = resampled/self.divisor
        return standardized_data[:,self.channel_idx,:]
    
class EEGPTDataTransformerWithChannelSelection:
    """
    Resample EEG to 256Hz for EEGPT
    Setting the unit to 1mV
    only use the supported channels
    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    
    def __init__(self, divisor, channel_idx, original_sfreq=128, new_sfreq=256):
        self.divisor = divisor
        self.channel_idx = channel_idx
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq

    def __call__(self, data):
        # Resample the time axis (axis=2)
        if self.original_sfreq != self.new_sfreq:
            resampled = resample_eeg(data, original_sfreq=self.original_sfreq, new_sfreq=self.new_sfreq)
            #print(f"resample from {self.original_sfreq} to {self.new_sfreq}")
        else:
            resampled = data
        # scaling it to be mostly (-1,1)
        standardized_data = resampled/self.divisor
        return standardized_data[:,self.channel_idx,:]
    
class BIOTDataTransformer:
    """
    Resample EEG to 200Hz for BIOT
    95% normalization
    only use the supported channels
    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    
    def __init__(self, original_sfreq=128, new_sfreq=200):
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq

    def __call__(self, data):
        # Resample the time axis (axis=2)
        if self.original_sfreq != self.new_sfreq:
            resampled = resample_eeg(data, original_sfreq=self.original_sfreq, new_sfreq=self.new_sfreq)
            #print(f"resample from {self.original_sfreq} to {self.new_sfreq}")
        else:
            resampled = data
        # 95-percentile standardization
        standardized_data = normalize_by_channel_percentile(resampled)
        #print("original:", data.shape, "after transformation:", standardized_data.shape)
        return standardized_data
    
class BENDRDataTransformer:
    """
    Resample EEG to 256Hz for BENDR
    apply min-max -1 +1 normalization
    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    
    def __init__(self, original_sfreq=128, new_sfreq=256):
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq

    def __call__(self, data):
        # Resample the time axis (axis=2)
        if self.original_sfreq != self.new_sfreq:
            resampled = resample_eeg(data, original_sfreq=self.original_sfreq, new_sfreq=self.new_sfreq)
            #print(f"resample from {self.original_sfreq} to {self.new_sfreq}")
        else:
            resampled = data
        # min:-1, max:+1 normalization
        standardized_data = normalize_to_pm1_numpy(resampled)
        #print("original:", data.shape, "after transformation:", standardized_data.shape)
        return standardized_data
    
    
class CBraModDataTransformer:
    """
    Resample EEG to 200Hz for CBraMod
    apply normalization
    Setting the unit to 0.1mV (the same as labram) 
    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    
    def __init__(self, divisor=100, original_sfreq=128, new_sfreq=200):
        self.divisor = divisor
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq

    def __call__(self, data):
        # Resample the time axis (axis=2)
        if self.original_sfreq != self.new_sfreq:
            resampled = resample_eeg(data, original_sfreq=self.original_sfreq, new_sfreq=self.new_sfreq)
            #print(f"resample from {self.original_sfreq} to {self.new_sfreq}")
        else:
            resampled = data
        # min:-1, max:+1 normalization
        standardized_data = resampled/self.divisor

        return standardized_data
    
class ViTDataTransformerWithChannelSelection:
    """
    Resample EEG to 128Hz for STEEGFormer
    apply normalization and channel selection

    Parameters:
    - data: np.ndarray of shape (n_trials, n_channels, n_timepoints)

    Returns:
    - standardized_data: np.ndarray of the same shape as input
    """
    
    def __init__(self, channel_idx, original_sfreq=128, new_sfreq=128):
        self.channel_idx = channel_idx
        self.original_sfreq = original_sfreq
        self.new_sfreq = new_sfreq

    def __call__(self, data):
        # Resample the time axis (axis=2)
        if self.original_sfreq != self.new_sfreq:
            resampled = resample_eeg(data, original_sfreq=self.original_sfreq, new_sfreq=self.new_sfreq)
            #print(f"resample from {self.original_sfreq} to {self.new_sfreq}")
        else:
            resampled = data
        # min:-1, max:+1 normalization
        standardized_data = standardize_per_channel_per_trial(resampled)

        return standardized_data[:,self.channel_idx,:]