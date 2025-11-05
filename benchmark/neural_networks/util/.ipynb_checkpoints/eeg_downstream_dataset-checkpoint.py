# Codes written by Liuyin Yang (liuyin.yang@kuleuven.be)
# All rights reserved.
# --------------------------------------------------------

import torch
import numpy as np
import pickle
import h5py
import os
import time
import pandas as pd
from typing import Sequence, Any
from scipy.signal import resample

class DTUDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, fold=0, regression_task="dtu", segment_time = 3, new_fs=256, hop_time=0.1, train=True, class_label=None, transform=None, chan_info=None):
        """
        Args:
            data_path: path for the .h5 file
            fold (int): not used
            regression_task (str): which classification task to use 
            data_length (int, optional): data_length to use
            train (bool): If True, select trials for training (i.e. those NOT marked as test).
                          If False, select test trials.
            class_label (dict): maps the class names to integers
            transform (callable, optional): Any additional transformation to apply on the sample.
            chan_info (torch tensor, optional): Any additional channel information that will be passed into the model 
        """
        self.subjectName = os.path.splitext(os.path.basename(data_path))[0]
        self.fold = fold
        self.orginal_fs = 256
        self.new_fs = new_fs
        self.segmentT = segment_time
        self.data_length = int(self.new_fs*segment_time)
        self.regression_task = regression_task
        self.train = train
        if self.train:
            self.hop_len = int(hop_time*self.new_fs)
        else:
            self.hop_len = int(1/64*self.new_fs) #64 hz prediction 
        self.transform = transform
        self.chan_info = chan_info
        self.eeg_recordings = []
        self.wav_recordings = []
        self.train_idx = []
        # load the dataset
        with h5py.File(data_path, 'r') as f:
            eeg_group = f['eeg']
            wav_group = f['wavA']
            n_trials = len(eeg_group)
            
            for i in range(n_trials):
                eeg = np.transpose(eeg_group[f'trial_{i}'][:])   # shape: [C, T]
                wav = wav_group[f'trial_{i}'][:]   # shape: [1, T]
                
                # Standardize across time axis
                mean = np.mean(wav, axis=1, keepdims=True)      # shape: [1, 1]
                std  = np.std(wav, axis=1, keepdims=True) + 1e-8  # add epsilon to avoid division by zero
                wav_standardized = (wav - mean) / std           # still shape: [1, T]
                
                # perform data transform:
                if self.transform:
                    # 1) add a dummy batch dimension → shape (1, C, L)
                    eeg_batch = eeg[np.newaxis, :, :]
                    eeg_batch = self.transform(eeg_batch)
                    # 3) squeeze back to (C, L)
                    eeg_transform = np.squeeze(eeg_batch, axis=0)
                else:
                    eeg_transform = eeg
                    
                if new_fs!=self.orginal_fs:
                    # resample the wav to the same fs
                    n_channels, n_times = wav_standardized.shape
                    new_n_times = int(round(n_times * self.new_fs / self.orginal_fs))
                    resampled_wav = resample(wav_standardized, new_n_times, axis=1)
                else:
                    resampled_wav = wav_standardized
                assert resampled_wav.shape[-1] == eeg_transform.shape[-1]
                        
                self.eeg_recordings.append(eeg_transform)
                self.wav_recordings.append(resampled_wav)
                
                total_len = eeg_transform.shape[1]
                max_start = total_len - self.data_length
                if max_start <= 0:
                    continue
                # Determine train/test split range
                if train:
                    end = int(0.8 * max_start)
                    starts = range(0, end + 1, self.hop_len)
                else:
                    start = int(0.8 * max_start)
                    starts = range(start, max_start + 1, self.hop_len) # in training, predict every step
                    
                for s in starts:
                    self.train_idx.append((i,s))
                
            print(f"In total: {len(self.train_idx)} training examples")

    def __len__(self):
        return len(self.train_idx)
    
    def __getitem__(self, idx):
        trail_idx, pos = self.train_idx[idx]
        eeg_win = self.eeg_recordings[trail_idx][:,pos:pos + self.data_length]
        wav_win = self.wav_recordings[trail_idx][:,pos:pos + self.data_length]
        eeg_tensor = torch.from_numpy(eeg_win).float()    # shape: [C,T]
        wav_tensor = torch.from_numpy(wav_win).float() # shape: [1,T]
        if self.chan_info is not None:
            return  eeg_tensor,  wav_tensor, self.chan_info
        else:
            return  eeg_tensor,  wav_tensor
        
        
class UpperLimbDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, fold=0, classification_task="motorimagination", data_length=None, train=True, class_label=None, transform=None, chan_info=None):
        """
        Args:
            data_path: path for the .h5 file
            fold (int): The index (0 to 4) specifying which CV fold parameters to use.
            classification_task (str): which classification task to use 
            data_length (int, optional): data_length to use
            train (bool): If True, select trials for training (i.e. those NOT marked as test).
                          If False, select test trials.
            class_label (dict): maps the class names to integers
            transform (callable, optional): Any additional transformation to apply on the sample.
            chan_info (torch tensor, optional): Any additional channel information that will be passed into the model 
        """
        self.subjectName = os.path.splitext(os.path.basename(data_path))[0]
        self.fold = fold
        self.classification_task = classification_task
        self.data_length = data_length
        self.train = train
        self.class_label = class_label
        self.transform = transform
        self.chan_info = chan_info
        
        # load the dataset
        with h5py.File(data_path,'r') as f5:
            X_loaded = f5['X'][()]
            classes   = f5['df/class'][()].astype(str)
            # get the current fold indices
            if self.train:
                indices = f5[f"folds/{classification_task}/fold_{fold}/train"][()]
            else:
                indices = f5[f"folds/{classification_task}/fold_{fold}/test"][()]
            self.data = X_loaded[indices,:,:]
            self.labels = classes[indices]
        
        if self.transform:
            self.data = self.transform(self.data)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        trial_data = self.data[idx,:,:]
        label = self.labels[idx]
        
        if self.data_length:
            trial_data = trial_data[:,:self.data_length]
            
        if self.chan_info is not None:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label], self.chan_info
        else:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label]

        
class ErrorDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, fold=0, classification_task="error", data_length=None, train=True, class_label=None, transform=None, chan_info=None):
        """
        Args:
            data_path: path for the .h5 file
            fold (int): The index (0 to 4) specifying which CV fold parameters to use.
            data_length (int, optional): data_length to use
            train (bool): If True, select trials for training (i.e. those NOT marked as test).
                          If False, select test trials.
            class_label (dict): maps the class names to integers
            transform (callable, optional): Any additional transformation to apply on the sample.
            chan_info (torch tensor, optional): Any additional channel information that will be passed into the model 
        """
        self.subjectName = os.path.splitext(os.path.basename(data_path))[0]
        self.fold = fold
        self.classification_task = classification_task
        self.data_length = data_length
        self.train = train
        self.class_label = class_label
        self.transform = transform
        self.chan_info = chan_info
        
        # load the dataset
        with h5py.File(data_path,'r') as f5:
            X_loaded = f5['X'][()]
            # 2) Load metadata from the /df group
            trial_idx = f5["df"]["trial_idx"][:]      # integer array
            classes   = f5["df"]["class"][:].astype(str)  # string array
            sets      = f5["df"]["set"][:].astype(str)    # "train" or "test"
            # get the current fold indices
            if self.train:
                train_mask = (sets == "train")
                self.data = X_loaded[train_mask]
                self.labels = classes[train_mask]
            
            else:
                test_mask  = (sets == "test")
                self.data = X_loaded[test_mask]
                self.labels = classes[test_mask]
        
        if self.transform:
            self.data = self.transform(self.data)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        trial_data = self.data[idx,:,:]
        label = self.labels[idx]
        
        if self.data_length:
            trial_data = trial_data[:,:self.data_length]
            
        if self.chan_info is not None:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label], self.chan_info
        else:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label]
        
class InnerSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, fold=0, classification_task="innerSpeech", data_length=None, train=True, class_label=None, transform=None, chan_info=None):
        """
        Args:
            data_path: path for the .h5 file
            fold (int): The index (0 to 4) specifying which CV fold parameters to use.
            data_length (int, optional): data_length to use
            train (bool): If True, select trials for training (i.e. those NOT marked as test).
                          If False, select test trials.
            class_label (dict): maps the class names to integers
            transform (callable, optional): Any additional transformation to apply on the sample.
            chan_info (torch tensor, optional): Any additional channel information that will be passed into the model 
        """
        self.subjectName = os.path.splitext(os.path.basename(data_path))[0]
        self.fold = fold
        self.classification_task = classification_task
        self.data_length = data_length
        self.train = train
        self.class_label = class_label
        self.transform = transform
        self.chan_info = chan_info
        
        # load the dataset
        with h5py.File(data_path, 'r') as f5:
            # 1) Full data & labels
            X_loaded = f5['X'][:]                  # shape (n_trials, n_ch, n_times)
            labels = f5['df/label'][:]      # array of bytes
            labels = labels.astype(str)    # → strings like "b'Izquierda/Imagined'"
            # get the current fold indices
            grp = f5[f'folds/fold_{fold}']
            train_idx = grp['train'][:]
            test_idx  = grp['test'][:]
            if self.train:
                # 3) Subset
                self.data = X_loaded[train_idx]
                self.labels = labels[train_idx]
            else:
                self.data  = X_loaded[test_idx]
                self.labels  = labels[test_idx]
        
        if self.transform:
            self.data = self.transform(self.data)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        trial_data = self.data[idx,:,:]
        label = self.labels[idx]
        
        if self.data_length:
            trial_data = trial_data[:,:self.data_length]
            
        if self.chan_info is not None:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label], self.chan_info
        else:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label]
        



class BinocularSSVEPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        fold: int = 0,
        classification_task: str = "sync",   # "sync" or "async"
        data_length: int = 250,             # window length in samples
        train: bool = True,
        class_label: dict = None,            # maps target → int
        transform=None,
        chan_info: torch.Tensor = None
    ):
        # basic attrs
        self.subjectName      = os.path.splitext(os.path.basename(data_path))[0]
        self.fold             = fold
        self.task             = classification_task
        self.data_length      = data_length
        self.train            = train
        self.class_label      = class_label or {}
        self.transform        = transform
        self.chan_info        = chan_info

        # load pickled data
        with open(data_path, "rb") as f:
            sub = pickle.load(f)
        # raw array: (num_targets, num_epochs, num_channels, num_time)
        self.data = sub["data"]
        # sliding-window metadata: DataFrame with columns
        #   example_idx, target, epoch, start_sample
        self.df   = sub["df"]
        # splits: list of (train_idx, sync_idx, async_idx)
        splits = sub["splits"]
        train_idx, sync_idx, async_idx = splits[fold]

        # pick which indices we want
        if self.train:
            self.example_idx = train_idx
        elif self.task == "sync":
            self.subjectName = self.subjectName+"_sync"
            self.example_idx = sync_idx
        else:
            self.subjectName = self.subjectName+"_async"
            self.example_idx = async_idx


    def __len__(self):
        return len(self.example_idx)

    def __getitem__(self, idx):
        # look up the metadata row
        df_row_idx = self.example_idx[idx]
        row = self.df.iloc[df_row_idx]
        t   = int(row["target"])      # which target (1…40)
        e   = int(row["epoch"])       # epoch index (0…4)
        s   = int(row["start_sample"])# start sample within that epoch

        # slice out the window
        # data[t, e, :, :] has shape (num_channels, num_time)
        arr = self.data[t, e, :, :]
        if self.data_length:
            arr = arr[:, s : s + self.data_length]
        else:
            arr = arr[:, s:]
        
        # 1) add a dummy batch dimension → shape (1, C, L)
        arr_batch = arr[np.newaxis, :, :]

        # 2) apply your numpy‐based transform (must accept & return shape (N, C, L))
        if self.transform:
            arr_batch = self.transform(arr_batch)

        # 3) squeeze back to (C, L)
        arr_out = np.squeeze(arr_batch, axis=0)

        # to torch tensor
        x = torch.from_numpy(arr_out).float()
        # map raw target → class int (or default to raw target)
        y = self.class_label[t]

        if self.chan_info is not None:
            return x, y, self.chan_info
        else:
            return x, y


class BCI2aDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        fold: int = 0,
        classification_task: str = "bci_iv2a",   # "sync" or "async"
        data_length: int = 1024,             # window length in samples
        train: bool = True,
        class_label: dict = None,            # maps target → int
        transform=None,
        chan_info: torch.Tensor = None
    ):
        # basic attrs
        self.subjectName      = os.path.splitext(os.path.basename(data_path))[0]
        self.fold             = fold
        self.task             = classification_task
        self.data_length      = data_length
        self.train            = train
        self.class_label      = class_label or {}
        self.transform        = transform
        self.chan_info        = chan_info

        # load pickled data
        with open(data_path, "rb") as f:
            sub = pickle.load(f)
            
        if self.train:
            self.data = sub["trainX"]
            self.labels = sub["trainY"]
        else:
            self.data = sub["testX"]
            self.labels = sub["testY"]    
            
        if self.transform:
            self.data = self.transform(self.data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        trial_data = self.data[idx,:,:]
        label = self.labels[idx]
        
        if self.data_length:
            trial_data = trial_data[:,:self.data_length]
            
        if self.chan_info is not None:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label], self.chan_info
        else:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[label]

        
class AlzheimerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        fold: int = 0,
        classification_task: str = "alzheimer",  
        data_length: int = 768,             # window length in samples
        train: bool = True,
        class_label: dict = None,            # maps target → int
        transform=None,
        chan_info: torch.Tensor = None
    ):
        # basic attrs
        self.subjectName      = os.path.splitext(os.path.basename(data_path))[0]
        self.fold             = fold
        self.task             = classification_task
        self.data_length      = data_length
        self.train            = train
        self.class_label      = class_label or {}
        self.transform        = transform
        self.chan_info        = chan_info

        # load pickled data
        with open(data_path, "rb") as f:
            sub = pickle.load(f)
            
        self.data = sub["eeg"]
        self.label = sub["group"]    
            
        if self.transform:
            self.data = self.transform(self.data)
        
        total_data = self.data.shape[0]
        train_num = int(0.8*total_data)
        if self.train:
            self.data = self.data[:train_num,:,:]
        else:
            self.data = self.data[train_num:,:,:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        trial_data = self.data[idx,:,:]
        
        if self.data_length:
            trial_data = trial_data[:,:self.data_length]
            
        if self.chan_info is not None:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[self.label], self.chan_info
        else:
            return  torch.from_numpy(trial_data).type(torch.FloatTensor),  self.class_label[self.label]