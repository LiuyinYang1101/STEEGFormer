import torch
import numpy as np
import pickle
import h5py
import os
import time
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


class SequentialLoader:
    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader
            
def get_pretrain_dataset(datasetName):
    if datasetName=="custom1":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom1", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom1/pretrain_data_indices.pkl", dataset_name="custom1", segment_length=int(5*128))
        return dataset
        
    elif datasetName=="custom2":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom2", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom2/pretrain_data_indices.pkl", dataset_name="custom2", segment_length=int(5*128))
        return dataset
        
    elif datasetName=="custom3":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom3", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom3/pretrain_data_indices.pkl", dataset_name="custom3", segment_length=int(5*128))
        return dataset

    elif datasetName=="custom4":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom4", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom4/pretrain_data_indices.pkl", dataset_name="custom4", segment_length=int(5*128))
        return dataset

    elif datasetName=="custom5":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom5", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom5/pretrain_data_indices.pkl", dataset_name="custom5", segment_length=int(5*128))
        return dataset

    elif datasetName=="custom6":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom6", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/custom6/pretrain_data_indices.pkl", dataset_name="custom6", segment_length=int(5*128))
        return dataset
        
    elif datasetName=="ecog_eeg_train":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/ecog_eeg/eeg_train", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/ecog_eeg/eeg_train/pretrain_data_indices.pkl", dataset_name="ecog_eeg", segment_length=int(5*128))
        return dataset
        
    elif datasetName=="ecog_eeg_valid":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/ecog_eeg/eeg_valid", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/ecog_eeg/eeg_valid/pretrain_data_indices.pkl", dataset_name="ecog_eeg", segment_length=int(5*128))
        return dataset
        
    elif datasetName=="ssvep":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/ssvep", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/ssvep/pretrain_data_indices.pkl", dataset_name="ssvep", segment_length=int(2*128))
        return dataset
    
    elif datasetName=="p300":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/p300", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/p300/pretrain_data_indices.pkl", dataset_name="p300", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="seed":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/seed", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/seed/pretrain_data_indices.pkl", dataset_name="seed", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="large_im":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/large_im", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/large_im/pretrain_data_indices.pkl", dataset_name="large_im", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="hgd_train":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/hgd", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/hgd/pretrain_data_indices.pkl", dataset_name="hgd", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="hgd_test":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/hgd", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/hgd/pretrain_test_data_indices.pkl", dataset_name="hgd", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="eeg_mi_bci":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/eeg_mi_bci", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/eeg_mi_bci/pretrain_data_indices.pkl", dataset_name="eeg_mi_bci", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="bci_comp_iv2a_train":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2a", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2a/pretrain_data_indices.pkl", dataset_name="bci_comp_iv2a", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="bci_comp_iv2a_test":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2a", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2a/pretrain_test_data_indices.pkl", dataset_name="bci_comp_iv2a", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="bci_comp_iv2b_train":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2b", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2b/pretrain_data_indices.pkl", dataset_name="bci_comp_iv2b", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="bci_comp_iv2b_test":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2b", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_comp_iv2b/pretrain_test_data_indices.pkl", dataset_name="bci_comp_iv2b", segment_length=int(5*128))
        return dataset

    elif datasetName=="auditory":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/auditory", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/auditory/pretrain_data_indices.pkl", dataset_name="auditory", segment_length=int(5*128))
        return dataset
    
    elif datasetName=="bci_meditation":
        dataset = CustomPretrainDataset(data_folder_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_meditation", 
                                        indice_path="/dodrio/scratch/projects/2025_038/optimized_pretrain_data/bci_meditation/pretrain_data_indices.pkl", dataset_name="bci_meditation", segment_length=int(5*128))
        return dataset
    
    else:
        return None

class CustomPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder_path, indice_path, dataset_name, segment_length):
        self.data_folder = data_folder_path
        self.dataset_name = dataset_name
        self.segment_length = segment_length


        # Load the indices as a pandas dataframe
        with open(indice_path, 'rb') as file:
            indice_df = pickle.load(file)
            self.recordings_list = indice_df['Recording'].tolist()
            self.positions_list = indice_df['Position'].tolist()

        # Load the channel indices
        with open("/dodrio/scratch/projects/2025_038/mae/senloc_file/sen_chan_idx.pkl", 'rb') as file:
            all_senloc = pickle.load(file)
            self.senloc = torch.from_numpy(all_senloc[dataset_name]).type(torch.IntTensor)
            self.num_channel = self.senloc.shape[0]

        # Placeholder for HDF5 file handles (initialized per worker)
        self.dataset = None

    def __len__(self):
        return len(self.positions_list)

    def _initialize_recordings_dict(self):
        # Initialize the recordings dictionary with file handles for each unique HDF5 file
        self.dataset = h5py.File(self.data_folder+"/"+"consolidated_data.h5", 'r', swmr=True)

    def __getstate__(self):
        # Called when a new worker process is spawned
        state = self.__dict__.copy()
        # Do not pickle the file handles
        state['recordings_dict'] = None
        return state

    def __setstate__(self, state):
        # Called to initialize the worker-specific state
        self.__dict__.update(state)
        self._initialize_recordings_dict()

    def __getitem__(self, idx):
        # Initialize the recordings_dict if it hasn't been done
        if self.dataset is None:
            #startT=time.time()
            self._initialize_recordings_dict()
            #print(self.dataset_name, "init T:", time.time()-startT)

        # Get recording name and position for the current sample
        recording = self.recordings_list[idx]
        pos = self.positions_list[idx]

        if recording.endswith(".h5"):
            eeg_segment = np.transpose(self.dataset[recording][pos:pos + self.segment_length,:])
        else:
            eeg_segment = np.transpose(self.dataset[str(recording)][pos:pos + self.segment_length,:])
        # Check for empty segments
        if eeg_segment.shape[-1] == 0:
            print(idx, pos, eeg_memmap.shape)

        # Clamp EEG values to ±500 if exceeds the range
        eeg_segment = np.clip(eeg_segment, -500, 500)
    
        # Standardize per channel (axis=1 after transpose)
        mean = np.mean(eeg_segment, axis=1, keepdims=True)
        std = np.std(eeg_segment, axis=1, keepdims=True) + 1e-8  # avoid divide-by-zero
        eeg_segment = (eeg_segment - mean) / std
    
        # Convert to torch tensor and return
        return torch.from_numpy(eeg_segment).type(torch.FloatTensor), self.senloc


class MultiDatasetBatchSampler(Sampler):
    def __init__(self, datasets, batch_size, shuffle=True, distributed=False, num_replicas=None, rank=None):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.distributed = distributed
        self.dataset_indices = []
        self.samplers = []
        
        if distributed:
            assert num_replicas is not None and rank is not None, "num_replicas and rank must be provided in DDP mode"
            # Use DistributedSampler for each dataset
            self.samplers = [
                DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=self.shuffle)
                for dataset in datasets
            ]
        else:
            # Standard shuffling for single GPU training
            for dataset in datasets:
                indices = torch.randperm(len(dataset)).tolist() if shuffle else list(range(len(dataset)))
                self.dataset_indices.append(indices)

    def set_epoch(self, epoch):
        # If distributed, set epoch for each DistributedSampler
        if self.distributed:
            for sampler in self.samplers:
                sampler.set_epoch(epoch)

    def __iter__(self):
        dataset_pointers = [0] * len(self.datasets)
        
        if self.distributed:
            # Use the DistributedSampler's indices for each dataset
            for sampler in self.samplers:
                indices = list(sampler)  # Get distributed indices for this epoch
                self.dataset_indices.append(indices)

        batches = []
        while any(len(indices) > ptr for indices, ptr in zip(self.dataset_indices, dataset_pointers)):
            for dataset_idx in range(len(self.datasets)):
                indices = self.dataset_indices[dataset_idx]
                ptr = dataset_pointers[dataset_idx]
                
                if ptr < len(indices):
                    batch = indices[ptr:ptr + self.batch_size]
                    batches.append([(dataset_idx, idx) for idx in batch])
                    dataset_pointers[dataset_idx] += len(batch)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.distributed:
            # In distributed mode, sum the lengths of all samplers
            return sum(len(sampler) for sampler in self.samplers)
        else:
            # Return the total number of batches in non-distributed mode
            total_samples = sum(len(indices) for indices in self.dataset_indices)
            return (total_samples + self.batch_size - 1) // self.batch_size




    
class MultiDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.total_length = sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, index):
        dataset_idx, sample_idx = index
        return self.datasets[dataset_idx][sample_idx]  # Retrieve sample from the appropriate dataset

    def __len__(self):
        # Return the sum of lengths of all datasets
        return self.total_length
    
    
class MultiEEGDataLoader:
    def __init__(self, dataloaders, distributed=False):
        self.dataloaders = dataloaders
        self.iterators = [iter(dl) for dl in dataloaders]
        self.done_dataloaders = [False] * len(dataloaders)
        self.distributed = distributed
        self.num_dataloaders = len(dataloaders)
        self.current_dataloader = 0
        
    def set_epoch(self, epoch):
        for dl in self.dataloaders:
            if self.distributed:
                if isinstance(dl.sampler, DistributedSampler):
                    dl.sampler.set_epoch(epoch)

    def __iter__(self):
        self.iterators = [iter(dl) for dl in self.dataloaders]
        self.done_dataloaders = [False] * len(self.dataloaders)
        self.current_dataloader = 0
        return self

    def __next__(self):
        if all(self.done_dataloaders):
            raise StopIteration

        while all(self.done_dataloaders)==False:
            if not self.done_dataloaders[self.current_dataloader]:
                try:
                    batch = next(self.iterators[self.current_dataloader])
                    self.current_dataloader = (self.current_dataloader + 1) % self.num_dataloaders
                    return batch
                except StopIteration:
                    self.done_dataloaders[self.current_dataloader] = True

            self.current_dataloader = (self.current_dataloader + 1) % self.num_dataloaders

        raise StopIteration

    def __len__(self):
        return sum(len(dl) for dl in self.dataloaders)