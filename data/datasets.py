import torch
import scipy.signal
import numpy as np
import os
from glob import glob
from torch.utils.data import Dataset

class RealSPFMDataset(Dataset):
    """ Generalized Dataset class for the Spectrally-Partitioned Flow Matching framework.
        Loads .npz files and pre-processes them by splitting into K frequency components 
        using a zero-phase 4th order Butterworth filter.
        
        Returns a dictionary consisting of 'x_partitions' [K, C, L] and 'y'.
    """
    def __init__(self, data_root, K, fs, filter_order, custom_boundaries, resource_name=None, epsilon=1e-8):
        """ Args:
            data_root: Path to the npz file or directory with npz files
            K: Number of frequency partitions
            fs: Sampling frequency of the data
            filter_order: Controls the sharpness of the filter's frequency cutoff
            custom_boundaries: List of cutoff frequencies [f_1, ..., f_K-1, f_max]
        """
        super().__init__()
        self.K = K
        self.fs = fs
        self.partition_boundaries = custom_boundaries

        data_path = os.path.join(data_root, resource_name)
        print(f"Initializing RealSPFMDataset from: {data_path}, partition boundaries: {custom_boundaries}")
        
        if os.path.isfile(data_path):
            data_npz = np.load(data_path)
            x_raw = data_npz['x'].astype(np.float32)
            y = data_npz['y']
        else: 
            samples, labels = [], []
            for s in sorted(glob(data_path + '/*.npz')):
                data_npz = np.load(s, allow_pickle=True)
                samples.append(data_npz['x'][:,:2,:])
                label = data_npz['y']
                label[label == 5] = 4
                labels.append(label)
            x_raw = np.concatenate(samples)
            y = np.concatenate(labels)

        print(f"Loaded raw data shape: {x_raw.shape}")
        N, C, L = x_raw.shape
        
        self.data = x_raw
        self.x_partitions = np.zeros((N, K, C, L))
        self.global_means = torch.zeros(K, C)
        self.global_stds = torch.zeros(K, C)
        
        current_signal = x_raw
        
        for k in range(self.K - 1):
            # idx maps to the specific band we are currently extracting
            idx = self.K - 1 - k 
            f_cut = self.partition_boundaries[idx - 1] # Use the boundary for this split
            
            # Use a Butterworth filter to extract the low-frequency component
            sos = scipy.signal.butter(filter_order, f_cut, btype='low', fs=fs, output='sos')
            low_part = scipy.signal.sosfiltfilt(sos, current_signal, axis=-1)
            
            # The band data is the high-frequency remainder of this split
            band_data = current_signal - low_part
            
            # Calculate and store global stats for model initialization
            self.global_means[idx] = torch.tensor(band_data.mean(axis=(0, 2)))
            self.global_stds[idx] = torch.tensor(band_data.std(axis=(0, 2)) + epsilon)
            
            # Sample-wise z-score normalization for training stability
            mu = band_data.mean(axis=-1, keepdims=True)
            std = band_data.std(axis=-1, keepdims=True) + epsilon
            self.x_partitions[:, idx, :, :] = (band_data - mu) / std
            
            # Continue splitting the remaining low-frequency component
            current_signal = low_part
            
        # Handle the final remaining (lowest) frequency band
        self.global_means[0] = torch.tensor(current_signal.mean(axis=(0, 2)))
        self.global_stds[0] = torch.tensor(current_signal.std(axis=(0, 2)) + epsilon)
        
        mu = current_signal.mean(axis=-1, keepdims=True)
        std = current_signal.std(axis=-1, keepdims=True) + epsilon
        self.x_partitions[:, 0, :, :] = (current_signal - mu) / std
        
        # Final tensors and labels[cite: 5]
        self.x_partitions = torch.from_numpy(self.x_partitions).float()
        self.global_means = self.global_means.float()
        self.global_stds = self.global_stds.float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x_partitions': self.x_partitions[idx],
            'y': self.y[idx]
            }

class NpzDataset(Dataset):
    """ General Dataset class for loading raw signals from npz files.
        It returns the raw data and its corresponding label: 'x', and 'y' as a dict.
    """
    def __init__(self, data_root, resource_name=None, **kwargs):
        """ Args:
            data_root: Path to the npz file or directory with npz files
            resource_name: e.g. ptbxl_train.npz or normalized_EEG/train
        """
        super().__init__()
        
        data_path = os.path.join(data_root, resource_name)

        print(f"Initializing NpzDataset from: {data_path}")
        # Load the raw data from the npz file
        if os.path.isfile(data_path):
            data_npz = np.load(data_path)
            x_raw = data_npz['x'].astype(np.float32)
            y = data_npz['y']
        else: # If it is a dir, e.g. EEG sleep data, load each npz file
            samples, labels = [], []
            for s in sorted(glob(data_path + '/*.npz')):
                data_npz = np.load(s, allow_pickle=True)
                samples.append(data_npz['x'][:,:2,:]) # Only the first 2 channels of EEGs
                label = data_npz['y']
                label[label == 5] = 4
                labels.append(label)
            x_raw = np.concatenate(samples)
            y = np.concatenate(labels)

        self.x = torch.from_numpy(x_raw)
        print(f"Loaded raw data shape: {x_raw.shape}")
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx]
            }
