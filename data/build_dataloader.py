import torch
import copy
import os
from torch.utils.data import DataLoader
from data.datasets import RealSPFMDataset, NpzDataset
from utils.io_utils import instantiate_from_config

def build_dataloader(config, mode='train', backbone='transformer'):
    """
    Builds a dataloader using the new NpzDataset class.
    
    Args:
        config (dict): The configuration dictionary from the YAML file.

    Returns:
        dict: A dictionary containing the dataloader instance.
    """
    ds_key, dl_key = f"{mode}_dataset", f"{mode}_dataloader"
    is_train = (mode == 'train')

    # Deepcopy to prevent mutating the global config
    dataset_config = copy.deepcopy(config[ds_key])

    # Extract backbone-specific boundaries if they exist in params
    if 'boundaries' in dataset_config.get('params', {}):
        dataset_config['params']['custom_boundaries'] = dataset_config['params'].pop('boundaries', [])

    # Instantiate the dataset (RealSPFMDataset or NpzDataset) cleanly
    dataset = instantiate_from_config(dataset_config)

    # Get the dataloader configuration parameters
    dataloader_params = config.get(dl_key, {})

    # Instantiate the DataLoader
    dataloader = DataLoader(
        dataset,
        **dataloader_params,
        pin_memory=True,
        drop_last=is_train,
        shuffle=is_train
    )

    return {'dataloader': dataloader}
