import os
import copy
import sys
import yaml
import json
import torch
import random
import datetime
import warnings
import importlib
import numpy as np
from torch import optim

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()

def save_dict_to_json(d, path, indent=None):
    json.dump(d, open(path, 'w'), indent=indent)

def load_dict_from_json(path):
    return json.load(open(path, 'r'))

def write_args(args, path):
    args_dict = dict((name, getattr(args, name)) for name in dir(args)if not name.startswith('_'))
    with open(path, 'a') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()

def seed_everything(seed=123):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    print(f"Global seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def merge_opts_to_config(config, opts):
    def modify_dict(c, nl, v):
        if len(nl) == 1:
            c[nl[0]] = type(c[nl[0]])(v)
        else:
            # print(nl)
            c[nl[0]] = modify_dict(c[nl[0]], nl[1:], v)
        return c

    if opts is not None and len(opts) > 0:
        assert len(opts) % 2 == 0, "each opts should be given by the name and values! The length shall be even number!"
        for i in range(len(opts) // 2):
            name = opts[2*i]
            value = opts[2*i+1]
            config = modify_dict(config, name.split('.'), value)
    return config 

def modify_config_for_debug(config):
    config['dataloader']['num_workers'] = 0
    config['dataloader']['batch_size'] = 1
    return config

def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # format the numbers
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)
    
    format_dict(parameters)
    return parameters

def format_seconds(seconds):
    h = int(seconds // 3600)
    m = int(seconds // 60 - h * 60)
    s = int(seconds % 60)

    d = int(h // 24)
    h = h - d * 24

    if d == 0:
        if h == 0:
            if m == 0:
                ft = '{:02d}s'.format(s)
            else:
                ft = '{:02d}m:{:02d}s'.format(m, s)
        else:
           ft = '{:02d}h:{:02d}m:{:02d}s'.format(h, m, s)
 
    else:
        ft = '{:d}d:{:02d}h:{:02d}m:{:02d}s'.format(d, h, m, s)

    return ft

def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def class_from_string(class_name):
    module, cls = class_name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls

def get_all_file(dir, end_with='.h5'):
    if isinstance(end_with, str):
        end_with = [end_with]
    filenames = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            for ew in end_with:
                if f.endswith(ew):
                    filenames.append(os.path.join(root, f))
                    break
    return filenames

def get_sub_dirs(dir, abs=True):
    sub_dirs = os.listdir(dir)
    if abs:
        sub_dirs = [os.path.join(dir, s) for s in sub_dirs]
    return sub_dirs

def get_model_buffer(model):
    state_dict = model.state_dict()
    buffers_ = {}
    params_ = {n: p for n, p in model.named_parameters()}

    for k in state_dict:
        if k not in params_:
            buffers_[k] = state_dict[k]
    return buffers_

def load_FM_config(experiment_config, key=None, backbones_path='./configs/backbones.yaml', 
        datasets_path='./configs/datasets.yaml', mode='single'):
    """ Loads main experiment file ad resolves the references to datasets and backbones, 
        returning a unified config dict. Currently loads both train and test datasets.
        mode determines whether it is SPFM (2 models) or normal FM (1 model)
    """
    config = load_yaml_config(experiment_config)
    # Load the libraries
    backbones_lib = load_yaml_config(backbones_path)
    datasets_lib = load_yaml_config(datasets_path)
    source_name = config['dataset_source']
    source_params = datasets_lib['sources'][source_name]

    # Resolve datasets
    for split in ['train', 'test']:
        ds_key, dl_key = f"{split}_dataset", f"{split}_dataloader"
        loader_key = f"{split}_loader_type"
        loader = config[loader_key]
        ds_cfg = copy.deepcopy(datasets_lib['loaders'][loader])
        # Pull up to top level dataset_params
        if 'params' not in ds_cfg:
            ds_cfg['params'] = {}
        if split in source_params:
            ds_cfg['params']['resource_name'] = source_params[split]
        for k, v in source_params.items():
            if k not in ['train', 'test']:
                ds_cfg['params'][k] = v
        
        dl_cfg = copy.deepcopy(datasets_lib['defaults']['dataloader'])
        config[ds_key] = ds_cfg
        config[dl_key] = dl_cfg

    # Resolve backbones
    backbone_conf = copy.deepcopy(backbones_lib['models'][key])
    if mode == 'dual':
        for model_type in ['structure_model', 'detail_model']:
            config[model_type] = copy.deepcopy(config['model'])
            config[model_type]['backbone'] = backbone_conf
            
            # Delete backbones def from configs
            del config[model_type]['backbones']
        del config['model']['backbones']
    elif mode == 'single': # Double the size for the single-model FM
        backbone_conf['params']['n_layers'] = backbone_conf['params']['n_layers']*2
        config['model']['backbone'] = backbone_conf
    
    return config

def load_SPFM_config(experiment_config, key=None, backbones_path='./configs/backbones.yaml', 
        datasets_path='./configs/datasets.yaml', partitions_path='./configs/partitions.yaml'):
    """ Loads main experiment file and resolves the references to datasets and backbones, 
        returning a unified config dict designed for the SPFM_Ensemble architecture.
    """
    config = load_yaml_config(experiment_config)
    # Load the libraries
    backbones_lib = load_yaml_config(backbones_path)
    datasets_lib = load_yaml_config(datasets_path)
    partitions_lib = load_yaml_config(partitions_path)

    source_name = config['dataset_source']
    source_params = datasets_lib['sources'][source_name]
    K_val = source_params.get('K', 2)

    strategy = config.get('partition_strategy', 'early_gradient')

    # Determine backbone type name from the execution 'key' (e.g., 'transformer_ecg' -> 'transformer')
    backbone_type = None
    if key:
        for b_type in ['transformer', 'mamformer', 'mamba']:
            if b_type in key:
                backbone_type = b_type
                break

    # Extract the boundaries for this specific dataset, backbone, and K combination
    boundaries = []
    if source_name in partitions_lib and backbone_type:
        strategy_dict = partitions_lib[source_name].get(strategy, {})
        backbone_partitions = strategy_dict.get(backbone_type, {})
        
        # Robust lookup supporting integer keys (2), string keys ("2"), or prefixed keys ("K2")
        boundaries = backbone_partitions.get(K_val, 
                        backbone_partitions.get(str(K_val), 
                            backbone_partitions.get(f"K{K_val}", [])))
    # Resolve datasets
    for split in ['train', 'test']:
        ds_key, dl_key = f"{split}_dataset", f"{split}_dataloader"
        loader_key = f"{split}_loader_type"
        loader = config[loader_key]
        ds_cfg = copy.deepcopy(datasets_lib['loaders'][loader])
        
        # Pull up to top level dataset_params
        if 'params' not in ds_cfg:
            ds_cfg['params'] = {}
        if split in source_params:
            ds_cfg['params']['resource_name'] = source_params[split]
            
        for k, v in source_params.items():
            if k not in ['train', 'test']:
                ds_cfg['params'][k] = v
        
        # Inject the dynamically resolved flat list of boundaries into the dataset parameters
        ds_cfg['params']['boundaries'] = boundaries

        dl_cfg = copy.deepcopy(datasets_lib['defaults']['dataloader'])
        config[ds_key] = ds_cfg
        config[dl_key] = dl_cfg

    # Resolve backbone
    backbone_conf = copy.deepcopy(backbones_lib['models'][key])
    
    # Attach backbone directly to the main model config
    config['model']['backbone'] = backbone_conf
    
    # Clean up old keys if they accidentally exist in the YAML
    if 'backbones' in config['model']:
        del config['model']['backbones']
    
    return config

def load_model(model_config, loss_type=None):
    bb_conf = copy.deepcopy(model_config['backbone'])
    if loss_type:
        bb_conf['params']['loss_type'] = loss_type
    model_bb = instantiate_from_config(bb_conf)
    ModelClass = get_obj_from_str(model_config['target'])
    
    return ModelClass(model=model_bb, **model_config['params'])

