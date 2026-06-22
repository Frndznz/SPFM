import os
import argparse
import torch
import copy
import itertools
from utils.io_utils import load_SPFM_config, merge_opts_to_config, seed_everything
from utils.logger import Logger
from engine.solver_SPFM import SPFMTrainer as Trainer
from train_SPFM import run_train

# New imports for dataloaders and unified ensemble logic
from data.build_dataloader import build_dataloader
from models.transformer_1d import Transformer1DModel as Transformer
from models.mamba_1d import Mamba1DModel as Mamba
from models.mamformer_1d import MambaTransformer1DModel as Mamformer
from models.SPFM import SPFM_Ensemble

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Script for Ablation Study of Losses for SFM')
    parser.add_argument('--config_file', type=str, default='./configs/ptbxl_uSPFM.yaml',
                        help='path of base config file')
    parser.add_argument('--name', type=str, default='PTBXL', help='name of dataset')
    parser.add_argument('--max_steps', type=int, default=100000, help='max train steps')
    parser.add_argument('--milestone', default=None, help='If set, loads checkpoint from milestone')
    parser.add_argument('--task_id', type=int, default=0, help='Slurm Array Task ID (0-8)')
    parser.add_argument('--variant', type=str, default='Transformer')
    parser.add_argument('--data_type', type=str, default='ECG')
    parser.add_argument('--arch_type', type=str, default='independent', help='independent or unified, defines how the SPFM Ensemble is initialized')
    parser.add_argument('--prior', type=str, default='white')
    parser.add_argument('--seed', type=int, default=123, help='seed for initializing training.')
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    
    # Define the grid
    loss_types = ["l1", "l2"]
    K = len(loss_types)
    loss_combinations = list(itertools.product(loss_types, loss_types))
    
    if args.task_id < 0 or args.task_id >= len(loss_combinations):
        print(f"Task ID {args.task_id} is out of bounds. Exiting.")
        exit(0)

    lf_loss, hf_loss = loss_combinations[args.task_id]
    loss_list = [lf_loss, hf_loss] # Current ablation combination
    print(f"Starting ablation task {args.task_id}: LF loss: {lf_loss} | HF loss: {hf_loss}")
    
    # Load and merge config
    model_key = f"{args.variant.lower()}_{args.data_type.lower()}-{K}"
    config = load_SPFM_config(args.config_file, key=model_key)
    config = merge_opts_to_config(config, args.opts)
    config['train_dataset']['params']['K'] = K
    config['test_dataset']['params']['K'] = K

    # Inject ablation parameters
    run_config = copy.deepcopy(config)
    run_config['solver']['max_steps'] = args.max_steps
    run_config['solver']['prior'] = args.prior

    # Instantiate Dataloaders (Required for run_train and extracting stats)
    train_dl = build_dataloader(run_config, mode='train', backbone=args.variant.lower())
    test_dl = build_dataloader(run_config, mode='test', backbone=args.variant.lower())
    
    dataset = train_dl['dataloader'].dataset
    means = dataset.global_means
    stds = dataset.global_stds

    # Instantiate the unified SPFM Ensemble
    if args.variant.lower() == 'transformer':
        bb_class = Transformer 
    elif args.variant.lower() == 'mamba':
        bb_class = Mamba
    elif args.variant.lower() == 'mamformer':
        bb_class = Mamformer
    
    model = SPFM_Ensemble(
        K=K,
        backbone_class=bb_class,
        means=means,
        stds=stds,
        solver=run_config['model'].get('solver', 'euler'),
        loss_types=loss_list, # The ablation losses
        **run_config['model']['backbone']['params']
    )
    
    partitions_dict = {'equal_variance': 'Eq_Var', 'uniform': 'Uniform', 'early_gradient': 'Early_Grad'}
    # Setup directories
    loss_slug = "_".join(loss_list)
    base_folder = run_config['solver']['results_folder']
    run_config['solver']['results_folder'] = os.path.join(base_folder, "Unified_SPFM", partitions_dict[run_config['partition_strategy']],  args.variant)
    
    run_args = copy.deepcopy(args)
    run_args.loss_types = loss_list
    run_args.save_dir = os.path.join(run_config['solver']['results_folder'], f"{loss_slug}_{args.prior}")
    os.makedirs(f"{run_args.save_dir}/logs/wandb", exist_ok=True)
    
    # Check if checkpoint exists
    ckpt_name = f"ckpt-{args.max_steps}.pt"
    if os.path.exists(os.path.join(run_args.save_dir, ckpt_name)): 
        print(f"Target checkpoint {ckpt_name} already exists. Skipping.")
        exit(0)

    # Execute Training with updated signature
    run_train(
        config=run_config, 
        args=run_args, 
        model=model,
        train_dl=train_dl,
        test_dl=test_dl
    )

    torch.cuda.empty_cache()
