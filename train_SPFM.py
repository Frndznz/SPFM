import os
import argparse
import copy
import torch
from utils.io_utils import load_yaml_config, load_SPFM_config, merge_opts_to_config, seed_everything, load_model
from utils.logger import Logger
from engine.solver_SPFM import SPFMTrainer as Trainer
from data.build_dataloader import build_dataloader
from models.transformer_1d import Transformer1DModel as Transformer
from models.mamba_1d import Mamba1DModel as Mamba
from models.SPFM import SPFM_Ensemble

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script for SPFM')
    parser.add_argument('--config_file', type=str, default='./configs/ptbxl_SPFM.yaml',
                        help='path of config file')
    # args for random
    parser.add_argument('--variant', type=str, default='Transformer')
    parser.add_argument('--data_type', type=str, default='ECG',
                        help='Type of data modality (e.g. ECG, EEG)')
    parser.add_argument('--arch_type', type=str, default='independent', help='independent or unified, defines how the SPFM Ensemble is initialized')
    parser.add_argument('--loss_types', nargs='+', default=['l1', 'l1'], 
                        help="Loss types for each partition (e.g., --loss_types l2 l1)")
    parser.add_argument('--prior', type=str, default='white')
    parser.add_argument('--seed', type=int, default=123, help='seed for initializing training.')
    parser.add_argument('--milestone', default=None, help='If set, trainer will load this ckpt.')
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    return args

def run_train(config, args, model, train_dl, test_dl) -> None:
    # Instantiate logger and save config
    logger = Logger(args, config)
    logger.save_config(config)
    logger.log_info(f'Instantiated Unified SPFM Ensemble with {args.variant} backbone, {args.loss_types} loss, and {args.prior} prior')

    if args.seed is not None:
        seed_everything(args.seed)

    model = model.cuda()

    trainer = Trainer(
            config=config,
            args=args,
            model=model,
            train_dataloader=train_dl,
            test_dataloader=test_dl,
            arch_type=args.arch_type,
            logger=logger
    )

    if args.milestone is not None:
        trainer.load(milestone=args.milestone)

    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    loss_list = args.loss_types 
    K = len(loss_list)
    model_key = f"{args.variant.lower()}_{args.data_type.lower()}-{K}"

    # Load configuration
    config = load_SPFM_config(args.config_file, key=model_key)
    
    config['solver']['prior'] = args.prior
    config = merge_opts_to_config(config, args.opts)
    config['train_dataset']['params']['K'] = K
    config['test_dataset']['params']['K'] = K

    # Instantiate Dataloaders to extract global stats
    train_dl = build_dataloader(config, mode='train', backbone=args.variant.lower())
    test_dl = build_dataloader(config, mode='test', backbone=args.variant.lower())

    dataset = train_dl['dataloader'].dataset

    # Extract partition properties from the loaded dataset
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
        solver=config['model'].get('solver', 'euler'),
        loss_types=loss_list,
        **config['model']['backbone']['params']
    )
    
    # Instantiate save dir
    strategies_dict = {'uniform': 'Uniform', 'equal_variance': 'Eq_Var'}

    # Setup directories
    loss_slug = "_".join(loss_list)
    if args.arch_type == 'independent':
        current_folder = f"{config['solver']['results_folder']}/SPFM/{strategies_dict[config['partition_strategy']]}/{args.variant}"
    elif args.arch_type == 'unified':
        current_folder = f"{config['solver']['results_folder']}/uSPFM/{strategies_dict[config['partition_strategy']]}/{args.variant}"
    
    args.save_dir = os.path.join(current_folder, f"{loss_slug}_{config['solver']['prior']}")
    os.makedirs(f"{args.save_dir}/logs/wandb", exist_ok=True)

    # Run the training loop
    run_train(config, args, model, train_dl, test_dl) 
