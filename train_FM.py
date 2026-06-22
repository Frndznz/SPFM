import os
import argparse
import yaml
import torch
from utils.io_utils import merge_opts_to_config, seed_everything, instantiate_from_config, load_FM_config, load_model
from utils.logger import Logger
from data.build_dataloader import build_dataloader
from engine.solver_FM import FMTrainer as Trainer
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training for Baseline FM')
    parser.add_argument('--config_file', type=str, default='./configs/ptbxl_FM.yaml')
    parser.add_argument('--variant', type=str, default='Transformer', 
                        help='Backbone variant to inject')
    parser.add_argument('--data_type', type=str, default='ECG',
                        help='Type of data modality (e.g. ECG, EEG)')
    parser.add_argument('--loss_type', type=str, default='standard', help='standard or freq_weighted')
    parser.add_argument('--prior', type=str, default='white')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--milestone', default=None)
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    return parser.parse_args()

def run_train(config, args, model):
    # Instantiate logger and save config
    logger = Logger(args, config)
    logger.save_config(config)
    logger.log_info(f'Instantiated FM model with {args.variant} backbone, {args.loss_type} loss, and {args.prior} prior.')

    if args.seed is not None:
        seed_everything(args.seed)
    
    # Instantiate DataLoaders
    train_dl = build_dataloader(config, 'train')
    test_dl = build_dataloader(config, 'test')
    
    model = model.cuda()
    
    # Initialize trainer
    trainer = Trainer(
        config=config,
        args=args,
        model=model,
        train_dataloader=train_dl,
        test_dataloader=test_dl,
        loss_type = args.loss_type,
        logger=logger
    )

    if args.milestone:
        trainer.load(args.milestone)

    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    model_key = f"{args.variant.lower()}_{args.data_type.lower()}-2" 
    config = load_FM_config(args.config_file, key=model_key)
    config['model']['backbone']['params']['loss_type'] = args.loss_type
    model = load_model(config['model'], loss_type=config['model']['params']['loss'])
    config['solver']['prior'] = args.prior
     
    config = merge_opts_to_config(config, args.opts)
    current_folder = config['solver']['results_folder']
    config['solver']['results_folder'] = f"{current_folder}/FM/{args.variant}"
    if args.loss_type == 'freq_weighted':
        loss_slug = f"fw_{config['model']['params']['loss']}_{config['solver']['prior']}"
    else:
        loss_slug = f"{config['model']['params']['loss']}_{config['solver']['prior']}"
    args.save_dir = os.path.join(config['solver']['results_folder'], loss_slug)
    os.makedirs(f"{args.save_dir}/logs/wandb", exist_ok=True)
    
    print(config)
    run_train(config, args, model)
