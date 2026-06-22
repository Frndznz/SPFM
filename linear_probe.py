import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
import numpy as np
import argparse
from tqdm.auto import tqdm
from ema_pytorch import EMA

from momentfm import MOMENTPipeline
from utils.io_utils import load_SPFM_config, seed_everything
from data.build_dataloader import build_dataloader
from utils.priors import PriorGenerator
from utils.classification_utils import *
from models.SPFM import SPFM_Ensemble
from models.transformer_1d import Transformer1DModel as Transformer
from models.mamba_1d import Mamba1DModel as Mamba
from models.mamformer_1d import MambaTransformer1DModel as Mamformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training for Baseline FM')
    parser.add_argument('--dataset', type=str, default='PTBXL')
    parser.add_argument('--data_type', type=str, default='ECG',
                        help='Type of data modality (e.g. ECG, EEG)')
    parser.add_argument('--prior', type=str, default='white')
    parser.add_argument('--num_runs', type=int, default=5)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_runs = [
        {"framework": "SPFM", "dataset": "PTBXL", "partitioning": "Uniform",
            "loss_types": ["l2", "l2"], "backbone": "Mamformer"},
        {"framework": "SPFM", "dataset": "Chapman", "partitioning": "Uniform",  
            "loss_types": ["l2", "l2"], "backbone": "Mamformer"},
        {"framework": "SPFM", "dataset": "ISRUC", "partitioning": "Uniform",  
            "loss_types": ["l1", "l2"], "backbone": "Mamba"},
        {"framework": "SPFM", "dataset": "Sleep-EDF", "partitioning": "Uniform", 
            "loss_types": ["l1", "l2"], "backbone": "Mamba"}
    ]

    runs_to_eval = [r for r in best_runs if r["dataset"] == args.dataset]

    bb_classes  = {'Transformer': Transformer, 'Mamba': Mamba, 'Mamformer': Mamformer}

    class_config = load_SPFM_config(f"./configs/{args.dataset.lower()}_class.yaml", 
            key=f"transformer_{args.data_type.lower()}")
    
    if args.data_type == "ECG":
        n_classes, n_channels = 7, 12
    elif args.data_type == "EEG":
        n_classes, n_channels = 5, 2
    
    train_dl = build_dataloader(class_config, mode='train')['dataloader']
    test_dl = build_dataloader(class_config, mode='test')['dataloader']

    moment_model = get_moment_model(n_classes=n_classes, n_channels=n_channels, 
            task="embedding", device=device)
    x_real, y_real = extract_moment_embeddings(train_dl, moment_model, device, is_test=False)
    x_test, y_test = extract_moment_embeddings(test_dl, moment_model, device, is_test=True)

    all_results = []
    print(f"Running real-only baseline for dataset {args.dataset}")
    for run in range(args.num_runs):
        seed_everything(123+run)
        metrics_real = train_and_eval_linear_probe(x_real, y_real, x_test, y_test)
        all_results.append({
            "Backbone": "-", "Loss": "-", "Framework": "-", "Partitioning": "-",
        "Trial": "TRTR (Real-Only)", "Run": run,
        **metrics_real,

        })
        
    for run, run_cfg in enumerate(runs_to_eval):
        # Load model checkpoint
        loss_slug = "_".join(run_cfg['loss_types'])
        base_dir  = f"./experiments/{run_cfg['dataset']}/{run_cfg['framework']}/{run_cfg['partitioning']}/{run_cfg['backbone']}/{loss_slug}_white"
        with open(f"{base_dir}/configs/config.yaml", 'r') as f:
            base_config = yaml.safe_load(f)
        
        K = base_config['train_dataset']['params']['K']
        model = SPFM_Ensemble(
                K=K,
                backbone_class = bb_classes[run_cfg['backbone']],
                means = torch.zeros(K, n_channels, device=device),
                stds = torch.zeros(K, n_channels, device=device),
                solver = 'euler',
                loss_types = run_cfg['loss_types'],
                **base_config['model']['backbone']['params']
        ).to(device)
        ckpt = torch.load(f"{base_dir}/ckpt-100000.pt", map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        emas = [EMA(model.models[k], beta=0.995, update_every=10).to(device) for k in range(K)]
        for k in range(K):
            emas[k].load_state_dict(ckpt['ema'][k])
            emas[k].to(device).eval()
            model.models[k] = emas[k]

        prior = PriorGenerator(prior_type=base_config['solver']['prior'])
        
        # Perform 
        for run in range(args.num_runs):
            seed_everything(123 + run)
            print(f"[{run_cfg['backbone']} | {loss_slug} | run {run + 1}/{args.num_runs}")
            
            syn_dataset, real_dataset = generate_classification_data(
                real_dataloader=train_dl,
                model=model, 
                prior=prior,
                data_type=args.data_type,
                device=device,
                sampling_steps=[20, 30]
            )
            train_syn_loader = torch.utils.data.DataLoader(syn_dataset,  batch_size=64, 
                    shuffle=True)
            train_mixed_loader = torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([real_dataset, syn_dataset]),
                batch_size=64, shuffle=True,
            )

            x_syn, y_syn = extract_moment_embeddings(train_syn_loader, moment_model, 
                    device, is_test=False)
            x_mixed, y_mixed = np.concatenate([x_real, x_syn]), np.concatenate([y_real, y_syn])

            metrics_syn   = train_and_eval_linear_probe(x_syn,   y_syn,   x_test, y_test)
            metrics_mixed = train_and_eval_linear_probe(x_mixed, y_mixed, x_test, y_test)

            for trial_name, metrics in [
                ("TSTR (Synthetic)",     metrics_syn),
                ("TSRTR (Augmentation)", metrics_mixed),
            ]:
                all_results.append({
                    "Backbone":     run_cfg['backbone'],
                    "Loss":         loss_slug,
                    "Framework":    run_cfg['framework'],
                    "Partitioning": run_cfg['partitioning'],
                    "Trial":        trial_name,
                    "Run":          run,
                    **metrics,
                })

        del model

    df_all = pd.DataFrame(all_results)
    metric_cols = [c for c in df_all.columns
               if c not in {"Backbone", "Loss", "Framework", "Partitioning", "Trial", "Run"}]
    group_cols  = ["Backbone", "Loss", "Trial"]

    df_mean = df_all.groupby(group_cols)[metric_cols].mean().round(4)
    df_std  = df_all.groupby(group_cols)[metric_cols].std().round(4)

    df_summary = df_mean.copy()
    for col in metric_cols:
        df_summary[col] = df_mean[col].map(str) + " ± " + df_std[col].map(str)

    print("Results saved to classification/{args.dataset.lower()}_all_runs.csv")
    df_all.to_csv("classification/{args.dataset.lower()}_all_runs.csv", index=False)
    df_summary.to_csv("classification/{args.dataset.lower()}_summary.csv")
