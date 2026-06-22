import os
import time
import torch
import numpy as np
import itertools
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from utils.io_utils import instantiate_from_config, get_model_parameters_info
from utils.priors import PriorGenerator, estimate_dataset_psd
from utils.evaluation import evaluate_FM
from momentfm import MOMENTPipeline

def cycle(dl):
    while True:
        for data in dl:
            yield data

def estimate_target_psd(dataloader, n_channels, device, fs=100, n_batches=100):
    """
    Estimate power spectral density of training data for frequency weighting.
    
    Args:
        dataloader: Training data loader
        n_channels: Number of channels
        device: torch device
        fs: Sampling frequency (Hz)
        n_batches: Number of batches to use for PSD estimation
    
    Returns:
        target_psd: (n_freq_bins,) tensor with target PSD
    """
    psds = []
    count = 0

    for batch in dataloader:
        x = batch['x'].to(device)  # (B, C, L)
        B, C, L = x.shape

        # Compute FFT
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')  # (B, C, n_freqs)
        psd = x_fft.abs().pow(2).mean(dim=(0, 1))  # (n_freqs,)

        psds.append(psd)
        count += 1

        if count >= n_batches:
            break

    # Average PSD across batches
    target_psd = torch.stack(psds).mean(dim=0)  # (n_freqs,)

    return target_psd

class FMTrainer(object):
    def __init__(self, config, args, model, train_dataloader, test_dataloader, loss_type='standard', logger=None):
        super().__init__()
        
        self.model = model
        self.device = next(self.model.parameters()).device
        
        self.config = config
        self.args = args
        self.logger = logger
        self.results_folder = args.save_dir
        
        self.train_steps = config["solver"]["max_steps"]
        self.save_cycle = config["solver"]["save_cycle"]
        self.log_frequency = config["solver"]["logger"]["log_freq"]
        self.gradient_accumulate_every = config["solver"].get("gradient_accumulate_every", 1)
        
        self.train_dataloader = cycle(train_dataloader["dataloader"])
        self.raw_train_loader = train_dataloader["dataloader"] 
        self.test_dataloader = test_dataloader["dataloader"]
        
        self.loss_type = loss_type

        prior_type = config["solver"].get("prior", "white")
        scale_vec = None
        
        if prior_type == "matched":
            if self.logger:
                self.logger.log_info("Prior is 'matched'. Computing dataset PSD...")
            
            if 'backbone' in config['model']:
                n_channels = config['model']['backbone']['params']['in_channels']
            else:
                n_channels = config['model']['params'].get('in_channels', 12)

            scale_vec = estimate_dataset_psd(self.raw_train_loader, n_channels, device=self.device)
            
        self.prior = PriorGenerator(prior_type, scale_vec)
        
        if self.loss_type == 'freq_weighted':
            # Estimate target PSD for weighting
            self.target_psd = estimate_target_psd(
                self.raw_train_loader,
                n_channels=config['model']['backbone']['params']['in_channels'],
                device=self.device,
                fs=100,
                n_batches=100
            )

            # Smoothly compress the extreme 1/f dynamic range using a fractional root (0.25)
            compressed_psd = torch.pow(self.target_psd, 0.25)

            # Compute raw inverse weights
            eps = 1e-4
            raw_weights = 1.0 / (compressed_psd + eps)

            # Normalize directly by the mean
            self.freq_weight = raw_weights / raw_weights.mean()

        start_lr = config["solver"].get("base_lr", 1.0e-4)
        ema_decay = config["solver"]["ema"]["decay"]
        ema_update_every = config["solver"]["ema"]["update_interval"]

        self.opt = Adam(self.model.parameters(), lr=start_lr, betas=[0.9, 0.95])
        
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        # Scheduler
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        self.step = 0
        self.best_ilse = float('inf')
        self.best_fid = float('inf')
        self.best_sinkhorn = float('inf')
        self.best_swd = float('inf')
        
        if self.logger is not None:
            self.logger.log_info(f"Initialized Baseline FM Solver with prior: {prior_type}, {loss_type}, {self.model.model.loss_type} loss")
            self.logger.log_info(str(get_model_parameters_info(self.model)))


    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(f"Saving model to {self.results_folder} at step {self.step}")
        
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
        }
        torch.save(data, os.path.join(self.results_folder, f'ckpt-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        path = os.path.join(self.results_folder, f'ckpt-{milestone}.pt')
        if os.path.exists(path):
            if self.logger: self.logger.log_info(f"Loading from {path}")
            data = torch.load(path, map_location=device)
            self.model.load_state_dict(data["model"])
            self.ema.load_state_dict(data["ema"])
            self.opt.load_state_dict(data["opt"])
            self.step = data["step"]
        else:
            print(f"Checkpoint {path} not found.")
    
    def compute_frequency_weighted_loss(self, model_output, x1, x0, y):
        """
        Compute frequency-weighted loss for FW-FM.
        
        Loss: || w(f) \odot (v_t - (x1 - x0)) ||_p^p
        where w(f) = 1 / (S_target(f) + eps) upweights high-frequency components,
        and p is the generalized p-norm of the FM loss (l1 or l2)
        
        Args:
            model_output: Model's velocity estimate v_t
            x1: Target data
            x0: Noise
            y: Class labels
        
        Returns:
            loss: Scalar loss value
        """
        # Residuals in time domain
        residuals = model_output - (x1 - x0)  # (B, C, L)

        # Transform to frequency domain
        res_fft = torch.fft.rfft(residuals, dim=-1, norm='ortho')  # (B, C, n_freqs)

        # Apply frequency weighting in frequency domain
        # w(f) upweights high frequencies (low PSD)
        weighted_res_fft = res_fft * torch.sqrt(self.freq_weight)

        # Compute MSE loss on weighted residuals
        loss = weighted_res_fft.abs().pow(2).mean()

        return loss


    def train(self):
        self.model.train()
        if self.logger: self.logger.log_info(f"{self.config['solver']['name']}: Starting training...")
        
        with tqdm(initial=self.step, total=self.train_steps) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.train_dataloader)
                    
                    # NpzDataset returns 'x' (Batch, Channels, Time)
                    x1 = batch["x"].to(self.device) 
                    y = batch["y"].to(self.device)
                    
                    # Sample Noise from Prior (White/Pink/Matched)
                    x0 = self.prior.sample(x1.shape, self.device)
                    
                    # Standard FM Loss: || v - (x1 - x0) ||^p for L_p loss
                    if self.loss_type == 'standard':
                        loss = self.model(x1=x1, x0=x0, y=y)
                    elif self.loss_type == 'freq_weighted':
                        t_1d = torch.rand(x1.shape[0], device=self.device)
                        t = t_1d.view(x1.shape[0], 1, 1)

                        # Calculate conditional flow path (x_t) matching the wrapper formulas
                        x_t = (1 - t) * x0 + t * x1

                        # Call the wrapper's inner backbone model directly using its required kwargs
                        model_output = self.model.model(
                            sample=x_t,
                            timestep=t_1d,
                            class_labels=y
                        )

                        loss = self.compute_frequency_weighted_loss(model_output, x1, x0, y)
                        
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()
                
                # Step Scheduler
                try: self.sch.step(total_loss)
                except: self.sch.step()
                
                self.ema.update()
                self.step += 1
                pbar.update(1)
                pbar.set_description(f"Loss: {total_loss:.4f} | LR: {self.opt.param_groups[0]['lr']:.2e}")

                # Logging & Evaluation
                if self.step % self.log_frequency == 0 and self.logger:
                    self.logger.log_metrics({'loss': total_loss}, step=self.step)
                
                if self.step % self.save_cycle == 0 and self.step > 0:
                    self.evaluate()
        
        self.save(self.step)

    @torch.no_grad()
    def sample(self, n_samples, seq_len, n_channels, y=None):
        self.ema.eval()
        
        # Start from Prior
        shape = (n_samples, n_channels, seq_len)
        x0 = self.prior.sample(shape, self.device)
        
        # Solve ODE using the class label as conditioning and the ODE solver in the config
        x_gen = self.ema.model.sample(x0=x0, y=y)
        
        return x_gen

    @torch.no_grad()
    def evaluate(self):
        """ Generates samples and computes SMSE / Sinkhorn / FID """
        metrics = evaluate_FM(
                model=self.ema.ema_model,
                dataloader=self.test_dataloader,
                prior=self.prior,
                device=self.device
        )

        if self.logger is not None:
            self.logger.log_info(f"Computed evaluation metrics at step {self.step}")
            self.logger.log_metrics(metrics)

        if metrics['ilse'] < self.best_ilse:
            self.best_ilse = metrics['ilse']
            if self.logger is not None:
                self.logger.log_info(f"New best ILSE: {self.best_ilse:.3f} @ step {self.step}. Saving model")
            self.save('best_ilse')

        if metrics['fid'] < self.best_fid:
            self.best_fid = metrics['fid']
            if self.logger is not None:
                self.logger.log_info(f"New best FID: {self.best_fid:.3f} @ step {self.step}. Saving model")
            self.save('best_fid')

        if metrics['sinkhorn'] < self.best_sinkhorn:
            self.best_sinkhorn = metrics['sinkhorn']
            if self.logger is not None:
                self.logger.log_info(f"New best Sinkhorn: {self.best_sinkhorn:.3f} @ step {self.step}. Saving model")
            self.save('best_sinkhorn')

        if metrics['swd'] < self.best_swd:
            self.best_swd = metrics['swd']
            if self.logger is not None:
                self.logger.log_info(f"New best SWD: {self.best_swd:.3f} @ step {self.step}. Saving model")
            self.save('best_swd')

        return metrics


