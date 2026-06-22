import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import wandb
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from utils.io_utils import instantiate_from_config, get_model_parameters_info
from utils.evaluation import evaluate_SPFM
from utils.priors import PriorGenerator, estimate_dataset_psd

def cycle(dl):
    while True:
        for data in dl:
            yield data

class SPFMTrainer(object):
    def __init__(self, config, args,
            model,
            train_dataloader,
            test_dataloader,
            arch_type='independent',
            logger = None):
        super().__init__()
        
        self.model = model
        self.models = model.models
        self.K = model.K
        self.device = next(model.parameters()).device

        self.train_steps = config["solver"]["max_steps"]
        self.gradient_accumulate_every = config["solver"]["gradient_accumulate_every"]
        self.save_cycle = config["solver"]["save_cycle"]
        self.train_dataloader = cycle(train_dataloader["dataloader"])
        self.test_dataloader = test_dataloader["dataloader"]
        
        prior_type = config["solver"]["prior"]
        self.arch_type = arch_type
        scale_vec = None

        self.step = 0
        self.milestone = 0
        self.best_ilse = float('inf')
        self.best_fid = float('inf')
        self.best_sinkhorn = float('inf')
        self.best_swd = float('inf')
        self.config = config
        self.args = args
        self.logger = logger

        self.results_folder = args.save_dir

        start_lr = config["solver"].get("base_lr", 1.0e-4)
        ema_decay = config["solver"]["ema"]["decay"]
        ema_update_every = config["solver"]["ema"]["update_interval"]

        self.data_noise_std = config["solver"].get("data_noise_std", 0.0)
        
        sc_cfg = config["solver"]["scheduler"]
        
        if self.arch_type == 'independent':
            self.optimizers = [
                Adam(self.models[k].parameters(),
                lr=start_lr,
                betas=[0.9, 0.95]) for k in range(self.K)
            ]
        
            self.schedulers = []
            for k in range(self.K):
                sc_cfg['params']['optimizer'] = self.optimizers[k]
                self.schedulers.append(instantiate_from_config(sc_cfg))
        
            self.emas = [EMA(self.models[k], beta=ema_decay, update_every=ema_update_every).to(self.device) for k in range(self.K)]

        elif self.arch_type == 'unified':
            self.opt = Adam(
                self.model.parameters(),
                lr=start_lr,
                betas=[0.9, 0.95]
            )
            self.scheduler = instantiate_from_config(sc_cfg)

            self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        
        # Spectral tracking
        self.num_freq_bins = self.models[0].num_freq_bins if hasattr(self.models[0], 'num_freq_bins') else None
        
        # Per-partition gradient tracking
        self.grad_spectrum_history = {k: [] for k in range(self.K)} 
        self.residual_psd_history = {k: [] for k in range(self.K)}
        self.gdr_history = {(k, k+1): [] for k in range(self.K)}
        
        # Current batch diagnostics
        self.current_grad_spectrum = {k: None for k in range(self.K)}
        self.current_residual_psd = {k: None for k in range(self.K)}
        
        if self.logger is not None:
            self.logger.log_info("SPFM Ensemble Parameters")
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        
        self.log_frequency = config["solver"]["logger"]["log_freq"]

        if prior_type == 'matched':
            if self.logger is not None:
                self.logger.log_info("Using spectrally-matched prior. Computing dataset PSD...")
            raw_dl = train_dataloader["dataloader"]
            n_channels = config['model']['backbone']['params']['in_channels']
            scale_vec = estimate_dataset_psd(
                raw_dl, 
                n_channels=n_channels, 
                device=self.device
            )
        self.prior = PriorGenerator(prior_type, scale_vec=scale_vec)

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info(f"Save current model to {self.results_folder} as milestone {milestone}")

        if self.arch_type == 'independent':
            data = {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": [ema.state_dict() for ema in self.emas],
                "opt": [opt.state_dict() for opt in self.optimizers],
                "scheduler": [sched.state_dict() for sched in self.schedulers]
            }
        elif self.arch_type == 'unified':
            data = {
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }

        torch.save(data, os.path.join(self.results_folder, f'ckpt-{milestone}.pt'))

    def load(self, milestone=None, verbose=False):
        if milestone is not None: 
            data = torch.load(f'{self.results_folder}/ckpt-{milestone}.pt', map_location=self.device)
            self.model.load_state_dict(data["model"], strict=False)
            self.step = data["step"]
            if self.arch_type == 'independent':
                for k in range(self.K):
                    self.emas[k].load_state_dict(data["emas"][k])
                    self.optimizers[k].load_state_dict(data["opts"][k])
                    self.schedulers[k].load_state_dict(data["schedulers"][k])
            elif self.arch_type == 'unified':
                self.ema.load_state_dict(data["ema"])
                self.opt.load_state_dict(data["opt"])
                self.scheduler.load_state_dict(data["scheduler"])

            self.milestone = milestone

    def _compute_partition_grad_spectrum(self, partition_k, n_grad_bins=257):
        """
        Compute FFT of gradients for partition k.
        Iterates over all parameters in partition_k and averages their gradient spectra.

        Returns:
            grad_spectrum: (n_freq_bins,) tensor or None
        """
        grad_spectra = []

        for param in self.models[partition_k].parameters():
            if param.grad is not None:
                # Flatten gradient and compute FFT
                grad_flat = param.grad.reshape(-1)
                grad_fft = torch.fft.rfft(grad_flat, norm='ortho')
                grad_mag = grad_fft.abs()

        if not grad_spectra:
            return None

        # Average spectra across parameters
        # Pad to same length if needed
        max_len = max(len(g) for g in grad_spectra)
        padded_spectra = [torch.nn.functional.pad(g, (0, max_len - len(g)))
                          for g in grad_spectra]
        full_spectrum = torch.stack(padded).mean(dim=0) # (max_len,)
        n_bins = len(full_spectrum)
        edges = np.geomspace(1, n_bins, n_grad_bins + 1).as_Type(int).clip(0, n_bins)
        summary = np.array([
            full_spectrum[edges[i]:edges[i+1]].pow(2).mean().item()
            for i in range(n_grad_bins)
        ])
        
        return summary

    def train(self):
        self.model.train()

        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: Start training...'.format(self.config['solver']['name']), check_primary=False)

        with tqdm(initial=self.step, total=self.train_steps) as pbar:
            while self.step < self.train_steps:
                total_loss = 0.
                accumulated_losses = {}

                for i in range(self.gradient_accumulate_every):
                    batch = next(self.train_dataloader)
                    x_partitions = batch["x_partitions"].to(self.device)
                    y = batch["y"].to(self.device)
                    
                    B, K, C, L = x_partitions.shape

                    if self.data_noise_std > 0.0:
                        noise_partitions = self.data_noise_std * torch.randn_like(x_partitions)
                        x_partitions = x_partitions + noise_partitions

                    x0 = self.prior.sample((B, C, L), self.device)

                    # Forward pass with spectral tracking
                    loss, partition_losses = self.model(x1=x_partitions, x0=x0, y=y)
                    
                    # Track residuals per partition if model supports it
                    if hasattr(self.model, '_last_residuals'):
                        for k, res in enumerate(self.model._last_residuals):
                            with torch.no_grad():
                                res_fft = torch.fft.rfft(res, dim=-1, norm='ortho')
                                res_psd = res_fft.abs().pow(2).mean(dim=(0, 1))
                                self.current_residual_psd[k] = res_psd

                    # Backward with gradient tracking hooks
                    scaled_loss = loss / self.gradient_accumulate_every
                    scaled_loss.backward()

                    total_loss += loss.item()
                    for k, v in partition_losses.items():
                        accumulated_losses[k] = accumulated_losses.get(k, 0) + v

                loss = total_loss / self.gradient_accumulate_every

                clip_grad_norm_(self.model.parameters(), 1.0)

                if self.step % (self.log_frequency // 5) == 0:
                    self._log_spectral_diagnostics()

                # Step independent optimizers, schedulers, and emas per partition
                if self.arch_type == 'independent':
                    for k in range(self.K):
                        self.optimizers[k].step()
                        try:
                            self.schedulers[k].step(loss)
                        except TypeError:
                            self.schedulers[k].step()
                        self.optimizers[k].zero_grad()
                        self.emas[k].update()
                elif self.arch_type == 'unified':
                    self.opt.step()
                    self.scheduler.step(loss)
                    self.opt.zero_grad()
                    self.ema.update()


                self.step += 1

                pbar.set_description(f"loss: {loss:.4f}, LF lr: {self.optimizers[0].param_groups[0]['lr']:.2e}")

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        print("Evaluating...")
                        self.evaluate()

                    if self.logger is not None and self.step % self.log_frequency == 0:
                        log_data = {
                            'total_loss': total_loss / self.gradient_accumulate_every,
                            'learning_rates': [opt.param_groups[0]['lr'] for opt in self.optimizers]
                        }
                        for k, v in accumulated_losses.items():
                            log_data[k] = v / self.gradient_accumulate_every
                        
                        self.logger.log_metrics(log_data, step=self.step)

                pbar.update(1)

        print('training complete')
        self.save(f'{self.step}')
        self._save_spectral_history()

        
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))
            self.logger.wandb_logger.finish()

    def _log_spectral_diagnostics(self):
        """Log spectral gradient and residual PSD for each partition."""
        for k in range(self.K):
            # Compute and log gradient spectrum for this partition
            grad_spectrum = self._compute_partition_grad_spectrum(k)
            if grad_spectrum is not None:
                self.current_grad_spectrum[k] = grad_spectrum
                grad_spectrum_np = grad_spectrum.cpu().numpy()
                self.grad_spectrum_history[k].append(grad_spectrum_np)
            
            # Log residual PSD
            if self.current_residual_psd[k] is not None:
                residual_psd_np = self.current_residual_psd[k].cpu().numpy()
                self.residual_psd_history[k].append(residual_psd_np)

        # Compute GDR proxy for adjacent partitions from current residual PSDs
        # GDR = mean_power(band_k) / mean_power(band_k+1); GDR = 1 -> Gradient Democracy
        gdr_log = {}
        for k in range(self.K - 1):
            psd_k = self.current_residual_psd[k]
            psd_k1 = self.current_residual_psd[k+1]
            if psd_k is not None and psd_k1 is not None:
                gdr = (psd_k.mean() / (psd_k1.mean() + 1e-10)).item()
                self.gdr_history[(k, k + 1)].append(gdr)
                gdr_log[f'gdr_proxy/band_{k}_vs_{k+1}'] = gdr

        if gdr_log and self.logger is not None:
            self.logger.log_metrics(gdr_log, step=self.step)
 
    def _save_spectral_history(self):
        """Save spectral tracking history to NPZ file."""
        spectral_data = {}
        
        # Convert to numpy arrays for each band
        for k in range(self.K):
            if self.grad_spectrum_history[k]:
                spectral_data[f'grad_spectrum_band_{k}'] = np.array(self.grad_spectrum_history[k])
            if self.residual_psd_history[k]:
                spectral_data[f'residual_psd_band_{k}'] = np.array(self.residual_psd_history[k])
        
        for (k, k1), values in self.gdr_history.items():
            if values:
                spectral_data[f'gdr_band_{k}_vs_{k1}'] = np.array(values)

        save_path = os.path.join(self.results_folder, 'spectral_history.npz')
        np.savez(save_path, **spectral_data)
        print(f"Spectral history saved to {save_path}")

    @torch.no_grad()
    def sample(self, n_samples, y=None, sampling_timesteps=None):
        """ Generates new samples using the establoished architecture type """
        n_channels = self.config['model']['backbone']['params']['in_channels']
        seq_len = 3000 if n_channels == 2 else 1000

        noise = self.prior.sample((n_samples, n_channels, seq_len), self.device)

        if y is not None:
            y = y.to(self.device)
            
        if sampling_timesteps is None:
            sampling_timesteps = [20] * self.config['model']['K']
        
        if self.arch_type == 'independent':
            for ema in self.emas:
                ema.eval()
            # Temporarily swap active model parameters with their corresponding EMA weights
            original_states = [{k_: v_.clone() for k_, v_ in m.state_dict().items()} for m in self.models]
            for k in range(self.K):
                self.models[k].load_state_dict(self.emas[k].ema_model.state_dict())

            # Forward sample execution through the updated wrapper model
            x_hat, _ = self.model.sample(x0=noise, sampling_timesteps=sampling_timesteps, y=y)

            # Restore original active weights back to sub-models
            for k in range(self.K):
                self.models[k].load_state_dict(original_states[k])

        elif self.arch_type == 'unified':
            self.ema.eval()
            x_hat, _ = self.ema.ema_model.sample(x0=noise, sampling_timesteps=sampling_timesteps, y=y)

        return x_hat

    @torch.no_grad()
    def evaluate(self):

        if self.arch_type == 'independent':
            for ema in self.emas:
                ema.eval()
        
            # Temporarily swap active model parameters with their corresponding EMA weights
            original_states = [{k_: v_.clone() for k_, v_ in m.state_dict().items()} for m in self.models]
            for k in range(self.K):
                self.models[k].load_state_dict(self.emas[k].ema_model.state_dict())

            metrics = evaluate_SPFM(
                    model=self.model,
                    dataloader=self.test_dataloader,
                    prior=self.prior,
                    device=self.device
            )

            # Restore original active weights back to sub-models
            for k in range(self.K):
                self.models[k].load_state_dict(original_states[k])
        
        elif self.arch_type == 'unified':
            metrics = evaluate_SPFM(
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
