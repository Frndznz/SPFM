import torch
import torch.fft
from tqdm.auto import tqdm
import numpy as np

def estimate_dataset_psd(dataloader, n_channels, device='cpu', max_batches=None):
    """
    Computes avg PSD magnitude per channel from a dataloader.
    Compatible with both NpzDataset ('x') and SpecPartDataset ('x_lf' + 'x_hf').
    """
    psd_accumulator = None
    count = 0
    
    print(f"Estimating Dataset PSD for Spectrally Matched Prior...")
    
    # Iterate through dataloader to calculate average PSD
    for batch in tqdm(dataloader, desc="Computing PSD"):
        # Reconstruct raw signal if using partitioned dataset
        if 'x' in batch:
            x = batch['x']
        elif 'x_lf' in batch and 'x_hf' in batch:
            x = batch['x_lf'] + batch['x_hf']
        else:
            continue # Skip if format unknown
            
        x = x.to(device)
        
        # FFT over time dimension
        fft_x = torch.fft.rfft(x, dim=-1)
        
        if psd_accumulator is None:
            n_bins = fft_x.shape[-1]
            psd_accumulator = torch.zeros(n_channels, n_bins, device=device)
            
        # Magnitude squared, averaged over batch
        batch_psd = torch.mean(torch.abs(fft_x)**2, dim=0)
        psd_accumulator += batch_psd
        count += 1
        
        if max_batches and count >= max_batches:
            break
    
    if count == 0:
        raise RuntimeError("Could not compute PSD: Dataloader empty or invalid keys.")

    avg_psd = psd_accumulator / count
    # Return scale_vec (Amplitude Spectrum)
    scale_vec = torch.sqrt(avg_psd).unsqueeze(0)
    return scale_vec

class PriorGenerator:
    """
    Handles noise generation: white, pink, brown, and spectrally matched.
    """
    def __init__(self, prior_type='white', scale_vec=None):
        self.prior_type = prior_type.lower()
        self.scale_vec = scale_vec
        
        # Pre-check requirements
        if self.prior_type == 'matched' and self.scale_vec is None:
            raise ValueError("Prior type 'matched' requires a 'scale_vec'.")

        if self.prior_type == 'white':
            self.beta = 0.0
        elif self.prior_type == 'pink':
            self.beta = 1.0
        elif self.prior_type == 'brown':
            self.beta = 2.0
        
    def sample(self, shape, device):
        """
        Generates noise of the specified shape.
        shape: (B, C, L)
        """
        if self.prior_type == 'white':
            return torch.randn(shape, device=device)
        elif self.prior_type == 'matched':
            return self._generate_matched(shape, device)
        else:
            return self._generate_colored_noise(shape, device)

    def _generate_matched(self, shape, device):
        """
        Generates noise matching the learned dataset PSD.
        """
        B, C, L = shape
        # Ensure scale_vec is on correct device
        scale = self.scale_vec.to(device)
        n_freqs = scale.shape[-1]
        
        # 1. Generate Complex Gaussian Noise
        # We divide by sqrt(2) so that the inverse FFT has unit variance before scaling
        real = torch.randn(B, C, n_freqs, device=device)
        imag = torch.randn(B, C, n_freqs, device=device)
        white_noise_freq = torch.complex(real, imag) / np.sqrt(2)
        
        # 2. Apply Learned Filter
        colored_freq = white_noise_freq * scale
        
        # 3. Inverse FFT
        colored_noise = torch.fft.irfft(colored_freq, n=L, dim=-1)
        
        return colored_noise

    def _generate_colored_noise(self, shape, device):
        """ Generates 1/f^beta noise via FFT. """
        B, C, L = shape
        wn = torch.randn(shape, device=device)
        wn_f = torch.fft.rfft(wn, dim=-1)
        freqs = torch.fft.rfftfreq(L, device=device)
        
        scaling = torch.ones_like(freqs)
        scaling[1:] = 1.0 / (freqs[1:] ** (self.beta / 2.0))
        
        cn_f = wn_f * scaling
        cn = torch.fft.irfft(cn_f, n=L, dim=-1)
        
        std = cn.std(dim=-1, keepdim=True)
        mean = cn.mean(dim=-1, keepdim=True)
        cn = (cn - mean) / (std + 1e-8)
        return cn
