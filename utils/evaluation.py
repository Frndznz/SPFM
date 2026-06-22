import torch
import numpy as np
import scipy.signal
from scipy import linalg
from geomloss import SamplesLoss
from momentfm import MOMENTPipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings("ignore")

def calculate_avg_psd(samples, fs=100.0, nperseg=256):
    """ Calculates the avg Power Spectral Density (PSD) for a batch of signals
    Args: samples: (Tensor) [B, C, S]
          fs: (float) Sampling frequency (in Hz)
          nperseg: (int) Length of each segment for Welch's method, determines the
                   frequency resolution (fs/nperseg). Ideally, we want 50 % overlap.
                   So, e.g. ECGs of length 1000 sampled @ 100 Hz would use 256 nperseg.
    Output: avg_psd: (array) Average PSD [C, N_freq-bins]
            freqs: (array) Frequencies [N_freq-bins]
    """
    if samples is None or len(samples) == 0:
        return None, None
    samples_np = samples.detach().cpu().numpy()
    all_psds = []
    for sample in samples_np:
        freqs, psd = scipy.signal.welch(
            sample,
            fs=fs,
            window='hamming',     
            nperseg=nperseg,      
            noverlap=nperseg // 2, # 50% overlap
            axis=-1 # PSD along time axis
        )
        all_psds.append(psd)

    # Average across all samples
    avg_psd = np.mean(all_psds, axis=0)
    return avg_psd, freqs

def compute_ilse(psd_true, psd_gen, freqs):
    """
    Compute Integrated Log Spectral Error (L1 distance in log-space)
    ILSE = Integral( |log10(psd_gen) - log10(psd_true)| )df
    """
    log_err = np.abs(np.log10(psd_true + 1e-12) - np.log10(psd_gen + 1e-12))
    # Integrate over the frequency axis (-1) and average across channels
    return np.trapz(log_err, freqs, axis=-1).mean()

def calculate_sinkhorn(real_data, fake_data, blur=10.0, scaling=0.9):
    """
    Calculates the Sinkhorn distance between two sets of multivariate signals.
    Args:
        real_data: (Tensor) [B, C, S]
        fake_data: (Tensor) [B, C, S]
    """
    B, C, S = real_data.shape
    # Flatten spatial/channel dimensions for OT: [B, C*S]
    real_flat = real_data.view(B, -1).contiguous()
    fake_flat = fake_data.view(B, -1).contiguous()
    
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=scaling)
    dist = loss_fn(real_flat, fake_flat)
    return dist.item()

def sliced_wasserstein_distance(real_data, fake_data, n_projections=100):
    """
    Wasserstein distance computed on random 1D projections.
    """
    real_flat = real_data.reshape(real_data.shape[0], -1)
    syn_flat = fake_data.reshape(fake_data.shape[0], -1)

    D = real_flat.shape[1]
    distances = []

    for _ in range(n_projections):
        # Random projection direction
        theta = np.random.randn(D)
        theta = theta / np.linalg.norm(theta)

        # Project both onto this direction
        real_proj = real_flat @ theta
        syn_proj = syn_flat @ theta

        # Compute 1D Wasserstein distance (closed form)
        real_proj_sorted = np.sort(real_proj)
        syn_proj_sorted = np.sort(syn_proj)

        # Interpolate to same size if needed
        if len(real_proj_sorted) != len(syn_proj_sorted):
            min_len = min(len(real_proj_sorted), len(syn_proj_sorted))
            real_proj_sorted = real_proj_sorted[:min_len]
            syn_proj_sorted = syn_proj_sorted[:min_len]

        dist = np.mean(np.abs(real_proj_sorted - syn_proj_sorted))
        distances.append(dist)

    return np.mean(distances)

# Helper functions for FID calculation
def _extract_embeddings(model, samples, batch_size, device):
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        # This loop prevents OOM by processing small chunks at a time
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size].to(device)
            res = model(x_enc=batch)
            
            # Keep the full embedding vector [B, d_model]
            features = res.embeddings if hasattr(res, 'embeddings') else res
            all_embeddings.append(features.cpu().numpy())

    # Stack them into [Total_Samples, d_model]
    return np.vstack(all_embeddings)

def prepare_moment_data(samples, window_size=512, num_windows=2):
    """
    Splits samples into num_windows to fit MOMENT's 512 context.
    The dataset size is effectively multiplied in size by n_windows.
    Args:
        samples: (Tensor) [N, C, L]
        window_size: (int) target length (MOMENT's context size)
        num_windows: (int) windows to extract per sample
    Returns:
        (Tensor) [N * num_windows, C, window_size]
    """
    N, C, L = samples.shape
    # Handle cases where seq length is shorter than window size
    if L < window_size:
        padding = torch.zeros(N, C, window_size - L).to(samples.device)
        return torch.cat([samples, padding], dim=-1)

    if num_windows > 1:
        # Calculate stride to distribute windows evenly across signal
        stride = (L - window_size) // (num_windows - 1)
    else:
        # If 1 window, take first window_size points
        return samples[:,:, :window_size]

    windows = []
    for i in range(num_windows):
        start = i * stride
        # Adjust the last window to ensure it doesn't exceed L
        if i == num_windows - 1:
            start = L - window_size
        end = start + window_size
        windows.append(samples[:,:,start:end])

    return torch.cat(windows, dim=0)

def calculate_FID(real_samples, syn_samples, model, batch_size=64, device='cuda'):
    if real_samples is None or syn_samples is None:
        return float('inf')

    # Extract latents using the batching helper
    real_feat = _extract_embeddings(model, real_samples, batch_size, device)
    syn_feat = _extract_embeddings(model, syn_samples, batch_size, device)

    # Calculate mean and covariance on the FULL feature set
    mu1, sigma1 = real_feat.mean(axis=0), np.cov(real_feat, rowvar=False)
    mu2, sigma2 = syn_feat.mean(axis=0), np.cov(syn_feat, rowvar=False)

    # Standard FID formula
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def evaluate_SPFM(
        model, 
        dataloader, 
        prior, 
        device, 
        moment_model=None, 
        sampling_timesteps=None
    ):
    """
    Evaluates the SPFM_Ensemble by computing ILSE, Sinkhorn, and FID.
    """
    model.eval()
    
    real_data = []
    fake_data = []
    batch_limit = 100

    # Get batch properties
    for idb, batch in enumerate(dataloader):
        if idb >= batch_limit:
            break
        real_x = batch['x'].to(device) # [B, C, S]
        y = batch['y'].to(device)

        B, C, S = real_x.shape

        # Sample shared noise for generation
        x0 = prior.sample((B, C, S), device)

        if sampling_timesteps is None:
            sampling_timesteps = [20] * model.K
            
        # Unified Sample Call
        fake_x, _ = model.sample(x0=x0, sampling_timesteps=sampling_timesteps, y=y)
        
        real_data.append(real_x)
        fake_data.append(fake_x)

    real_data = torch.cat(real_data, dim=0)
    fake_data = torch.cat(fake_data, dim=0)

    metrics = {}
    real_np, fake_np = real_data.cpu().numpy(), fake_data.cpu().numpy()
    
    # Calculate ILSE
    real_psd, freqs = calculate_avg_psd(real_data, fs=100)
    fake_psd, _ = calculate_avg_psd(fake_data, fs=100)
    ilse = compute_ilse(real_psd, fake_psd, freqs)
    print(f"ILSE: {ilse:.3f}")
    metrics['ilse'] = ilse
    
    # Calculate Sinkhorn
    sinkhorn = calculate_sinkhorn(real_data, fake_data)
    print(f"Sinkhorn: {sinkhorn:.3f}")
    metrics['sinkhorn'] = sinkhorn
    
    # Calculate Sliced Wasserstein Distance
    swd = sliced_wasserstein_distance(real_np, fake_np)
    print(f"SWD: {swd:.3f}")
    metrics['swd'] = swd

    # Calculate FID using MOMENT
    if moment_model is None:
        moment_model = MOMENTPipeline.from_pretrained(
            './Foundational/moment_large_embedding',
            model_kwargs={'task_name': 'embedding'},
            torch_dtype=torch.float32,
            local_files_only=True
        )
    moment_model.init()
    moment_model.to(device)
    moment_model.eval()
    
    real_fid_data = prepare_moment_data(real_data, window_size=512, num_windows=2)
    fake_fid_data = prepare_moment_data(fake_data, window_size=512, num_windows=2)
    
    # Use a small batch_size (e.g., 16 or 32) here if you still face OOM
    fid = calculate_FID(real_fid_data, fake_fid_data, moment_model, batch_size=16, device=device)

    print(f"FID: {fid:.3f}")
    metrics['fid'] = fid

    return metrics

def evaluate_FM(
        model, 
        dataloader, 
        prior, 
        device, 
        moment_model=None, 
        sampling_timesteps=None
    ):
    """
    Evaluates the FM model by computing ILSE, Sinkhorn, SWD, and FID.
    """
    model.eval()
    
    real_data = []
    fake_data = []
    batch_limit = 100

    # Get batch properties
    for idb, batch in enumerate(dataloader):
        if idb >= batch_limit:
            break
        real_x = batch['x'].to(device) # [B, C, S]
        y = batch['y'].to(device)

        B, C, S = real_x.shape

        # Sample shared noise for generation
        x0 = prior.sample((B, C, S), device)

        if sampling_timesteps is None:
            sampling_timesteps = 40
            
        # Unified Sample Call
        fake_x = model.sample(x0=x0, timesteps=sampling_timesteps, y=y)
        
        real_data.append(real_x)
        fake_data.append(fake_x)

    real_data = torch.cat(real_data, dim=0)
    fake_data = torch.cat(fake_data, dim=0)

    metrics = {}
    real_np, fake_np = real_data.cpu().numpy(), fake_data.cpu().numpy()
    
    # Calculate ILSE
    real_psd, freqs = calculate_avg_psd(real_data, fs=100)
    fake_psd, _ = calculate_avg_psd(fake_data, fs=100)
    ilse = compute_ilse(real_psd, fake_psd, freqs)
    print(f"ILSE: {ilse:.5f}")
    metrics['ilse'] = ilse
    
    # Calculate Sinkhorn
    sinkhorn = calculate_sinkhorn(real_data, fake_data)
    print(f"Sinkhorn: {sinkhorn:.5f}")
    metrics['sinkhorn'] = sinkhorn
    
    # Calculate Sliced Wasserstein Distance
    swd = sliced_wasserstein_distance(real_np, fake_np)
    print(f"SWD: {swd:.5f}")
    metrics['swd'] = swd

    # Calculate FID using MOMENT
    if moment_model is None:
        moment_model = MOMENTPipeline.from_pretrained(
            '/home/fnunez/Foundational/moment_large_embedding',
            model_kwargs={'task_name': 'embedding'},
            torch_dtype=torch.float32,
            local_files_only=True
        )
    moment_model.init()
    moment_model.to(device)
    moment_model.eval()
    
    real_fid_data = prepare_moment_data(real_data, window_size=512, num_windows=2)
    fake_fid_data = prepare_moment_data(fake_data, window_size=512, num_windows=2)
    
    fid = calculate_FID(real_fid_data, fake_fid_data, moment_model, batch_size=16, device=device)

    print(f"FID: {fid:.5f}")
    metrics['fid'] = fid

    return metrics
