import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    """ Helper class for ODE solver compatible with torchdiffeq """
    def __init__(self, model, y, data_cond):
        super().__init__()
        self.model = model
        self.y = y
        self.data_cond = data_cond

    def forward(self, t, x):
        t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
        return self.model(sample=x, timestep=t_batch, class_labels=self.y, data_cond=self.data_cond)

class SPFM_Ensemble(nn.Module):
    def __init__(self, K, backbone_class, means, stds, loss_types, solver='dopri5', **backbone_kwargs):
        """
        Args:
            K: The number of frequency partitions
            backbone_class: Transformer, Mamba or Mamformer
            means/stds: Tensor of means/stds [K, C]
            loss_types: List of strings (e.g., ['l2', 'l1']) 
            solver: ODE solver type ('euler' default)
            backbone_kwargs: Arguments for the backbone class
        """
        super().__init__()
        self.K = K
        self.C = means.shape[1]
        self.solver = solver
        
        if isinstance(loss_types, str):
            self.loss_types = [loss_types] * K
        else:
            self.loss_types = loss_types
        
        assert len(self.loss_types) == K, f"Expected {K} loss types, got {len(self.loss_types)}"

        self.register_buffer('means', means.view(1, K, self.C, 1))
        self.register_buffer('stds', stds.view(1, K, self.C, 1))

        # Create K independent backbone models
        self.models = nn.ModuleList([
            backbone_class(**backbone_kwargs) for _ in range(K)
        ])
        
        # For tracking residuals during training
        self._last_residuals = None

    def _compute_loss(self, pred, target, loss_type):
        """ Internal helper to handle different loss objectives """
        if loss_type == "l1":
            return (pred - target).abs().mean()
        elif loss_type == "l2":
            return (pred - target).pow(2).mean()
        elif loss_type == "l15":
            return (pred - target).abs().pow(1.5).mean()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, x1, x0, y=None):
        """ Hierarchical training loop with band-specific losses """
        B, K, C, L = x1.shape
        total_loss = 0.0
        partition_losses = {}
        cumulative_cond = None
        residuals = []  # Track residuals for spectral analysis

        for k in range(self.K):
            t_1d = torch.rand(B, device=x1.device)
            t = t_1d.view(B, 1, 1)
            
            clean_k = x1[:, k]
            x_t = (1 - t) * x0 + t * clean_k
            target_velocity = clean_k - x0

            predicted_velocity = self.models[k](
                sample=x_t,
                timestep=t_1d,
                class_labels=y,
                data_cond=cumulative_cond
            )

            # Compute residual for spectral tracking
            residual = predicted_velocity - target_velocity
            residuals.append(residual)

            loss_k = self._compute_loss(predicted_velocity, target_velocity, self.loss_types[k])
            partition_losses[f'loss_band_{k}'] = loss_k.item()
            total_loss += loss_k

            cumulative_cond = clean_k if cumulative_cond is None else cumulative_cond + clean_k
        
        # Store residuals for the trainer to access
        self._last_residuals = residuals
        
        return total_loss, partition_losses

    @torch.no_grad()
    def sample(self, x0, sampling_timesteps, y=None):
        """ Hierarchical multi-rate sampling with dopri5 solver """
        B, C, L = x0.shape
        gen_partitions = []
        cumulative_gen = None

        for k in range(self.K):
            model_k = self.models[k]
            model_k.eval()

            steps = sampling_timesteps[k]
            t_eval = torch.linspace(0., 1., steps + 1, device=x0.device)

            # Always use dopri5 for accuracy
            ode_func = ODEFunc(model_k, y, cumulative_gen)
            x_t_all = odeint(ode_func, x0, t_eval, method='dopri5', atol=1e-4, rtol=1e-4)
            pk = x_t_all[-1]

            gen_partitions.append(pk)
            cumulative_gen = pk if cumulative_gen is None else cumulative_gen + pk

        gen_partitions = torch.stack(gen_partitions, dim=1)
        physical_partitions = (gen_partitions * self.stds) + self.means
        final_signal = physical_partitions.sum(dim=1)

        return final_signal, gen_partitions
