import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from torchdiffeq import odeint
import os

class Flow_Matching_TS(nn.Module):
    """ Wrapper class for a conditional Flow Matching model
        Takes a backbone model (e.g. Transformer1DModel) and handles training and sampling
    """
    def __init__(
            self,
            feature_size,
            model,
            sampling_timesteps=None,
            solver=None,
            **kwargs):
        super().__init__()
        self.model = model
        self.sampling_timesteps = sampling_timesteps
        self.device = next(self.model.parameters()).device
        self.solver = solver

    # Training method
    def forward(self, x1, x0, y=None, data_cond=None, reconstruct=False):
        """ Calculates FM loss.
            x1: target data [B, C, S]
            x0: source data [B, C, S]
            y: class labels [B]
            data_cond: (optional) data condition (e.g. x_lf) [B, C, S]
        """
        # First we sample t ~ U(0,1) for each item in the batch
        t = torch.rand(x1.shape[0], 1, 1, device=x1.device)

        # Calculate interpolated path x_t and target velocity v_t
        x_t = (1 - t) * x0 + t * x1
        target_velocity = x1 - x0

        # Get model prediction (velocity)
        predicted_velocity = self.model(sample=x_t, 
                timestep=t.squeeze(), 
                class_labels=y,
                data_cond=data_cond)

        # Calculate the loss (MSE b/w predicted and target velocity)
        if self.model.loss_type == "l1": # Absolute error
            loss_time = (predicted_velocity - target_velocity).abs().mean()
        elif self.model.loss_type == "l2": # Mean squared error
            loss_time = (predicted_velocity - target_velocity).pow(2).mean()
        elif self.model.loss_type == "l15": # Custom L_1.5 loss in b/w L_1 and L_2
            loss_time = (predicted_velocity - target_velocity).abs().pow(1.5).mean()

        if not reconstruct:
            return loss_time
        else:
            # Reconstruct estimated x1: Follow predicted velocity from t=0 to t=1
            x1_hat = predicted_velocity + x0
            return loss_time , x1_hat 

    @torch.no_grad()
    def sample(self, x0, y=None, data_cond=None, timesteps=None):
        """ Generates x1 from x0 by solving the ODE from t=0 to t=1
            x0: source data [B, C, S]
            y: class labels [B]
            data_cond (optional) data condition (e.g. x_lf) [B, C, S]
        """
        self.model.eval()
        solver = self.solver
        b = x0.shape[0]
        # Create the timesteps for the ODE solver
        if timesteps:
            self.sampling_timesteps = timesteps
        timesteps = torch.linspace(0., 1., self.sampling_timesteps + 1, device=x0.device)

        # Option 1: ODEInt library (with Runge-Kutta solver)
        if solver != 'euler':
            ode_func = ODEFunc(self.model, y, data_cond)
            x_t = odeint(ode_func, x0, timesteps, 
                    method=solver, atol=1e-5, rtol=1e-5)
            # Return the final sample (at t=1)
            x1 = x_t[-1]

        # Option 2: Euler solver
        else:
            x_t = x0.clone()
            for i in range(self.sampling_timesteps):
                t_curr = timesteps[i]
                t_next = timesteps[i+1]
                dt = t_next - t_curr # Step size

                # Prepare the time tensor for the model
                t_tensor = torch.full((x0.shape[0],), t_curr.item(), device=x0.device)

                # Get the predicted velocity v(x_t, t, y)
                predicted_velocity = self.model(sample=x_t, 
                        timestep=t_tensor, 
                        class_labels=y,
                        data_cond=data_cond)

                # Euler step: x_{t+dt} = x_t + v(x_t, t) * dt
                x_t = x_t + predicted_velocity * dt
            x1 = x_t

        return x1 # The final sample is x_t at t=1

# Helper method for different ODE solvers
class ODEFunc(nn.Module):
    """ Helper class for ODE solver. Needed for torchdiffeq library which takes (t, x) as input
        This class wraps our (x, t, y, data_cond) model to match
    """
    def __init__(self, model, y, data_cond):
        super().__init__()
        self.model = model
        self.y = y
        self.data_cond = data_cond

    def forward(self, t, x):
        # Solver gives us a single float 't', we need to broadcast it to the batch shape
        t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
        # Get the velocity from our model
        v = self.model(sample=x, timestep=t_batch, class_labels=self.y, data_cond=self.data_cond)

        return v
