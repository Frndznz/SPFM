import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        
        # Shared Mamba mixer for both directions
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        
        # Forward Pass
        out_fwd = self.mamba(x)
        
        # Backward Pass (Flip -> Process -> Flip)
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.mamba(x_rev)
        out_bwd = torch.flip(out_rev, dims=[1])
        
        # Combine Directions (Average to keep scale consistent)
        out_combined = (out_fwd + out_bwd) / 2
        
        # Residual Connection + Dropout
        return self.norm(x + self.dropout(out_combined))

class Mamba1DModel(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        n_layers=8, 
        d_model=256, 
        num_classes=None,
        time_embed_dim=256,
        class_embed_dim=256,
        dropout=0.1,
        d_state=32,
        d_conv=4,
        expand=4,
        seq_length=0,
        loss_type='l1'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.d_model = d_model
        
        # Input Projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Time Conditioning
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, d_model)
        )

        # Class Conditioning
        self.class_embed = None
        if num_classes:
            self.class_embed = nn.Sequential(
                nn.Embedding(num_classes, class_embed_dim),
                nn.SiLU(), 
                nn.Linear(class_embed_dim, d_model)
            )
        
        # Bidirectional Mamba Stack
        self.layers = nn.ModuleList([
            BidirectionalMambaBlock(
                d_model=d_model,
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand,
                dropout=dropout
            ) 
            for _ in range(n_layers)
        ])
        
        # Output Projection
        self.output_proj = nn.Linear(d_model, out_channels)
        self.final_act = nn.SiLU()

    def forward(self, sample, timestep, class_labels=None, data_cond=None):
        # Input shape: [B, C, L] -> Need [B, L, C] for Mamba
        x = sample.transpose(1, 2)
        
        # Embed Signal
        x = self.input_proj(x)
        
        # Handle Time Conditioning
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(-1)

        t_emb = self.time_mlp(timestep).unsqueeze(1)
        x = x + t_emb
        
        # Handle Class Conditioning
        if class_labels is not None and self.class_embed is not None:
            c_emb = self.class_embed(class_labels).unsqueeze(1)
            x = x + c_emb
            
        # Handle Data Conditioning (for Detail model conditioning on Structure)
        if data_cond is not None:
            d_cond = data_cond.transpose(1, 2)
            d_emb = self.input_proj(d_cond) 
            x = x + d_emb

        # Pass through layers
        for layer in self.layers:
            x = layer(x)
            
        # Output
        x = self.final_act(x)
        output = self.output_proj(x)
        
        # Return to [B, C, L]
        return output.transpose(1, 2)

