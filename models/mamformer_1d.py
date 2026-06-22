import torch
import torch.nn as nn
import math
from einops import rearrange
from mamba_ssm import Mamba

# Helper: Positional Encoding (for transformer)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, C]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Helper: Bidirectional Mamba Block (from mamba)
class BidirectionalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # forward pass
        out_fwd = self.mamba(x)
        # backward pass (flip -> mamba -> flip)
        x_rev = torch.flip(x, dims=[1])
        out_rev = self.mamba(x_rev)
        out_bwd = torch.flip(out_rev, dims=[1])
        
        # Mean pooling of directions
        out_combined = (out_fwd + out_bwd) / 2
        return self.norm(x + self.dropout(out_combined))

# Hybrid mamba-transformer backbone 
class MambaTransformer1DModel(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        d_model=256, 
        class_embed_dim=256,
        n_layers=8, 
        mamba_ratio=0.75, # mamba-to-transformer layers ratio
        n_heads=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        num_classes=None,
        seq_length=0,
        loss_type=None,
        time_embed_dim=None, # Inherited from Mamba model instantiation
        dim_feedforward=256, # Inherited from Transformer model instantiation

    ):
        super().__init__()
        
        self.d_model = d_model
        self.loss_type = loss_type        
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # For detail model
        self.data_cond_proj = nn.Linear(in_channels, d_model) 

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Class embedding
        self.num_classes = num_classes
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, class_embed_dim)
            
        # Positional encoding (for attention layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Hybrid layers
        self.layers = nn.ModuleList()
        
        # Calculate transformer layers to insert
        num_transformer = int(n_layers * (1 - mamba_ratio))
        # e.g. n_layers=8, ratio=0.75 then 2 Transformer layers. Indices roughly [3, 7]
        attn_interval = n_layers // (num_transformer + 1) if num_transformer > 0 else n_layers + 1

        for i in range(n_layers):
            # Check if this index should be a Transformer
            is_attn = (i + 1) % attn_interval == 0 and num_transformer > 0
            
            if is_attn:
                # Transformer layer
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True 
                )
                num_transformer -= 1
            else:
                # Bidirectional mamba layer
                layer = BidirectionalMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand, 
                    dropout=dropout
                )
            
            self.layers.append(layer)

        self.norm_f = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_channels)

    def forward(self, sample, timestep, class_labels=None, data_cond=None):
        # Input: [B, C, L] , Target: [B, L, C]
        x = sample.transpose(1, 2)
        x = self.input_proj(x)
        
        # Handle data conditioning
        if data_cond is not None:
            d_cond = data_cond.transpose(1, 2)
            d_emb = self.data_cond_proj(d_cond)
            x = x + d_emb
            
        # Handle time conditioning
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(-1)
        t_emb = self.time_mlp(timestep).unsqueeze(1)
        x = x + t_emb
        
        # Handle class conditioning
        if class_labels is not None and self.num_classes is not None:
             c_emb = self.class_embed(class_labels).unsqueeze(1)
             x = x + c_emb

        # Apply positional encoding 
        x = self.pos_encoder(x)

        # Hybrid block processing
        for layer in self.layers:
            if isinstance(layer, nn.TransformerEncoderLayer):
                x = layer(x)
            else:
                x = layer(x)

        # Output projection
        x = self.norm_f(x)
        x = self.output_proj(x)
        
        # Return [B, C, L]
        return x.transpose(1, 2)
