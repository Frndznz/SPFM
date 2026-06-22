import torch
import torch.nn as nn
import math
from einops import rearrange

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding, corrected for batch_first=True.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding tensor in the correct shape (1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SinusoidalTimeEmbedding(nn.Module):
    """
    Optimized version which pre-calculates embeddings and registers them as a buffer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim) * -embeddings)
        self.register_buffer('freqs', freqs)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # This is now a very fast lookup and broadcast operation.
        args = time[:, None] * self.freqs[None, :]
        embeddings = torch.cat((args.sin(), args.cos()), dim=-1)
        if self.dim % 2 != 0:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))
        return embeddings

class Transformer1DModel(nn.Module):
    """
    A stable, architecturally sound Transformer encoder for time series.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        seq_length: int = None,
        class_embed_dim: int = None,
        num_classes: int = None,
        time_embed_dim: int = None,
        dropout: float = 0.1,
        loss_type: str = 'l2',
        spectral_weight: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.spectral_weight = spectral_weight
       
        # Input projection: [B, S, C_in] -> [B, S, C_model]
        self.input_proj = nn.Linear(in_channels, d_model)

        # Simplified Time Embedding
        if time_embed_dim is None:
            time_embed_dim = d_model
        self.time_proj = SinusoidalTimeEmbedding(dim=time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, d_model),
        )

        # Class Embedding
        if num_classes is not None:
            if class_embed_dim is not None:
                class_embed_dim = d_model
            self.class_embed = nn.Embedding(num_classes + 1, class_embed_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(class_embed_dim, class_embed_dim * 4),
                nn.SiLU(),
                nn.Linear(class_embed_dim * 4, d_model),
            )

        # For spectrally-partitioned mode: use x_lf as an additional conditioning
        self.data_cond_proj = nn.Linear(in_channels, d_model)
        
        # Positional encoder
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout, max_len=seq_length)
        
        # Main Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Output Projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, out_channels)
        
        # Zero initialize output layer for stability
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sample: torch.Tensor, 
            timestep: torch.Tensor, 
            class_labels: torch.Tensor = None,
            data_cond: torch.Tensor = None) -> torch.Tensor:
        # Process timestep
        timestep_embed = self.time_proj(timestep)
        timestep_embed = self.time_mlp(timestep_embed)

        # Process label
        if self.num_classes is not None:
            class_embed = self.class_embed(class_labels)
            class_embed = self.class_mlp(class_embed)
            conditioning_embed = timestep_embed + class_embed # [B, C_model]
        else:
            conditioning_embed = timestep_embed

        # Process signal
        sample = sample.permute(0, 2, 1) # [B, C_in, S] -> [B, S, C_in]
        x = self.input_proj(sample) # [B, S, C_model]
        
        if data_cond is not None:
            data_cond = data_cond.permute(0, 2, 1)
            data_embed = self.data_cond_proj(data_cond)
            x = x + data_embed

        x = x + conditioning_embed.unsqueeze(1)
        x = self.pos_encoder(x)
        
        # Pass through transformer
        output = self.transformer_encoder(x)
        output = self.output_norm(output)
        output = self.output_proj(output)
        
        output = output.permute(0, 2, 1) # [B, S, C] -> [B, C, S]
        
        return output
