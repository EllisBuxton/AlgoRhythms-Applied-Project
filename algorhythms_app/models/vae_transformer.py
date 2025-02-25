import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Tuple

class VAETransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Encoder
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # VAE components
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        decoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer_decoder = TransformerEncoder(decoder_layers, num_layers)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_embedding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)  # Average pooling
        
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z: torch.Tensor, seq_length: int) -> torch.Tensor:
        z = self.latent_to_hidden(z)
        z = z.unsqueeze(1).repeat(1, seq_length, 1)
        z = self.transformer_decoder(z)
        return self.output_layer(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, x.size(1))
        return recon_x, mu, log_var
    
    def generate(self, num_samples: int, seq_length: int, device: str = 'cuda') -> torch.Tensor:
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            return self.decode(z, seq_length) 