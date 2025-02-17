import torch
import torch.nn as nn
from algorhythms_app.models.generator import MusicGenerator
from algorhythms_app.models.discriminator import MusicDiscriminator
from algorhythms_app.models.lstm import MusicLSTM

class MusicGAN:
    def __init__(self, latent_dim, sequence_length, output_dim):
        self.generator = MusicGenerator(latent_dim, sequence_length, output_dim)
        self.discriminator = MusicDiscriminator(output_dim, sequence_length)
        self.lstm = MusicLSTM(
            input_size=output_dim,
            hidden_size=256,
            num_layers=2,
            output_size=output_dim
        )
        self.latent_dim = latent_dim
        self.criterion = nn.BCELoss()
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=0.001)
        
    def train_step(self, sequences, device='cuda'):
        # Move models to device
        self.generator.to(device)
        self.discriminator.to(device)
        self.lstm.to(device)
        
        batch_size = sequences.size(0)
        real_sequences = sequences.to(device)
        
        # Train LSTM
        self.lstm.train()
        self.lstm_optimizer.zero_grad()
        lstm_output, _ = self.lstm(real_sequences)
        lstm_loss = nn.MSELoss()(lstm_output, real_sequences)
        lstm_loss.backward()
        self.lstm_optimizer.step()
        
        # Train Discriminator
        self.discriminator.train()
        self.d_optimizer.zero_grad()
        
        # Real sequences
        label_real = torch.ones(batch_size, 1).to(device)
        output_real = self.discriminator(real_sequences)
        d_loss_real = self.criterion(output_real, label_real)
        
        # Fake sequences
        noise = torch.randn(batch_size, self.latent_dim).to(device)
        fake_sequences = self.generator(noise)
        label_fake = torch.zeros(batch_size, 1).to(device)
        output_fake = self.discriminator(fake_sequences.detach())
        d_loss_fake = self.criterion(output_fake, label_fake)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.generator.train()
        self.g_optimizer.zero_grad()
        
        output_fake = self.discriminator(fake_sequences)
        g_loss = self.criterion(output_fake, label_real)  # Try to fool discriminator
        
        # Add LSTM refinement loss
        refined_fake, _ = self.lstm(fake_sequences)
        refinement_loss = nn.MSELoss()(refined_fake, fake_sequences)
        g_loss += 0.5 * refinement_loss
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
        
    def generate(self, num_sequences=1, device='cuda'):
        self.generator.eval()
        self.lstm.eval()
        with torch.no_grad():
            noise = torch.randn(num_sequences, self.latent_dim).to(device)
            generated = self.generator(noise)
            
            # Refine with LSTM
            refined, _ = self.lstm(generated)
            return refined.cpu().numpy()
