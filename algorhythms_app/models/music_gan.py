import torch
from algorhythms_app.models.generator import MusicGenerator
from algorhythms_app.models.discriminator import MusicDiscriminator
from algorhythms_app.models.lstm import MusicLSTM
from algorhythms_app.training.train_lstm import train_lstm
from algorhythms_app.training.train_gan import train_gan

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
        
    def train_step(self, train_loader, device='cuda'):
        # Train LSTM first
        self.lstm = train_lstm(
            self.lstm,
            train_loader,
            num_epochs=1,
            device=device
        )
        
        # Then train GAN
        self.generator, self.discriminator = train_gan(
            self.generator,
            self.discriminator,
            train_loader,
            num_epochs=1,
            latent_dim=self.generator.latent_dim,
            device=device
        )
        
    def generate(self, num_sequences=1, device='cuda'):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_sequences, self.generator.latent_dim).to(device)
            generated = self.generator(noise)
            
            # Refine with LSTM
            refined, _ = self.lstm(generated)
            return refined.cpu().numpy()
