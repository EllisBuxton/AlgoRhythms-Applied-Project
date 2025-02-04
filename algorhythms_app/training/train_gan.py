import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_gan(generator, discriminator, train_loader, num_epochs, latent_dim, device='cuda'):
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    for epoch in range(num_epochs):
        g_losses = []
        d_losses = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for real_sequences in pbar:
                batch_size = real_sequences.size(0)
                real_sequences = real_sequences.to(device)
                
                # Train Discriminator
                d_optimizer.zero_grad()
                
                label_real = torch.ones(batch_size, 1).to(device)
                label_fake = torch.zeros(batch_size, 1).to(device)
                
                # Real sequences
                output_real = discriminator(real_sequences)
                d_loss_real = criterion(output_real, label_real)
                
                # Fake sequences
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_sequences = generator(noise)
                output_fake = discriminator(fake_sequences.detach())
                d_loss_fake = criterion(output_fake, label_fake)
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train Generator
                g_optimizer.zero_grad()
                
                output_fake = discriminator(fake_sequences)
                g_loss = criterion(output_fake, label_real)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                
                pbar.set_postfix({
                    'D_loss': sum(d_losses) / len(d_losses),
                    'G_loss': sum(g_losses) / len(g_losses)
                })
        
        print(f'Epoch {epoch+1} - D_loss: {sum(d_losses)/len(d_losses):.4f}, G_loss: {sum(g_losses)/len(g_losses):.4f}')
    
    return generator, discriminator
