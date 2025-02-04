import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_combined_model(music_gan, train_loader, num_epochs, device='cuda'):
    """Train both LSTM and GAN models together."""
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, sequences in enumerate(pbar):
                # Train step expects a batch of sequences
                music_gan.train_step(train_loader, device)  # Pass the entire loader instead of just sequences
                
                pbar.set_postfix({
                    'status': 'Training in progress'
                })
    
    return music_gan 