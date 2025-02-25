import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import json
import os
from datetime import datetime
from typing import Optional, Dict

def vae_loss(recon_x, x, mu, log_var, beta: float = 1.0, min_kl: float = 0.1):
    """Compute VAE loss with reconstruction and KL divergence terms"""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Apply minimum KL divergence
    kl_loss = torch.max(kl_loss, torch.tensor(min_kl).to(kl_loss.device))
    
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()

def train_vae_transformer(
    model,
    train_loader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    checkpoint_dir: Optional[str] = None,
    kl_warmup_epochs: int = 10
):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    
    # Create checkpoint directory if specified
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'epoch_losses': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        # Calculate beta for KL warm-up
        beta = min(1.0, epoch / kl_warmup_epochs)
        
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for batch in pbar:
                batch = batch.to(device)
                
                # Forward pass
                recon_batch, mu, log_var = model(batch)
                loss, recon, kl = vae_loss(recon_batch, batch, mu, log_var, beta=beta)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                epoch_recon_loss += recon
                epoch_kl_loss += kl
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon,
                    'kl': kl
                })
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = epoch_recon_loss / len(train_loader)
        avg_kl_loss = epoch_kl_loss / len(train_loader)
        
        # Update history
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon_loss)
        history['kl_loss'].append(avg_kl_loss)
        history['epoch_losses'].append({
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        })
        
        print(f'Epoch {epoch + 1}:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Reconstruction Loss: {avg_recon_loss:.4f}')
        print(f'  KL Loss: {avg_kl_loss:.4f}')
        
        # Save checkpoint if it's the best model so far
        if checkpoint_dir and avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            
            # Save training history
            history_path = os.path.join(checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
    
    return model, history 