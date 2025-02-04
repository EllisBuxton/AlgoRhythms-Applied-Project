import numpy as np
import torch

def compute_metrics(model, val_loader, device):
    model.eval()
    total_loss = 0
    metrics = {
        'reconstruction_loss': 0,
        'novelty_score': 0,
        'consistency_score': 0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            # Compute various metrics
            sequences = batch.to(device)
            reconstructed = model.generate(sequences.size(0), device)
            
            # Reconstruction loss
            metrics['reconstruction_loss'] += torch.nn.functional.mse_loss(
                torch.tensor(reconstructed), sequences.cpu()
            ).item()
            
            # Add other metrics...
            
    # Average metrics
    for key in metrics:
        metrics[key] /= len(val_loader)
        
    return metrics 