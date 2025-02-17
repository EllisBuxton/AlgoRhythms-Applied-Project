import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import psutil
import datetime

def train_combined_model(music_gan, train_loader, num_epochs, device='cuda'):
    """Train both LSTM and GAN models together."""
    
    start_time = time.time()
    best_loss = float('inf')
    
    print("\nStarting Training:")
    print(f"{'Epoch':^10} | {'G Loss':^12} | {'D Loss':^12} | {'Time/Batch':^12} | {'Memory':^12}")
    print("-" * 65)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_g_loss = 0
        total_d_loss = 0
        batch_times = []
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, sequences in enumerate(pbar):
                batch_start = time.time()
                
                # Train step expects a batch of sequences
                g_loss, d_loss = music_gan.train_step(sequences, device)
                
                # Calculate metrics
                total_g_loss += g_loss
                total_d_loss += d_loss
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Memory usage
                memory_used = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                # Update progress bar
                pbar.set_postfix({
                    'G_Loss': f'{g_loss:.4f}',
                    'D_Loss': f'{d_loss:.4f}',
                    'Time/Batch': f'{batch_time:.3f}s',
                    'Memory': f'{memory_used:.1f}MB'
                })
        
        # Calculate epoch statistics
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_loss = total_d_loss / len(train_loader)
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"{epoch+1:^10} | {avg_g_loss:^12.4f} | {avg_d_loss:^12.4f} | "
              f"{avg_batch_time:^12.3f} | {memory_used:^12.1f}")
        
        # Save best model if loss improved
        current_loss = avg_g_loss + avg_d_loss
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"✨ New best loss: {best_loss:.4f}")
    
    # Training complete summary
    total_time = time.time() - start_time
    print("\nTraining Complete!")
    print("-" * 65)
    print(f"Total training time: {str(datetime.timedelta(seconds=int(total_time)))}")
    print(f"Best combined loss: {best_loss:.4f}")
    print(f"Final memory usage: {memory_used:.1f}MB")
    
    return music_gan 