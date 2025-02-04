import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_lstm(model, train_loader, num_epochs, learning_rate=0.001, device='cuda'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
            for batch_idx, sequences in enumerate(pbar):
                sequences = sequences.to(device)
                
                # Prepare input and target sequences
                # sequences shape: (batch_size, sequence_length, features)
                input_seq = sequences[:, :-1, :]
                target_seq = sequences[:, 1:, :]
                
                # Forward pass
                optimizer.zero_grad()
                output, _ = model(input_seq)
                
                # Calculate loss
                loss = criterion(output, target_seq)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
        
        print(f'Epoch {epoch+1} Loss: {total_loss / len(train_loader)}')
    
    return model
