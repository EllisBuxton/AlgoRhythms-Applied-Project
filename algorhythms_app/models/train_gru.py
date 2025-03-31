import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gru_model import GRUModel
from utils.pop909_processor import POP909Processor

# Set random seed for reproducibility
torch.manual_seed(42)

def train_model(model, train_data, train_labels, val_data=None, val_labels=None, 
                num_epochs=10, batch_size=64, learning_rate=0.001):
    """Train the GRU model using GPU if available"""
    # Move model to the appropriate device
    model.to_device()
    device = model.device
    
    # Print which device we're using
    print(f"Training on: {device}")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert data to TensorDataset for batching
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    
    if val_data is not None and val_labels is not None:
        val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
    else:
        val_loader = None
    
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, labels in progress_bar:
            # Move data to appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': total_train_loss / len(train_loader)})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if val_loader:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move data to device
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                
                avg_val_loss = total_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')
    
    # Plot training (and validation) loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_loss.png')
    plt.close()
    
    return model, train_losses, val_losses

def main():
    # Dataset and model parameters
    pop909_path = 'algorhythms_app/POP909'  # Update with your actual path
    sequence_length = 16
    hidden_size = 256
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001
    model_save_path = 'algorhythms_app/models/saved_models/gru_pop909.pth'
    
    # Create directory for saved models if it doesn't exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Process dataset
    processor = POP909Processor(pop909_path, sequence_length=sequence_length)
    X, y, note_range = processor.prepare_data_for_training()
    
    # Save note range for generation later
    np.save('algorhythms_app/models/saved_models/note_range.npy', note_range)
    
    # Split data into training and validation sets (80/20)
    dataset_size = len(X)
    train_size = int(dataset_size * 0.8)
    
    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    
    # Define input size based on note range
    input_size = note_range[1] - note_range[0] + 1
    output_size = input_size  # Same range for output
    
    print(f"Input/output size: {input_size}")
    
    # Create and train model
    print("Creating GRU model...")
    model = GRUModel(input_size, hidden_size, output_size)
    
    print("Training model...")
    model, train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val,
        num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate
    )
    
    # Save the trained model
    print(f"Saving model to {model_save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'note_range': note_range,
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'sequence_length': sequence_length
    }, model_save_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 