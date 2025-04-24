import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gru_model import GRUModel
from utils.pop909_processor import POP909Processor

# Set random seed for reproducibility
torch.manual_seed(42)

def train_model(model, train_data, train_labels, val_data=None, val_labels=None, 
                num_epochs=50, batch_size=64, learning_rate=0.001, 
                patience=5, min_lr=1e-6, grad_clip=1.0):
    """Train the GRU model using GPU if available with enhanced training process"""
    # Move model to the appropriate device
    model.to_device()
    device = model.device
    
    # Print which device we're using
    print(f"Training on: {device}")
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2,
        min_lr=min_lr
    )
    
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
    best_val_loss = float('inf')
    patience_counter = 0
    
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
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
                
                # Update learning rate
                scheduler.step(avg_val_loss)
                
                print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                        # Load best model
                        model.load_state_dict(torch.load('best_model.pth'))
                        break
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
    
    # Save loss history arrays
    np.save('train_losses.npy', np.array(train_losses))
    if val_losses:
        np.save('val_losses.npy', np.array(val_losses))
    
    return model, train_losses, val_losses

def main():
    # Dataset and model parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    pop909_path = os.path.join(project_root, 'POP909')
    sequence_length = 16
    hidden_size = 512  # Increased hidden size
    num_epochs = 50  # Increased number of epochs
    batch_size = 64
    learning_rate = 0.001
    
    # Create directories for saved models and plots
    saved_models_dir = os.path.join(project_root, 'models', 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)
    
    model_save_path = os.path.join(saved_models_dir, 'gru_pop909.pth')
    note_range_path = os.path.join(saved_models_dir, 'note_range.npy')
    
    # Process dataset
    processor = POP909Processor(pop909_path, sequence_length=sequence_length)
    X, y, note_range = processor.prepare_data_for_training()
    
    # Save note range for generation later
    np.save(note_range_path, note_range)
    
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
    
    # Define input size based on note range and chord vocabulary
    note_range_size = note_range[1] - note_range[0] + 1
    chord_vocab_size = len(processor.chord_vocab)
    input_size = note_range_size + chord_vocab_size  # Combined input size
    output_size = note_range_size  # Output size remains the same
    
    print(f"Note range size: {note_range_size}")
    print(f"Chord vocabulary size: {chord_vocab_size}")
    print(f"Total input size: {input_size}")
    print(f"Output size: {output_size}")
    
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
        'sequence_length': sequence_length,
        'chord_vocab_size': chord_vocab_size
    }, model_save_path)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 