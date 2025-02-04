import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from algorhythms_app.models.music_gan import MusicGAN
from algorhythms_app.utils.data_processing import create_data_loader, MusicDataset
from algorhythms_app.training.train_combined import train_combined_model

def create_synthetic_dataset(num_sequences=10, sequence_length=32, output_dim=128):
    """Create a small synthetic dataset for testing."""
    sequences = []
    for _ in range(num_sequences):
        # Create random sequences with some structure
        # Using sine waves + noise to simulate musical patterns
        t = np.linspace(0, 4*np.pi, sequence_length)
        base_signal = np.sin(t).reshape(-1, 1)
        
        # Create multiple channels
        sequence = np.zeros((sequence_length, output_dim))
        for i in range(0, output_dim, 12):  # Add pattern every octave
            if i + 12 <= output_dim:
                sequence[:, i:i+12] = base_signal + 0.1 * np.random.randn(sequence_length, 1)
        
        # Normalize to [0, 1]
        sequence = (sequence - sequence.min()) / (sequence.max() - sequence.min())
        # Reshape to (1, sequence_length, output_dim)
        sequence = sequence.reshape(1, sequence_length, output_dim)
        sequences.append(sequence)
    
    return sequences

def test_training_pipeline():
    print("Starting training pipeline test...")
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    latent_dim = 100
    sequence_length = 32
    output_dim = 128
    batch_size = 2
    num_epochs = 2
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    sequences = create_synthetic_dataset(
        num_sequences=10,
        sequence_length=sequence_length,
        output_dim=output_dim
    )
    
    # Verify sequence shapes
    for i, seq in enumerate(sequences):
        print(f"Sequence {i} shape: {seq.shape}")
    
    # Create data loader
    train_loader = create_data_loader(
        sequences,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Initialize model
    print("Initializing MusicGAN...")
    music_gan = MusicGAN(latent_dim, sequence_length, output_dim)
    
    try:
        # Test forward pass
        print("Testing forward pass...")
        for batch in train_loader:
            print(f"Batch shape: {batch.shape}")
            music_gan.lstm(batch.to(device))
            test_noise = torch.randn(batch_size, latent_dim).to(device)
            music_gan.generator(test_noise)
            music_gan.discriminator(batch.to(device))
            print("Forward pass successful!")
            break
        
        # Test training
        print("\nStarting test training...")
        trained_model = train_combined_model(
            music_gan,
            train_loader,
            num_epochs=num_epochs,
            device=device
        )
        
        # Test generation
        print("\nTesting generation...")
        generated = trained_model.generate(num_sequences=1, device=device)
        print(f"Generated sequence shape: {generated.shape}")
        
        # Basic validation
        assert generated.shape == (1, sequence_length, output_dim), \
            f"Expected shape (1, {sequence_length}, {output_dim}), got {generated.shape}"
        assert np.all((generated >= 0) & (generated <= 1)), \
            "Generated values should be between 0 and 1"
        
        print("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    test_training_pipeline() 