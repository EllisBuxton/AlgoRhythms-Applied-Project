import torch
from algorhythms_app.utils.nes_data_loader import create_nes_data_loader
from algorhythms_app.models.vae_transformer import VAETransformer
from algorhythms_app.training.train_vae_transformer import train_vae_transformer
import os
from datetime import datetime
import json

def main():
    # Configuration
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "data", "midi_files", "nesmdb_midi", "nesmdb_midi", "train")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = f"models/vae_transformer_{timestamp}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Hyperparameters
    batch_size = 32
    sequence_length = 256
    hidden_dim = 512
    latent_dim = 256
    num_layers = 6
    learning_rate = 1e-4
    num_epochs = 100
    
    # Save hyperparameters
    hyperparams = {
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'hidden_dim': hidden_dim,
        'latent_dim': latent_dim,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    
    with open(os.path.join(model_save_dir, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Create data loader
    train_loader = create_nes_data_loader(
        data_dir,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    # Initialize model
    model = VAETransformer(
        input_dim=128,  # MIDI pitch range
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers
    )
    
    # Train model
    trained_model, history = train_vae_transformer(
        model,
        train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=model_save_dir
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(model_save_dir, 'final_model.pt'))
    
    print(f"Training completed. Model and history saved to {model_save_dir}")

if __name__ == "__main__":
    main() 