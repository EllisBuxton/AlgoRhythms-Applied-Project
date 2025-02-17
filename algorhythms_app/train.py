import torch
from algorhythms_app.models.music_gan import MusicGAN
from algorhythms_app.utils.data_processing import create_data_loader
from algorhythms_app.utils.midi_utils import MidiProcessor
from algorhythms_app.training.train_combined import train_combined_model
from algorhythms_app.utils.model_utils import save_combined_model, load_combined_model
import os
import pickle

def main():
    # Check CUDA availability
    print("\n=== System Info ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 30)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    sequence_length = 32
    output_dim = 128  # MIDI note range
    batch_size = 32
    num_epochs = 100
    
    print("\n=== MIDI Music GAN Training ===")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Sequence Length: {sequence_length}")
    print(f"Number of Epochs: {num_epochs}")
    print("=" * 30)
    
    # Initialize MIDI processor
    midi_processor = MidiProcessor(sequence_length=sequence_length)
    
    # Check for existing checkpoint
    checkpoint_dir = 'algorhythms_app/models/checkpoints'
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')])
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            print(f"\nLoading checkpoint: {latest_checkpoint}")
            with open(os.path.join(checkpoint_dir, latest_checkpoint), 'rb') as f:
                sequences_by_instrument = pickle.load(f)
        else:
            # Process from scratch
            sequences_by_instrument = midi_processor.process_directory('algorhythms_app/data')
    else:
        # Process from scratch
        sequences_by_instrument = midi_processor.process_directory('algorhythms_app/data')
    
    # Train separate models for each instrument
    total_instruments = len(sequences_by_instrument)
    for idx, (instrument_name, sequences) in enumerate(sequences_by_instrument.items(), 1):
        print(f"\n[{idx}/{total_instruments}] Training model for {instrument_name}")
        print("-" * 50)
        
        # Skip if not enough sequences for this instrument
        if len(sequences) < batch_size:
            print(f"⚠️ Skipping {instrument_name} - not enough sequences ({len(sequences)} < {batch_size})")
            continue
            
        print(f"Total sequences for {instrument_name}: {len(sequences)}")
        
        # Initialize model for this instrument
        music_gan = MusicGAN(latent_dim, sequence_length, output_dim)
        
        # Create data loader for this instrument
        train_loader = create_data_loader(sequences, batch_size=batch_size)
        num_batches = len(train_loader)
        print(f"Number of batches per epoch: {num_batches}")
        
        # Train model
        print(f"\nStarting training for {instrument_name}...")
        music_gan = train_combined_model(
            music_gan,
            train_loader,
            num_epochs=num_epochs,
            device=device
        )
        
        # Save trained model with instrument name
        model_path = f'algorhythms_app/models/saved/combined_model_{instrument_name}'
        save_combined_model(music_gan, model_path)
        print(f"✅ Model saved to {model_path}")

if __name__ == '__main__':
    main() 