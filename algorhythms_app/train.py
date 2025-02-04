import torch
from algorhythms_app.models.music_gan import MusicGAN
from algorhythms_app.utils.data_processing import create_data_loader
from algorhythms_app.utils.midi_utils import MidiProcessor
from algorhythms_app.training.train_combined import train_combined_model
from algorhythms_app.utils.model_utils import save_combined_model, load_combined_model

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 100
    sequence_length = 32
    output_dim = 128  # MIDI note range
    batch_size = 32
    num_epochs = 100
    
    # Initialize models
    music_gan = MusicGAN(latent_dim, sequence_length, output_dim)
    
    # Load data
    midi_processor = MidiProcessor(sequence_length=sequence_length)
    sequences = midi_processor.process_directory('algorhythms_app/data/midi_files')
    train_loader = create_data_loader(sequences, batch_size=batch_size)
    
    # Train model
    music_gan = train_combined_model(
        music_gan,
        train_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    # Save trained model
    save_combined_model(music_gan, 'algorhythms_app/models/saved/combined_model')

if __name__ == '__main__':
    main() 