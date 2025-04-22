import os
import torch
import numpy as np
import pretty_midi
import sys
import random
import torch.serialization
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gru_model import GRUModel
from utils.pop909_processor import POP909Processor

def generate_midi_file(note_sequence, output_file='output.mid', 
                       instrument=0, velocity=80, duration=0.5, start_time=0.0,
                       note_range=None):
    """Convert a sequence of MIDI note numbers to a MIDI file"""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument)
    
    current_time = start_time
    for note in note_sequence:
        # If note_range provided, adjust note back to MIDI range
        if note_range is not None:
            note_number = int(note) + note_range[0]  # Shift back to original range
        else:
            note_number = int(note)
            
        if 0 <= note_number <= 127:  # Valid MIDI note range
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=note_number,
                start=current_time,
                end=current_time + duration
            )
            instrument.notes.append(note)
        current_time += duration
    
    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f"MIDI file saved to {output_file}")

def load_model(model_path):
    """Load a trained GRU model"""
    print("Loading model...")
    try:
        # Try loading with weights_only=False
        checkpoint = torch.load(
            model_path, 
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            weights_only=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
    
    # Extract model parameters
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    output_size = checkpoint['output_size']
    note_range = checkpoint['note_range']
    chord_vocab_size = checkpoint['chord_vocab_size']
    
    # Create model
    model = GRUModel(input_size, hidden_size, output_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model.to_device()
    model.eval()
    
    print(f"Model loaded and running on: {model.device}")
    
    return model, checkpoint

def generate_random_seed(note_range, chord_vocab_size, seq_length=16):
    """Generate a random seed sequence within the trained note range"""
    min_note, max_note = 0, note_range[1] - note_range[0]  # Adjusted to 0-based index
    
    # Generate random sequence
    note_indices = np.random.randint(0, max_note + 1, size=seq_length)
    
    # Convert to one-hot encoding for notes
    note_range_size = max_note + 1
    note_onehot = np.zeros((seq_length, note_range_size))
    for i, note in enumerate(note_indices):
        note_onehot[i, note] = 1
    
    # Generate random chord context
    chord_onehot = np.zeros((seq_length, chord_vocab_size))
    for i in range(seq_length):
        chord_idx = np.random.randint(0, chord_vocab_size)
        chord_onehot[i, chord_idx] = 1
    
    # Combine note and chord information
    seed = np.concatenate([note_onehot, chord_onehot], axis=1)
    
    # Convert to actual MIDI notes for display
    midi_notes = [idx + note_range[0] for idx in note_indices]
    
    return torch.FloatTensor(seed), midi_notes

def extract_real_seed_from_dataset(dataset_path, note_range, chord_vocab_size, seq_length=16):
    """Extract a real sequence from the dataset to use as a seed"""
    # Find MIDI files
    midi_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    if not midi_files:
        print(f"No MIDI files found in {dataset_path}, using random seed")
        return generate_random_seed(note_range, chord_vocab_size, seq_length)
    
    # Select a random MIDI file
    midi_file = random.choice(midi_files)
    print(f"Using seed from: {os.path.basename(midi_file)}")
    
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        if len(midi_data.instruments) > 0:
            melody_track = midi_data.instruments[0]
            notes = sorted(melody_track.notes, key=lambda x: x.start)
            
            if len(notes) >= seq_length:
                # Get a random sequence
                start_idx = random.randint(0, len(notes) - seq_length)
                
                # Get pitches and adjust to model range
                pitches = [note.pitch for note in notes[start_idx:start_idx + seq_length]]
                adjusted_pitches = [max(0, min(pitch - note_range[0], note_range[1] - note_range[0])) 
                                   for pitch in pitches]
                
                # Create one-hot encoding for notes
                note_range_size = note_range[1] - note_range[0] + 1
                note_onehot = np.zeros((seq_length, note_range_size))
                for i, note_idx in enumerate(adjusted_pitches):
                    note_onehot[i, int(note_idx)] = 1
                
                # Get chord context
                song_dir = os.path.dirname(midi_file)
                chord_file = os.path.join(song_dir, "chord_midi.txt")
                chords = []
                if os.path.exists(chord_file):
                    with open(chord_file, 'r') as f:
                        for line in f:
                            start, end, chord = line.strip().split('\t')
                            chords.append({
                                'start': float(start),
                                'end': float(end),
                                'chord': chord
                            })
                
                # Create one-hot encoding for chords
                chord_onehot = np.zeros((seq_length, chord_vocab_size))
                for i in range(seq_length):
                    note_time = notes[start_idx + i].start
                    current_chord = None
                    for chord in chords:
                        if chord['start'] <= note_time < chord['end']:
                            current_chord = chord['chord']
                            break
                    if current_chord:
                        chord_idx = hash(current_chord) % chord_vocab_size
                        chord_onehot[i, chord_idx] = 1
                
                # Combine note and chord information
                seed = np.concatenate([note_onehot, chord_onehot], axis=1)
                
                return torch.FloatTensor(seed), pitches
    except Exception as e:
        print(f"Error extracting seed from {midi_file}: {e}")
    
    # Fall back to random seed
    print("Falling back to random seed")
    return generate_random_seed(note_range, chord_vocab_size, seq_length)

def generate_melody(model, seed_sequence, length=16, temperature=1.0):
    """Generate a melody using the trained model"""
    model.eval()
    device = model.device
    
    with torch.no_grad():
        # Make sure seed_sequence is on the correct device
        seed_sequence = seed_sequence.to(device)
        
        # The model expects input shape [batch_size, seq_length, input_size]
        # Add batch dimension if it doesn't exist
        if len(seed_sequence.shape) == 2:
            current_sequence = seed_sequence.unsqueeze(0)
        else:
            current_sequence = seed_sequence
            
        print(f"Input sequence shape: {current_sequence.shape}")
        
        # Get dimensions
        input_size = current_sequence.size(2)
        
        # Generate melody
        generated_melody = []
        print(f"Generating {length} notes...")
        for i in tqdm(range(length)):
            # Forward pass
            output, _ = model(current_sequence)
            
            # Apply temperature
            output = output / temperature
            
            # Sample from distribution
            probs = torch.exp(output).squeeze()
            note_idx = torch.multinomial(probs, 1).item()
            
            # Add to melody
            generated_melody.append(note_idx)
            
            # Update sequence by removing first timestep and adding new note
            new_note = torch.zeros(1, 1, input_size, device=device)
            new_note[0, 0, note_idx] = 1
            current_sequence = torch.cat([current_sequence[:, 1:, :], new_note], dim=1)
            
    return generated_melody

def main():
    # Parameters
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'models', 'saved_models', 'gru_pop909.pth')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'data', 'generated')
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'POP909')
    melody_length = 64  # Number of notes to generate
    temperature = 0.8  # Lower for more conservative, higher for more random
    use_random_seed = False  # Use real seed from dataset if False
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    output_midi = os.path.join(output_dir, f'gru_generated_{int(temperature*10)}.mid')
    
    try:
        # Load the model
        model, checkpoint = load_model(model_path)
        
        # Get parameters
        note_range = checkpoint['note_range']
        sequence_length = checkpoint.get('sequence_length', 16)
        chord_vocab_size = checkpoint.get('chord_vocab_size', 0)
        
        print(f"Model parameters: Note range {note_range}, Sequence length {sequence_length}, Chord vocab size {chord_vocab_size}")
        
        # Generate seed
        print("Generating seed sequence...")
        if use_random_seed:
            seed_sequence, seed_notes = generate_random_seed(note_range, chord_vocab_size, sequence_length)
        else:
            seed_sequence, seed_notes = extract_real_seed_from_dataset(
                dataset_path, note_range, chord_vocab_size, sequence_length
            )
        
        print(f"Seed notes: {seed_notes}")
        
        # Generate melody
        print(f"Generating melody of length {melody_length} with temperature {temperature}...")
        generated_melody = generate_melody(model, seed_sequence, melody_length, temperature)
        
        # Convert to MIDI notes
        generated_notes = [idx + note_range[0] for idx in generated_melody]
        
        print("Generated melody (MIDI note values):")
        print(generated_notes)
        
        # Save MIDI file
        generate_midi_file(generated_melody, output_midi, note_range=note_range)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 