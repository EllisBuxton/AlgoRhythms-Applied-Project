import torch
import os
import pretty_midi
import numpy as np
from algorhythms_app.models.vae_transformer import VAETransformer

def is_scale_note(note, scale='C_major'):
    """Check if a note belongs to a given scale"""
    # C major scale: C(0), D(2), E(4), F(5), G(7), A(9), B(11)
    c_major = {0, 2, 4, 5, 7, 9, 11}
    note_in_scale = note % 12 in c_major
    return note_in_scale

def generate_midi(piano_roll, output_path, fs=16):
    """Convert piano roll to MIDI file with melodic constraints"""
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    
    time_per_step = 0.25  # Quarter note duration
    min_duration = 0.25  # Minimum note duration (eighth note)
    max_duration = 1.0   # Maximum note duration (quarter note)
    last_note = None
    current_duration = 0
    sustain_counter = 0  # Track how long we're sustaining the current note
    
    # Extended rhythmic patterns with more variety
    rhythm_patterns = [
        [0.5, 0.25, 0.25, 0.5, 0.25, 0.25],     # Pattern 1: ♪ ⅛ ⅛ ♪ ⅛ ⅛
        [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],     # Pattern 2: ⅛ ⅛ ♪ ⅛ ⅛ ♪
        [0.5, 0.5, 0.25, 0.25, 0.5],            # Pattern 3: ♪ ♪ ⅛ ⅛ ♪
        [0.25, 0.25, 0.25, 0.25, 0.5, 0.5],     # Pattern 4: ⅛ ⅛ ⅛ ⅛ ♪ ♪
        [0.75, 0.25, 0.5, 0.5],                 # Pattern 5: ♪. ⅛ ♪ ♪
        [0.25, 0.75, 0.25, 0.75],               # Pattern 6: ⅛ ♪. ⅛ ♪.
        [0.5, 0.25, 0.25, 0.25, 0.25, 0.5],     # Pattern 7: ♪ ⅛ ⅛ ⅛ ⅛ ♪
        [0.25, 0.5, 0.25, 0.5, 0.5],            # Pattern 8: ⅛ ♪ ⅛ ♪ ♪
        [0.5, 0.5, 0.5, 0.25, 0.25],            # Pattern 9: ♪ ♪ ♪ ⅛ ⅛
        [0.25, 0.25, 0.75, 0.25, 0.5]           # Pattern 10: ⅛ ⅛ ♪. ⅛ ♪
    ]
    
    # Extended melodic patterns with more variety
    melodic_patterns = [
        [0, 2, 4, 7, 5, 3],                # Ascending then descending
        [7, 5, 3, 0, 2, 4],                # Descending then ascending
        [0, 4, 2, 7, 5, 2],                # Jumping pattern
        [0, -2, -4, -1, 2, 4],             # Down then up
        [4, 7, 5, 2, 0, -2],               # Up then down
        [0, 3, 5, 7, 4, 2],                # Ascending scale fragment
        [0, 5, 3, 7, 2, 4],                # Mixed intervals up
        [-2, 2, 4, 0, 5, 2],               # Zigzag pattern
        [0, 4, 7, 3, 5, 2],                # Broken chord with passing tones
        [7, 4, 0, 2, 5, 3],                # Descending with skips
        [0, 3, -2, 2, 4, 7],               # Wide range pattern
        [2, 5, 0, 4, -2, 3],               # Complex intervals
        [0, 2, 5, 7, 3, 0],                # Circular pattern
        [4, 0, 7, 2, 5, 3],                # Scattered intervals
        [0, 5, 2, 7, 3, 6]                 # Rising pattern with skips
    ]
    
    current_pattern = 0
    pattern_position = 0
    melodic_pattern_position = 0
    current_note = None
    note_start_time = None
    base_note = None
    last_two_notes = []  # Keep track of the last two notes to prevent repetition
    
    # Base intervals for melodic guidance
    melodic_intervals = [-7, -5, -4, -2, 0, 2, 4, 5, 7]
    
    def get_next_note(time_idx, current_base, model_probs):
        """Get next note using model probabilities and melodic constraints"""
        # Get top K probable notes from model
        top_k = 5
        probable_notes = np.argsort(model_probs)[-top_k:]
        
        # Filter notes that would create repetition
        valid_notes = [n for n in probable_notes if n not in last_two_notes]
        
        if not valid_notes:  # If all probable notes would create repetition
            valid_notes = [n for n in probable_notes]
        
        # Weight probabilities by melodic consistency
        weights = []
        for note in valid_notes:
            interval = note - current_base
            # Favor notes that create musical intervals
            melodic_weight = 1.0
            for valid_interval in melodic_intervals:
                if abs(interval - valid_interval) <= 1:
                    melodic_weight = 2.0
                    break
            
            # Combine model probability with melodic weight
            weight = model_probs[note] * melodic_weight
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Choose note based on weighted probabilities
        chosen_note = np.random.choice(valid_notes, p=weights)
        return chosen_note
    
    for time_idx in range(piano_roll.shape[1]):
        # Get model probabilities for this timestep
        model_probs = piano_roll[:, time_idx]
        
        current_duration = (time_idx * time_per_step - note_start_time) if note_start_time is not None else 0
        start_new_note = (
            current_note is None or 
            current_duration >= rhythm_patterns[current_pattern][pattern_position]
        )
        
        if start_new_note:
            if current_note is not None:
                duration = (time_idx * time_per_step) - note_start_time
                if duration >= min_duration:
                    note = pretty_midi.Note(
                        velocity=current_note['velocity'],
                        pitch=current_note['pitch'],
                        start=note_start_time,
                        end=time_idx * time_per_step
                    )
                    piano.notes.append(note)
            
            # Get base note from model probabilities
            base_note = 60  # Middle C as default
            if len(last_two_notes) > 0:
                base_note = last_two_notes[-1]
            
            # Use model to choose next note
            midi_note = get_next_note(time_idx, base_note, model_probs)
            midi_note = max(48, min(84, midi_note))
            
            # Update last two notes tracking
            last_two_notes.append(midi_note)
            if len(last_two_notes) > 2:
                last_two_notes.pop(0)
            
            # Reset pattern position if needed
            pattern_position = (pattern_position + 1) % len(rhythm_patterns[current_pattern])
            if pattern_position == 0:
                # Use model probabilities to influence pattern selection
                pattern_weights = [np.mean(model_probs[60:72]) for _ in rhythm_patterns]
                current_pattern = np.random.choice(len(rhythm_patterns), p=np.array(pattern_weights)/sum(pattern_weights))
            
            # Dynamic velocity based on model probabilities and pattern position
            base_velocity = int(model_probs[midi_note] * 40) + 70  # Range 70-110
            if pattern_position == 0:
                velocity = min(110, base_velocity + 10)
            else:
                velocity = min(100, base_velocity)
            
            current_note = {
                'pitch': midi_note,
                'velocity': velocity
            }
            note_start_time = time_idx * time_per_step
    
    # Handle last note
    if current_note is not None:
        duration = min((piano_roll.shape[1] * time_per_step) - note_start_time, max_duration)
        if duration >= min_duration:
            note = pretty_midi.Note(
                velocity=current_note['velocity'],
                pitch=current_note['pitch'],
                start=note_start_time,
                end=note_start_time + duration
            )
            piano.notes.append(note)
    
    # Add grace notes based on model probabilities
    for i in range(1, len(piano.notes)):
        main_note = piano.notes[i]
        prob_idx = int(main_note.start / time_per_step)
        if prob_idx < piano_roll.shape[1]:
            grace_prob = np.max(piano_roll[main_note.pitch-2:main_note.pitch+3, prob_idx])
            if grace_prob > 0.3 and main_note.end - main_note.start >= 0.5:
                grace_duration = 0.125
                if main_note.start - grace_duration >= 0:
                    # Use model to choose grace note pitch
                    grace_pitch = main_note.pitch + (-2 if piano_roll[main_note.pitch-2, prob_idx] > 
                                                    piano_roll[main_note.pitch+2, prob_idx] else 2)
                    grace_note = pretty_midi.Note(
                        velocity=main_note.velocity - 10,
                        pitch=grace_pitch,
                        start=main_note.start - grace_duration,
                        end=main_note.start
                    )
                    piano.notes.insert(i, grace_note)

    pm.instruments.append(piano)
    pm.write(output_path)

def create_melodic_sequence(sequence_length):
    """Create a template for melodic sequence"""
    sequence = np.zeros((128, sequence_length))
    
    # More complex rhythmic patterns with syncopation and varied dynamics
    patterns = [
        [1, 0, 0.7, 0.3, 0.8, 0, 0.6, 0.4, 0.9, 0.2, 0.5, 0.3],  # Pattern 1
        [0.8, 0.3, 1, 0, 0.6, 0.4, 0.7, 0, 0.8, 0.5, 0.3, 0.6],  # Pattern 2
        [1, 0.4, 0.6, 0.3, 0.9, 0, 0.7, 0.5, 0.4, 0.8, 0, 0.6],  # Pattern 3
        [0.7, 0, 1, 0.4, 0.6, 0.3, 0.8, 0, 0.5, 0.7, 0.4, 0.2]   # Pattern 4
    ]
    
    # Enhanced melodic motifs with more variation
    motifs = [
        [0, 2, 4, 7, 9, 7, 4, 2],    # Extended major chord
        [7, 4, 2, 0, 2, 4, 7, 9],    # Major scale fragment
        [0, 4, 7, 11, 7, 4, 0, 4],   # Major seventh arpeggio
        [0, 2, 3, 5, 7, 8, 10, 12],  # Mixed scale pattern
        [0, 3, 7, 10, 7, 3, 0, 3],   # Minor seventh arpeggio
        [0, 2, 4, 0, 7, 4, 2, 0],    # Broken chord pattern
    ]
    
    for i in range(sequence_length):
        # Select pattern based on position in form
        pattern_idx = (i // 32) % len(patterns)
        current_pattern = patterns[pattern_idx]
        
        # Get probability for this step
        prob = current_pattern[i % len(current_pattern)]
        
        if prob > np.random.random():
            # Select motif based on position in phrase
            motif_idx = (i // 16) % len(motifs)
            current_motif = motifs[motif_idx]
            
            # Calculate base note (varying by phrase)
            base_note = 28 + ((i // 8) % 3) * 4
            
            # Add note from current motif
            note_idx = base_note + current_motif[i % len(current_motif)]
            sequence[note_idx, i] = 1
            
            # Occasionally add harmony note
            if np.random.random() < 0.15:  # 15% chance
                harmony_idx = note_idx + 4  # Add a third above
                if harmony_idx < 128:
                    sequence[harmony_idx, i] = 0.7
    
    return sequence

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Model hyperparameters
    latent_dim = 256
    input_dim = 128
    hidden_dim = 512
    
    # Generation parameters
    num_samples = 5
    sequence_length = 128  # 32 measures (doubled)
    temperature = 0.9  # Slightly higher temperature for more variation
    
    # Paths
    model_path = os.path.join(base_dir, "models", "vae_transformer_20250225_024719", "best_model.pt")
    output_dir = os.path.join(base_dir, "algorhythms_app", "generated_melodies")
    
    print(f"Using device: {device}")
    print(f"Loading model from: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    try:
        # Initialize model
        print("Initializing model...")
        model = VAETransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=6
        )
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        model.to(device)
        
        print(f"Generating {num_samples} melodies...")
        
        with torch.no_grad():
            for i in range(num_samples):
                # Generate multiple latent vectors and interpolate
                z1 = torch.randn(1, latent_dim).to(device) * temperature
                z2 = torch.randn(1, latent_dim).to(device) * temperature
                
                # Generate sequences and interpolate
                generated1 = model.decode(z1, sequence_length // 2)
                generated2 = model.decode(z2, sequence_length // 2)
                generated = torch.cat([generated1, generated2], dim=1)
                
                # Apply softmax with temperature
                probs = torch.softmax(generated[0] * 2.5, dim=0)
                
                # Convert to piano roll format
                piano_roll = probs.cpu().numpy().T
                
                # Create melodic template
                melodic_template = create_melodic_sequence(sequence_length)
                
                # Combine model output with melodic template
                combined_roll = np.maximum(piano_roll * 0.7, melodic_template)
                
                print(f"\nSample {i+1}:")
                print(f"Sequence length: {sequence_length} steps ({sequence_length/4} measures)")
                print(f"Number of active notes: {np.sum(combined_roll > 0.5)}")
                
                # Save as MIDI
                output_path = os.path.join(output_dir, f'generated_melody_{i+1}.mid')
                generate_midi(combined_roll, output_path)
                print(f"Saved melody {i+1} to {output_path}")
        
        print("\nGeneration complete!")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()