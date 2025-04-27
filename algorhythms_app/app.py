from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import random
import numpy as np
import torch
from models.gru_model import GRUModel
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')

# Try to load the model checkpoint
try:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint format
        input_size = checkpoint['input_size']
        hidden_size = checkpoint['hidden_size']
        output_size = checkpoint['output_size']
        note_range = checkpoint['note_range']
        chord_vocab_size = checkpoint['chord_vocab_size']
        model_state_dict = checkpoint['model_state_dict']
    else:
        # State dict only format - extract dimensions from the state dict
        model_state_dict = checkpoint
        # Extract dimensions from the state dict
        input_size = model_state_dict['input_bn.weight'].size(0)
        hidden_size = model_state_dict['gru_layers.0.weight_ih_l0'].size(0) // 3  # GRU has 3 gates
        output_size = model_state_dict['fc2.weight'].size(0)
        # Use default values for other parameters
        note_range = (60, 72)  # Middle C to C5
        chord_vocab_size = 12
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

print(f"Model parameters: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")

# Initialize the model with the correct architecture
model = GRUModel(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=3,
    dropout=0.2
)

# Load the state dict
model.load_state_dict(model_state_dict)

# Move to device and set to eval mode
model.to(device)
model.eval()

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
    
    # Pad or truncate to match model's input size
    if seed.shape[1] < input_size:
        # Pad with zeros
        pad_width = ((0, 0), (0, input_size - seed.shape[1]))
        seed = np.pad(seed, pad_width, mode='constant')
    elif seed.shape[1] > input_size:
        # Truncate
        seed = seed[:, :input_size]
    
    # Convert to actual MIDI notes for display
    midi_notes = [idx + note_range[0] for idx in note_indices]
    
    return torch.FloatTensor(seed), midi_notes

def generate_melody_with_model(seed_sequence, length=64, temperature=0.8):
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
            
        # Get dimensions
        input_size = current_sequence.size(2)
        
        # Generate melody
        generated_melody = []
        current_cell = 0
        max_cells = 256  # Maximum number of cells in piano roll
        spacing = 4  # Fixed spacing between notes (4 cells = quarter note at 16th note resolution)
        
        # Keep generating until we either hit the max cells or reach the desired length
        while current_cell < max_cells and len(generated_melody) < length:
            # Forward pass
            output, _ = model(current_sequence)
            
            # Apply temperature
            output = output / temperature
            
            # Sample from distribution
            probs = torch.exp(output).squeeze()
            note_idx = torch.multinomial(probs, 1).item()
            
            # Add to melody with fixed spacing
            if note_idx > 0:  # Only add non-rest notes
                current_cell += spacing
                
                if current_cell < max_cells:
                    generated_melody.append({
                        'note': note_idx,
                        'cell': current_cell
                    })
                else:
                    break  # Stop if we've reached the end of the piano roll
            
            # Update sequence by removing first timestep and adding new note
            new_note = torch.zeros(1, 1, input_size, device=device)
            new_note[0, 0, note_idx] = 1
            current_sequence = torch.cat([current_sequence[:, 1:, :], new_note], dim=1)
            
    return generated_melody

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Instruments route
@app.route('/instruments', methods=['GET', 'POST'])
def select_instruments():
    if request.method == 'POST':
        selected_instruments = request.form.getlist('instruments')
        return jsonify({'instruments': selected_instruments})
    return render_template('instruments.html')

# Route to generate melodies
@app.route('/generate-melody', methods=['POST'])
def generate_melody():
    try:
        data = request.get_json()
        bpm = data.get('bpm', 120)
        
        # Generate a random seed sequence
        seed_sequence, seed_notes = generate_random_seed(note_range, chord_vocab_size)
        
        # Generate melody using the trained model with increased length
        notes = generate_melody_with_model(seed_sequence, length=64, temperature=0.8)
        
        # Format for frontend
        formatted_melody = {
            'notes': [
                {
                    'midiNote': note['note'] + note_range[0],  # Convert back to MIDI range
                    'cell': note['cell']  # Use the calculated cell position
                }
                for note in notes
            ],
            'bpm': bpm,
            'instrument': 'piano'
        }
        
        return jsonify(formatted_melody)
        
    except Exception as e:
        print(f"Error generating melody: {str(e)}")
        return jsonify({'error': str(e)}), 500

# In-memory storage for ratings
melody_ratings = {}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)