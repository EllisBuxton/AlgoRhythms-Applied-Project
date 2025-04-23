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

# Genetic Algorithm Parameters
NOTE_RANGE = note_range  # Use the note range from the trained model
POPULATION_SIZE = 10
MELODY_LENGTH = 8
MUTATION_RATE = 0.1

def initialize_population(population_size, melody_length):
    """Initialize a population of random melodies."""
    population = []
    for _ in range(population_size):
        melody = []
        for _ in range(melody_length):
            note = random.randint(NOTE_RANGE[0], NOTE_RANGE[1])
            duration = random.choice([0.25, 0.5, 1.0])  # Quarter, half, or whole note
            melody.append({
                'midiNote': note,
                'duration': duration,
                'velocity': random.randint(60, 100)
            })
        population.append(melody)
    return population

def fitness_function(melody, ratings):
    """Calculate fitness based on user ratings."""
    if not ratings:
        return 0.5  # Default fitness if no ratings
    return np.mean(ratings) / 5.0  # Normalize to 0-1 range

def crossover(parent1, parent2):
    """Perform crossover between two parent melodies."""
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(melody, mutation_rate):
    """Apply mutations to a melody."""
    mutated = melody.copy()
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Randomly change note, duration, or velocity
            mutation_type = random.choice(['note', 'duration', 'velocity'])
            if mutation_type == 'note':
                mutated[i]['midiNote'] = random.randint(NOTE_RANGE[0], NOTE_RANGE[1])
            elif mutation_type == 'duration':
                mutated[i]['duration'] = random.choice([0.25, 0.5, 1.0])
            else:
                mutated[i]['velocity'] = random.randint(60, 100)
    return mutated

def evolve_population(population, ratings, num_generations=5, mutation_rate=0.1):
    """Evolve the population over multiple generations."""
    for _ in range(num_generations):
        # Calculate fitness for each melody
        fitnesses = [fitness_function(melody, ratings.get(i, [])) for i, melody in enumerate(population)]
        
        # Select parents based on fitness
        parents = []
        for _ in range(len(population) // 2):
            parent1 = random.choices(population, weights=fitnesses)[0]
            parent2 = random.choices(population, weights=fitnesses)[0]
            parents.append((parent1, parent2))
        
        # Create new generation through crossover and mutation
        new_population = []
        for parent1, parent2 in parents:
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population
    
    return population

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

@app.route('/rate', methods=['POST'])
def rate_melody():
    data = request.json
    melody_index = data['melodyIndex']
    rating = int(data['rating'])

    if melody_index in melody_ratings:
        melody_ratings[melody_index].append(rating)
    else:
        melody_ratings[melody_index] = [rating]

    return jsonify({'message': f'Melody {melody_index + 1} rated {rating}'})

# Initialize population on app start
population = initialize_population(POPULATION_SIZE, MELODY_LENGTH)

@app.route('/evolve', methods=['GET'])
def evolve_melodies():
    global population
    population = evolve_population(population, melody_ratings, num_generations=5, mutation_rate=MUTATION_RATE)
    return jsonify({'melodies': population})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)