import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pretty_midi
import pandas as pd
from typing import List, Dict, Tuple
import time
import torch
from models.gru_model import GRUModel
from utils.pop909_processor import POP909Processor
from models.generate_with_gru import extract_real_seed_from_dataset, generate_melody, generate_midi_file

def create_piano_roll(midi_file: str, save_path: str = None):
    """
    Create a piano roll visualization of a MIDI file
    
    Args:
        midi_file: Path to the MIDI file
        save_path: Path to save the visualization (optional)
    """
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    
    # Get piano roll
    piano_roll = midi_data.get_piano_roll()
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Velocity')
    plt.xlabel('Time (frames)')
    plt.ylabel('Pitch')
    plt.title('Piano Roll Visualization')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_generated_melody(midi_file: str) -> Dict:
    """
    Analyze a generated melody and return statistics
    
    Args:
        midi_file: Path to the MIDI file
        
    Returns:
        Dictionary containing melody statistics
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    melody_track = midi_data.instruments[0]
    
    stats = {
        'num_notes': len(melody_track.notes),
        'total_duration': sum(note.end - note.start for note in melody_track.notes),
        'pitch_counts': Counter(),
        'duration_counts': Counter(),
        'unique_pitch_classes': set()
    }
    
    for note in melody_track.notes:
        stats['pitch_counts'][note.pitch] += 1
        stats['duration_counts'][round(note.end - note.start, 2)] += 1
        stats['unique_pitch_classes'].add(note.pitch % 12)
    
    return stats

def plot_pitch_distribution(stats: Dict, training_stats: Dict = None, save_path: str = None):
    """
    Plot the distribution of pitches in generated melodies
    
    Args:
        stats: Dictionary containing generated melody statistics
        training_stats: Dictionary containing training data statistics (optional)
        save_path: Path to save the plot (optional)
    """
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    counts = [0] * 12
    
    # Count pitches for generated melody
    for pitch, count in stats['pitch_counts'].items():
        counts[pitch % 12] += count
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(pitch_classes))
    width = 0.35
    
    if training_stats:
        # Count pitches for training data
        train_counts = [0] * 12
        for pitch, count in training_stats['pitch_counts'].items():
            train_counts[pitch % 12] += count
        
        # Normalize counts to make them comparable
        train_total = sum(train_counts)
        gen_total = sum(counts)
        train_counts = [count/train_total for count in train_counts]
        counts = [count/gen_total for count in counts]
        
        plt.bar(x - width/2, train_counts, width, label='Training', alpha=0.7)
        plt.bar(x + width/2, counts, width, label='Generated', alpha=0.7)
    else:
        plt.bar(x, counts, width, label='Generated')
    
    plt.xlabel('Pitch Class')
    plt.ylabel('Normalized Count')
    plt.title('Pitch Class Distribution')
    plt.xticks(x, pitch_classes, rotation=45)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_duration_distribution(stats: Dict, training_stats: Dict = None, save_path: str = None):
    """
    Plot the distribution of note durations in generated melodies
    
    Args:
        stats: Dictionary containing generated melody statistics
        training_stats: Dictionary containing training data statistics (optional)
        save_path: Path to save the plot (optional)
    """
    # Get all unique durations from both generated and training data
    all_durations = set()
    all_durations.update(stats['duration_counts'].keys())
    if training_stats:
        all_durations.update(training_stats['duration_counts'].keys())
    all_durations = sorted(list(all_durations))
    
    # Create counts for both datasets
    gen_counts = [stats['duration_counts'].get(d, 0) for d in all_durations]
    train_counts = [training_stats['duration_counts'].get(d, 0) for d in all_durations] if training_stats else None
    
    plt.figure(figsize=(12, 6))
    
    if training_stats:
        # Normalize counts
        train_total = sum(train_counts)
        gen_total = sum(gen_counts)
        train_counts = [count/train_total for count in train_counts]
        gen_counts = [count/gen_total for count in gen_counts]
        
        plt.bar(all_durations, train_counts, width=0.1, label='Training', alpha=0.7)
        plt.bar(all_durations, gen_counts, width=0.1, label='Generated', alpha=0.7)
    else:
        plt.bar(all_durations, gen_counts, width=0.1, label='Generated')
    
    plt.xlabel('Note Duration (seconds)')
    plt.ylabel('Normalized Count')
    plt.title('Note Duration Distribution')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_ngram_repetition(melody_notes: List[int], n: int = 2) -> Dict:
    """
    Analyze n-gram repetition in a generated melody
    
    Args:
        melody_notes: List of note pitches
        n: Size of n-grams to analyze
        
    Returns:
        Dictionary containing n-gram statistics
    """
    ngrams = Counter()
    for i in range(len(melody_notes) - n + 1):
        ngram = tuple(melody_notes[i:i+n])
        ngrams[ngram] += 1
    
    unique_ngrams = len(ngrams)
    repeated_ngrams = sum(1 for count in ngrams.values() if count > 1)
    
    return {
        'unique_ngrams': unique_ngrams,
        'repeated_ngrams': repeated_ngrams,
        'repetition_ratio': repeated_ngrams / unique_ngrams if unique_ngrams > 0 else 0
    }

def plot_ngram_repetition(stats: Dict, save_path: str = None):
    """
    Plot n-gram repetition statistics
    
    Args:
        stats: Dictionary containing n-gram statistics
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.bar(['Unique', 'Repeated'], 
            [stats['unique_ngrams'], stats['repeated_ngrams']])
    plt.ylabel('Count')
    plt.title('N-gram Repetition Analysis')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_user_evaluation(results: Dict, save_path: str = None):
    """
    Plot user evaluation results
    
    Args:
        results: Dictionary containing evaluation scores
        save_path: Path to save the plot (optional)
    """
    metrics = list(results.keys())
    scores = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, scores)
    plt.ylim(0, 10)  # Assuming scores are out of 10
    plt.ylabel('Score')
    plt.title('User Evaluation Results')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_training_data(processor: POP909Processor) -> Dict:
    """
    Analyze the training data and return statistics
    
    Args:
        processor: POP909Processor instance
        
    Returns:
        Dictionary containing training data statistics
    """
    stats = {
        'num_files': len(processor.midi_files),
        'total_notes': 0,
        'total_duration': 0,
        'pitch_counts': Counter(),
        'duration_counts': Counter(),
        'unique_pitch_classes': set()
    }
    
    for midi_file in processor.midi_files:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            if len(midi_data.instruments) > 0:
                melody_track = midi_data.instruments[0]
                for note in melody_track.notes:
                    stats['total_notes'] += 1
                    stats['total_duration'] += note.end - note.start
                    stats['pitch_counts'][note.pitch] += 1
                    stats['duration_counts'][round(note.end - note.start, 2)] += 1
                    stats['unique_pitch_classes'].add(note.pitch % 12)
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
    
    return stats

def generate_and_analyze_melody(model: GRUModel, processor: POP909Processor, 
                              output_dir: str = 'generation_results'):
    """
    Generate a melody and perform all analyses
    
    Args:
        model: Trained GRU model
        processor: POP909Processor instance
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate melody
    start_time = time.time()
    generated_midi = os.path.join(output_dir, 'generated_melody.mid')
    
    # Get model parameters from checkpoint
    checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       'saved_models', 'gru_pop909.pth'),
                          weights_only=False)
    note_range = checkpoint['note_range']
    chord_vocab_size = checkpoint['chord_vocab_size']
    
    # Generate seed sequence
    seed_sequence, seed_notes = extract_real_seed_from_dataset(
        processor.dataset_path,
        note_range,
        chord_vocab_size
    )
    
    # Move seed sequence to the same device as the model
    seed_sequence = seed_sequence.to(model.device)
    
    # Generate melody
    generated_sequence = generate_melody(model, seed_sequence, length=64, temperature=0.8)
    
    # Convert to MIDI file
    generate_midi_file(
        generated_sequence,
        output_file=generated_midi,
        instrument=0,  # Piano
        velocity=80,
        duration=0.5,
        note_range=note_range
    )
    
    generation_time = time.time() - start_time
    
    # Create piano roll
    create_piano_roll(generated_midi, os.path.join(output_dir, 'piano_roll.png'))
    
    # Analyze generated melody
    stats = analyze_generated_melody(generated_midi)
    print("\nGenerated melody statistics:")
    print(f"Number of notes: {stats['num_notes']}")
    print(f"Pitch counts: {dict(stats['pitch_counts'])}")
    print(f"Duration counts: {dict(stats['duration_counts'])}")
    
    # Create statistics table
    stats_table = pd.DataFrame({
        'Metric': [
            'Average notes per melody',
            'Average pitch',
            'Average note duration',
            'Unique pitch classes',
            'Generation time'
        ],
        'Value': [
            stats['num_notes'],
            sum(pitch * count for pitch, count in stats['pitch_counts'].items()) / stats['num_notes'],
            stats['total_duration'] / stats['num_notes'],
            len(stats['unique_pitch_classes']),
            generation_time
        ]
    })
    stats_table.to_csv(os.path.join(output_dir, 'generation_stats.csv'), index=False)
    
    # Load training statistics for comparison
    training_stats = analyze_training_data(processor)
    print("\nTraining data statistics:")
    print(f"Number of files: {training_stats['num_files']}")
    print(f"Total notes: {training_stats['total_notes']}")
    print(f"Pitch counts: {dict(training_stats['pitch_counts'])}")
    print(f"Duration counts: {dict(training_stats['duration_counts'])}")
    
    # Create comparison plots
    plot_pitch_distribution(stats, training_stats, 
                          os.path.join(output_dir, 'pitch_distribution.png'))
    plot_duration_distribution(stats, training_stats, 
                             os.path.join(output_dir, 'duration_distribution.png'))
    
    # Analyze n-gram repetition
    melody_notes = [note.pitch for note in pretty_midi.PrettyMIDI(generated_midi).instruments[0].notes]
    ngram_stats = analyze_ngram_repetition(melody_notes)
    plot_ngram_repetition(ngram_stats, 
                         os.path.join(output_dir, 'ngram_repetition.png'))
    
    print(f"All generation analyses have been saved to {output_dir}")

if __name__ == '__main__':
    # Initialize processor and load model
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'POP909')
    processor = POP909Processor(dataset_path)
    processor.scan_dataset()
    
    # Load trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'saved_models', 'gru_pop909.pth')
    model = GRUModel(input_size=151, hidden_size=512, output_size=57)
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to_device()
    
    # Generate and analyze melody
    generate_and_analyze_melody(model, processor) 