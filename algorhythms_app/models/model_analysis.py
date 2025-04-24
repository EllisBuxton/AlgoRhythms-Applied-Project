import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from tqdm import tqdm
import pretty_midi
from utils.pop909_processor import POP909Processor
from models.gru_model import GRUModel
import pandas as pd
from typing import List, Dict, Tuple
import networkx as nx
from matplotlib.patches import Rectangle

def plot_training_curves(train_losses: List[float], val_losses: List[float], save_path: str = None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_model_architecture_diagram(model: GRUModel, save_path: str = None):
    """
    Create a diagram of the GRU model architecture
    
    Args:
        model: The GRU model instance
        save_path: Path to save the diagram (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        'Input\n(Batch Norm)',
        'GRU Layer 1',
        'GRU Layer 2',
        'GRU Layer 3',
        'Attention',
        'Dense 1',
        'Dense 2',
        'Output\n(Softmax)'
    ]
    
    for node in nodes:
        G.add_node(node)
    
    # Add edges
    edges = [
        ('Input\n(Batch Norm)', 'GRU Layer 1'),
        ('GRU Layer 1', 'GRU Layer 2'),
        ('GRU Layer 2', 'GRU Layer 3'),
        ('GRU Layer 3', 'Attention'),
        ('Attention', 'Dense 1'),
        ('Dense 1', 'Dense 2'),
        ('Dense 2', 'Output\n(Softmax)')
    ]
    
    for edge in edges:
        G.add_edge(*edge)
    
    # Draw the graph
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold',
            arrows=True, arrowsize=20)
    
    plt.title('GRU Model Architecture')
    
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
    
    for midi_file in tqdm(processor.midi_files, desc='Analyzing MIDI files'):
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

def plot_pitch_class_distribution(stats: Dict, save_path: str = None):
    """
    Plot the distribution of pitch classes in the training data
    
    Args:
        stats: Dictionary containing training data statistics
        save_path: Path to save the plot (optional)
    """
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    counts = [0] * 12
    
    for pitch, count in stats['pitch_counts'].items():
        counts[pitch % 12] += count
    
    plt.figure(figsize=(12, 6))
    plt.bar(pitch_classes, counts)
    plt.xlabel('Pitch Class')
    plt.ylabel('Count')
    plt.title('Pitch Class Distribution in Training Data')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_note_duration_distribution(stats: Dict, save_path: str = None):
    """
    Plot the distribution of note durations in the training data
    
    Args:
        stats: Dictionary containing training data statistics
        save_path: Path to save the plot (optional)
    """
    durations = sorted(stats['duration_counts'].keys())
    counts = [stats['duration_counts'][d] for d in durations]
    
    plt.figure(figsize=(12, 6))
    plt.hist(durations, weights=counts, bins=50)
    plt.xlabel('Note Duration (seconds)')
    plt.ylabel('Count')
    plt.title('Note Duration Distribution in Training Data')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_ngram_variety(processor: POP909Processor, n: int = 2) -> Dict:
    """
    Analyze n-gram variety in the training sequences
    
    Args:
        processor: POP909Processor instance
        n: Size of n-grams to analyze (default: 2 for bigrams)
        
    Returns:
        Dictionary containing n-gram statistics
    """
    ngrams = Counter()
    sequences = processor.create_training_sequences()
    
    # We only want to analyze the note sequences (first array)
    note_sequences = sequences[0]  # This is the (294879, 16) array
    
    # Analyze each sequence
    for seq in note_sequences:
        # Convert to list of integers
        seq_list = [int(x) for x in seq]
        
        # Create n-grams
        for i in range(len(seq_list) - n + 1):
            ngram = tuple(seq_list[i:i+n])
            ngrams[ngram] += 1
    
    # Print some statistics
    print(f"\nN-gram Analysis:")
    print(f"Total n-grams: {sum(ngrams.values())}")
    print(f"Unique n-grams: {len(ngrams)}")
    print("\nTop 10 most common n-grams:")
    for ngram, count in ngrams.most_common(10):
        print(f"{ngram}: {count} times")
    
    return {
        'total_ngrams': sum(ngrams.values()),
        'unique_ngrams': len(ngrams),
        'ngram_counts': ngrams
    }

def plot_ngram_variety(ngram_stats: Dict, save_path: str = None):
    """
    Plot the distribution of n-gram frequencies
    
    Args:
        ngram_stats: Dictionary containing n-gram statistics
        save_path: Path to save the plot (optional)
    """
    # Get top 20 most common n-grams
    top_ngrams = ngram_stats['ngram_counts'].most_common(20)
    ngrams, counts = zip(*top_ngrams)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(ngrams)), counts)
    plt.xlabel('N-gram')
    plt.ylabel('Frequency')
    plt.title('Top 20 Most Common N-grams')
    plt.xticks(range(len(ngrams)), [str(ng) for ng in ngrams], rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_visualizations(processor: POP909Processor, model: GRUModel, 
                              train_losses: List[float], val_losses: List[float],
                              output_dir: str = 'analysis_results'):
    """
    Generate all visualizations and save them to the specified directory
    
    Args:
        processor: POP909Processor instance
        model: GRUModel instance
        train_losses: List of training losses
        val_losses: List of validation losses
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training vs Validation Loss Curve
    plot_training_curves(train_losses, val_losses, 
                        os.path.join(output_dir, 'training_curves.png'))
    
    # 2. Model Architecture Diagram
    create_model_architecture_diagram(model, 
                                    os.path.join(output_dir, 'model_architecture.png'))
    
    # 3. Training Data Statistics
    stats = analyze_training_data(processor)
    
    # Create statistics table
    stats_table = pd.DataFrame({
        'Metric': [
            'Number of files',
            'Total notes',
            'Average notes per file',
            'Average pitch',
            'Average note duration',
            'Unique pitch classes'
        ],
        'Value': [
            stats['num_files'],
            stats['total_notes'],
            stats['total_notes'] / stats['num_files'],
            sum(pitch * count for pitch, count in stats['pitch_counts'].items()) / stats['total_notes'],
            stats['total_duration'] / stats['total_notes'],
            len(stats['unique_pitch_classes'])
        ]
    })
    stats_table.to_csv(os.path.join(output_dir, 'training_stats.csv'), index=False)
    
    # 4. Pitch Class Distribution
    plot_pitch_class_distribution(stats, 
                                os.path.join(output_dir, 'pitch_class_distribution.png'))
    
    # 5. Note Duration Distribution
    plot_note_duration_distribution(stats, 
                                  os.path.join(output_dir, 'note_duration_distribution.png'))
    
    # 6. N-gram Variety
    ngram_stats = analyze_ngram_variety(processor)
    plot_ngram_variety(ngram_stats, 
                      os.path.join(output_dir, 'ngram_variety.png'))
    
    print(f"All visualizations and statistics have been saved to {output_dir}") 