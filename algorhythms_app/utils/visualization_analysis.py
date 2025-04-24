import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os
import json
from typing import List, Dict, Any
import torch
from collections import defaultdict

class TrainingVisualizer:
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.training_metrics = defaultdict(list)
        self.generation_metrics = defaultdict(list)
        
    def log_training_metric(self, metric_name: str, value: float, epoch: int):
        """Log a training metric for visualization."""
        self.training_metrics[metric_name].append((epoch, value))
        
    def log_generation_metric(self, metric_name: str, value: float, generation: int):
        """Log a generation metric for visualization."""
        self.generation_metrics[metric_name].append((generation, value))
        
    def plot_training_metrics(self):
        """Plot all training metrics over time."""
        plt.figure(figsize=(12, 8))
        for metric_name, values in self.training_metrics.items():
            epochs, metric_values = zip(*values)
            plt.plot(epochs, metric_values, label=metric_name, marker='o')
        
        plt.title('Training Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'))
        plt.close()
        
    def plot_generation_metrics(self):
        """Plot all generation metrics over time."""
        plt.figure(figsize=(12, 8))
        for metric_name, values in self.generation_metrics.items():
            generations, metric_values = zip(*values)
            plt.plot(generations, metric_values, label=metric_name, marker='o')
        
        plt.title('Generation Metrics Over Time')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'generation_metrics.png'))
        plt.close()
        
    def plot_melody_distribution(self, melodies: List[List[Dict[str, Any]]]):
        """Plot the distribution of notes in generated melodies."""
        all_notes = []
        for melody in melodies:
            for note in melody:
                all_notes.append(note['midiNote'])
                
        plt.figure(figsize=(12, 6))
        sns.histplot(all_notes, bins=50)
        plt.title('Distribution of Notes in Generated Melodies')
        plt.xlabel('MIDI Note Number')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, 'note_distribution.png'))
        plt.close()
        
    def plot_duration_distribution(self, melodies: List[List[Dict[str, Any]]]):
        """Plot the distribution of note durations."""
        all_durations = []
        for melody in melodies:
            for note in melody:
                all_durations.append(note['duration'])
                
        plt.figure(figsize=(12, 6))
        sns.histplot(all_durations, bins=20)
        plt.title('Distribution of Note Durations')
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, 'duration_distribution.png'))
        plt.close()
        
    def plot_velocity_distribution(self, melodies: List[List[Dict[str, Any]]]):
        """Plot the distribution of note velocities."""
        all_velocities = []
        for melody in melodies:
            for note in melody:
                all_velocities.append(note['velocity'])
                
        plt.figure(figsize=(12, 6))
        sns.histplot(all_velocities, bins=20)
        plt.title('Distribution of Note Velocities')
        plt.xlabel('Velocity')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.output_dir, 'velocity_distribution.png'))
        plt.close()
        
    def plot_melody_heatmap(self, melodies: List[List[Dict[str, Any]]], num_melodies: int = 10):
        """Create a heatmap of note patterns in melodies."""
        # Take the first num_melodies melodies
        sample_melodies = melodies[:num_melodies]
        
        # Create a matrix of notes
        max_length = max(len(melody) for melody in sample_melodies)
        note_matrix = np.zeros((num_melodies, max_length))
        
        for i, melody in enumerate(sample_melodies):
            for j, note in enumerate(melody):
                note_matrix[i, j] = note['midiNote']
                
        plt.figure(figsize=(15, 8))
        sns.heatmap(note_matrix, cmap='viridis')
        plt.title('Note Pattern Heatmap')
        plt.xlabel('Time Step')
        plt.ylabel('Melody Index')
        plt.savefig(os.path.join(self.output_dir, 'melody_heatmap.png'))
        plt.close()
        
    def save_metrics(self):
        """Save all metrics to a JSON file."""
        metrics = {
            'training_metrics': dict(self.training_metrics),
            'generation_metrics': dict(self.generation_metrics)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.output_dir, f'metrics_{timestamp}.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def generate_all_visualizations(self, melodies: List[List[Dict[str, Any]]]):
        """Generate all available visualizations."""
        self.plot_training_metrics()
        self.plot_generation_metrics()
        self.plot_melody_distribution(melodies)
        self.plot_duration_distribution(melodies)
        self.plot_velocity_distribution(melodies)
        self.plot_melody_heatmap(melodies)
        self.save_metrics() 