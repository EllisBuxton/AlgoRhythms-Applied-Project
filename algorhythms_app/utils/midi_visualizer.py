import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import os
from typing import List, Optional
import seaborn as sns

class MidiVisualizer:
    def __init__(self):
        self.style_setup()
    
    def style_setup(self):
        """Setup plotting style."""
        plt.style.use('dark_background')
        sns.set_palette("husl")
    
    def plot_piano_roll(self, 
                       piano_roll: np.ndarray, 
                       title: str = "Piano Roll Visualization",
                       save_path: Optional[str] = None):
        """
        Visualize a piano roll matrix.
        
        Args:
            piano_roll: Shape (time_steps, pitch_range) numpy array
            title: Title for the plot
            save_path: If provided, save the plot to this path
        """
        plt.figure(figsize=(15, 8))
        plt.imshow(piano_roll.T, 
                  aspect='auto', 
                  origin='lower',
                  cmap='magma')
        
        plt.colorbar(label='Velocity')
        plt.xlabel('Time Steps')
        plt.ylabel('MIDI Pitch')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_pitch_distribution(self, 
                              sequences: List[np.ndarray],
                              title: str = "Pitch Distribution",
                              save_path: Optional[str] = None):
        """
        Plot the distribution of MIDI pitches in the dataset.
        """
        # Combine all sequences
        all_pitches = []
        for seq in sequences:
            active_pitches = np.where(seq > 0)
            all_pitches.extend(active_pitches[1])  # Get pitch values
        
        plt.figure(figsize=(15, 6))
        plt.hist(all_pitches, bins=88, color='purple', alpha=0.7)
        plt.xlabel('MIDI Pitch')
        plt.ylabel('Frequency')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_sequence_heatmap(self,
                            sequence: np.ndarray,
                            title: str = "Sequence Heatmap",
                            save_path: Optional[str] = None):
        """
        Create a heatmap visualization of a single sequence.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(sequence.T, 
                   cmap='magma',
                   xticklabels=5,
                   yticklabels=10)
        
        plt.xlabel('Time Steps')
        plt.ylabel('MIDI Pitch')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def plot_velocity_over_time(self,
                              sequence: np.ndarray,
                              title: str = "Velocity Over Time",
                              save_path: Optional[str] = None):
        """
        Plot the average velocity over time for a sequence.
        """
        avg_velocity = np.mean(sequence, axis=1)
        
        plt.figure(figsize=(15, 5))
        plt.plot(avg_velocity, color='cyan', alpha=0.7)
        plt.fill_between(range(len(avg_velocity)), 
                        avg_velocity,
                        alpha=0.3,
                        color='cyan')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Average Velocity')
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def create_visualization_report(midi_file_path: str, output_dir: str):
    """
    Create a complete visualization report for a MIDI file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    piano_roll = midi_data.get_piano_roll(fs=16)  # 16 samples per beat
    
    # Initialize visualizer
    visualizer = MidiVisualizer()
    
    # Create visualizations
    visualizer.plot_piano_roll(
        piano_roll.T,
        title=f"Piano Roll - {os.path.basename(midi_file_path)}",
        save_path=os.path.join(output_dir, "piano_roll.png")
    )
    
    visualizer.plot_sequence_heatmap(
        piano_roll.T,
        title=f"Heatmap - {os.path.basename(midi_file_path)}",
        save_path=os.path.join(output_dir, "heatmap.png")
    )
    
    visualizer.plot_velocity_over_time(
        piano_roll.T,
        title=f"Velocity Profile - {os.path.basename(midi_file_path)}",
        save_path=os.path.join(output_dir, "velocity.png")
    ) 