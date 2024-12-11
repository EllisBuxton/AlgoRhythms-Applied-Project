import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from algorhythms_app.utils.midi_utils import MidiProcessor
from algorhythms_app.utils.midi_visualizer import MidiVisualizer, create_visualization_report

def test_visualizations():
    # Setup paths
    current_dir = os.path.dirname(os.path.dirname(__file__))
    midi_dir = os.path.join(current_dir, 'data', 'midi_files')
    output_dir = os.path.join(current_dir, 'data', 'visualizations')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor and visualizer
    processor = MidiProcessor()
    visualizer = MidiVisualizer()
    
    # Process MIDI files
    sequences = processor.process_directory(midi_dir)
    
    if sequences:
        print(f"Processing {len(sequences)} sequences...")
        
        # Create visualizations for the first sequence
        visualizer.plot_piano_roll(
            sequences[0],
            title="Sample Piano Roll",
            save_path=os.path.join(output_dir, "sample_piano_roll.png")
        )
        print("Created piano roll visualization")
        
        visualizer.plot_pitch_distribution(
            sequences,
            save_path=os.path.join(output_dir, "pitch_distribution.png")
        )
        print("Created pitch distribution visualization")
        
        visualizer.plot_sequence_heatmap(
            sequences[0],
            save_path=os.path.join(output_dir, "sequence_heatmap.png")
        )
        print("Created sequence heatmap")
        
        visualizer.plot_velocity_over_time(
            sequences[0],
            save_path=os.path.join(output_dir, "velocity_profile.png")
        )
        print("Created velocity profile")
        
        print(f"Visualizations saved to {output_dir}")
        
        # Create full reports for each MIDI file
        for midi_file in os.listdir(midi_dir):
            if midi_file.endswith(('.mid', '.midi')):
                midi_path = os.path.join(midi_dir, midi_file)
                report_dir = os.path.join(output_dir, f"report_{os.path.splitext(midi_file)[0]}")
                create_visualization_report(midi_path, report_dir)
                print(f"Report created for {midi_file}")
    else:
        print("No sequences found to visualize")

if __name__ == "__main__":
    test_visualizations() 