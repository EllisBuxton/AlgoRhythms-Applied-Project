from utils.visualization_analysis import TrainingVisualizer
import numpy as np
import random

def create_mock_melody():
    """Create a mock melody for visualization purposes."""
    melody = []
    for _ in range(16):  # 16 notes per melody
        melody.append({
            'midiNote': random.randint(60, 72),  # Middle C to C5
            'duration': random.choice([0.25, 0.5, 1.0]),  # Quarter, half, or whole note
            'velocity': random.randint(60, 100)  # MIDI velocity
        })
    return melody

def demo_visualizations():
    # Initialize the visualizer
    visualizer = TrainingVisualizer(output_dir="dissertation_visualizations")
    
    # Example: Log some training metrics
    for epoch in range(10):
        # Example metrics
        visualizer.log_training_metric("loss", np.random.random(), epoch)
        visualizer.log_training_metric("accuracy", np.random.random(), epoch)
    
    # Generate mock melodies for analysis
    melodies = []
    for _ in range(20):
        melody = create_mock_melody()
        melodies.append(melody)
        
        # Log generation metrics
        visualizer.log_generation_metric("melody_length", len(melody), _)
        visualizer.log_generation_metric("average_note", np.mean([note['midiNote'] for note in melody]), _)
    
    # Example of evolution visualization
    for gen in range(5):
        # Log evolution metrics with mock data
        visualizer.log_generation_metric("population_fitness", 
                                       np.random.random(), 
                                       gen)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(melodies)
    
    print("Visualizations have been generated in the 'dissertation_visualizations' directory.")

if __name__ == "__main__":
    demo_visualizations() 