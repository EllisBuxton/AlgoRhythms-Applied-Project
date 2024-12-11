import os
import sys
import logging

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from algorhythms_app.utils.midi_utils import MidiProcessor
from algorhythms_app.utils.data_processing import create_data_loader

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_pipeline():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize processor
    processor = MidiProcessor()
    
    # Get path to MIDI files
    midi_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'midi_files')
    
    # Check if directory exists and has files
    if not os.path.exists(midi_dir):
        logger.error(f"Error: Directory {midi_dir} does not exist")
        return
        
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith(('.mid', '.midi'))]
    if not midi_files:
        logger.warning(f"No MIDI files found in {midi_dir}")
        return
        
    logger.info(f"Found {len(midi_files)} MIDI files: {midi_files}")
    
    # Process files
    sequences = processor.process_directory(midi_dir)
    logger.info(f"Created {len(sequences)} sequences")
    
    # Additional sequence information
    if sequences:
        logger.info(f"Sequence shape: {sequences[0].shape}")
        logger.info(f"Value range: [{min([seq.min() for seq in sequences]):.2f}, "
                   f"{max([seq.max() for seq in sequences]):.2f}]")
    
    # Create data loader
    batch_size = 32
    data_loader = create_data_loader(sequences, batch_size=batch_size)
    
    # Test data loader
    for i, batch in enumerate(data_loader):
        logger.info(f"Batch {i + 1} shape: {batch.shape}")
        logger.info(f"Batch value range: [{batch.min():.2f}, {batch.max():.2f}]")
        if i == 0:  # Just test the first batch
            break

if __name__ == "__main__":
    test_pipeline() 