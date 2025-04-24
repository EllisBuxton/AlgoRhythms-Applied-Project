import os
import sys
import torch
import numpy as np
import torch.serialization

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gru_model import GRUModel
from utils.pop909_processor import POP909Processor
from models.model_analysis import generate_all_visualizations

def main():
    # Initialize the data processor
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'POP909')
    processor = POP909Processor(dataset_path)
    processor.scan_dataset()
    
    # Load the trained model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'saved_models', 'gru_pop909.pth')
    model = GRUModel(input_size=151, hidden_size=512, output_size=57)  # Using the actual parameters from training
    
    # Load model state dict with weights_only=False for compatibility
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load training history
    train_losses = np.load('train_losses.npy')
    val_losses = np.load('val_losses.npy')
    
    # Generate all visualizations
    output_dir = 'analysis_results'
    generate_all_visualizations(processor, model, train_losses, val_losses, output_dir)

if __name__ == '__main__':
    main() 