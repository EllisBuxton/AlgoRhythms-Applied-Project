import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import os
from typing import List, Tuple
import json

class NESMusicDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 256):
        self.sequence_length = sequence_length
        self.data = []
        self.load_nes_data(data_dir)
        
    def load_nes_data(self, data_dir: str):
        """Load and preprocess NES music data"""
        for file in os.listdir(data_dir):
            if file.endswith('.mid'):
                midi_path = os.path.join(data_dir, file)
                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                    piano_roll = midi_data.get_piano_roll(fs=16)  # 16 samples per beat
                    
                    # Normalize to [0, 1]
                    piano_roll = (piano_roll - piano_roll.min()) / (piano_roll.max() - piano_roll.min() + 1e-6)
                    
                    # Split into sequences
                    for i in range(0, piano_roll.shape[1] - self.sequence_length, self.sequence_length // 2):
                        sequence = piano_roll[:, i:i + self.sequence_length]
                        if sequence.shape[1] == self.sequence_length:
                            self.data.append(torch.FloatTensor(sequence.T))
                            
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_nes_data_loader(
    data_dir: str,
    batch_size: int = 32,
    sequence_length: int = 256,
    shuffle: bool = True
) -> DataLoader:
    dataset = NESMusicDataset(data_dir, sequence_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 