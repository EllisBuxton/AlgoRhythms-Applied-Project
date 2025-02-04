import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple

def collate_music_sequences(batch):
    """Collate function for music sequences."""
    return torch.cat(batch, dim=0)

class MusicDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], sequence_length: int = 32):
        self.sequences = sequences
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Ensure sequence is 3D: (batch_size, sequence_length, features)
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)
        return torch.FloatTensor(sequence)

def create_data_loader(
    sequences: List[np.ndarray],
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    dataset = MusicDataset(sequences)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues during testing
        collate_fn=collate_music_sequences
    )
