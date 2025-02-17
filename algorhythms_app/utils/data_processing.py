import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple

def collate_music_sequences(batch):
    """Collate function for music sequences."""
    return torch.cat(batch, dim=0)

class MusicDataset(Dataset):
    def __init__(self, sequences):
        """Initialize dataset with sequences."""
        if isinstance(sequences, dict):
            # If sequences is a dictionary, flatten it into a list
            self.sequences = []
            for instrument_sequences in sequences.values():
                self.sequences.extend(instrument_sequences)
        else:
            # If sequences is already a list
            self.sequences = sequences
            
    def __len__(self):
        """Return the number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Get a sequence by index."""
        if idx >= len(self.sequences):
            raise IndexError(f"Index {idx} out of bounds for dataset with {len(self.sequences)} items")
        return torch.FloatTensor(self.sequences[idx])

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
