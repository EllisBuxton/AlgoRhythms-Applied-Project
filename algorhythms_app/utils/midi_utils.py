import pretty_midi
import numpy as np
import os
from typing import List, Tuple

class MidiProcessor:
    def __init__(self, sequence_length: int = 32):
        self.sequence_length = sequence_length
        self.min_pitch = 21  # A0
        self.max_pitch = 108  # C8
        self.pitch_range = self.max_pitch - self.min_pitch + 1

    def load_midi_file(self, file_path: str) -> np.ndarray:
        """Load and process a MIDI file into a piano roll."""
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            piano_roll = midi_data.get_piano_roll(fs=4)  # 4 steps per beat
            # Transpose to shape (time, pitch)
            piano_roll = piano_roll.T
            return piano_roll
        except Exception as e:
            print(f"Error loading MIDI file {file_path}: {e}")
            return None

    def process_directory(self, directory_path: str) -> List[np.ndarray]:
        """Process all MIDI files in a directory."""
        sequences = []
        for file in os.listdir(directory_path):
            if file.endswith('.mid') or file.endswith('.midi'):
                file_path = os.path.join(directory_path, file)
                piano_roll = self.load_midi_file(file_path)
                if piano_roll is not None:
                    sequences.extend(self.create_sequences(piano_roll))
        return sequences

    def create_sequences(self, piano_roll: np.ndarray) -> List[np.ndarray]:
        """Create fixed-length sequences from piano roll."""
        sequences = []
        for i in range(0, len(piano_roll) - self.sequence_length, self.sequence_length // 2):
            sequence = piano_roll[i:i + self.sequence_length]
            if sequence.shape[0] == self.sequence_length:
                sequences.append(sequence)
        return sequences

    def normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize the sequence values between 0 and 1."""
        return np.clip(sequence / 100.0, 0, 1)  # Assuming velocity max is 100

    def denormalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """Convert normalized values back to MIDI velocity range."""
        return np.clip(sequence * 100.0, 0, 100).astype(np.int16)
