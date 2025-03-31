import os
import numpy as np
import pretty_midi
import torch
from tqdm import tqdm

class POP909Processor:
    def __init__(self, dataset_path, sequence_length=16):
        """
        Process the POP909 dataset for training GRU models
        
        Args:
            dataset_path: Path to the POP909 dataset
            sequence_length: Length of sequences to generate for training
        """
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.midi_files = []
        self.note_range = [0, 127]  # Full MIDI range

    def scan_dataset(self):
        """Scan the dataset directory for MIDI files"""
        print(f"Scanning POP909 dataset at {self.dataset_path}...")
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.mid'):
                    self.midi_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.midi_files)} MIDI files")
        return self.midi_files

    def extract_melody_from_midi(self, midi_file):
        """Extract melody notes from a MIDI file"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            # Find melody track (usually the highest part)
            # In POP909, melody is usually in the first track
            if len(midi_data.instruments) > 0:
                melody_track = midi_data.instruments[0]
                
                # Extract notes
                notes = []
                for note in melody_track.notes:
                    notes.append({
                        'pitch': note.pitch,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity
                    })
                
                # Sort by start time
                notes = sorted(notes, key=lambda x: x['start'])
                
                return notes
            else:
                return []
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            return []

    def create_training_sequences(self):
        """Create training sequences from all MIDI files"""
        X = []  # Input sequences
        y = []  # Target notes
        
        if not self.midi_files:
            self.scan_dataset()
        
        print("Processing MIDI files into training sequences...")
        for midi_file in tqdm(self.midi_files):
            melody_notes = self.extract_melody_from_midi(midi_file)
            
            if len(melody_notes) > self.sequence_length + 1:
                # Extract pitches for one-hot encoding
                pitches = [note['pitch'] for note in melody_notes]
                
                # Create sequences of sequence_length notes
                for i in range(len(pitches) - self.sequence_length):
                    sequence = pitches[i:i + self.sequence_length]
                    target = pitches[i + self.sequence_length]
                    
                    X.append(sequence)
                    y.append(target)
        
        if not X:
            raise ValueError("No valid training sequences extracted from dataset")
        
        print(f"Created {len(X)} training sequences")
        return np.array(X), np.array(y)

    def prepare_data_for_training(self):
        """Prepare data for training: one-hot encode and create tensors"""
        # Get raw sequences
        X_raw, y_raw = self.create_training_sequences()
        
        # Determine the actual range used in the dataset
        min_note = min(np.min(X_raw), np.min(y_raw))
        max_note = max(np.max(X_raw), np.max(y_raw))
        self.note_range = [min_note, max_note]
        note_range_size = max_note - min_note + 1
        
        print(f"Note range in dataset: {min_note}-{max_note} ({note_range_size} values)")
        
        # One-hot encode the input sequences
        X_onehot = np.zeros((len(X_raw), self.sequence_length, note_range_size))
        for i, sequence in enumerate(X_raw):
            for j, note in enumerate(sequence):
                X_onehot[i, j, note - min_note] = 1
        
        # Convert target to shifted indices (relative to min_note)
        y_shifted = y_raw - min_note
        
        return torch.FloatTensor(X_onehot), torch.LongTensor(y_shifted), self.note_range 