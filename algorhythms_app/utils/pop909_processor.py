import os
import numpy as np
import pretty_midi
import torch
from tqdm import tqdm
import re

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
        self.chord_files = []
        self.note_range = [0, 127]  # Full MIDI range
        self.chord_vocab = set()  # Set to store unique chords

    def scan_dataset(self):
        """Scan the dataset directory for MIDI and chord files"""
        print(f"Scanning POP909 dataset at {self.dataset_path}...")
        # Look for MIDI and chord files in each song directory
        for song_dir in os.listdir(self.dataset_path):
            song_path = os.path.join(self.dataset_path, song_dir)
            if os.path.isdir(song_path):
                # Look for the main MIDI file in the song directory
                midi_file = os.path.join(song_path, f"{song_dir}.mid")
                chord_file = os.path.join(song_path, "chord_midi.txt")
                
                if os.path.exists(midi_file):
                    self.midi_files.append(midi_file)
                if os.path.exists(chord_file):
                    self.chord_files.append(chord_file)
        
        print(f"Found {len(self.midi_files)} MIDI files and {len(self.chord_files)} chord files")
        return self.midi_files, self.chord_files

    def parse_chord(self, chord_str):
        """Parse chord string into root note and chord type"""
        if chord_str == 'N':
            return None
        
        # Extract root note and chord type
        match = re.match(r'([A-G][#b]?)(:?)([a-zA-Z]*)', chord_str)
        if match:
            root, _, chord_type = match.groups()
            return (root, chord_type)
        return None

    def extract_chords_from_file(self, chord_file):
        """Extract chord information from chord file"""
        chords = []
        with open(chord_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    chord_str = parts[2]
                    chord_info = self.parse_chord(chord_str)
                    if chord_info:
                        self.chord_vocab.add(chord_info)
                        chords.append({
                            'start': start_time,
                            'end': end_time,
                            'chord': chord_info
                        })
        return chords

    def extract_melody_from_midi(self, midi_file):
        """Extract melody notes from MIDI file with chord information"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            if len(midi_data.instruments) > 0:
                melody_track = midi_data.instruments[0]
                notes = sorted(melody_track.notes, key=lambda x: x.start)
                
                # Find corresponding chord file
                song_dir = os.path.dirname(midi_file)
                chord_file = os.path.join(song_dir, "chord_midi.txt")
                chords = []
                if os.path.exists(chord_file):
                    chords = self.extract_chords_from_file(chord_file)
                
                # Combine notes with chord information
                processed_notes = []
                for note in notes:
                    # Find current chord
                    current_chord = None
                    for chord in chords:
                        if chord['start'] <= note.start < chord['end']:
                            current_chord = chord['chord']
                            break
                    
                    processed_notes.append({
                        'pitch': note.pitch,
                        'velocity': note.velocity,
                        'duration': note.end - note.start,
                        'chord': current_chord
                    })
                
                return processed_notes
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
        return []

    def create_training_sequences(self):
        """Create training sequences from all MIDI files with chord information"""
        X = []  # Input sequences
        y = []  # Target notes
        chord_context = []  # Chord context for each sequence
        
        if not self.midi_files:
            self.scan_dataset()
        
        print("Processing MIDI files into training sequences...")
        for midi_file in tqdm(self.midi_files):
            melody_notes = self.extract_melody_from_midi(midi_file)
            
            if len(melody_notes) > self.sequence_length + 1:
                # Extract features for each note
                pitches = [note['pitch'] for note in melody_notes]
                velocities = [note['velocity'] for note in melody_notes]
                durations = [note['duration'] for note in melody_notes]
                chords = [note['chord'] for note in melody_notes]
                
                # Create sequences
                for i in range(len(pitches) - self.sequence_length):
                    sequence = pitches[i:i + self.sequence_length]
                    target = pitches[i + self.sequence_length]
                    
                    # Get chord context
                    current_chord = chords[i + self.sequence_length - 1]
                    
                    X.append(sequence)
                    y.append(target)
                    chord_context.append(current_chord)
        
        if not X:
            raise ValueError("No valid training sequences extracted from dataset")
        
        print(f"Created {len(X)} training sequences")
        print(f"Unique chords in dataset: {len(self.chord_vocab)}")
        return np.array(X), np.array(y), chord_context

    def prepare_data_for_training(self):
        """Prepare data for training: one-hot encode and create tensors"""
        # Get raw sequences
        X_raw, y_raw, chord_context = self.create_training_sequences()
        
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
        
        # Convert chord context to one-hot encoding
        chord_vocab_list = sorted(list(self.chord_vocab))
        chord_onehot = np.zeros((len(chord_context), len(chord_vocab_list)))
        for i, chord in enumerate(chord_context):
            if chord in chord_vocab_list:
                chord_onehot[i, chord_vocab_list.index(chord)] = 1
        
        # Reshape chord information to match sequence length
        chord_onehot = np.repeat(chord_onehot[:, np.newaxis, :], self.sequence_length, axis=1)
        
        # Combine note and chord information
        X_combined = np.concatenate([X_onehot, chord_onehot], axis=2)
        
        return torch.FloatTensor(X_combined), torch.LongTensor(y_shifted), self.note_range 