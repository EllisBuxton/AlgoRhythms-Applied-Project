import pretty_midi
import numpy as np
import os
import re
from typing import List, Dict, Tuple
import gc
import torch
import pickle
import psutil
import time

class MidiProcessor:
    def __init__(self, sequence_length: int = 32):
        self.sequence_length = sequence_length
        self.min_pitch = 21  # A0
        self.max_pitch = 108  # C8
        self.pitch_range = self.max_pitch - self.min_pitch + 1

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing/replacing invalid characters."""
        # Replace invalid characters with underscore
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove any other non-ASCII characters
        sanitized = ''.join(char for char in sanitized if ord(char) < 128)
        return sanitized

    def load_midi_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load and process a MIDI file into separate instrument piano rolls."""
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            instrument_rolls = {}
            
            for idx, instrument in enumerate(midi_data.instruments):
                # Skip drum tracks
                if instrument.is_drum:
                    continue
                    
                # Create piano roll for this instrument
                piano_roll = instrument.get_piano_roll(fs=4)  # 4 steps per beat
                # Transpose to shape (time, pitch)
                piano_roll = piano_roll.T
                
                # Use program name if available, otherwise use index
                instrument_name = (
                    f"{pretty_midi.program_to_instrument_name(instrument.program)}"
                    f"_{idx}"
                )
                instrument_name = self.sanitize_filename(instrument_name)
                
                instrument_rolls[instrument_name] = piano_roll
                
            return instrument_rolls
            
        except Exception as e:
            print(f"Error loading MIDI file {file_path}: {e}")
            return None

    def process_directory(self, directory_path: str) -> Dict[str, List[np.ndarray]]:
        """Process all MIDI files in a directory and its subdirectories."""
        sequences_by_instrument = {}
        checkpoint_dir = 'algorhythms_app/models/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Find all MIDI files first
        midi_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.mid') or file.endswith('.midi'):
                    midi_files.append(os.path.join(root, file))
        
        total_files = len(midi_files)
        processed_files_count = 0
        
        # Try to load from last checkpoint
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]) if x.split('_')[1].split('.')[0].isdigit() else 0)
                
                for checkpoint in reversed(checkpoints):
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                    print(f"\nTrying to load checkpoint: {checkpoint}")
                    try:
                        with open(checkpoint_path, 'rb') as f:
                            checkpoint_data = pickle.load(f)
                            sequences_by_instrument = checkpoint_data['sequences']
                            processed_files_count = checkpoint_data['processed_files']
                            
                            # Verify data integrity
                            total_sequences = sum(len(seqs) for seqs in sequences_by_instrument.values())
                            print(f"Successfully loaded checkpoint with {processed_files_count} processed files")
                            print(f"Total sequences loaded: {total_sequences}")
                            break
                    except (EOFError, pickle.UnpicklingError, KeyError) as e:
                        print(f"Error loading checkpoint {checkpoint}: {e}")
                        print("Trying older checkpoint...")
                        continue
        
        print(f"\nProcessing {total_files} files from {directory_path}")
        print(f"Already processed: {processed_files_count}")
        print("-" * 50)
        
        # Continue from where we left off
        for file_path in midi_files[processed_files_count:]:
            processed_files_count += 1
            relative_path = os.path.relpath(file_path, directory_path)
            print(f"\nProcessing file {processed_files_count}/{total_files}: {relative_path}")
            
            try:
                instrument_rolls = self.load_midi_file(file_path)
                
                if instrument_rolls:
                    print(f"Found {len(instrument_rolls)} instruments:")
                    for instrument_name, piano_roll in instrument_rolls.items():
                        if instrument_name not in sequences_by_instrument:
                            sequences_by_instrument[instrument_name] = []
                        
                        piano_roll = piano_roll.astype(np.float32)
                        sequences = self.create_sequences(piano_roll)
                        
                        prev_count = len(sequences_by_instrument[instrument_name])
                        sequences_by_instrument[instrument_name].extend(sequences)
                        new_count = len(sequences_by_instrument[instrument_name])
                        print(f"  - {instrument_name}: +{new_count - prev_count} sequences")
                        
                        del sequences
                    
                    del instrument_rolls
                    del piano_roll
                    
                    # Save checkpoint every 1000 files
                    if processed_files_count % 1000 == 0:
                        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{processed_files_count}.pkl')
                        temp_checkpoint_path = checkpoint_path + '.tmp'
                        
                        print(f"\n💾 Saving checkpoint at {processed_files_count} files...")
                        print("Current sequence counts:")
                        total_sequences = 0
                        for instrument, sequences in sequences_by_instrument.items():
                            count = len(sequences)
                            total_sequences += count
                            print(f"  - {instrument}: {count} sequences")
                        print(f"Total sequences: {total_sequences}")
                        
                        # Save to temporary file first
                        with open(temp_checkpoint_path, 'wb') as f:
                            pickle.dump({
                                'sequences': sequences_by_instrument,
                                'processed_files': processed_files_count
                            }, f)
                        
                        # If save was successful, rename to final filename
                        os.replace(temp_checkpoint_path, checkpoint_path)
                        print(f"Saved: {checkpoint_path}")
                        
                        # Basic cleanup after saving
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"  ⚠️ Error processing file {relative_path}: {e}")
                continue
            
            # Show memory usage every 100 files
            if processed_files_count % 100 == 0:
                memory_percent = psutil.virtual_memory().percent
                print(f"Memory usage: {memory_percent}%")
        
        # Save final checkpoint
        final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_final.pkl')
        temp_final_checkpoint_path = final_checkpoint_path + '.tmp'
        
        with open(temp_final_checkpoint_path, 'wb') as f:
            pickle.dump({
                'sequences': sequences_by_instrument,
                'processed_files': processed_files_count
            }, f)
        os.replace(temp_final_checkpoint_path, final_checkpoint_path)
        
        print("\nProcessing Complete!")
        print("-" * 50)
        print("Final sequence counts by instrument:")
        for instrument, sequences in sequences_by_instrument.items():
            print(f"  - {instrument}: {len(sequences)} sequences")
            
        return sequences_by_instrument

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
