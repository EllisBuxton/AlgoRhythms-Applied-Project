<template>
  <div v-if="show" class="piano-popup-overlay" @click="close">
    <div 
      class="piano-popup" 
      @click.stop
      @dragenter.prevent="handleDragOver"
      @dragover.prevent="handleDragOver"
      @dragleave.prevent="handleDragLeave"
      @drop.prevent="handleFileDrop"
    >
      <button class="close-button" @click="close">×</button>
      <h2>Piano Roll</h2>
      <div class="controls">
        <select 
          class="instrument-select"
          v-model="selectedInstrument"
          @change="handleInstrumentChange"
        >
          <option value="piano">Piano</option>
          <option value="sawtooth">Sawtooth</option>
          <option value="brass">Brass</option>
          <option value="strings">Strings</option>
          <option value="drums">Drums</option>
        </select>
        <button 
          class="play-button" 
          @click="playSequence"
          :disabled="isPlaying"
        >
          {{ isPlaying ? '▶️ Playing...' : '▶️ Play Sequence' }}
        </button>
        <button 
          class="clear-button" 
          @click="clearAllNotes"
          :disabled="placedNotes.size === 0"
        >
          🗑️ Clear All
        </button>
        <button 
          class="save-button" 
          @click="saveMelody"
          :disabled="placedNotes.size === 0"
        >
          💾 Save Melody
        </button>
      </div>
      <div class="piano-content">
        <div 
          class="piano-roll-container"
          :class="{ 'drag-over': isDraggingOver }"
        >
          <div class="piano-roll-scroll-container" ref="scrollContainer">
            <div class="piano-keys" ref="pianoKeys">
              <div 
                v-for="note in notes" 
                :key="note.midi"
                class="piano-key"
                :class="{ 
                  'black-key': note.isBlack,
                  'white-key': !note.isBlack
                }"
                @mousedown="playNote(note)"
              >
                <span class="note-label">{{ note.label }}</span>
              </div>
            </div>
            <div class="grid-container" ref="gridContainer" @scroll="handleGridScroll">
              <div class="note-rows">
                <div 
                  v-for="note in notes" 
                  :key="note.midi"
                  class="note-row"
                >
                  <div 
                    v-for="cell in 256" 
                    :key="cell"
                    class="grid-cell"
                    :class="{ 'current-beat': currentPlaybackCell === cell }"
                    @click="toggleNote(note.midi, cell)"
                    @contextmenu.prevent="removeNote(note.midi, cell)"
                  >
                    <div 
                      v-if="hasNote(note.midi, cell)" 
                      class="note-block"
                      :class="{ 'playing': isNotePlaying(note.midi, cell) }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import '../style/PianoRollPopup.css'
import * as Tone from 'tone'
import { Midi } from '@tonejs/midi'

export default {
  name: 'PianoRollPopup',
  props: {
    show: {
      type: Boolean,
      required: true
    },
    onNotePlay: {
      type: Function,
      required: true
    },
    bpm: {
      type: Number,
      default: 120
    }
  },
  data() {
    return {
      notes: [],
      placedNotes: new Map(),
      isPlaying: false,
      currentPlaybackTime: 0,
      currentlyPlayingNotes: new Set(),
      currentPlaybackCell: null,
      selectedInstrument: 'piano',
      isDraggingOver: false,
      polySynth: null
    }
  },
  created() {
    this.polySynth = new Tone.PolySynth(Tone.Synth).toDestination();
    
    // Generate notes based on selected instrument
    this.updateNotes();
  },
  watch: {
    show(newValue) {
      if (newValue) {
        this.$nextTick(() => {
          if (this.$refs.scrollContainer) {
            const c5Index = this.notes.findIndex(note => note.midi === 72);
            if (c5Index !== -1) {
              const scrollPosition = c5Index * 30;
              this.$refs.scrollContainer.scrollTop = scrollPosition;
            }
          }
        });
      }
    },
    selectedInstrument(newValue) {
      this.updateNotes();
      this.handleInstrumentChange();
      // Clear notes when switching instruments
      this.placedNotes.clear();
      this.placedNotes = new Map(this.placedNotes);
      // Reset scroll position when switching to drums
      if (newValue === 'drums') {
        this.$nextTick(() => {
          if (this.$refs.scrollContainer) {
            this.$refs.scrollContainer.scrollTop = 0;
          }
        });
      }
    }
  },
  mounted() {
    this.$nextTick(() => {
      if (this.$refs.gridContainer) {
        this.$refs.gridContainer.addEventListener('scroll', this.handleGridScroll);
      }
    });
  },
  beforeUnmount() {
    if (this.$refs.gridContainer) {
      this.$refs.gridContainer.removeEventListener('scroll', this.handleGridScroll);
    }
  },
  methods: {
    close() {
      this.$emit('close');
    },
    handleInstrumentChange() {
      // Emit the instrument change to parent
      this.$emit('instrument-changed', this.selectedInstrument);
    },
    async playNote(note) {
      this.onNotePlay(note.midi, this.selectedInstrument);
    },
    toggleNote(midiNote, cellIndex) {
      const noteKey = `${midiNote}-${cellIndex}`;
      if (this.placedNotes.has(noteKey)) {
        return; // Don't toggle off on left click anymore
      }
      this.placedNotes.set(noteKey, true);
      this.onNotePlay(midiNote, this.selectedInstrument);
      this.placedNotes = new Map(this.placedNotes);
    },
    removeNote(midiNote, cellIndex) {
      const noteKey = `${midiNote}-${cellIndex}`;
      if (this.placedNotes.has(noteKey)) {
        this.placedNotes.delete(noteKey);
        this.placedNotes = new Map(this.placedNotes);
      }
    },
    hasNote(midiNote, cellIndex) {
      return this.placedNotes.has(`${midiNote}-${cellIndex}`);
    },
    async playSequence() {
      if (this.isPlaying) return;
      
      this.isPlaying = true;
      const beatDuration = 60000 / this.bpm;
      const gridCellDuration = beatDuration / 4;
      
      // Reset playback position
      this.currentPlaybackCell = 0;
      
      try {
        // Ensure audio context is started before sequence
        if (Tone.context.state !== 'running') {
          await Tone.start();
        }
        
        // Get max cell index to know how long to play
        const maxCell = Math.max(...Array.from(this.placedNotes.keys())
          .map(key => parseInt(key.split('-')[1])), 16);  // minimum 16 cells
        
        // Play through all cells, even empty ones
        for (let cell = 0; cell <= maxCell; cell++) {
          this.currentPlaybackCell = cell;
          
          // Find and play notes in current cell
          const currentNotes = Array.from(this.placedNotes.keys())
            .filter(key => parseInt(key.split('-')[1]) === cell)
            .map(key => parseInt(key.split('-')[0]));
          
          // Play all notes in the cell at once
          if (currentNotes.length > 0) {
            currentNotes.forEach(midiNote => {
              const noteKey = `${midiNote}-${cell}`;
              this.currentlyPlayingNotes.add(noteKey);
              this.onNotePlay(midiNote, this.selectedInstrument);
              
              setTimeout(() => {
                this.currentlyPlayingNotes.delete(noteKey);
              }, 200);
            });
          }
          
          await new Promise(resolve => setTimeout(resolve, gridCellDuration));
        }
      } catch (error) {
        console.error('Error playing sequence:', error);
      } finally {
        // Reset at end
        await new Promise(resolve => setTimeout(resolve, 500));
        this.isPlaying = false;
        this.currentlyPlayingNotes.clear();
        this.currentPlaybackCell = null;
      }
    },
    isNotePlaying(midiNote, cellIndex) {
      return this.currentlyPlayingNotes.has(`${midiNote}-${cellIndex}`);
    },
    clearAllNotes() {
      this.placedNotes.clear();
      this.placedNotes = new Map(this.placedNotes);
    },
    handleDragLeave(event) {
      event.preventDefault();
      this.isDraggingOver = false;
    },
    handleDragOver(event) {
      event.preventDefault();
      this.isDraggingOver = true;
    },
    async handleFileDrop(event) {
      event.preventDefault();
      event.stopPropagation();
      this.isDraggingOver = false;

      const file = event.dataTransfer.files[0];
      if (!file || !file.name.toLowerCase().endsWith('.mid')) {
        alert('Please drop a valid MIDI file');
        return;
      }

      try {
        const buffer = await file.arrayBuffer();
        const midi = new Midi(buffer);
        
        this.clearAllNotes();
        
        // Find the track with the most notes (usually the melody track)
        const mainTrack = midi.tracks.reduce((prev, current) => 
          current.notes.length > prev.notes.length ? current : prev
        );
        
        // Calculate the time per cell based on BPM
        const beatDuration = 60 / this.bpm; // Duration of one beat in seconds
        const sixteenthNoteDuration = beatDuration / 4; // Duration of one cell in seconds
        
        // Find the earliest and latest note times
        const noteTimes = mainTrack.notes.map(note => note.time);
        const earliestTime = Math.min(...noteTimes);
        const latestTime = Math.max(...noteTimes);
        
        // Process each note
        mainTrack.notes.forEach(note => {
          // Calculate relative time from the start of the sequence
          const relativeTime = note.time - earliestTime;
          
          // Convert time to cell index, ensuring we start from cell 0
          const cellIndex = Math.floor(relativeTime / sixteenthNoteDuration);
          
          // Only add notes within our 256-cell range
          if (cellIndex < 256) {
            // Add note directly to placedNotes without triggering sound
            const noteKey = `${note.midi}-${cellIndex}`;
            this.placedNotes.set(noteKey, true);
          }
        });
        
        // Update the placedNotes reference to trigger reactivity
        this.placedNotes = new Map(this.placedNotes);
        
        // If the melody is longer than 256 cells, show a message
        const totalCells = Math.ceil((latestTime - earliestTime) / sixteenthNoteDuration);
        if (totalCells > 256) {
          console.log(`Note: The MIDI file contains ${totalCells} cells. Only the first 256 cells will be shown.`);
        }
      } catch (error) {
        console.error('Error processing MIDI file:', error);
        alert('Unable to process this MIDI file');
      }
    },
    saveMelody() {
      // Convert placed notes to a serializable format
      const melody = {
        notes: Array.from(this.placedNotes.keys()).map(key => {
          const [midiNote, cell] = key.split('-').map(Number);
          return { midiNote, cell };
        }),
        instrument: this.selectedInstrument,
        bpm: this.bpm
      };
      
      // Emit save event to parent
      this.$emit('save-melody', melody);
    },
    handleGridScroll(event) {
      if (this.$refs.pianoKeys) {
        this.$refs.pianoKeys.scrollTop = event.target.scrollTop;
      }
    },
    updateNotes() {
      if (this.selectedInstrument === 'drums') {
        // Generate drum notes (General MIDI drum mapping)
        this.notes = [
          { midi: 35, label: 'Bass Drum 2', isBlack: false },
          { midi: 36, label: 'Bass Drum 1', isBlack: false },
          { midi: 37, label: 'Side Stick', isBlack: false },
          { midi: 38, label: 'Snare 1', isBlack: false },
          { midi: 39, label: 'Hand Clap', isBlack: false },
          { midi: 40, label: 'Snare 2', isBlack: false },
          { midi: 41, label: 'Low Tom 2', isBlack: false },
          { midi: 42, label: 'Closed Hi-hat', isBlack: false },
          { midi: 43, label: 'Low Tom 1', isBlack: false },
          { midi: 44, label: 'Pedal Hi-hat', isBlack: false },
          { midi: 45, label: 'Mid Tom 2', isBlack: false },
          { midi: 46, label: 'Open Hi-hat', isBlack: false },
          { midi: 47, label: 'Mid Tom 1', isBlack: false },
          { midi: 48, label: 'High Tom 2', isBlack: false },
          { midi: 49, label: 'Crash 1', isBlack: false },
          { midi: 50, label: 'High Tom 1', isBlack: false },
          { midi: 51, label: 'Ride 1', isBlack: false },
          { midi: 52, label: 'Chinese Cymbal', isBlack: false },
          { midi: 53, label: 'Ride Bell', isBlack: false },
          { midi: 54, label: 'Tambourine', isBlack: false },
          { midi: 55, label: 'Splash Cymbal', isBlack: false },
          { midi: 56, label: 'Cowbell', isBlack: false },
          { midi: 57, label: 'Crash 2', isBlack: false },
          { midi: 58, label: 'Vibraslap', isBlack: false },
          { midi: 59, label: 'Ride 2', isBlack: false }
        ].reverse(); // Reverse to show from top to bottom
      } else {
        // Generate regular piano notes
        const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const notes = [];
        
        // Start from MIDI note 12 (C1) to MIDI note 108 (B8)
        for (let midi = 12; midi <= 108; midi++) {
          const octave = Math.floor(midi / 12) - 1;
          const noteIndex = midi % 12;
          const noteName = noteNames[noteIndex];
          const isBlack = noteName.includes('#');
          
          notes.push({
            midi,
            label: `${noteName}${octave}`,
            isBlack
          });
        }
        
        this.notes = notes.reverse();
      }
    }
  }
}
</script> 

<style>
.controls {
  margin-bottom: 20px;
  display: flex;
  justify-content: center;
  gap: 10px;
}

.play-button {
  background-color: #894ab6;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.play-button:hover:not(:disabled) {
  background-color: #6b3a91;
  transform: scale(1.05);
}

.play-button:disabled {
  background-color: #666;
  cursor: not-allowed;
  opacity: 0.7;
}

.note-block {
  background-color: rgba(76, 175, 80, 0.9);
  border: 1px solid rgba(76, 175, 80, 1);
  border-radius: 4px;
  height: 90%;
  width: 90%;
  margin: auto;
  transition: all 0.2s ease;
  box-shadow: 
    inset 0 1px 1px rgba(255, 255, 255, 0.3),
    inset 0 -1px 1px rgba(0, 0, 0, 0.3),
    2px 2px 4px rgba(0, 0, 0, 0.2);
  position: relative;
  top: 5%;
}

.note-block:hover {
  background-color: rgba(76, 175, 80, 1);
  transform: scale(1.02) translateY(-1px);
  box-shadow: 
    inset 0 1px 1px rgba(255, 255, 255, 0.3),
    inset 0 -1px 1px rgba(0, 0, 0, 0.3),
    2px 4px 6px rgba(0, 0, 0, 0.3);
}

.note-block.playing {
  background-color: rgba(76, 175, 80, 1);
  transform: scale(1.05) translateY(-2px);
  box-shadow: 
    inset 0 1px 1px rgba(255, 255, 255, 0.4),
    inset 0 -1px 1px rgba(0, 0, 0, 0.3),
    0 0 15px rgba(76, 175, 80, 0.8),
    2px 6px 8px rgba(0, 0, 0, 0.4);
}

.grid-cell {
  height: 100%;
  width: 20px;
  border-right: 1px solid #2a2a2a;
  transition: background-color 0.2s ease;
  cursor: pointer;
  position: relative;
  flex-shrink: 0;
}

.grid-cell.current-beat {
  background-color: rgba(137, 74, 182, 0.15);
}

.instrument-select {
  background-color: #2a2a2a;
  color: white;
  border: 1px solid #894ab6;
  border-radius: 4px;
  padding: 8px 12px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
  outline: none;
}

.instrument-select:hover {
  border-color: #6b3a91;
  background-color: #3a3a3a;
}

.instrument-select:focus {
  border-color: #894ab6;
  box-shadow: 0 0 0 2px rgba(137, 74, 182, 0.3);
}

.instrument-select option {
  background-color: #2a2a2a;
  color: white;
}

.clear-button {
  background-color: #d32f2f;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.clear-button:hover:not(:disabled) {
  background-color: #b71c1c;
  transform: scale(1.05);
}

.clear-button:disabled {
  background-color: #666;
  cursor: not-allowed;
  opacity: 0.7;
}

.save-button {
  background-color: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.save-button:hover:not(:disabled) {
  background-color: #1976D2;
  transform: scale(1.05);
}

.save-button:disabled {
  background-color: #666;
  cursor: not-allowed;
  opacity: 0.7;
}

.piano-content {
  height: 70vh;
  overflow: hidden;
  width: 100%;
}

.piano-roll-container {
  position: relative;
  border: 2px dashed transparent;
  transition: all 0.3s ease;
  height: 100%;
  width: 100%;
}

.piano-roll-scroll-container {
  height: 100%;
  display: flex;
  overflow: auto;
  width: 100%;
}

.piano-keys {
  flex-shrink: 0;
  width: 60px;
}

.grid-container {
  flex-grow: 1;
  overflow: auto;
  width: calc(100% - 60px);
}

.note-row {
  display: flex;
  height: 30px;
  min-width: 5120px; /* 256 cells * 20px per cell */
}

.piano-roll-container.drag-over {
  border: 2px dashed #894ab6;
  background-color: rgba(137, 74, 182, 0.1);
  border-radius: 4px;
}
</style>