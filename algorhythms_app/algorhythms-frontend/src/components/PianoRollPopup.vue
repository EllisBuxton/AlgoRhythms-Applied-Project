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
                    v-for="cell in 128" 
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
      isDraggingOver: false
    }
  },
  created() {
    // Generate notes from C1 to C8 (8 octaves)
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
        
        midi.tracks.forEach(track => {
          track.notes.forEach(note => {
            const startTime = note.time * (this.bpm / 60) * 4;
            const midiNote = note.midi;
            const cellIndex = Math.floor(startTime);
            
            if (cellIndex < 128 && midiNote >= 21 && midiNote <= 108) {
              this.toggleNote(midiNote, cellIndex);
            }
          });
        });
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
  border-right: 1px solid #2a2a2a;
  transition: background-color 0.2s ease;
  cursor: pointer;
  position: relative;
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
}

.piano-roll-container {
  position: relative;
  border: 2px dashed transparent;
  transition: all 0.3s ease;
  height: 100%;
}

.piano-roll-scroll-container {
  height: 100%;
  display: flex;
  overflow: auto;
}

.piano-keys {
  flex-shrink: 0;
}

.grid-container {
  flex-grow: 1;
  overflow: auto;
}

.piano-roll-container.drag-over {
  border: 2px dashed #894ab6;
  background-color: rgba(137, 74, 182, 0.1);
  border-radius: 4px;
}
</style>