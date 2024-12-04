<template>
  <div v-if="show" class="piano-popup-overlay" @click="close">
    <div class="piano-popup" @click.stop>
      <button class="close-button" @click="close">×</button>
      <h2>Piano Roll</h2>
      <div class="controls">
        <button 
          class="play-button" 
          @click="playSequence"
          :disabled="isPlaying"
        >
          {{ isPlaying ? '▶️ Playing...' : '▶️ Play Sequence' }}
        </button>
      </div>
      <div class="piano-content">
        <div class="piano-roll-container">
          <div class="piano-roll-scroll-container" ref="scrollContainer">
            <div class="piano-keys">
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
            <div class="grid-container">
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
                    @click="toggleNote(note.midi, cell)"
                    @contextmenu.prevent="removeNote(note.midi, cell)"
                  >
                    <div 
                      v-if="hasNote(note.midi, cell)" 
                      class="note-block"
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
      currentPlaybackTime: 0
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
  methods: {
    close() {
      this.$emit('close');
    },
    playNote(note) {
      this.onNotePlay(note.midi);
    },
    toggleNote(midiNote, cellIndex) {
      const noteKey = `${midiNote}-${cellIndex}`;
      if (this.placedNotes.has(noteKey)) {
        return; // Don't toggle off on left click anymore
      }
      this.placedNotes.set(noteKey, true);
      this.onNotePlay(midiNote);
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
      
      // Get all placed notes and sort them by time (cell index)
      const sortedNotes = Array.from(this.placedNotes.keys())
        .map(key => {
          const [midi, cell] = key.split('-').map(Number);
          return { midi, time: cell * gridCellDuration };
        })
        .sort((a, b) => a.time - b.time);

      // Play each note at its scheduled time
      const startTime = performance.now();
      
      for (const note of sortedNotes) {
        const waitTime = note.time;
        await new Promise(resolve => setTimeout(resolve, waitTime - (performance.now() - startTime)));
        this.onNotePlay(note.midi);
      }

      // Wait a bit after the last note before enabling the play button again
      await new Promise(resolve => setTimeout(resolve, 500));
      this.isPlaying = false;
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
</style>