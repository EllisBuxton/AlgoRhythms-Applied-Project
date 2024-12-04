<template>
  <div v-if="show" class="piano-popup-overlay" @click="close">
    <div class="piano-popup" @click.stop>
      <button class="close-button" @click="close">×</button>
      <h2>Piano Roll</h2>
      <div class="piano-content">
        <div class="piano-roll-container">
          <div class="piano-keys" ref="pianoKeys">
            <div 
              v-for="note in notes" 
              :key="note.midi"
              class="piano-key"
              :class="{ 
                'black-key': note.isBlack,
                'white-key': !note.isBlack
              }"
            >
              <span class="note-label">{{ note.label }}</span>
            </div>
          </div>
          <div class="grid-container">
            <div class="grid">
              <!-- Grid lines will be added via CSS -->
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
    }
  },
  data() {
    return {
      notes: []
    }
  },
  created() {
    // Generate notes from C0 to C10 (11 octaves)
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const notes = [];
    
    // Start from MIDI note 0 (C-1) to MIDI note 127 (G9)
    for (let midi = 0; midi <= 127; midi++) {
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
    
    // Reverse the array so higher notes are at the top
    this.notes = notes.reverse();
  },
  watch: {
    show(newValue) {
      if (newValue) {
        // Wait for the next tick to ensure the DOM is updated and visible
        this.$nextTick(() => {
          if (this.$refs.pianoKeys) {
            // Find the index of C5 (MIDI note 72)
            const c5Index = this.notes.findIndex(note => note.midi === 72);
            if (c5Index !== -1) {
              // Calculate scroll position (30px per key)
              const scrollPosition = c5Index * 30;
              this.$refs.pianoKeys.scrollTop = scrollPosition;
            }
          }
        });
      }
    }
  },
  methods: {
    close() {
      this.$emit('close');
    }
  }
}
</script> 