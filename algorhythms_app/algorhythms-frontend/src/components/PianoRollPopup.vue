<template>
  <div v-if="show" class="piano-popup-overlay" @click="close">
    <div class="piano-popup" @click.stop>
      <button class="close-button" @click="close">×</button>
      <h2>Piano Roll</h2>
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
                  <!-- Note placement will go here in future updates -->
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
    }
  },
  data() {
    return {
      notes: []
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
    }
  }
}
</script> 