<template>
  <div v-if="show" class="piano-popup-overlay" @click="close">
    <div class="piano-popup" @click.stop>
      <button class="close-button" @click="close">×</button>
      <h2>Piano Roll</h2>
      <div class="piano-content">
        <div class="piano-roll-container">
          <div class="piano-keys">
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
      notes: [
        { midi: 72, label: 'C5', isBlack: false },
        { midi: 71, label: 'B4', isBlack: false },
        { midi: 70, label: 'A#4', isBlack: true },
        { midi: 69, label: 'A4', isBlack: false },
        { midi: 68, label: 'G#4', isBlack: true },
        { midi: 67, label: 'G4', isBlack: false },
        { midi: 66, label: 'F#4', isBlack: true },
        { midi: 65, label: 'F4', isBlack: false },
        { midi: 64, label: 'E4', isBlack: false },
        { midi: 63, label: 'D#4', isBlack: true },
        { midi: 62, label: 'D4', isBlack: false },
        { midi: 61, label: 'C#4', isBlack: true },
        { midi: 60, label: 'C4', isBlack: false },
      ]
    }
  },
  methods: {
    close() {
      this.$emit('close');
    }
  }
}
</script> 