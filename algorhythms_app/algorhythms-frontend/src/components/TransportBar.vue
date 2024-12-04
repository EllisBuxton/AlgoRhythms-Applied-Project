<template>
  <div class="transport-bar">
    <button class="transport-button piano-button" @click="togglePianoPopup">🎹</button>
    <button class="transport-button test-note-button" @click="playTestNote">🎵</button>
    <div class="centered-controls">
      <div class="transport-controls">
        <button class="transport-button" @click="togglePlayPause">
          <span v-if="!isPlaying">▶</span>
          <span v-else>⏸</span>
        </button>
        <button class="transport-button" @click="stop">⏹</button>
      </div>
      
      <div class="time-display">
        {{ formatTime(currentTime) }}
      </div>
      
      <div class="tempo-control">
        <label for="bpm">BPM:</label>
        <input 
          type="range" 
          id="bpm" 
          v-model="bpm" 
          min="60" 
          max="200" 
          step="1"
        >
        <span class="bpm-value">{{ bpm }}</span>
      </div>
    </div>

    <piano-roll-popup 
      :show="showPianoPopup"
      @close="closePianoPopup"
    />
  </div>
</template>

<script>
import '../style/TransportBar.css'
import PianoRollPopup from './PianoRollPopup.vue'
import * as Tone from 'tone'

export default {
  name: 'TransportBar',
  components: {
    PianoRollPopup
  },
  props: {
    currentTime: {
      type: Number,
      default: 0
    }
  },
  data() {
    return {
      isPlaying: false,
      bpm: 120,
      timer: null,
      lastTimestamp: null,
      showPianoPopup: false
    }
  },
  methods: {
    togglePlayPause() {
      this.isPlaying = !this.isPlaying;
      if (this.isPlaying) {
        this.startTimer();
      } else {
        this.pauseTimer();
      }
      this.$emit('playback-changed', this.isPlaying);
    },
    stop() {
      this.isPlaying = false;
      this.lastTimestamp = null;
      this.$emit('time-updated', 0);
      this.pauseTimer();
      this.$emit('playback-changed', false);
    },
    startTimer() {
      this.pauseTimer(); // Clear any existing timer
      this.lastTimestamp = performance.now();
      
      const tick = () => {
        const now = performance.now();
        const delta = (now - this.lastTimestamp) / 1000; // Convert to seconds
        this.lastTimestamp = now;
        
        this.$emit('time-updated', this.currentTime + delta);
        this.timer = requestAnimationFrame(tick);
      };
      
      this.timer = requestAnimationFrame(tick);
    },
    pauseTimer() {
      if (this.timer) {
        cancelAnimationFrame(this.timer);
        this.timer = null;
      }
      this.lastTimestamp = null;
    },
    formatTime(time) {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60);
      const milliseconds = Math.floor((time % 1) * 1000);
      return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds.toString().padStart(3, '0')}`;
    },
    togglePianoPopup() {
      this.showPianoPopup = !this.showPianoPopup;
    },
    closePianoPopup() {
      this.showPianoPopup = false;
    },
    async playTestNote() {
      // Create and connect a synth
      const synth = new Tone.Synth().toDestination();
      
      // Play C5 (MIDI note 72)
      await Tone.start();
      synth.triggerAttackRelease("C5", "8n");
    }
  },
  watch: {
    bpm(newValue) {
      this.$emit('bpm-changed', newValue);
    }
  },
  beforeUnmount() {
    this.pauseTimer();
  }
}
</script> 