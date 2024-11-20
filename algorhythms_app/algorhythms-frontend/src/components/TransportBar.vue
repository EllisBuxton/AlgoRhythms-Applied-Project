<template>
  <div class="transport-bar">
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
</template>

<script>
import '../style/TransportBar.css'

export default {
  name: 'TransportBar',
  data() {
    return {
      isPlaying: false,
      currentTime: 0,
      bpm: 120,
      timer: null
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
      this.currentTime = 0;
      this.pauseTimer();
      this.$emit('playback-changed', false);
    },
    startTimer() {
      this.timer = setInterval(() => {
        this.currentTime += 0.1;
      }, 100);
    },
    pauseTimer() {
      if (this.timer) {
        clearInterval(this.timer);
      }
    },
    formatTime(time) {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60);
      const milliseconds = Math.floor((time % 1) * 10);
      return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${milliseconds}`;
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