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
          v-model.number="bpm"
          min="60" 
          max="200" 
          step="1"
          @input="handleBpmChange"
        >
        <span class="bpm-value">{{ bpm }}</span>
      </div>
    </div>

    <piano-roll-popup 
      :show="showPianoPopup"
      :onNotePlay="playMidiNote"
      :bpm="bpm"
      @close="closePianoPopup"
    />
  </div>
</template>

<script>
import '../style/TransportBar.css'
import PianoRollPopup from './PianoRollPopup.vue'
import * as Tone from 'tone'

let polySynth = null;

export default {
  name: 'TransportBar',
  components: {
    PianoRollPopup
  },
  created() {
    // Initialize polySynth once
    polySynth = new Tone.PolySynth(Tone.Synth).toDestination();
  },
  beforeUnmount() {
    if (polySynth) {
      polySynth.dispose();
    }
    this.pauseTimer();
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
    async playMidiNote(midiNote, instrument = 'piano') {
      try {
        // Ensure audio context is started
        await Tone.start();
        
        const now = Tone.now();
        
        // Update synth settings based on instrument
        polySynth.set({
          oscillator: {
            type: this.getOscillatorType(instrument)
          },
          envelope: this.getEnvelopeSettings(instrument),
          volume: -6
        });
        
        const freq = Tone.Frequency(midiNote, "midi");
        await polySynth.triggerAttackRelease(freq, "8n", now);
      } catch (error) {
        console.error('Error playing note:', error);
      }
    },
    getOscillatorType(instrument) {
      switch (instrument) {
        case 'sawtooth':
          return 'sawtooth';
        case 'brass':
          return 'square';
        case 'strings':
          return 'sine';
        default: // piano
          return 'triangle';
      }
    },
    getEnvelopeSettings(instrument) {
      switch (instrument) {
        case 'sawtooth':
          return {
            attack: 0.05,
            decay: 0.2,
            sustain: 0.5,
            release: 1
          };
        case 'brass':
          return {
            attack: 0.1,
            decay: 0.3,
            sustain: 0.6,
            release: 0.8
          };
        case 'strings':
          return {
            attack: 0.2,
            decay: 0.3,
            sustain: 0.8,
            release: 1.5
          };
        default: // piano
          return {
            attack: 0.005,
            decay: 0.1,
            sustain: 0.3,
            release: 1
          };
      }
    },
    async playTestNote() {
      try {
        // Add a user interaction check
        if (Tone.context.state !== 'running') {
          await Tone.start();
        }
        await this.playMidiNote(72, 'piano'); // C5 is MIDI note 72
      } catch (error) {
        console.error('Error playing test note:', error);
      }
    },
    handleBpmChange() {
      this.$emit('bpm-changed', this.bpm);
    }
  },
  watch: {
    bpm(newValue) {
      this.$emit('bpm-changed', newValue);
    }
  }
}
</script> 