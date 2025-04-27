<template>
  <div class="transport-bar">
    <synth-engine ref="synthEngine" />
    <button class="transport-button piano-button" @click="togglePianoPopup">🎹</button>
    <button class="transport-button test-note-button" @click="playTestNote">🎵</button>
    
    <div class="saved-melodies-container">
      <div class="saved-melodies-list" :class="{ 'show-melodies': showMelodies }">
        <div 
          v-for="(melody, index) in savedMelodies" 
          :key="index"
          class="saved-melody"
          draggable="true"
          @dragstart="handleMelodyDragStart($event, melody)"
        >
          <span class="melody-name">Melody {{ index + 1 }}</span>
          <button class="play-melody-btn" @click="playMelody(melody)">▶</button>
          <button class="delete-melody-btn" @click="deleteMelody(index)">×</button>
        </div>
      </div>
      <div class="melody-buttons">
        <button 
          class="transport-button melodies-toggle" 
          @click="toggleMelodies"
          :class="{ active: showMelodies }"
        >
          <span class="file-icon">📂</span>
          <span class="melody-count">{{ savedMelodies.length }}</span>
        </button>
        <button class="transport-button generate-button" @click="generateMelody">
          <span class="generate-icon">🎵</span>
          <span class="generate-text">Generate</span>
        </button>
      </div>
    </div>

    <div class="centered-controls">
      <div class="transport-controls">
        <button @click="togglePlayback" class="play-button">
          {{ isPlaying ? '⏸' : '▶' }}
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
      @save-melody="saveMelody"
    />

    <track-list 
      :currentTime="currentTime"
      :isPlaying="isPlaying"
      :bpm="bpm"
      @track-selected="handleTrackSelect"
      @track-muted="handleTrackMute"
      @playhead-moved="handlePlayheadMoved"
      @drag-started="handleDragStart"
      @drag-ended="handleDragEnd"
      @play-note="handlePlayNote"
      ref="trackList"
    />
  </div>
</template>

<script>
import '../style/TransportBar.css'
import PianoRollPopup from './PianoRollPopup.vue'
import SynthEngine from './SynthEngine.vue'
import * as Tone from 'tone'
import TrackList from './TrackList.vue'

export default {
  name: 'TransportBar',
  components: {
    PianoRollPopup,
    SynthEngine,
    TrackList
  },
  beforeUnmount() {
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
      showPianoPopup: false,
      savedMelodies: JSON.parse(localStorage.getItem('savedMelodies') || '[]'),
      showMelodies: false,
      wasPlaying: false
    }
  },
  methods: {
    togglePlayback() {
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
      // Reset all melody playback positions when stopping
      if (this.$refs.trackList) {
        this.$refs.trackList.tracks.forEach(track => {
          track.melodies.forEach(melody => {
            melody.lastPlayedCell = -1;
          });
        });
      }
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
      await this.$refs.synthEngine.playNote(midiNote, instrument);
    },
    async playTestNote() {
      try {
        if (Tone.context.state !== 'running') {
          await Tone.start();
        }
        await this.playMidiNote(72, 'piano');
      } catch (error) {
        console.error('Error playing test note:', error);
      }
    },
    handleBpmChange() {
      this.$emit('bpm-changed', this.bpm);
    },
    toggleMelodies() {
      this.showMelodies = !this.showMelodies;
    },
    saveMelody(melody) {
      // Add a unique ID to the melody
      const melodyWithId = {
        ...melody,
        id: Date.now().toString()
      };
      this.savedMelodies.push(melodyWithId);
      this.showMelodies = true; // Show the melodies list when saving
    },
    async playMelody(melody) {
      if (this.isPlaying) return;
      
      this.isPlaying = true;
      const beatDuration = 60000 / melody.bpm; // Duration of one beat in ms
      const sixteenthNoteDuration = beatDuration / 4; // Duration of one grid cell
      
      try {
        if (Tone.context.state !== 'running') {
          await Tone.start();
        }
        
        // Sort notes by cell position to play in order
        const sortedNotes = melody.notes.sort((a, b) => a.cell - b.cell);
        let lastCell = -1;
        
        for (const note of sortedNotes) {
          // Calculate delay based on cell position
          const cellDelay = note.cell - lastCell;
          if (cellDelay > 0) {
            await new Promise(resolve => setTimeout(resolve, cellDelay * sixteenthNoteDuration));
          }
          
          await this.playMidiNote(note.midiNote, melody.instrument);
          lastCell = note.cell;
        }
      } catch (error) {
        console.error('Error playing melody:', error);
      } finally {
        this.isPlaying = false;
      }
    },
    deleteMelody(index) {
      this.savedMelodies.splice(index, 1);
    },
    handleMelodyDragStart(event, melody) {
      if (event && melody) {
        event.dataTransfer.setData('application/json', JSON.stringify(melody));
        event.dataTransfer.effectAllowed = 'copy';
      }
    },
    handlePlayheadMoved(time) {
      // Emit the time update instead of mutating the prop
      this.$emit('time-updated', time);
      
      // If we were playing before dragging, continue playing from the new position
      if (this.wasPlaying) {
        this.lastTimestamp = performance.now();
        this.startTimer();
      }
    },
    handlePlayheadDragStart() {
      this.wasPlaying = this.isPlaying;
      if (this.wasPlaying) {
        this.pauseTimer();
      }
      this.$emit('playhead-drag-started');
    },
    handlePlayheadDragEnd() {
      // Don't restart the timer here, as it's handled in handlePlayheadMoved
      this.wasPlaying = false;
      this.$emit('playhead-drag-ended');
    },
    handlePlayNote({ midiNote, instrument }) {
      this.playMidiNote(midiNote, instrument);
    },
    async generateMelody() {
      try {
        console.log('Attempting to generate melody...');
        const response = await fetch('http://localhost:5000/generate-melody', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            bpm: this.bpm,
            instrument: 'piano'
          })
        });
        
        if (!response.ok) {
          throw new Error(`Server returned status ${response.status}`);
        }
        
        const melody = await response.json();
        console.log('Generated melody:', melody);
        
        // Save the generated melody
        this.saveMelody({
          ...melody,
          bpm: this.bpm,
          instrument: 'piano'
        });
        
        // Show the melodies list
        this.showMelodies = true;
        
      } catch (error) {
        console.error('Error generating melody:', error);
        if (error.message.includes('NetworkError')) {
          alert('Could not connect to the server. Please make sure the backend server is running on port 5000.');
        } else {
          alert(`Failed to generate melody: ${error.message}`);
        }
      }
    }
  },
  watch: {
    savedMelodies: {
      handler(newMelodies) {
        localStorage.setItem('savedMelodies', JSON.stringify(newMelodies));
      },
      deep: true
    },
    bpm(newValue) {
      this.$emit('bpm-changed', newValue);
    }
  }
}
</script> 