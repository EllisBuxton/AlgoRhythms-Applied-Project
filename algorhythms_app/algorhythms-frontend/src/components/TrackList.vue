<template>
  <div class="track-list">
    <div 
      class="playhead-container"
      :style="{ left: `${playheadPosition}px` }"
      @mousedown="startDragging"
      @touchstart="startDragging"
    >
      <div class="playhead-line"></div>
      <div class="playhead-handle"></div>
    </div>
    <div 
      v-for="(track, index) in tracks" 
      :key="index"
      class="track-container"
    >
      <div 
        class="track-item"
        :class="{ active: selectedTrack === index, muted: track.muted }"
        @click="selectTrack(index)"
        @dragover.prevent="handleDragOver($event)"
        @drop.prevent="handleDrop($event, index)"
      >
        <span class="track-icon">🎵</span>
        <span class="track-name">{{ track.name }}</span>
        <button 
          class="mute-button"
          :class="{ muted: track.muted }"
          @click.stop="toggleMute(index)"
        >
          {{ track.muted ? '🔇' : '🔊' }}
        </button>
      </div>
      <div class="timeline">
        <div class="timeline-notches">
          <div v-for="second in 60" :key="second" class="notch-container">
            <div class="notch"></div>
            <span v-if="second % 5 === 0" class="time-label">{{ second }}s</span>
          </div>
        </div>
        <div class="track-melodies">
          <div 
            v-for="(melody, mIndex) in track.melodies" 
            :key="mIndex"
            class="track-melody"
            :style="{ 
              left: melody.startTime + 'px',
              width: calculateMelodyWidth(melody) + 'px'
            }"
            draggable="true"
            @dragstart="handleMelodyDragStart($event, index, mIndex)"
            @dragend="handleMelodyDragEnd"
          >
            <div class="melody-block">
              <div class="melody-name">{{ melody.name || `Melody ${mIndex + 1}` }}</div>
              <div class="mini-piano-roll">
                <div 
                  v-for="note in melody.notes" 
                  :key="note.cell + '-' + note.midiNote"
                  class="mini-note"
                  :style="{
                    left: `${(note.cell / (calculateLastCell(melody) + 1)) * 100}%`,
                    bottom: `${((note.midiNote - 21) / 88) * 100}%`
                  }"
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="track-buttons">
      <button 
        class="add-track-button" 
        @click.stop="addTrack"
      >
        + Add Track
      </button>
      <button 
        class="remove-track-button" 
        @click.stop="removeTrack"
        :disabled="tracks.length <= 1"
      >
        - Remove Track
      </button>
    </div>
  </div>
</template>

<script>
import '../style/TrackList.css'

export default {
  name: 'TrackList',
  props: {
    currentTime: {
      type: Number,
      default: 0
    },
    isPlaying: {
      type: Boolean,
      default: false
    },
    bpm: {
      type: Number,
      default: 120
    }
  },
  data() {
    return {
      tracks: [
        { name: 'Track 1', instrument: 'piano', muted: false, melodies: [] },
        { name: 'Track 2', instrument: 'piano', muted: false, melodies: [] },
        { name: 'Track 3', instrument: 'piano', muted: false, melodies: [] }
      ],
      selectedTrack: 0,
      playheadPosition: 250,
      isDragging: false,
      startX: 0,
      startPos: 0,
      isDraggingMelody: false,
      draggedMelodyIndex: null,
      draggedTrackIndex: null
    }
  },
  watch: {
    currentTime(newTime) {
      if (!this.isDragging) {
        this.playheadPosition = 250 + (newTime * 30) - 15;
        this.checkAndPlayMelodies(newTime);
      }
    },
    isPlaying(newValue) {
      if (newValue) {
        this.checkAndPlayMelodies(this.currentTime);
      }
    },
    isDragging(newValue) {
      if (!newValue) {
        // When dragging stops, update the currentTime based on the final position
        const timePosition = ((this.playheadPosition - 250 + 15) / 30);
        this.$emit('playhead-moved', timePosition);
      }
    }
  },
  mounted() {
    window.addEventListener('mousemove', this.onDrag);
    window.addEventListener('mouseup', this.stopDragging);
    window.addEventListener('touchmove', this.onDrag);
    window.addEventListener('touchend', this.stopDragging);
  },
  beforeUnmount() {
    window.removeEventListener('mousemove', this.onDrag);
    window.removeEventListener('mouseup', this.stopDragging);
    window.removeEventListener('touchmove', this.onDrag);
    window.removeEventListener('touchend', this.stopDragging);
  },
  methods: {
    selectTrack(index) {
      this.selectedTrack = index;
      this.$emit('track-selected', index);
    },
    addTrack(event) {
      // Prevent event propagation
      event.stopPropagation();
      event.preventDefault();
      
      const newTrackNumber = this.tracks.length + 1;
      this.tracks.push({
        name: `Track ${newTrackNumber}`,
        instrument: 'piano',
        muted: false,
        melodies: []
      });
    },
    toggleMute(index) {
      this.tracks[index].muted = !this.tracks[index].muted;
      this.$emit('track-muted', { index, muted: this.tracks[index].muted });
    },
    removeTrack(event) {
      // Prevent event propagation
      event.stopPropagation();
      event.preventDefault();
      
      if (this.tracks.length > 1) {
        this.tracks.pop();
        if (this.selectedTrack >= this.tracks.length) {
          this.selectedTrack = this.tracks.length - 1;
          this.$emit('track-selected', this.selectedTrack);
        }
      }
    },
    startDragging(event) {
      if (event) {
        this.isDragging = true;
        this.startX = event.type === 'mousedown' ? event.clientX : event.touches[0].clientX;
        this.startPos = this.playheadPosition;
        this.$emit('playhead-drag-started');
      }
    },
    onDrag(event) {
      if (!this.isDragging) return;
      
      const currentX = event.type === 'mousemove' ? event.clientX : event.touches[0].clientX;
      const delta = currentX - this.startX;
      
      const trackList = document.querySelector('.track-list');
      const minX = 250;
      const maxX = trackList.offsetWidth;
      
      let newPosition = this.startPos + delta;
      newPosition = Math.max(minX, Math.min(maxX, newPosition));
      
      this.playheadPosition = newPosition;
      
      // Calculate and emit the new time position during dragging
      const timePosition = ((newPosition - 250 + 15) / 30);
      this.$emit('playhead-moved', timePosition);
    },
    stopDragging() {
      this.isDragging = false;
      this.$emit('playhead-drag-ended');
    },
    handleDragOver(event) {
      event.preventDefault();
      event.currentTarget.style.backgroundColor = '#894ab6';
    },
    handleDrop(event, index) {
      event.preventDefault();
      event.currentTarget.style.backgroundColor = '';
      
      try {
        const melodyData = JSON.parse(event.dataTransfer.getData('application/json'));
        const dropPosition = event.clientX - 250; // Adjust for the track label width
        
        // Find the last melody in the track
        const lastMelody = this.tracks[index].melodies.length > 0 
          ? this.tracks[index].melodies[this.tracks[index].melodies.length - 1] 
          : null;
        
        // Calculate start time based on the last melody's end position
        const startTime = lastMelody 
          ? lastMelody.startTime + this.calculateMelodyWidth(lastMelody)
          : Math.max(0, dropPosition);
        
        this.tracks[index].melodies.push({
          ...melodyData,
          startTime,
          lastPlayedCell: -1 // Track playback position
        });
        
        this.$emit('melody-added', { trackIndex: index, melody: melodyData });
      } catch (error) {
        console.error('Error adding melody to track:', error);
      }
    },
    checkAndPlayMelodies(currentTime) {
      if (!this.isPlaying) return;

      const timeInMs = currentTime * 1000; // Convert to milliseconds
      
      this.tracks.forEach(track => {
        if (track.muted) return;
        
        track.melodies.forEach(melody => {
          const melodyStartTime = melody.startTime / 30 * 1000; // Convert pixels to ms
          const beatDuration = 60000 / this.bpm;
          const cellDuration = beatDuration / 4;
          
          // Calculate which cell should be playing at current time
          const currentCell = Math.floor((timeInMs - melodyStartTime) / cellDuration);
          
          // Remove the cell range check to allow all notes to play
          if (currentCell >= 0 && currentCell !== melody.lastPlayedCell) {
            // Find notes that should play in this cell
            const notesToPlay = melody.notes.filter(note => note.cell === currentCell);
            
            notesToPlay.forEach(note => {
              this.$emit('play-note', {
                midiNote: note.midiNote,
                instrument: melody.instrument
              });
            });
            
            melody.lastPlayedCell = currentCell;
          }
        });
      });
    },
    calculateMelodyWidth(melody) {
      // Find the last cell that has a note
      const lastCell = Math.max(...melody.notes.map(note => note.cell), 0);
      
      // Calculate duration based on current BPM instead of melody's BPM
      const beatDuration = 60 / this.bpm; // Duration of one beat in seconds
      const sixteenthNoteDuration = beatDuration / 4; // Duration of one cell in seconds
      const totalDuration = (lastCell + 1) * sixteenthNoteDuration; // Total duration in seconds
      
      // Convert duration to pixels (30px = 1 second)
      return totalDuration * 30;
    },
    calculateLastCell(melody) {
      return Math.max(...melody.notes.map(note => note.cell), 0);
    },
    handleMelodyDragStart(event, trackIndex, melodyIndex) {
      this.isDraggingMelody = true;
      this.draggedTrackIndex = trackIndex;
      this.draggedMelodyIndex = melodyIndex;
      event.dataTransfer.setData('text/plain', ''); // Required for drag to work
      event.dataTransfer.effectAllowed = 'move';
    },
    
    handleMelodyDragEnd(event) {
      if (!this.isDraggingMelody) return;
      
      // Check if the melody was dragged outside the track area
      const trackList = document.querySelector('.track-list');
      const rect = trackList.getBoundingClientRect();
      
      if (
        event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom
      ) {
        // Remove the melody if it was dragged outside
        this.tracks[this.draggedTrackIndex].melodies.splice(this.draggedMelodyIndex, 1);
        this.$emit('melody-removed', { 
          trackIndex: this.draggedTrackIndex, 
          melodyIndex: this.draggedMelodyIndex 
        });
      }
      
      this.isDraggingMelody = false;
      this.draggedTrackIndex = null;
      this.draggedMelodyIndex = null;
    }
  }
}
</script>

<style>
/* Add to your existing styles */
.timeline {
  position: relative;
}

.track-melodies {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.track-melody {
  position: absolute;
  top: 5px;
  bottom: 5px;
  pointer-events: auto;
}

.melody-block {
  height: 100%;
  width: 100%;
  background-color: rgba(137, 74, 182, 0.3);
  border: 1px solid #894ab6;
  border-radius: 4px;
  padding: 4px;
  display: flex;
  flex-direction: column;
}

.melody-name {
  font-size: 12px;
  color: #e1bee7;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 2px;
}

.mini-piano-roll {
  flex-grow: 1;
  position: relative;
  background-color: rgba(0, 0, 0, 0.2);
  border-radius: 2px;
  min-height: 24px;
}

.mini-note {
  position: absolute;
  width: 4px;
  height: 4px;
  background-color: #4caf50;
  border-radius: 2px;
  transform: translate(-50%, 50%);
  box-shadow: 0 0 2px rgba(76, 175, 80, 0.5);
}
</style> 