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
    }
  },
  data() {
    return {
      tracks: [
        { name: 'Track 1', instrument: 'piano', muted: false },
        { name: 'Track 2', instrument: 'piano', muted: false },
        { name: 'Track 3', instrument: 'piano', muted: false }
      ],
      selectedTrack: 0,
      playheadPosition: 250,
      isDragging: false,
      startX: 0,
      startPos: 0
    }
  },
  watch: {
    currentTime(newTime) {
      if (!this.isDragging) {
        // Each notch is 30px apart, and each second is one notch
        this.playheadPosition = 250 + (newTime * 30) - 15; // Subtract 15px to center on notches
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
        muted: false
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
      this.isDragging = true;
      this.startX = event.type === 'mousedown' ? event.clientX : event.touches[0].clientX;
      this.startPos = this.playheadPosition;
      this.$emit('drag-started');
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
      
      // Adjust the time calculation to match the position
      const timePosition = ((newPosition - 250 + 15) / 30); // Add 15px to compensate for centering
      this.$emit('playhead-moved', timePosition);
    },
    stopDragging() {
      this.isDragging = false;
      this.$emit('drag-ended');
    }
  }
}
</script> 