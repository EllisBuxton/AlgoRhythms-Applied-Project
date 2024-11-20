<template>
  <div class="track-list">
    <div 
      v-for="(track, index) in tracks" 
      :key="index"
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
    <div class="track-buttons">
      <button class="add-track-button" @click="addTrack">
        + Add Track
      </button>
      <button 
        class="remove-track-button" 
        @click="removeTrack"
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
  data() {
    return {
      tracks: [
        { name: 'Track 1', instrument: 'piano', muted: false },
        { name: 'Track 2', instrument: 'piano', muted: false },
        { name: 'Track 3', instrument: 'piano', muted: false }
      ],
      selectedTrack: 0
    }
  },
  methods: {
    selectTrack(index) {
      this.selectedTrack = index;
      this.$emit('track-selected', index);
    },
    addTrack() {
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
    removeTrack() {
      if (this.tracks.length > 1) {
        this.tracks.pop();
        if (this.selectedTrack >= this.tracks.length) {
          this.selectedTrack = this.tracks.length - 1;
          this.$emit('track-selected', this.selectedTrack);
        }
      }
    }
  }
}
</script> 