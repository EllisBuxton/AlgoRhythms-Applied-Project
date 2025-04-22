<template>
  <div id="app">
    <transport-bar 
      :currentTime="currentTime"
      @playback-changed="handlePlaybackChange"
      @bpm-changed="handleBpmChange"
      @time-updated="handleTimeUpdate"
      ref="transportBar"
    />
    <track-list 
      :currentTime="currentTime"
      @track-selected="handleTrackSelect"
      @track-muted="handleTrackMute"
      @playhead-moved="handlePlayheadMoved"
      @playhead-drag-started="handlePlayheadDragStart"
      @playhead-drag-ended="handlePlayheadDragEnd"
    />
    <div class="main-content">
    </div>
  </div>
</template>

<script>
import TransportBar from './components/TransportBar.vue'
import TrackList from './components/TrackList.vue'
import './style/App.css'

export default {
  name: 'App',
  components: {
    TransportBar,
    TrackList
  },
  data() {
    return {
      currentTime: 0,
      wasPlaying: false,
      isPlaying: false
    }
  },
  methods: {
    handlePlaybackChange(isPlaying) {
      console.log('Playback state:', isPlaying);
      this.isPlaying = isPlaying;
    },
    handleBpmChange(bpm) {
      console.log('New BPM:', bpm);
    },
    handleTrackSelect(trackIndex) {
      console.log('Selected track:', trackIndex);
    },
    handleTrackMute({ index, muted }) {
      console.log(`Track ${index + 1} muted:`, muted);
    },
    handleTimeUpdate(time) {
      if (this.isPlaying) {
        this.currentTime = time;
      }
    },
    handlePlayheadMoved(time) {
      this.currentTime = time;
    },
    handlePlayheadDragStart() {
      this.$refs.transportBar.handlePlayheadDragStart();
    },
    handlePlayheadDragEnd() {
      this.$refs.transportBar.handlePlayheadDragEnd();
    }
  }
}
</script>