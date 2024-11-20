<template>
  <div id="app">
    <transport-bar 
      @playback-changed="handlePlaybackChange"
      @bpm-changed="handleBpmChange"
    />
    <track-list 
      @track-selected="handleTrackSelect"
      @track-muted="handleTrackMute"
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
  methods: {
    handlePlaybackChange(isPlaying) {
      console.log('Playback state:', isPlaying);
    },
    handleBpmChange(bpm) {
      console.log('New BPM:', bpm);
    },
    handleTrackSelect(trackIndex) {
      console.log('Selected track:', trackIndex);
    },
    handleTrackMute({ index, muted }) {
      console.log(`Track ${index + 1} muted:`, muted);
      // Add your mute logic here
    }
  }
}
</script>