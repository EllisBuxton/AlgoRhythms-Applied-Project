<template>
  <div id="app">
    <transport-bar 
      @playback-changed="handlePlaybackChange"
      @bpm-changed="handleBpmChange"
    />
    <div class="main-content">
      <h1>Welcome to AlgoRhythms</h1>
      <p>Generate and evolve musical melodies using AI</p>
      <melody-generator />
    </div>
  </div>
</template>

<script>
import MelodyGenerator from './components/MelodyGenerator.vue'
import TransportBar from './components/TransportBar.vue'
import './style/App.css'

export default {
  name: 'App',
  components: {
    MelodyGenerator,
    TransportBar
  },
  methods: {
    handlePlaybackChange(isPlaying) {
      console.log('Playback state:', isPlaying);
    },
    handleBpmChange(bpm) {
      console.log('New BPM:', bpm);
    }
  }
}
</script>