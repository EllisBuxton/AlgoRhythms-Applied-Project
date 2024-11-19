<template>
  <div id="app">
    <h1>Welcome to AlgoRhythms</h1>
    <p>Generate and evolve musical melodies using AI</p>
    <melody-generator />
  </div>
</template>

<script>
import MelodyGenerator from './components/MelodyGenerator.vue'
import './App.css'
export default {
  name: 'App',
  components: {
    MelodyGenerator
  }
}
</script>