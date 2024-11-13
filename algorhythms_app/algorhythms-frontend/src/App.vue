<template>
  <div id="app">
    <h1>Welcome to AlgoRhythms</h1>
    <p>Generate and evolve musical melodies using AI</p>
    <melody-generator />
  </div>
</template>

<script>
import MelodyGenerator from './components/MelodyGenerator.vue'

export default {
  name: 'App',
  components: {
    MelodyGenerator
  }
}
</script>

<style>
#app {
  font-family: Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

h1 {
  color: #34495e;
  margin-bottom: 10px;
}

p {
  color: #7f8c8d;
  margin-bottom: 30px;
}
</style>