<template>
  <div v-if="show" class="piano-popup-overlay" @click="close">
    <div class="piano-popup" @click.stop>
      <button class="close-button" @click="close">×</button>
      <h2>Piano</h2>
      <div class="piano-content">
        <!-- Piano interface content will go here -->
        <p>Piano interface coming soon!</p>
      </div>
    </div>
  </div>
</template>

<script>
import '../style/PianoRollPopup.css'

export default {
  name: 'PianoRollPopup',
  props: {
    show: {
      type: Boolean,
      required: true
    }
  },
  methods: {
    close() {
      this.$emit('close');
    }
  }
}
</script> 