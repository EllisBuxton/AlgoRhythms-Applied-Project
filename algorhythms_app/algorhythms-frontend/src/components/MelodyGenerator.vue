<template>
  <div class="melody-generator">
    <h1>Select an Instrument</h1>
    
    <div class="instrument-buttons">
      <button 
        @click="toggleInstrument('piano')"
        :class="{ active: selectedInstrument === 'piano' }"
      >Piano</button>
      <button disabled>Guitar</button>
      <button disabled>Drums</button>
    </div>

    <div v-if="selectedInstrument === 'piano'" class="instrument-section">
      <h2>Piano Melodies</h2>

      <div class="melody-list" v-if="melodies.length > 0">
        <div v-for="(melody, index) in melodies" :key="index" class="melody-item">
          <p>Melody {{ index + 1 }}: {{ melody.join(', ') }}</p>
          <button @click="playMelody(melody)">Play Melody {{ index + 1 }}</button>
          
          <div class="rating-section">
            <label :for="'rating' + index">Rate this melody:</label>
            <select 
              v-model="ratings[index]" 
              :id="'rating' + index"
            >
              <option value="" disabled>Select rating</option>
              <option v-for="n in 5" :key="n" :value="n">{{ n }}</option>
            </select>
            <button @click="submitRating(index)">Submit Rating</button>
          </div>
          <hr>
        </div>

        <button 
          @click="evolveMelodies" 
          :disabled="!allMelodiesRated"
          class="evolve-button"
        >Evolve Melodies</button>
      </div>
    </div>
  </div>
</template>

<script>
import * as Tone from 'tone'
import './MelodyGenerator.css'

export default {
  name: 'MelodyGenerator',
  data() {
    return {
      selectedInstrument: null,
      melodies: [],
      ratings: {},
      ratedMelodies: new Set()
    }
  },
  computed: {
    allMelodiesRated() {
      return this.melodies.length > 0 && 
             this.ratedMelodies.size === this.melodies.length
    }
  },
  methods: {
    async toggleInstrument(instrument) {
      this.selectedInstrument = instrument
      this.melodies = []
      this.ratings = {}
      this.ratedMelodies.clear()
      
      try {
        const response = await fetch('http://localhost:5000/generate')
        const data = await response.json()
        this.melodies = data.melodies
      } catch (error) {
        console.error('Error fetching melodies:', error)
      }
    },
    playMelody(melody) {
      const synth = new Tone.Synth().toDestination()
      let now = Tone.now()

      melody.forEach((note, index) => {
        synth.triggerAttackRelease(
          Tone.Frequency(note, "midi").toFrequency(),
          "8n",
          now + index * 0.5
        )
      })
    },
    async submitRating(index) {
      if (!this.ratings[index]) {
        alert("Please select a rating before submitting.")
        return
      }

      try {
        const response = await fetch('http://localhost:5000/rate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            melodyIndex: index,
            rating: this.ratings[index]
          })
        })
        const data = await response.json()
        alert(data.message)
        this.ratedMelodies.add(index)
      } catch (error) {
        console.error('Error submitting rating:', error)
      }
    },
    async evolveMelodies() {
      try {
        const response = await fetch('http://localhost:5000/evolve')
        const data = await response.json()
        this.melodies = data.melodies
        this.ratedMelodies.clear()
        this.ratings = {}
      } catch (error) {
        console.error('Error evolving melodies:', error)
      }
    }
  }
}
</script>
