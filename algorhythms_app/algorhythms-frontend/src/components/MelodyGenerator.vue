<template>
  <div class="melody-generator">
    <h1>Generated Melodies</h1>
    <button @click="fetchMelodies">Generate Melodies</button>
    <button 
      @click="evolveMelodies" 
      :disabled="!allMelodiesRated"
    >Evolve Melodies</button>

    <div class="melody-list">
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
    </div>
  </div>
</template>

<script>
import * as Tone from 'tone'

export default {
  name: 'MelodyGenerator',
  data() {
    return {
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
    async fetchMelodies() {
      try {
        const response = await fetch('http://localhost:5000/generate')
        const data = await response.json()
        this.melodies = data.melodies
        this.ratedMelodies.clear()
        this.ratings = {}
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

<style scoped>
.melody-generator {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.melody-item {
  margin-bottom: 20px;
  padding: 15px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

.rating-section {
  margin-top: 10px;
}

button {
  margin: 5px;
  padding: 8px 16px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

select {
  margin: 0 10px;
  padding: 5px;
}
</style> 