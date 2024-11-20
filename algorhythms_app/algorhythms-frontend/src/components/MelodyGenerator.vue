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
            <div class="star-rating">
              <span 
                v-for="star in 5" 
                :key="star"
                class="star"
                :class="{ 
                  active: (ratings[index] || 0) >= star,
                  hover: (hoverRatings[index] || 0) >= star 
                }"
                @click="rateMelody(index, star)"
                @mouseenter="setHoverRating(index, star)"
                @mouseleave="setHoverRating(index, 0)"
              >
                ★
              </span>
            </div>
            <div class="rating-status">
              <button 
                @click="submitRating(index)" 
                :disabled="!ratings[index] || ratedMelodies.has(index)"
              >
                Submit Rating
              </button>
              <span v-if="ratedMelodies.has(index)" class="rating-submitted">✓</span>
            </div>
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
import '../style/MelodyGenerator.css'

export default {
  name: 'MelodyGenerator',
  data() {
    return {
      selectedInstrument: null,
      melodies: [],
      ratings: {},
      ratedMelodies: new Set(),
      hoverRatings: {}
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
    rateMelody(index, rating) {
      this.ratings[index] = rating;
    },
    async submitRating(index) {
      if (!this.ratings[index]) return;
      
      try {
        const response = await fetch('http://localhost:5000/rate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            melodyIndex: index,
            rating: this.ratings[index]
          }),
        });

        if (response.ok) {
          this.ratedMelodies.add(index);
        }
      } catch (error) {
        console.error('Error submitting rating:', error);
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
    },
    setHoverRating(index, rating) {
      this.hoverRatings[index] = rating;
    }
  }
}
</script>
