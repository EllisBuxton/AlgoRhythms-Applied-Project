<script>
import * as Tone from 'tone'

export default {
  name: 'SynthEngine',
  created() {
    this.polySynth = new Tone.PolySynth(Tone.Synth).toDestination();
  },
  beforeUnmount() {
    if (this.polySynth) {
      this.polySynth.dispose();
    }
  },
  methods: {
    async playNote(midiNote, instrument = 'piano') {
      try {
        await Tone.start();
        const now = Tone.now();
        
        this.polySynth.set({
          oscillator: {
            type: this.getOscillatorType(instrument)
          },
          envelope: this.getEnvelopeSettings(instrument),
          volume: -6
        });
        
        const freq = Tone.Frequency(midiNote, "midi");
        await this.polySynth.triggerAttackRelease(freq, "8n", now);
      } catch (error) {
        console.error('Error playing note:', error);
      }
    },
    getOscillatorType(instrument) {
      switch (instrument) {
        case 'sawtooth':
          return 'sawtooth';
        case 'brass':
          return 'square';
        case 'strings':
          return 'sine';
        default: // piano
          return 'triangle';
      }
    },
    getEnvelopeSettings(instrument) {
      switch (instrument) {
        case 'sawtooth':
          return {
            attack: 0.05,
            decay: 0.2,
            sustain: 0.5,
            release: 1
          };
        case 'brass':
          return {
            attack: 0.1,
            decay: 0.3,
            sustain: 0.6,
            release: 0.8
          };
        case 'strings':
          return {
            attack: 0.2,
            decay: 0.3,
            sustain: 0.8,
            release: 1.5
          };
        default: // piano
          return {
            attack: 0.005,
            decay: 0.1,
            sustain: 0.3,
            release: 1
          };
      }
    }
  }
}
</script> 