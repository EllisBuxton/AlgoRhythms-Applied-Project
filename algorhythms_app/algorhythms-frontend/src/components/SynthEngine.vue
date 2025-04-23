<template>
  <div class="synth-engine"></div>
</template>

<script>
import * as Tone from 'tone'

export default {
  name: 'SynthEngine',
  data() {
    return {
      audioContextStarted: false
    }
  },
  created() {
    // Create separate synths for each instrument type
    this.synths = {
      piano: new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' },
        envelope: {
          attack: 0.005,
          decay: 0.1,
          sustain: 0.3,
          release: 1
        }
      }).toDestination(),
      sawtooth: new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'sawtooth' },
        envelope: {
          attack: 0.05,
          decay: 0.2,
          sustain: 0.5,
          release: 1
        }
      }).toDestination(),
      brass: new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'square' },
        envelope: {
          attack: 0.1,
          decay: 0.3,
          sustain: 0.6,
          release: 0.8
        }
      }).toDestination(),
      strings: new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'sine' },
        envelope: {
          attack: 0.2,
          decay: 0.3,
          sustain: 0.8,
          release: 1.5
        }
      }).toDestination()
    };
    
    // Create drum synthesizers with distinct sounds
    this.kickSynth = new Tone.MembraneSynth({
      pitchDecay: 0.05,
      octaves: 10,
      oscillator: {
        type: "sine"
      },
      envelope: {
        attack: 0.001,
        decay: 0.2,
        sustain: 0.01,
        release: 1.2,
        attackCurve: "exponential"
      }
    }).toDestination();

    this.snareSynth = new Tone.NoiseSynth({
      noise: {
        type: "white"
      },
      envelope: {
        attack: 0.001,
        decay: 0.2,
        sustain: 0.01,
        release: 0.2
      }
    }).chain(
      new Tone.Filter({
        type: "highpass",
        frequency: 2000,
        Q: 1
      }),
      new Tone.Filter({
        type: "lowpass",
        frequency: 8000,
        Q: 1
      })
    ).toDestination();

    this.hihatSynth = new Tone.MetalSynth({
      frequency: 200,
      envelope: {
        attack: 0.001,
        decay: 0.1,
        release: 0.01
      },
      harmonicity: 5.1,
      modulationIndex: 32,
      resonance: 4000,
      octaves: 1.5
    }).chain(
      new Tone.Filter({
        type: "highpass",
        frequency: 5000,
        Q: 1
      })
    ).toDestination();

    this.tomSynth = new Tone.MembraneSynth({
      pitchDecay: 0.05,
      octaves: 8,
      oscillator: {
        type: "sine"
      },
      envelope: {
        attack: 0.001,
        decay: 0.2,
        sustain: 0.01,
        release: 0.8,
        attackCurve: "exponential"
      }
    }).chain(
      new Tone.Filter({
        type: "bandpass",
        frequency: 400,
        Q: 2
      })
    ).toDestination();

    this.rideSynth = new Tone.MetalSynth({
      frequency: 400,
      envelope: {
        attack: 0.001,
        decay: 0.2,
        release: 0.1
      },
      harmonicity: 3.1,
      modulationIndex: 24,
      resonance: 3000,
      octaves: 1.2
    }).chain(
      new Tone.Filter({
        type: "bandpass",
        frequency: 2000,
        Q: 2
      })
    ).toDestination();

    this.crashSynth = new Tone.MetalSynth({
      frequency: 300,
      envelope: {
        attack: 0.001,
        decay: 0.1,
        release: 0.01
      },
      harmonicity: 5.1,
      modulationIndex: 32,
      resonance: 4000,
      octaves: 1.5
    }).chain(
      new Tone.Filter({
        type: "highpass",
        frequency: 3000,
        Q: 1
      })
    ).toDestination();

    this.chineseCymbalSynth = new Tone.MetalSynth({
      frequency: 500,
      envelope: {
        attack: 0.001,
        decay: 0.3,
        release: 0.2
      },
      harmonicity: 4.1,
      modulationIndex: 40,
      resonance: 5000,
      octaves: 1.8
    }).chain(
      new Tone.Filter({
        type: "bandpass",
        frequency: 3000,
        Q: 3
      })
    ).toDestination();
  },
  beforeUnmount() {
    // Dispose of all synths
    Object.values(this.synths).forEach(synth => synth.dispose());
    if (this.kickSynth) this.kickSynth.dispose();
    if (this.snareSynth) this.snareSynth.dispose();
    if (this.hihatSynth) this.hihatSynth.dispose();
    if (this.tomSynth) this.tomSynth.dispose();
    if (this.rideSynth) this.rideSynth.dispose();
    if (this.crashSynth) this.crashSynth.dispose();
    if (this.chineseCymbalSynth) this.chineseCymbalSynth.dispose();
  },
  methods: {
    async startAudioContext() {
      if (!this.audioContextStarted) {
        try {
          await Tone.start();
          this.audioContextStarted = true;
          console.log("AudioContext started successfully");
        } catch (error) {
          console.error("Error starting AudioContext:", error);
        }
      }
    },
    async playNote(midiNote, instrument = 'piano') {
      try {
        await this.startAudioContext();
        const now = Tone.now();
        
        if (instrument === 'drums') {
          console.log("Attempting to play drum note:", midiNote);
          
          // Declare variables outside switch statement
          let freq;
          let decay;
          const tomNotes = ["C2", "D2", "E2", "F2", "G2", "A2"];
          let tomIndex;
          
          // Standard General MIDI drum mapping
          switch (midiNote) {
            // Bass Drum
            case 35: // Acoustic Bass Drum
            case 36: // Bass Drum 1
              this.kickSynth.triggerAttackRelease("C1", "8n", now);
              break;
              
            // Snare Drum
            case 38: // Acoustic Snare
            case 40: // Electric Snare
              this.snareSynth.triggerAttackRelease("8n", now);
              break;
              
            // Hi-Hat
            case 42: // Closed Hi-Hat
            case 44: // Pedal Hi-Hat
            case 46: // Open Hi-Hat
              freq = midiNote === 42 ? "C6" : midiNote === 44 ? "C#6" : "D6";
              decay = midiNote === 42 ? 0.05 : midiNote === 44 ? 0.1 : 0.15;
              this.hihatSynth.envelope.decay = decay;
              this.hihatSynth.triggerAttackRelease(freq, "8n", now);
              break;
              
            // Toms
            case 41: // Low Floor Tom
            case 43: // High Floor Tom
            case 45: // Low Tom
            case 47: // Low-Mid Tom
            case 48: // Hi-Mid Tom
            case 50: // High Tom
              tomIndex = Math.min(midiNote - 41, tomNotes.length - 1);
              this.tomSynth.triggerAttackRelease(tomNotes[tomIndex], "8n", now);
              break;
              
            // Cymbals
            case 49: // Crash Cymbal 1
            case 57: // Crash Cymbal 2
              this.crashSynth.triggerAttackRelease(midiNote === 49 ? "C6" : "D6", "8n", now);
              break;
              
            case 51: // Ride Cymbal 1
            case 59: // Ride Cymbal 2
              this.rideSynth.triggerAttackRelease(midiNote === 51 ? "E6" : "F6", "8n", now);
              break;
              
            case 52: // Chinese Cymbal
              this.chineseCymbalSynth.triggerAttackRelease("G6", "8n", now);
              break;
              
            case 53: // Ride Bell
              this.rideSynth.triggerAttackRelease("A6", "8n", now);
              break;
              
            case 55: // Splash Cymbal
              this.crashSynth.triggerAttackRelease("B6", "8n", now);
              break;
              
            default:
              console.warn("No drum sound mapped for MIDI note:", midiNote);
          }
        } else {
          // Use the appropriate synth for the instrument
          const synth = this.synths[instrument];
          if (synth) {
            const freq = Tone.Frequency(midiNote, "midi");
            await synth.triggerAttackRelease(freq, "8n", now);
          }
        }
      } catch (error) {
        console.error('Error playing note:', error);
      }
    }
  }
}
</script> 