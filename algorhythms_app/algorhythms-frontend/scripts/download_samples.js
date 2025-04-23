import fetch from 'node-fetch';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const samples = {
  kick: 'https://raw.githubusercontent.com/Tonejs/audio/master/drum-samples/kick.mp3',
  snare: 'https://raw.githubusercontent.com/Tonejs/audio/master/drum-samples/snare.mp3',
  hihat: 'https://raw.githubusercontent.com/Tonejs/audio/master/drum-samples/hihat.mp3',
  tom1: 'https://raw.githubusercontent.com/Tonejs/audio/master/drum-samples/tom1.mp3',
  tom2: 'https://raw.githubusercontent.com/Tonejs/audio/master/drum-samples/tom2.mp3',
  crash: 'https://raw.githubusercontent.com/Tonejs/audio/master/drum-samples/crash.mp3'
};

const samplesDir = path.join(__dirname, '../public/samples');

// Create samples directory if it doesn't exist
if (!fs.existsSync(samplesDir)) {
  fs.mkdirSync(samplesDir, { recursive: true });
}

// Download each sample
for (const [name, url] of Object.entries(samples)) {
  try {
    const response = await fetch(url);
    const buffer = await response.buffer();
    fs.writeFileSync(path.join(samplesDir, `${name}.mp3`), buffer);
    console.log(`Downloaded ${name}.mp3`);
  } catch (err) {
    console.error(`Error downloading ${name}.mp3:`, err.message);
  }
} 