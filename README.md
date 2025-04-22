# AlgoRhythms

A web application for generating and evolving musical melodies using genetic algorithms.

## Project Structure
```
algorhythms_app/
├── app.py              # Main Flask application
├── templates/          # HTML templates
├── utils/             # Utility functions
├── models/            # ML models
├── algorhythms-frontend/  # Frontend code
└── POP909/            # Dataset directory
```

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Ubuntu WSL (Windows Subsystem for Linux)

## WSL Setup

1. Install WSL and Ubuntu:
```bash
# In PowerShell as Administrator
wsl --install
wsl --install -d Ubuntu
```

2. Launch Ubuntu WSL and update packages:
```bash
sudo apt update
sudo apt upgrade
```

3. Install required system packages:
```bash
sudo apt install python3-pip python3-venv build-essential
```

4. Clone the repository in your WSL home directory:
```bash
cd ~
git clone [your-repository-url]
cd AlgoRhythms-Applied-Project
```

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application in WSL

1. Make sure you're in the project root directory and your virtual environment is activated.

2. Start the Flask server:
```bash
python3 algorhythms_app/app.py
```

3. Open your web browser on Windows and navigate to:
```
http://localhost:5000
```

Note: The Flask server running in WSL will be accessible from your Windows browser.

## Running the AI Model in WSL

The project includes a GRU (Gated Recurrent Unit) model for melody generation. Here's how to use it in WSL:

### Step 1: Training the Model

1. Make sure you're in the project root directory and your virtual environment is activated.

2. Install additional system dependencies for MIDI processing:
```bash
sudo apt install libasound2-dev
```

3. Run the training script:
```bash
python3 algorhythms_app/models/train_gru.py
```

This will:
- Process the POP909 dataset
- Train the GRU model for 20 epochs
- Save the trained model to `algorhythms_app/models/saved_models/gru_pop909.pth`
- Generate a training loss plot

### Step 2: Generating Melodies

After training, you can generate new melodies:

1. Run the generation script:
```bash
python3 algorhythms_app/models/generate_with_gru.py
```

This will:
- Load the trained model
- Generate a new melody
- Save it as a MIDI file in `algorhythms_app/data/generated/`

### Generation Parameters

You can modify these parameters in `generate_with_gru.py`:
- `melody_length`: Number of notes to generate (default: 64)
- `temperature`: Controls randomness (0.8 is default, lower = more conservative, higher = more random)
- `use_random_seed`: Whether to use a random seed or a real sequence from the dataset

The generated MIDI files will be saved with names like `gru_generated_8.mid` (where 8 represents the temperature × 10).

### Important Notes for WSL:
1. Make sure you have the POP909 dataset in the `algorhythms_app/POP909` directory
2. The model will automatically use GPU if available in WSL, otherwise it will use CPU
3. To play the generated MIDI files from Windows:
   - The files will be accessible in your WSL filesystem at `\\wsl$\Ubuntu\home\[username]\AlgoRhythms-Applied-Project\algorhythms_app\data\generated\`
   - You can copy them to your Windows system using:
     ```bash
     cp algorhythms_app/data/generated/*.mid /mnt/c/Users/[your_windows_username]/Music/
     ```
4. For MIDI playback on Windows, you can use:
   - Windows Media Player
   - VLC Media Player
   - Any DAW (Digital Audio Workstation) software

## Features
- Generate musical melodies using genetic algorithms
- Rate and evolve melodies based on user feedback
- Select different instruments for melody generation
- Interactive web interface

## Development
- The application uses Flask for the backend
- Frontend is built with HTML, CSS, and JavaScript
- Uses PyTorch for machine learning components
- MIDI processing with pretty_midi and mido

## Contributing
Feel free to submit issues and enhancement requests!

## 🚀 Quick Start

### Install dependencies

pip install -r requirements.txt

### Run the Flask Server

python app.py

### Frontend Setup

cd algorhythms_app/algorhythms-frontend

npm install

npm run serve

## 🧪 Testing

Run various tests to verify the setup:

### Test data pipeline 

python -m algorhythms_app.utils.test_data_pipeline

### Test visualizations

python -m algorhythms_app.utils.test_visualizer

## 📊 Data Visualization

The project includes tools for visualizing MIDI data:
- Piano roll visualizations
- Pitch distributions
- Sequence heatmaps
- Velocity profiles

Visualizations are automatically saved to `algorhythms_app/data/visualizations/`

## Frontend Setup in WSL

1. Install Node.js and npm:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

2. Navigate to the frontend directory:
```bash
cd algorhythms_app/algorhythms-frontend
```

3. Install dependencies and run the development server:
```bash
npm install
npm run serve
```

The frontend will be accessible from your Windows browser at the URL shown in the terminal.
