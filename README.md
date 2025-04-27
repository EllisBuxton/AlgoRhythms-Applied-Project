# 🎵 AlgoRhythms

AlgoRhythms is a music generation application that uses machine learning to create unique musical compositions. The project uses a GRU (Gated Recurrent Unit) neural network model to generate music sequences, trained on MIDI data to learn musical patterns and structures. The GRU model's ability to capture long-term dependencies makes it particularly effective for music generation tasks. The project consists of a Flask backend for music generation and a Vue.js frontend for user interaction.

## 🚀 Setup and Running

### 🔧 Backend Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Flask backend:
```bash
python app.py
```

### 🎨 Frontend Setup
1. Navigate to the frontend directory:
```bash
cd algorhythms_app
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run serve
```

The application will be available at:
- Frontend: http://localhost:8080
- Backend: http://localhost:5000

## 🎼 Training and Generation

### 🏋️ Training the Model
To train the GRU model on your MIDI dataset:
```bash
python train_model.py --data_path path/to/your/midi/files --epochs 100 --batch_size 32
```

Parameters:
- `--data_path`: Path to directory containing MIDI files
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Size of training batches (default: 32)

### 🎹 Generating Music
To generate new music using the trained model:
```bash
python generate_music.py --model_path path/to/trained/model --output_path path/to/save/midi --length 100
```

Parameters:
- `--model_path`: Path to the trained model file
- `--output_path`: Where to save the generated MIDI file
- `--length`: Number of steps to generate (default: 100)
