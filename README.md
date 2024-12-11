# AlgoRhythms Project

A machine learning-based music generation project using LSTM and GAN architectures.

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
