import torch
import torch.nn as nn

class MusicGenerator(nn.Module):
    def __init__(self, latent_dim, sequence_length, output_dim):
        super(MusicGenerator, self).__init__()
        
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        # Initial dense layer to reshape noise
        self.fc1 = nn.Linear(latent_dim, sequence_length * 128)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(512, 256),  # 512 due to bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        # Reshape noise vector
        x = self.fc1(z)
        x = x.view(-1, self.sequence_length, 128)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Generate output
        output = self.fc_out(lstm_out)
        
        return output
