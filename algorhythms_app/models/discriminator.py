import torch
import torch.nn as nn

class MusicDiscriminator(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(MusicDiscriminator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Calculate the size after convolutions
        conv_out_size = sequence_length // 4 * 128
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input shape: [batch_size, sequence_length, input_dim]
        x = x.transpose(1, 2)  # Convert to [batch_size, input_dim, sequence_length]
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.transpose(1, 2)  # Convert back for LSTM
        
        lstm_out, _ = self.lstm(x)
        
        # Use last output from LSTM
        x = lstm_out[:, -1, :]
        
        # Final classification
        output = self.fc(x)
        
        return output
