import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        """
        Simple GRU (Gated Recurrent Unit) model
        
        Args:
            input_size: Number of expected features in the input
            hidden_size: Number of features in the hidden state
            output_size: Number of output features (e.g., number of possible MIDI notes)
            num_layers: Number of recurrent layers
            dropout: Dropout probability (applies when num_layers > 1)
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Optional: add more layers as needed
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)
        
        # Set device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model will use device: {self.device}")
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Initial hidden state
            
        Returns:
            output: Output tensor after passing through the model
            hidden: Final hidden state
        """
        # Move input data to device
        x = x.to(self.device)
        
        # Initialize hidden state with zeros if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        else:
            hidden = hidden.to(self.device)
            
        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)
        
        # Reshape for dense layer (consider only the last output)
        gru_out = gru_out[:, -1, :]
        
        # Apply dropout
        gru_out = self.dropout(gru_out)
        
        # Dense layer
        output = self.fc(gru_out)
        
        # Apply log softmax for classification tasks
        output = self.softmax(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        """
        Initialize hidden state with zeros
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Hidden state tensor initialized with zeros
        """
        weight = next(self.parameters()).data
        return weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
        
    def to_device(self):
        """Move the entire model to the configured device"""
        self.to(self.device)
        return self 