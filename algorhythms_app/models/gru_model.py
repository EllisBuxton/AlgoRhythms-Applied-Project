import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
        """
        Enhanced GRU (Gated Recurrent Unit) model with multiple layers and attention
        
        Args:
            input_size: Number of expected features in the input (notes + chords)
            hidden_size: Number of features in the hidden state
            output_size: Number of output features (e.g., number of possible MIDI notes)
            num_layers: Number of recurrent layers
            dropout: Dropout probability
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input layer with batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # GRU layers with residual connections
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0
            ) for i in range(num_layers)
        ])
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(num_layers)
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        # Set device for model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Model will use device: {self.device}")
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the network with attention mechanism
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            hidden: Initial hidden state
            
        Returns:
            output: Output tensor after passing through the model
            hidden: Final hidden state
        """
        # Move input data to device
        x = x.to(self.device)
        
        # Apply input batch normalization
        batch_size, seq_length, input_size = x.size()
        x = x.view(-1, input_size)
        x = self.input_bn(x)
        x = x.view(batch_size, seq_length, input_size)
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = [self.init_hidden(batch_size) for _ in range(self.num_layers)]
        else:
            hidden = [h.to(self.device) for h in hidden]
        
        # Process through GRU layers with residual connections
        current_input = x
        new_hidden = []
        
        for i, (gru, bn) in enumerate(zip(self.gru_layers, self.bn_layers)):
            # GRU forward pass
            gru_out, h = gru(current_input, hidden[i])
            
            # Apply batch normalization
            gru_out = gru_out.permute(0, 2, 1)  # (batch, hidden, seq)
            gru_out = bn(gru_out)
            gru_out = gru_out.permute(0, 2, 1)  # (batch, seq, hidden)
            
            # Add residual connection if not first layer
            if i > 0:
                gru_out = gru_out + current_input
                
            current_input = gru_out
            new_hidden.append(h)
        
        # Apply attention mechanism
        attention_weights = self.attention(current_input)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(current_input * attention_weights, dim=1)  # (batch, hidden)
        
        # Apply dropout
        attended = self.dropout(attended)
        
        # Final output layers
        output = self.fc1(attended)
        output = torch.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        
        return output, new_hidden
    
    def init_hidden(self, batch_size):
        """Initialize hidden state with zeros"""
        weight = next(self.parameters()).data
        return weight.new(1, batch_size, self.hidden_size).zero_().to(self.device)
        
    def to_device(self):
        """Move the entire model to the configured device"""
        self.to(self.device)
        return self 