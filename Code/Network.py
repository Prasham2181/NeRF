import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NeRFNetwork(nn.Module):
    """
    NeRFNetwork implements the NeRF MLP architecture as described in 
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" by Mildenhall et al.
    """
    def __init__(self, Freq_L=10, direction_L=4, hidden_layer1=256, hidden_layer2=128):
        super(NeRFNetwork, self).__init__()
        self.Freq_L = Freq_L
        self.direction_L = direction_L

        # Calculate dimensions after positional encoding:
        self.pos_dim = 3 + 3 * 2 * Freq_L  
        self.dir_dim = 3 + 3 * 2 * direction_L  

        # Input layer for position
        self.input_layer = nn.Linear(self.pos_dim, hidden_layer1)  

        # First block of hidden layers
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_layer1, hidden_layer1), nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer1), nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer1), nn.ReLU()
        )

        # Skip connection
        self.skip_layer = nn.Linear(hidden_layer1 + self.pos_dim, hidden_layer1)

        # Second block of hidden layers
        self.hidden_layers_2 = nn.Sequential(
            nn.Linear(hidden_layer1, hidden_layer1), nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer1), nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer1), nn.ReLU()
        )

        # Density output
        self.sigma_layer = nn.Linear(hidden_layer1, 1)

        # Feature layer for color prediction
        self.feature_layer = nn.Linear(hidden_layer1, hidden_layer1)

        # Layers for processing direction (for RGB)
        self.color_layer_1 = nn.Linear(hidden_layer1 + self.dir_dim, hidden_layer2)
        self.color_layer_2 = nn.Linear(hidden_layer2, 3)

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights for the network. Setting the sigma layer bias
        to a negative value helps avoid high initial density.
        """
        nn.init.constant_(self.sigma_layer.bias, -1.0)

    def positional_encoding(self, x, L):
        """
        Applies positional encoding to the input tensor x (assumed in [-1, 1]).
        """
        out = [x]
        for i in range(L):
            out.append(torch.sin((2**i * np.pi) * x))
            out.append(torch.cos((2**i * np.pi) * x))
        return torch.cat(out, dim=-1)

    def forward(self, pos, direction):
        # Positional encoding for both position and direction.
        encoded_pos = self.positional_encoding(pos, self.Freq_L)
        encoded_dir = self.positional_encoding(direction, self.direction_L)

        # Process position
        x = self.input_layer(encoded_pos)
        x = self.relu(x)
        x = self.hidden_layers(x)

        # Skip connection
        x = torch.cat([x, encoded_pos], dim=-1)
        x = self.skip_layer(x)
        x = self.relu(x)

        # Additional hidden layers
        x = self.hidden_layers_2(x)

        # Density
        sigma = self.sigma_layer(x).squeeze(-1)
        sigma = self.relu(sigma)

        # Features for color
        features = self.feature_layer(x)

        # Combine with direction
        color_input = torch.cat([features, encoded_dir], dim=-1)
        color = self.color_layer_1(color_input)
        color = self.relu(color)
        color = self.color_layer_2(color)

        # Tanh output for color in [-1,1]
        rgb = self.tanh(color)

        return rgb, sigma
