import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_nodes=128, hidden_layers=2):
        super(DQN, self).__init__()
        # CNN feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, *state_dim)
            cnn_out_size = self.conv(dummy).shape[1]
        
        # Fully connected layers
        fc_layers = []
        fc_layers.append(nn.Linear(cnn_out_size, hidden_nodes))
        fc_layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            fc_layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            fc_layers.append(nn.ReLU())
        
        fc_layers.append(nn.Linear(hidden_nodes, action_dim))
        self.fc = nn.Sequential(*fc_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        features = self.conv(x)
        return self.fc(features)