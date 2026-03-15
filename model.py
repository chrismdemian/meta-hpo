import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, DROPOUT

# Build a class for the neural network
class HPOModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 3 linear layers with dropout and relu activation
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )
    
    # Forward method
    def forward(self, x):
        return self.network(x)