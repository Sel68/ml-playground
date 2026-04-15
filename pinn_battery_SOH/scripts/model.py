import torch
import torch.nn as nn

class BatteryNaturePINN(nn.Module):
    """
    PINN architecture based on Nature 2024 paper.
    Inputs: 16 features + 1 cycle (all normalized to [-1, 1])
    """
    def __init__(self, input_dim=17, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # SOH is typically 0-1
        )
        
    def forward(self, x):
        return self.net(x)
