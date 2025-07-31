import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        return self.layers[-1](x)