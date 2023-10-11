import torch as th
from torch import nn

class DNN(nn.Module):
    def __init__(self, c, h, w, out_dim):
        super(DNN, self).__init__()
        
        self.input = nn.Flatten()
        
        self.hidden = nn.Sequential(
            nn.Linear(c * h * w, 128),
            nn.ReLU()
        )
        
        self.output = nn.Linear(128, out_dim)
        
    def forward(self, X):
        for layer in [self.input, self.hidden, self.output]:
            X = layer(X)
            
        return X