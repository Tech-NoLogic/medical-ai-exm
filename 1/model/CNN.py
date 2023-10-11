import torch as th
from torch import nn

# LeNet
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        
        self.net = nn.Sequential(
            # [1,28,28]
            
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
            nn.ReLU(),
            # [6,24,24]
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            # [6,12,12]
            
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            # [16,8,8]
            
            nn.AvgPool2d(kernel_size=2, stride=2),
            # [16,4,4]
            
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, out_channels)
        )
        
    def forward(self, x):
        return self.net(x)
        
        