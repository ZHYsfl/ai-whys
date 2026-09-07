import torch
from torch import nn

class Reshape(torch.nn.Module):
    def forward(self, X):
        return X.view(-1, 1, 28, 28)

net = torch.nn.Sequential(
    Reshape(), # batch_size, 1, 28, 28
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), # batch_size, 6, (28+2*2-5)/1+1 = 28, (28+2*2-5)/1+1 = 28
    nn.AvgPool2d(kernel_size=2, stride=2), # batch_size, 6, (28-2)/2+1 = 14, (28-2)/2+1 = 14
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), # batch_size, 16, (14-5)/1+1 = 10, (14-5)/1+1 = 10
    nn.AvgPool2d(kernel_size=2, stride=2), # batch_size, 16, (10-2)/2+1 = 5, (10-2)/2+1 = 5
    nn.Flatten(), # batch_size, 16 * 5 * 5
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

