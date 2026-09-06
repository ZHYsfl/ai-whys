import torch
from torch import nn

def comp_conv2d(conv2d, X):
    """Compute 2D convolution."""
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
# Y = comp_conv2d(conv2d, X)
# print(Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(5,3), padding=(2,1))
# Y = comp_conv2d(conv2d, X)
# print(Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
# Y = comp_conv2d(conv2d, X)
# print(Y.shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3,5), padding=(0,1), stride=(3,4))
# Y = comp_conv2d(conv2d, X)
# print(Y.shape)