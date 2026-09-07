import torch
from torch import nn
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn'))
from lenet import train_v2
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled(): # inference mode, use the moving mean and variance
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else: # training mode, use the mini-batch mean and variance
        assert len(X.shape) in (2, 4)  # X must be a fully-connected layer or a convolutional layer
        if len(X.shape) == 2: # X is a fully-connected layer, compute mean and variance along the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # X is a convolutional layer, compute mean and variance along the channel dimension
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)

        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y

if __name__ == '__main__':
    net = torch.nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),
        BatchNorm(6, num_dims=4), 
        nn.Sigmoid(), 
        nn.MaxPool2d(kernel_size=2, stride=2), 
        nn.Conv2d(6, 16, kernel_size=5), 
        BatchNorm(16, num_dims=4),
        nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120), 
        BatchNorm(120, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        BatchNorm(84, num_dims=2),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_v2(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())