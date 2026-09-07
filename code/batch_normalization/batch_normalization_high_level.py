import torch
from torch import nn
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn'))
from lenet import train_v2
from d2l import torch as d2l

if __name__ == '__main__':
    net = torch.nn.Sequential(
        # nn.BatchNorm2d(1),
        nn.Conv2d(1, 6, kernel_size=5),
        nn.BatchNorm2d(6), 
        nn.Sigmoid(), 
        nn.MaxPool2d(kernel_size=2, stride=2), 
        # nn.BatchNorm2d(6),
        nn.Conv2d(6, 16, kernel_size=5), 
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        # nn.BatchNorm1d(16 * 4 * 4),
        nn.Linear(16 * 4 * 4, 120), 
        nn.BatchNorm1d(120),
        nn.Sigmoid(),
        # nn.BatchNorm1d(120),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_v2(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())