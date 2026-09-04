import torch
from torch import nn
from fashion_mnist_dataset import load_data_fashion_mnist
from softmax_low_level import train_v1

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

loss = nn.CrossEntropyLoss()  # CrossEntropyLoss combines softmax and cross-entropy loss

trainer = torch.optim.SGD(net.parameters(), lr=0.1)  # Stochastic Gradient Descent optimizer

num_epochs = 10

if __name__ == '__main__':
    train_v1(net, train_iter, test_iter, loss, num_epochs, trainer)