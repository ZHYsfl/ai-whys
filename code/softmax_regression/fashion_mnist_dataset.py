import os
from time import time
from d2l import torch as d2l
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor() # use ToTensor to convert images data from PIL format to tensor format,and normalize the pixel values to [0,1](divided by 255)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')

mnist_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=trans,download=True)
mnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=trans,download=True)

# print(len(mnist_train), len(mnist_test)) # 60000 10000

# print(mnist_train[0][0].shape, mnist_test[0][0].shape) # torch.Size([1, 28, 28]) torch.Size([1, 28, 28])

# X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))

batch_size = 256

def get_dataloader_workers():
    """use 4 processes to read data"""
    return 4

def main():
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

    # warm up one round to exclude the time of starting workers
    for X, y in train_iter:
        break

    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    transforms_list = [transforms.ToTensor()]
    if resize:
        transforms_list.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(transforms_list)
    mnist_train = torchvision.datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True)
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers())
    return train_iter, test_iter

if __name__ == '__main__':
    main()
