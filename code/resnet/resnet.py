import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cnn'))
from lenet import train_v2

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return nn.Sequential(*blk)

if __name__ == "__main__":
    # blk = Residual(3, 3)
    # X = torch.rand(4, 3, 6, 6)
    # Y = blk(X)
    # print(Y.shape)

    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), # batch_size, 64, (224-7+2*3)/2+1=112, 112
        nn.BatchNorm2d(64), # batch_size, 64, 112, 112
        nn.ReLU(), # batch_size, 64, 112, 112
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # batch_size, 64, (112-3+2*1)/2+1=56, 56
    )

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True)) # batch_size, 64, 56, 56
    b3 = nn.Sequential(*resnet_block(64, 128, 2)) # batch_size, 128, (56-3+2*1)/2+1=28, 28
    b4 = nn.Sequential(*resnet_block(128, 256, 2)) # batch_size, 256, (28-3+2*1)/2+1=14, 14
    b5 = nn.Sequential(*resnet_block(256, 512, 2)) # batch_size, 512, (14-3+2*1)/2+1=7, 7

    net = nn.Sequential(
        b1,b2,b3,b4,b5, # batch_size, 512, 7, 7
        nn.AdaptiveAvgPool2d((1,1)), # batch_size, 512, 1, 1
        nn.Flatten(), # batch_size, 512
        nn.Linear(512,10) # batch_size, 10
    )

    X = torch.rand(1, 1, 224, 224)
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.02, 10, 256

    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_v2(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())