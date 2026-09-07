import torch
from torch import nn
from d2l import torch as d2l
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
from lenet import train_v2

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

if __name__ == "__main__":
    net = nn.Sequential(
        nin_block(1, 96, kernel_size=11, strides=4, padding=0), # batch_size,96,(224-11+0)/4+1=54,(224-11+0)/4+1=54
        nn.MaxPool2d(kernel_size=3, stride=2), # batch_size,96,(54-3)/2+1=26,(54-3)/2+1=26,(54-3)/2+1=26
        nin_block(96, 256, kernel_size=5, strides=1, padding=2), # batch_size,256,(26-5+2*2)/1+1=26,(26-5+2*2)/1+1=26
        nn.MaxPool2d(kernel_size=3, stride=2), # batch_size,256,(26-3)/2+1=12,(26-3)/2+1=12
        nin_block(256, 384, kernel_size=3, strides=1, padding=1), # batch_size,384,(12-3+2*1)/1+1=12,(12-3+2*1)/1+1=12
        nn.MaxPool2d(kernel_size=3, stride=2), # batch_size,384,(12-3)/2+1=5,(12-3)/2+1=5
        nn.Dropout(0.5), 
        nin_block(384, 10, kernel_size=3, strides=1, padding=1), # batch_size,10,(5-3+2*1)/1+1=5,(5-3+2*1)/1+1=5
        nn.AdaptiveAvgPool2d((1, 1)), # batch_size,10,1,1
        nn.Flatten() # batch_size,10
    )
    X = torch.rand(size=(1, 1, 224, 224))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, 'output shape:\t', X.shape)

    lr,num_epochs,batch_size = 0.1, 10, 128
    train_iter,test_iter = load_data_fashion_mnist(batch_size=batch_size,resize=224)
    train_v2(net, train_iter,test_iter,num_epochs,lr,d2l.try_gpu())