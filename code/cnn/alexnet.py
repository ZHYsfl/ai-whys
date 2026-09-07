from d2l import torch as d2l
import torch
from torch import nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
from lenet import train_v2

if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), # batch_size, 96, (224-11+2*1)/4 + 1 = 54, (224-11+2*1)/4 + 1 = 54
        nn.MaxPool2d(kernel_size=3, stride=2), # batch_size, 96, (54-3)/2 + 1 =26,  (54-3)/2 + 1 =26
        nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), # batch_size, 256, (26-5+2*2)/1 + 1 = 26, (26-5+2*2)/1 + 1 = 26
        nn.MaxPool2d(kernel_size=3, stride=2), # batch_size, 256, (26-3)/2 + 1 = 12, (26-3)/2 + 1 = 12
        nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), # batch_size, 384, (12-3+2*1)/1 + 1 = 12, (12-3+2*1)/1 + 1 = 12
        nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(), # batch_size, 384, (12-3+2*1)/1 + 1 = 12, (12-3+2*1)/1 + 1 = 12
        nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), # batch_size, 256, (12-3+2*1)/1 + 1 = 12, (12-3+2*1)/1 + 1 = 12
        nn.MaxPool2d(kernel_size=3, stride=2), # batch_size, 256, (12-3)/2 + 1 = 5, (12-3)/2 + 1 = 5
        nn.Flatten(), # batch_size, 256*5*5 = 6400
        nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5), # batch_size, 4096
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5), # batch_size, 4096 
        nn.Linear(4096, 10) # batch_size, 10
    )
    
    X = torch.rand(size=(1, 1, 224, 224)) # batch_size, channel, height, width
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    # you can see the pics/resize_demo.png to see the effect of resize

    lr, num_epochs = 0.01, 10
    train_v2(net, train_iter, test_iter,num_epochs,lr, d2l.try_gpu())