import torch
from torch import nn
from torch.nn import functional as F
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
from lenet import train_v2
from d2l import torch as d2l

class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1_1(x)) # batch_size, c1, (h-1+2*0)/1+1=h, (w-1+2*0)/1+1=w
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x)))) # batch_size, c2[1], (h-3+2*1)/1+1=h, (w-3+2*1)/1+1=w
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x)))) # batch_size, c3[1], (h-5+2*2)/1+1=h, (w-5+2*2)/1+1=w
        p4 = F.relu(self.p4_2(self.p4_1(x))) # batch_size, c4, (h-3+2*1)/1+1=h, (w-3+2*1)/1+1=w
        return torch.cat((p1,p2,p3,p4),dim=1) # batch_size, c1+c2[1]+c3[1]+c4, h, w

if __name__ == '__main__':
    b1 = nn.Sequential(
        nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),nn.ReLU(), # batch_size, 64, (96-7+3*2)/2+1=48, (96-7+3*2)/2+1=48
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # batch_size, 64, (48-3+2*1)/2+1=24, (48-3+2*1)/2+1=24
    )

    b2 = nn.Sequential(
        nn.Conv2d(64,64,kernel_size=1),nn.ReLU(), # batch_size, 64, (24-1+2*0)/1+1=24, (24-1+2*0)/1+1=24
        nn.Conv2d(64,192,kernel_size=3,padding=1),nn.ReLU(), # batch_size, 192, (24-3+2*1)/1+1=24, (24-3+2*1)/1+1=24
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # batch_size, 192, (24-3+2*1)/2+1=12, (24-3+2*1)/2+1=12
    )

    b3 = nn.Sequential(
        Inception(192,64,(96,128),(16,32),32), # batch_size, 256, 12, 12
        Inception(256,128,(128,192),(32,96),64), # batch_size, 480, 12, 12
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # batch_size, 480, (12-3+2*1)/2+1=6, (12-3+2*1)/2+1=6
    )

    b4 = nn.Sequential(
        Inception(480,192,(96,208),(16,48),64), # batch_size, 512, 6, 6
        Inception(512,160,(112,224),(24,64),64), # batch_size, 512, 6, 6
        Inception(512,128,(128,256),(24,64),64), # batch_size, 512, 6, 6
        Inception(512,112,(144,288),(32,64),64), # batch_size, 528, 6, 6
        Inception(528,256,(160,320),(32,128),128), # batch_size, 832, 6, 6
        nn.MaxPool2d(kernel_size=3,stride=2,padding=1) # batch_size, 832, (6-3+2*1)/2+1=3, (6-3+2*1)/2+1=3
    )

    b5 = nn.Sequential(
        Inception(832,256,(160,320),(32,128),128), # batch_size, 832, 3, 3
        Inception(832,384,(192,384),(48,128),128), # batch_size, 1024, 3, 3
        nn.AdaptiveAvgPool2d((1,1)), # batch_size, 1024, 1, 1
        nn.Flatten()  # batch_size, 1024
    )

    net = nn.Sequential(
        b1,b2,b3,b4,b5,
        nn.Linear(1024,10) # batch_size, 10
    )

    X = torch.rand(1,1,96,96)

    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__,'output shape:\t',X.shape)

    lr, num_epochs, batch_size = 0.1, 10, 128

    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    train_v2(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())