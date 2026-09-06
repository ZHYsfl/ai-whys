import torch
from torch import nn

def corr2d(X,K):
    '''compute the cross-correlation of 2D matrices'''
    h, w = K.shape
    assert X.shape[0] >= h and X.shape[1] >= w, "Kernel size must be equal to or smaller than input size"
    Y=torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
    return Y

class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,X):
        return corr2d(X,self.weight)+self.bias


if __name__ == '__main__':
    # X = torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
    # K = torch.tensor([[0.0,1.0],[2.0,3.0]])
    # print(corr2d(X,K))


    # edge detection example 
    X = torch.ones((6,8))
    X[:,2:6] = 0
    K = torch.tensor([[1.0,-1.0]])
    Y = corr2d(X,K)
    # print(Y)
    # Y_ = corr2d(X.t(),K)
    # print(Y_)

    conv2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)

    X = X.reshape((1,1,6,8)) # N, C, H, W
    Y = Y.reshape((1,1,6,7))

    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat-Y)**2
        conv2d.zero_grad()
        l.sum().backward()
        conv2d.weight.data[:] -= 3e-2*conv2d.weight.grad
        if (i+1)%2==0:
            print(f'batch {i+1}, loss {l.sum():.3f}')

    print(conv2d.weight.data.reshape((1,2)))