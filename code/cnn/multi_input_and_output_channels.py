import torch
from cross_correlation import corr2d

def corr2d_multi_in(X, K):
    """compute 2D cross-correlation operation on multi-input channels."""
    return sum(corr2d(x, k) for x, k in zip(X, K)) # sum is different from torch.sum, it is a python built-in function, which can sum up all the elements in a list.

def corr2d_multi_in_out(X, K):
    """compute 2D cross-correlation operation on multi-input and multi-output channels."""
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

if __name__ == '__main__':
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    # X is a 2x3x3 tensor, representing 2 input channels, each of size 3x3.
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    # K is a 2x2x2 tensor, representing 1 output channel.

    # print(corr2d_multi_in(X, K)) #  2x2 , representing 1 output channel.

    K = torch.stack((K, K + 1, K + 2), 0) # 3x2x2x2, representing 3 output channels.

    print(corr2d_multi_in_out(X, K)) #  3x2x2 , representing 3 output channels.