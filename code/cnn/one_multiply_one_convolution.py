import torch
from torch import nn
from multi_input_and_output_channels import corr2d_multi_in_out

def corr2d_multi_in_out_1x1(X, K):
    """compute 2D cross-correlation operation on multi-input and multi-output channels with 1x1 kernel."""
    c_i, h, w = X.shape
    c_o, c_i_, kh, kw = K.shape
    assert c_i == c_i_, "Input channels must match kernel input channels."
    assert kh == 1 and kw == 1, "Kernel size must be 1x1."
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w)) 

X = torch.normal(0,1,(3, 3, 3)) # 3 input channels, each of size 3x3
K = torch.normal(0,1,(2, 3, 1, 1)) # 2 output channels, each with 3 input channels and kernel size 1x1
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
# Using nn.Linear to perform the same operation: each spatial position is a "sample"
# whose features are the c_i channel values at that position (channel-last view).
linear = nn.Linear(3, 2)
with torch.no_grad():
    linear.weight.copy_(K.reshape(2, 3))
    linear.bias.zero_() # the conv version above has no bias
Y3 = linear(X.permute(1, 2, 0).reshape(3 * 3, 3)) # (h*w, c_i) -> (h*w, c_o)
Y3 = Y3.reshape(3, 3, 2).permute(2, 0, 1) # back to (c_o, h, w)

print(Y1)
print(Y2)
print(Y3)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6, "The results from the two methods should be the same."
assert float(torch.abs(Y1 - Y3).sum().detach()) < 1e-6, "The Linear version should match the 1x1 convolution."