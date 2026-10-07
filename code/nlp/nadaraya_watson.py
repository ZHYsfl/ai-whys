import torch
from torch import nn
from d2l import torch as d2l

if __name__ == "__main__":
    n_train = 50
    x_train, _ = torch.sort(torch.rand(n_train) * 5)
    # print(x_train.shape) # torch.Size([50])

    def f(x):
        return 2 * torch.sin(x) + x ** 0.8

    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
    x_test = torch.arange(0, 5, 0.1)
    y_truth = f(x_test)
    n_test = len(x_test)
    
    X_repeat = x_test.repeat_interleave(n_train).reshape(-1, n_train) # (n_test, n_train)
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1) # (n_test, n_train),第i行第j列表示x_test[i]和x_train[j]的注意力权重
    # see pics/nadaraya_watson_attention_weights.png
    y_hat = torch.matmul(attention_weights, y_train) # (n_test, n_train) @ (n_train, ) → (n_test, )
    print(y_hat.shape) # torch.Size([50])