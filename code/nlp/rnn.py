import math
import torch
from torch import nn
from torch.nn import functional as F
from seqdataloader import load_data_time_machine

if __name__ == "__main__":

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

    X = torch.arange(10).reshape((2, 5))
    F.one_hot(X.T,28).shape