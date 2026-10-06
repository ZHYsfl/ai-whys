import math
import torch
from torch import device, nn
from torch.nn import functional as F
from seqdataloader import load_data_time_machine
from d2l import torch as d2l
from rnn_low_level import predict_v1, train_v3

class RNNModel(nn.Module):
    '''循环神经网络模型'''
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元个数)
        # 它的输出形状是(时间步数*批量大小,词表大小)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.RNN和nn.GRU的隐藏状态是一个张量
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            # nn.LSTM的隐藏状态是一个元组
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens), device=device))

if __name__ == '__main__':
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    state = torch.zeros((1, batch_size, num_hiddens))

    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state) # Y 是**隐藏层**的输出
    print(Y.shape, state_new.shape)  # torch.Size([35, 32, 256]) torch.Size([1, 32, 256])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = RNNModel(rnn_layer, vocab_size = len(vocab))
    net = net.to(device)
    print("Model initialized and moved to device:", device)
    print("Predicted text:", predict_v1('time traveller ', 10, net, vocab, device))

    print("Training the model...")

    num_epochs, lr = 500, 1
    train_v3(net, train_iter, vocab, lr, num_epochs, device)