import torch
from d2l import torch as d2l
from seqdataloader import load_data_time_machine
from rnn_low_level import train_v3
from rnn_high_level import RNNModel

if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
    num_inputs = num_outputs = vocab_size
    num_layers = 2
    lstm_layer = torch.nn.LSTM(input_size=num_inputs, hidden_size=num_hiddens, num_layers=num_layers, bidirectional=True)
    model = RNNModel(lstm_layer, vocab_size)
    model = model.to(device)
    num_epochs, lr = 500, 1
    train_v3(model, train_iter, vocab, lr, num_epochs, device)