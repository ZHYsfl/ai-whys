import torch
from torch import nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'softmax_regression'))
from fashion_mnist_dataset import load_data_fashion_mnist
from softmax_low_level import Accumulator,accuracy
from d2l import torch as d2l

class Reshape(torch.nn.Module):
    def forward(self, X):
        return X.view(-1, 1, 28, 28)

def evaluate_accuracy_gpu(net,data_iter,device=None):
    if isinstance(net,torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]

def train_v2(net, train_iter, test_iter, num_epochs, lr, device):
    '''train a model with a GPU'''
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss() # do softmax and cross entropy loss in one step, the reduction is mean by default
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = Accumulator(3) # # total_loss, num_correct_predictions, num_total_predictions
        net.train() # set the model to training mode
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'epoch {epoch + 1}, step {i + 1}, loss {train_l:.4f}, train acc {train_acc:.3f}')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
    print(f'epoch {epoch + 1}, loss {metric[0] / metric[2]:.4f}, train acc {metric[1] / metric[2]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

if __name__ == '__main__':

    net = torch.nn.Sequential(
        Reshape(), # batch_size, 1, 28, 28
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(), # batch_size, 6, (28+2*2-5)/1+1 = 28, (28+2*2-5)/1+1 = 28
        nn.AvgPool2d(kernel_size=2, stride=2), # batch_size, 6, (28-2)/2+1 = 14, (28-2)/2+1 = 14
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(), # batch_size, 16, (14-5)/1+1 = 10, (14-5)/1+1 = 10
        nn.AvgPool2d(kernel_size=2, stride=2), # batch_size, 16, (10-2)/2+1 = 5, (10-2)/2+1 = 5
        nn.Flatten(), # batch_size, 16 * 5 * 5
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(), # batch_size, 120
        nn.Linear(120, 84), nn.Sigmoid(), # batch_size, 84
        nn.Linear(84, 10) # batch_size, 10
    )
        
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)

    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    lr, num_epochs = 0.9, 10
    train_v2(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())