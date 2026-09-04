import torch
from fashion_mnist_dataset import load_data_fashion_mnist

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # here we broadcast the division operation

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# firstly understand the meaning of y_hat[[0, 1], y]
# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# print(y_hat[[0, 1], y])

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])  # y_hat[range(len(y_hat)), y] is the predicted probability of the correct class

# print(cross_entropy(y_hat,y))

def accuracy(y_hat, y):
    # the number of correct predictions / the total number of predictions
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # get the index of the max value in each row
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

# print(accuracy(y_hat, y) / len(y))  # the accuracy of the model on the given batch

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()  # set the model to evaluation mode
    metric = Accumulator(2)  # num_correct_predictions, num_total_predictions
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_v1(net, train_iter, loss, updater):
    """Train a model within one epoch."""
    if isinstance(net, torch.nn.Module):
        net.train()  # set the model to training mode
    metric = Accumulator(3)  # total_loss, num_correct_predictions, num_total_predictions
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        # Using PyTorch built-in optimizer & loss criterion
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    return metric[0] / metric[2], metric[1] / metric[2]

def train_v1(net, train_iter, test_iter, loss, num_epochs, updater):
    """Train a model."""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_v1(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch {epoch + 1}, loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')

lr = 0.1
num_epochs = 10

if __name__ == '__main__':
    print(evaluate_accuracy(net, test_iter))  # evaluate the accuracy of the model on the test set
    train_v1(net, train_iter, test_iter, cross_entropy, num_epochs, torch.optim.SGD([W, b], lr=lr))  # train the model