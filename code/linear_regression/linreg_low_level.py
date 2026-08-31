'''
the complementary code for linear regression using PyTorch.
It includes functions to generate synthetic data, iterate over batches, define the linear regression model, compute the loss, and perform stochastic gradient descent. The code also trains the model and prints the estimated parameters compared to the true parameters.
'''
import random
import torch

def synthetic_data(w,b,num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# print("Features:", features)
# print("Labels:", labels)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # Shuffle the indices
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]

batch_size = 10

# for X,y in data_iter(batch_size, features, labels):
#     print(X, y)
#     break

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    """Linear regression model."""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y, batch_size):
    """Mean squared error loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 / batch_size

def sgd(params, lr):
    """Mini-batch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y, batch_size)
        l.sum().backward()
        sgd([w, b], lr)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels, len(labels))
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'True w - Estimated w: {true_w - w.reshape(true_w.shape)}')
print(f'True b - Estimated b: {true_b - b}')