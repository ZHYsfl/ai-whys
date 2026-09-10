import torch
from torch import nn
from d2l import torch as d2l

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weight)
    return net

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss {d2l.evaluate_loss(net, train_iter, loss):f}')

if __name__ == "__main__":
    T = 1000
    time = torch.arange(1, T + 1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

    # goal:
    # [x0 x1 x2 x3] -> x4
    # [x1 x2 x3 x4] -> x5
    # [x2 x3 x4 x5] -> x6
    # ...
    # [x995 x996 x997 x998] -> x999

    tau = 4
    features = torch.zeros((T - tau, tau)) # 996, 4
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))

    batch_size, n_train = 16, 600
    # 时间序列有先后顺序，不能随机打乱划分（否则等于用"未来"预测"过去"，数据泄漏）
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True) # 打乱的是样本的读取顺序，但样本内容（哪个窗口对应哪个标签）不变，所以不破坏时间序列的因果关系。

    loss = nn.MSELoss()
    net = get_net()
    train(net, train_iter, loss, 5, 0.01)

    onestep_preds = net(features)

    multistep_preds = torch.zeros(T)
    multistep_preds[:n_train + tau] = x[:n_train + tau]
    for i in range(n_train + tau, T):
        multistep_preds[i] = net(multistep_preds[i - tau:i].reshape((1, -1))) # 连续预测几百个点，会预测错的很离谱，误差在累计，连续预测的点越少，误差越小

    #  可视化to do