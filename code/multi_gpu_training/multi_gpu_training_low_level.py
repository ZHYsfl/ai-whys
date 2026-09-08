import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

def allreduce(data):
    for i in range(1,len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1,len(data)):
        data[i] = data[0].to(data[i].device)

def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_batch(X, y, device_params, devices, lr, loss): # 2. get the current weight
    X_shards, y_shards = split_batch(X, y, devices) # 1. read the data block
    # 在每个GPU上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    for l in ls:  # 反向传播在每个GPU上分别执行
        l.backward() # 3. compute the grad
    # 将每个GPU的所有梯度相加，并将其广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))]) # 4. send the grad back
    # 5. update the weight
    # 在每个GPU上分别更新模型参数
    for param in device_params:
        sgd(param, lr, X.shape[0]) # 在这里，我们使用全尺寸的小批量 

def train(num_gpus, batch_size, lr, params, loss):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到num_gpus个GPU
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            # 为单个小批量执行多GPU训练
            train_batch(X, y, device_params, devices, lr, loss)
            torch.cuda.synchronize() # 让 CPU 停下来，等 GPU 把队列里所有任务真正算完再走    
        timer.stop()
        # 在GPU0上评估模型
        print('epoch ', epoch + 1, 'test acuracy: ', d2l.evaluate_accuracy_gpu(lambda X: lenet(X, device_params[0]), test_iter, devices[0]))
    print(f'{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')

if __name__ == "__main__":
    scale = 0.01
    W1 = torch.randn((20, 1, 3, 3)) * scale
    b1 = torch.zeros(20)
    W2 = torch.randn((50, 20, 5, 5)) * scale
    b2 = torch.zeros(50)
    W3 = torch.randn(size = (800, 128)) * scale
    b3 = torch.zeros(128)   
    W4 = torch.randn(size = (128, 10)) * scale
    b4 = torch.zeros(10)
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    loss = nn.CrossEntropyLoss(reduction='none')

    # new_params = get_params(params, d2l.try_gpu(0))
    # print('b1 weight:',new_params[1])
    # print('b1 grad:',new_params[1].grad)

    # data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
    # print('allreduce之前：\n', data[0], '\n', data[1])
    # allreduce(data)
    # print('allreduce之后：\n', data[0], '\n', data[1])

    # data = torch.arange(20).reshape(4, 5)
    # devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    # split = nn.parallel.scatter(data, devices)

    # print('input:',data)
    # print('load info',devices)
    # print('output:',split)

    train(num_gpus=1, batch_size=256, lr=0.1,params=params,loss = loss)
    # train(num_gpus=2, batch_size=256*2, lr=0.2,params=params,loss = loss)