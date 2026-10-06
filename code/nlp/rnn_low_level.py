import math
import torch
from torch import device, nn
from torch.nn import functional as F
from seqdataloader import load_data_time_machine
from d2l import torch as d2l

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)

    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )

def rnn(inputs, state, params):
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs: # 按照时间维度遍历
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q # Y's shape : (batch_size, vocab_size) 
        outputs.append(Y)
    # 返回的 state 包成单元素元组 (H,)，是为了和 LSTM 的 (H, C) 保持统一接口，
    # 使训练循环无需关心具体是哪种 RNN
    return torch.cat(outputs, dim=0), (H, ) # return shape : (num_steps * batch_size, vocab_size), (batch_size, num_hiddens)

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params, init_rnn_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_rnn_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T.long(), self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_v1(prefix, num_preds, net, vocab, device):
    """在 prefix 后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测 num_preds 步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params)) # 可以理解成整个net的所有参数的梯度排成一个1D向量求他的模长    
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_v2(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期

    顺序分区模式下跨 batch 传递 state 包含两层近似（牺牲数学完备性换工程可行性）：
    近似一（截断BPTT）：detach 剪断计算图，梯度只回溯 num_steps 步而非整个语料历史，
        换取恒定的显存和计算开销（理论上梯度有偏，见 Tallec & Ollivier 2017）
    近似二（state过时）：state 是用更新前的参数算出的，参数更新后并未用新参数重算，
        因单步更新很小，此误差可忽略
    """
    state, timer = None, 0.0
    metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
    for X, Y in train_iter:
        # 两种数据迭代方式（以语料 0~34、batch_size=2、num_steps=5 为例）：
        # 随机采样(random)：语料切成小段后打乱组 batch，相邻 batch 在原文中不连续，
        #     因此每个 batch 都必须重建 state，无法跨 batch 传递上下文
        #     batch1 X=[[10..14],[0..4]]，batch2 X=[[5..9],[25..29]]（段间无关联）
        # 顺序分区(seq)：语料切成 batch_size 条平行流，相邻 batch 的第 i 行在原文中
        #     首尾相接，因此可以跨 batch 传递 state（数值保留，梯度截断）
        #     batch1 X=[[0..4],[17..21]]，batch2 X=[[5..9],[22..26]]（每行恰好接续）
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机采样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 近似一（截断BPTT）发生在这里：剪掉梯度历史，保留 state 数值
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 如果是nn.GRU实例，state是张量
                state.detach_()
            else:
                # 否则，state是张量或元组
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        # 近似二（state过时）发生在这里：传入的 state 是上一个 batch 用【参数更新前】的
        # 权重算出来的（上一轮 updater.step() 之后并未用新参数重算），此处直接沿用
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()

        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(1) # 上面 loss 已调用 mean()，梯度已按 batch 归一化，这里不能再除 batch_size

        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1])

def train_v3(net,train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型"""
    loss = nn.CrossEntropyLoss()
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr=lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_v1(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl = train_epoch_v2(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
    print(f'困惑度 {ppl:.1f}, {predict("time traveller")}')
    print(predict('traveller'))

if __name__ == "__main__":

    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)

    print(F.one_hot(torch.tensor([0, 2]), len(vocab)))

    X = torch.arange(10).reshape((2, 5))
    print(F.one_hot(X.T,len(vocab)).shape) # 5, 2, 28

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, device, get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], device)
    Y, new_state = net(X.to(device), state)
    print(Y.shape, len(new_state), new_state[0].shape) # 10, 28 ; 1 ; 2, 512

    print(predict_v1('time traveller ', 10, net, vocab, device))

    train_v3(net, train_iter, vocab, lr=1, num_epochs=500, device=device)