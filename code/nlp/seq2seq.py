import torch
from torch import nn
from d2l import torch as d2l
from encoder_decoder import Encoder, Decoder, EncoderDecoder
from machine_translation_dataset import load_data_nmt,truncate_pad
import math
import collections

# nn.RNN/GRU/LSTM 接口注意点（高层API，数学原理与低层手写版一致）：
# 1. 形状约定：默认输入和output都是 (num_steps, batch_size, feature)，时间维在最前；
#    传 batch_first=True 可改成 batch 在前，但输入和output会一起变，要变都变。
# 2. state 类型：RNN/GRU 的 state 是单张量，LSTM 是 (H, C) 元组——
#    低层手写版把 state 包成 (H,) 元组就是为了和 LSTM 接口统一，三种模型可对换。
# 3. state 仍需手动 detach：截断BPTT时 state.detach_() 要自己做，nn.GRU不会代劳，
#    低层版注释过的两个近似（梯度截断、state过时）在高层版照样存在。
# 4. dropout 只在层间生效：加在每层输出上（喂给下一层之前），顶层输出之后不加，
#    顶层之后要不要dropout由你自己接的层决定；num_layers=1 时该参数无效。
#    如 num_layers=3：层1输出→dropout→层2，层2输出→dropout→层3，层3输出直接返回。
# 5. 变长序列：batch内句子长短不一，必须pad补齐成矩形才能叠成张量（如"go ."补成
#    "go . <eos> <pad> <pad>"）。但GRU不知道pad是假词，会照常读入并更新状态——
#    编码器终态本该定格在读入<eos>那一刻（这才是对整句的"总结"，要传给解码器当上下文），
#    却被后面的pad步继续搅和，污染了总结。模型理论上能学会"读pad时近似不更新状态"，
#    但这是学出来的近似，白占容量且不严格。
#    规范做法：nn.utils.rnn.pack_padded_sequence 打包（按各句valid_len跑到<eos>就停），
#    pad根本不进GRU，终态严格干净。注意loss掩码是另一码事：它管的是标签侧（Y）的pad
#    （防止强迫模型学"<eos>之后输出<pad>"），管不到编码器输入侧的污染。
#    本文件按d2l教学版实现，未打包；升级点：forward接收X_valid_len（现被*args吞掉），
#    打包喂GRU，output需pad_packed_sequence解包还原，state直接就是干净的。

class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器。"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # 将词元索引转换为词向量
        X = self.embedding(X)
        # 在时间步维度上展开
        X = X.permute(1, 0, 2)
        # nn.GRU内部没有输出层（没有W_ho、不产logits，构造参数里连vocab_size都没有），
        # 它只负责隐状态更新 H_t = f(...)，返回的两部分都是H：
        # output: 顶层在每个时间步的H，(num_steps, batch_size, num_hiddens)，时间步全留、只留顶层
        # state:  每层在最后时间步的H，(num_layers, batch_size, num_hiddens)，层全留、只留末时刻
        # 交集：output[-1] == state[-1层]（dropout=0时）；单层时 state[0] == output[-1]
        # 笔记公式里的 O_t = H_t@W_ho + B_o 要另加 nn.Linear 算（见Decoder的self.dense），
        # 编码器不预测词元，所以这里没有Linear，output也没人用（留到注意力机制才有用）
        output, state = self.rnn(X)
        return output, state

class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器。"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                          num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        # 在特征维度上拼接上下文
        context = state[-1].repeat(X.shape[0], 1, 1)  # （batch_size, num_hiddens）→（num_steps, batch_size, num_hiddens）
        X_and_context = torch.cat((X, context), 2) # (num_steps, batch_size, embed_size + num_hiddens)
        # 将输入传入 RNN 中
        # output.shape is (num_steps, batch_size, num_hiddens), state.shape is (num_layers, batch_size, num_hiddens)
        output, state = self.rnn(X_and_context, state)
        # 在特征维度上变换输出
        output = self.dense(output).permute(1, 0, 2) # (batch_size, num_steps, vocab_size),Permute是因为算交叉熵损失的时候Pytorch约定类别紧贴在batch维度后面，batch在前，类别在后
        return output, state

def sequence_mask(X, valid_len, value=0): # X.shape is (batch_size, num_steps), valid_len.shape is (batch_size, num_steps)
    """在序列中屏蔽不相关的项。"""
    maxlen = X.size(1) # maxlen is num_steps
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None] # (1, maxlen) < (batch_size, 1) → (batch_size, maxlen)
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的交叉熵损失函数。"""

    def forward(self, pred, label, valid_len):
        '''
        loss的掩码是根据输出的pad来做的，和输入没关系。
        pred.shape is (batch_size, num_steps, vocab_size),模型的预测，就是解码器 dense 层的输出   
        label.shape is (batch_size, num_steps),标准答案，就是 load_data_nmt 吐出的 Y   
        valid_len.shape is (batch_size, )
        '''
        weights = torch.ones_like(label) # (batch_size, num_steps)
        weights = sequence_mask(weights, valid_len) # (batch_size, num_steps)
        self.reduction = 'none' # 不做平均，保留每个样本的损失，我们后面手动平均
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label) # (batch_size, vocab_size, num_steps)和(batch_size, num_steps)做交叉熵损失，返回 (batch_size, num_steps),每格是该位置的交叉熵
        weighted_loss = (unweighted_loss * weights).mean(dim=1) # (batch_size)
        return weighted_loss    

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型。"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)  # 训练损失总和, 词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch] # X or Y shape is (batch_size, num_steps), X_valid_len or Y_valid_len shape is (batch_size, )
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1) # bos.shape is (batch_size, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len) # Y_hat.shape is (batch_size, num_steps, vocab_size)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            optimizer.step()
            metric.add(l.sum(), Y_valid_len.sum())
        if (epoch + 1) % 10 == 0:
            print('epoch', epoch + 1, 'loss',
                         (metric[0] / metric[1],)) 
        # metric[0] / metric[1] ≈ 平均每个词元的损失。为什么要除词元 
        # 数而不是除 batch 数或句子数？因为句子长短不一，按句平均会被短句 
        # 带偏；摊到每个词元上才是可比的指标——而且这正是困惑度的定义域：困
        # 惑度 = exp（平均每词元损失）
    print(f'final loss {metric[0] / metric[1]:.3f}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列的预测。"""
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # 添加批量轴
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long,
                                         device=device), dim=0) # enc_X.shape is (batch_size=1, num_steps)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    # 传enc_valid_len：注意力解码器要用它屏蔽源端pad位置（基础解码器的init_state有*args，多传也不报错）
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab['<bos>']],
                                         dtype=torch.long, device=device), dim=0) # dec_X.shape is (batch_size=1, num_steps=1)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 取最后时间步的输出词元
        dec_X = Y.argmax(dim=2)  # (batch_size, num_steps, vocab_size) → (batch_size, num_steps)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item() # (batch_size=1, num_steps=1) → (batch_size=1,) -> int
        # 保存注意力权重
        if save_attention_weights:
            # Save attention weights (to be covered later)   ← 以后才讲  
            attention_weight_seq.append(net.decoder.attention_weights)
        # 遇到'<eos>'时，输出结束
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """计算BLEU。"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            # clipping：参考答案里的每个n-gram只能被消耗一次（>0才计匹配，匹配后减1），
            # 防止复读常见词刷分（如"the the the the"配只含一个the的参考）
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

if __name__ == "__main__":
    encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
    encoder.eval()
    X = torch.zeros((4, 7), dtype=torch.long)
    output, state = encoder(X)
    print('encoder output shape: ' + str(output.shape))
    print('encoder state shape: ' + str(state.shape))

    decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,num_layers=2)
    decoder.eval()
    state = decoder.init_state((output, state))
    output, state = decoder(X, state)
    print('decoder output shape: ' + str(output.shape))
    print('decoder state shape: ' + str(state.shape))

    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    valid_len = torch.tensor([1, 2])
    print(sequence_mask(X, valid_len))

    loss = MaskedSoftmaxCELoss()
    loss(torch.ones((3, 4, 10)), torch.ones((3, 4), dtype=torch.long), torch.tensor([4, 2, 0]))

    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')