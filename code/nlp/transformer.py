import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from multi_head_attention import MultiHeadAttention
from bahdanau_attention import AttentionDecoder
from machine_translation_dataset import load_data_nmt
from encoder_decoder import EncoderDecoder
from seq2seq import train_seq2seq, predict_seq2seq, bleu

class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self,key_size, query_size, value_size, num_hiddens, norm_shape, 
                 ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads,
            dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout,
                use_bias))

    def forward(self, X, valid_lens):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    """Transformer decoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads,
            dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads,
            dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # state[0]：enc_outputs （编码器的最终输出，用来做 Cross Attention 的 K 和 V）
        # state[1]：enc_valid_lens （编码器输出的有效长度，防止看填充符）
        # state[2]：key_values 缓存列表（这是一个长度为 num_layers 的列表，用来存每一层过去生成的 K 和 V，这是推理加速的核心！）
        # X 是当前步的输入 (batch_size, current_steps, hidden_size), 推理时 X 一直是 (batch_size, 1, hidden_size)

        enc_outputs, enc_valid_lens = state[0], state[1]
        # Self-attention with the first attention block
        if state[2][self.i] is None: 
            # 如果是训练，X 就是整个句子 [<bos>, 我, 爱]，直接一次性进来，state[2][self.i] 初始是 None，所以 key_values = X。训练时其实不需要缓存，因为下一次前向传播是新的批次，会把整个 state 重新初始化。这里写这个逻辑，纯粹是为了训练和推理共用同一套代码。
            # 也可能是第一步推理。
            key_values = X 
        else: # 如果是推理，这次只输入一个词 X 形状是 (batch_size, current_steps=1, hidden_size)。此时 state[2][self.i] 里存着之前所有时间步的 K 和 V。通过 torch.cat 把当前词和历史拼起来，形成完整的 key_values。
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, seq_len, _ = X.shape
            # Shape of `dec_valid_lens`: (batch_size * num_heads, seq_len)
            dec_valid_lens = torch.arange(1, seq_len + 1,
                                          device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention with the second attention block
        Y2 = self.attention2(Y, enc_outputs, enc_outputs,
                             enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state # (batch_size, current_steps, hidden_size)

class TransformerDecoder(AttentionDecoder):
    """Transformer decoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers= num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", DecoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens,
                [None] * self.num_layers] # 初始化缓存列表

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == "__main__":
    # ln = nn.LayerNorm(2)
    # bn = nn.BatchNorm1d(2)
    # X = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32)
    # print('LayerNorm:', ln(X))
    # print('BatchNorm1d:', bn(X))

    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 2, 100
    lr, num_epochs, device = 0.005, 50, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=True)
        print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')