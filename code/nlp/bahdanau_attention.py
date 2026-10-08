import torch
from torch import nn
from d2l import torch as d2l
from encoder_decoder import Decoder,EncoderDecoder
from attention_score import AdditiveAttention
from seq2seq import train_seq2seq,predict_seq2seq
from machine_translation_dataset import load_data_nmt
class AttentionDecoder(Decoder):
    '''The base class of decoders with attention mechanisms.'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    '''The base class of decoders for sequence-to-sequence models with attention mechanisms.'''
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.attention = AdditiveAttention(num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs: (num_steps, batch_size, num_hiddens)
        # hidden_state: (num_layers, batch_size, num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens) # Permute outputs to (batch_size, num_steps, num_hiddens) for attention computation

    def forward(self, X, state):
        # X: (batch_size, num_steps)
        # enc_outputs: (batch_size, num_steps, num_hiddens)
        # hidden_state: (num_layers, batch_size, num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        X = self.embedding(X).permute(1, 0, 2) # (num_steps, batch_size, embed_size)
        outputs = []
        for x in X:
            # query: (batch_size, num_querys/seq_len=1, num_hiddens),理论上是能获取翻译的原文信息的，只不过学习的时候角色分配不会太关注原文，而是已经翻译好的句子。最开始用的是编码器的最后一层的最后一步的输出
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens) # (batch_size, num_queries=1, value_size),代表锁定到的原文信息，这里大部分应该是原文接下来要被翻译的部分
            # x: (batch_size, 1, embed_size)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1) # (batch_size, 1, embed_size + num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state) # (seq_len=1, batch_size, num_hiddens), (num_layers, batch_size, num_hiddens)
            outputs.append(out)
        outputs = self.dense(torch.cat(outputs, dim=0)) # (num_steps, batch_size, num_hiddens) -> (num_steps, batch_size, vocab_size)
        return outputs.permute(1, 0, 2), (enc_outputs, hidden_state, enc_valid_lens) # (outputs: (batch_size, num_steps, vocab_size), state: (enc_outputs, hidden_state, enc_valid_lens))

    @property
    def attention_weights(self):
        return self.attention.attention_weights

if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, dropout = 8, 16, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(vocab_size=len(src_vocab), embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
    decoder = Seq2SeqAttentionDecoder(vocab_size=len(tgt_vocab), embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', "j'ai perdu .", 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weights = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}',f'bleu: {d2l.bleu(translation, fra, k=2):.3f}')