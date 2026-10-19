import torch
from torch import nn
from d2l import torch as d2l
from transformer import EncoderBlock

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # Add the special tokens '<cls>' and '<sep>'
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nn.Module):
    """The BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens) # (num_hiddens,)
        self.segment_embedding = nn.Embedding(2, num_hiddens) # (num_hiddens,)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock( # post-LN
                                    key_size, query_size, value_size,
                                    num_hiddens, norm_shape,
                                    ffn_num_input, ffn_num_hiddens,
                                    num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens)) # (1, max_len, num_hiddens)

    def forward(self, tokens, segments, valid_lens):
        # tokens: (batch size, max sequence length), segments: (batch size, max sequence length)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        # The shape of `X` is (batch size, max sequence length, num_hiddens)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X # (batch size, max sequence length, num_hiddens)

class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            # input: (batch size, num_pred_positions, num_inputs)
            nn.Linear(num_inputs, num_hiddens), nn.ReLU(), # output: (batch size, num_pred_positions, num_hiddens)
            nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, vocab_size)) # output: (batch size, num_pred_positions, vocab_size)

    def forward(self, X, pred_positions):
        # X is the output of the BERT encoder, with shape (batch size, max sequence length, num_hiddens)
        # pred_positions is the index of the masked tokens in each sequence, with shape (batch size, num_pred_positions)
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1) # (batch size * num_pred_positions,)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions) # (batch size * num_pred_positions,)
        masked_X = X[batch_idx, pred_positions] # (batch size * num_pred_positions, num_hiddens)
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1)) # (batch size, num_pred_positions, num_hiddens)
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_hiddens, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_hiddens, 2) # output: (batch size, 2)

    def forward(self, X):
        # X是<cls>位置的向量：BERTModel里已取出encoded_X[:, 0, :]，这里直接分类
        return self.output(X) # (batch size, 2)

class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768,mlm_in_features=768,
                 nsp_in_features=768,
                 **kwargs):
        super(BERTModel, self).__init__(**kwargs)
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                                   ffn_num_input, ffn_num_hiddens,
                                   num_heads, num_layers, dropout,
                                   max_len=max_len, key_size=key_size,
                                   query_size=query_size,
                                   value_size=value_size)
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens), nn.Tanh())

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        # tokens: (batch size, max sequence length), segments: (batch size, max sequence length)
        encoded_X = self.encoder(tokens, segments, valid_lens)
        # (batch size, max sequence length, num_hiddens)

        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
            # (batch size, num_pred_positions, vocab_size)
        else:
            mlm_Y_hat = None

        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :])) # (batch size, 2)
        
        return encoded_X, mlm_Y_hat, nsp_Y_hat

if __name__ == "__main__":
    pass