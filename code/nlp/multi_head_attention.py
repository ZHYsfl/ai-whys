import math
import torch
from torch import nn
from d2l import torch as d2l

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries, keys, values: (batch_size, no. of queries or key-value pairs, num_hiddens)
        # valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, output `queries`, `keys`, and `values` have shapes:
        # (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # Copy the first axis of valid_lens by `num_heads` times
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output: (batch_size * num_heads, no. of queries, num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # 把注意力权重暴露到MHA这一层（d2l原书是让调用方穿透两层去取，
        # 这里在MHA上存一份，transformer.py里 blk.attention.attention_weights 即可取到）；
        # 形状 (batch_size * num_heads, num_queries, num_kv)，绘图时再按head拆开
        self.attention_weights = self.attention.attention_weights

        # output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        # Shape of input `X`:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of output `X`:
        # (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1) # Reshape `X` to (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3) # Transpose `X` to (batch_size, num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3]) # (batch_size * num_heads, no. of queries or key-value pairs, num_hiddens / num_heads)

    def transpose_output(self, X, num_heads):
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2]) # (batch_size, num_heads, no. of queries, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3) # (batch_size, no. of queries, num_heads, num_hiddens / num_heads)
        return X.reshape(X.shape[0], X.shape[1], -1) # (batch_size, no. of queries, num_hiddens)

if __name__ == "__main__":
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(key_size=num_hiddens, query_size=num_hiddens, value_size=num_hiddens, num_hiddens=num_hiddens, num_heads=num_heads, dropout=0.5)
    attention.eval()
    batch_size, num_queries, num_kvpairs, valid_lens = 2, 4, 6, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    print(attention(X, Y, Y, valid_lens).shape)

    