import math
import torch
from torch import nn
from d2l import torch as d2l

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    # X.shape is (batch_size, num_queries, num_kv)
    # valid_lens.shape is (batch_size,) or (batch_size, num_queries)
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
            # valid_lens.shape is (batch_size * num_queries,)
        else:
            valid_lens = valid_lens.reshape(-1)
            # valid_lens.shape is (batch_size * num_queries,)
        # Fill masked elements with a large negative value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6) # X.reshape(-1, shape[-1]).shape is (batch_size * num_queries, num_kv)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    """Additive attention."""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super().__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: (batch_size, num_queries, query_size)
        # keys: (batch_size, num_kv, key_size)
        # values: (batch_size, num_kv, value_size)
        # valid_lens: (batch_size,) or (batch_size, num_queries)
        queries, keys = self.W_q(queries), self.W_k(keys) # (batch_size, num_queries, num_hiddens), (batch_size, num_kv, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1) # (batch_size, num_queries, num_kv, num_hiddens)
        features = torch.tanh(features) # (batch_size, num_queries, num_kv, num_hiddens)
        scores = self.w_v(features).squeeze(-1) # (batch_size, num_queries, num_kv)
        self.attention_weights = masked_softmax(scores, valid_lens) # (batch_size, num_queries, num_kv)
        return torch.bmm(self.dropout(self.attention_weights), values) # (batch_size, num_queries, value_size)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries: (batch_size, num_queries, d)
        # keys: (batch_size, num_kv, d)
        # values: (batch_size, num_kv, value_size)
        # valid_lens: (batch_size,) or (batch_size, num_queries)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d) # (batch_size, num_queries, num_kv)
        self.attention_weights = masked_softmax(scores, valid_lens) # (batch_size, num_queries, num_kv)
        return torch.bmm(self.dropout(self.attention_weights), values) # (batch_size, num_queries, value_size)

if __name__ == "__main__":
    X = torch.rand(2, 2, 4)
    print(X)
    print(masked_softmax(X, torch.tensor([2, 3])))
    print(masked_softmax(X, torch.tensor([[1, 3], [2, 4]])))
    print('-' * 20)
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
    print('-' * 20)
    attention = DotProductAttention(dropout=0.1)
    attention.eval()
    # 点积注意力没有投影层，要求query和key的最后一维相同（都是d），
    # 所以不能复用加性注意力那份 query_size=20 的queries，需另造 d=2 的
    queries = torch.normal(0, 1, (2, 1, 2))
    print(attention(queries, keys, values, valid_lens))
