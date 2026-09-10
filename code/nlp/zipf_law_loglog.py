import os
import random
import torch
from d2l import torch as d2l
from text_pre_process import read_time_machine,tokenize,Vocab

PICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'pics')

def plot_zipf_law(vocab_dict, save_path=None):
    """Plot n-gram frequency vs. rank on log-log axes (Zipf's law).

    vocab_dict: ordered dict of {label: vocab}, e.g.
        {'unigram': vocab, 'bigram': bigram_vocab, 'trigram': trigram_vocab}
    """
    styles = ['-', '--', '-.', ':']
    d2l.plt.figure(figsize=(6, 4))
    for (label, vocab), style in zip(vocab_dict.items(), styles):
        freqs = [freq for token, freq in vocab.token_freqs]
        d2l.plt.plot(freqs, style, label=label)
    d2l.plt.xscale('log')
    d2l.plt.yscale('log')
    d2l.plt.xlabel('token: x')
    d2l.plt.ylabel('frequency: n(x)')
    d2l.plt.grid(True)
    d2l.plt.legend()
    if save_path:
        d2l.plt.savefig(save_path, bbox_inches='tight')
    d2l.plt.show()

if __name__ == "__main__":
    tokens = tokenize(read_time_machine())
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    print(vocab.token_freqs[:10]) # 最流行的词被称为 停用词（stop words)

    bigram_tokens = [pair for pair in zip(corpus[:-1],corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10]) # 最流行的词对被称为 停用词对（stop word pairs)

    trigram_tokens = [triple for triple in zip(corpus[:-2],corpus[1:-1],corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    quadgram_tokens = [quadruple for quadruple in zip(corpus[:-3],corpus[1:-2],corpus[2:-1],corpus[3:])]
    quadgram_vocab = Vocab(quadgram_tokens)
    print(quadgram_vocab.token_freqs[:10])

    plot_zipf_law({'unigram': vocab, 'bigram': bigram_vocab,
                   'trigram': trigram_vocab, 'quadgram': quadgram_vocab},
                  os.path.join(PICS_DIR, 'zipf_law_loglog.png'))
