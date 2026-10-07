import os
import zipfile
import torch
from d2l import torch as d2l
from text_pre_process import Vocab

# 注册数据集：名字 -> (下载URL, sha1校验和)
# sha1用于缓存校验：本地文件已存在且sha1匹配就直接用，不匹配则重新下载
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')


def read_data_nmt():
    """载入"英语-法语"数据集。"""
    # 注意：d2l.download_extract 的 folder 参数不是下载目录（它只是拼在返回路径末尾），
    # 其内部调 download(name) 时下载目录写死为默认的 '../data'（相对当前工作目录），
    # 所以这里自己 download(folder=DATA_DIR) + zipfile 解压，保证落到项目 data/ 下。
    # 解压后结构：data/fra-eng/fra.txt，每行形如 "Go.\tVa !"（英文\t法文）
    fra_path = os.path.join(DATA_DIR, 'fra-eng', 'fra.txt')
    if not os.path.exists(fra_path):  # 已解压过就直接用，避免重复下载
        fname = d2l.download('fra-eng', folder=DATA_DIR)
        with zipfile.ZipFile(fname, 'r') as fp:
            fp.extractall(DATA_DIR)
        os.remove(fname)  # zip只是中间产物，解压完即删，不留残留
    with open(fra_path, 'r') as f:
        return f.read()


def preprocess_nmt(text):
    """预处理"英语-法语"数据集。"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 把不间断空格替换为普通空格，并全部转小写
    # （法语文本里存在\u202f和\xa0，肉眼和普通空格一样，但split(' ')切不开）
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点之间插入空格："go." -> "go ."，这样按空格切分时标点能独立成词元
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化"英语-法语"数据数据集。"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            # 例：line = "hi .\tsalut !" -> source += [['hi', '.']], target += [['salut', '!']]
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列，使长度统一为num_steps（同一个batch才能叠成张量）。"""
    # 例：num_steps=5，[12, 4] -> [12, 4, 1, 1, 1]（假设<pad>的索引是1）
    #          [12, 4, 9, 34, 88, 21] -> [12, 4, 9, 34, 88]
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量。"""
    lines = [vocab[l] for l in lines]  # 词元 -> 索引
    lines = [l + [vocab['<eos>']] for l in lines]  # 每条序列末尾加<eos>
    array = torch.tensor([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # valid_len：每条序列的实际长度（不含<pad>），后面seq2seq靠它忽略填充部分
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和源语言、目标语言两种词表。"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    # 源语言（英语）和目标语言（法语）各建一个词表，二者索引互不相干
    src_vocab = Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # print(truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>']))
    tgt_vocab = Vocab(target, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


if __name__ == '__main__':
    raw_text = read_data_nmt()
    print(raw_text[:75])

    text = preprocess_nmt(raw_text)
    print(text[:80])

    source, target = tokenize_nmt(text)
    print('source:', source[:6])
    print('target:', target[:6])


    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
   
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break
