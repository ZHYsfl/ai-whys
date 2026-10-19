import os
import random
import zipfile
import torch
from d2l import torch as d2l
from text_pre_process import Vocab, tokenize
from bert import get_tokens_and_segments

# 注册数据集：名字 -> (下载URL, sha1校验和)
# sha1用于缓存校验：本地文件已存在且sha1匹配就直接用，不匹配则重新下载
# 课程里的原始链接（s3.amazonaws.com/research.metamind.io）已失效（403），
# 这里改用 Wayback Machine 的同文件快照，sha1 与原文件一致
d2l.DATA_HUB['wikitext-2'] = ('https://web.archive.org/web/20230301182406id_/'
                              'https://s3.amazonaws.com/research.metamind.io/'
                              'wikitext/wikitext-2-v1.zip',
                              '3c914d17d80b1459be871a5039ac23e752a53cbe')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data')


def _read_wiki(data_dir):
    """读取 WikiText-2 训练集，返回段落列表（每个段落是句子的列表）。"""
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # WikiText-2 里标点是独立词元，句子以 " . " 结尾，
    # 按 " . " 切分句子，只保留至少含2个句子的段落（NSP需要句子对），最后打散段落顺序
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def _get_next_sentence(sentence, next_sentence, paragraphs):
    """构造一句"句子B"：50%概率是真正的下一句（标签True），
    50%从任意段落随机挑一句（标签False）。"""
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """把段落里相邻的句子两两配对，生成NSP任务样本：(tokens, segments, is_next)。"""
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # +3 是给 <cls> 和两个 <sep> 留的位置，超长的句对直接丢弃
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """按BERT的规则替换选中的token：80%换成<mask>，10%保持不变，10%换成随机词。
    返回替换后的token序列，以及 [(被掩位置, 原词), ...]。"""
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """对一条token序列（已含<cls>/<sep>）做MLM掩码：
    返回 (词元id序列, 被掩位置列表, 被掩位置的原词id列表)。"""
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        # 特殊token不参与预测
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15%的词参与预测，至少1个
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    # 打乱后挑出的位置顺序乱了，按位置排序并拆成位置和标签两个列表
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    # 标签也要从token转成id
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    """把所有样本补齐到max_len：token ids和segments补<pad>，
    MLM的预测位置和标签补到固定个数（配套mlm_weights标出哪些是真预测位）。"""
    # 序列最长max_len，预测位最多也是max_len的15%，据此统一补齐长度
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        # token ids 补 <pad>，segments 补 0
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # 有效长度（不含<pad>），attention靠它忽略填充位
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        # 每条样本的实际预测数不同，统一补到 max_num_mlm_preds
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # 真实预测位权重为1，补出来的padding位权重为0，算loss时padding位不计
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(is_next)
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    """WikiText-2 的 BERT 预训练数据集：NSP + MLM 样本的集合。"""

    def __init__(self, paragraphs, max_len):
        # 词元化：每个段落从"句子的列表"变成"token列表的列表"
        paragraphs = [tokenize(paragraph, token='word')
                      for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 先在原始句子上做NSP配对，再在配对好的token序列上做MLM掩码
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                     + (segments, is_next))
                    for tokens, segments, is_next in examples]
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels,
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """下载 WikiText-2 并生成 BERT 预训练样本，返回数据迭代器和词表。"""
    num_workers = d2l.get_dataloader_workers()
    # 与 machine_translation_dataset.py 相同的下载约定：
    # download 下载 zip 到项目 data/，解压出 data/wikitext-2/ 后删除 zip
    data_dir = os.path.join(DATA_DIR, 'wikitext-2')
    if not os.path.exists(os.path.join(data_dir, 'wiki.train.tokens')):
        fname = d2l.download('wikitext-2', cache_dir=DATA_DIR)
        with zipfile.ZipFile(fname, 'r') as fp:
            fp.extractall(DATA_DIR)
        os.remove(fname)  # zip只是中间产物，解压完即删，不留残留
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    return train_iter, train_set.vocab


if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)

    # 打印一个batch里七个分量的形状，与课程中的形状对照
    for (tokens_X, segments_X, valid_lens_X, pred_positions_X,
         mlm_weights_X, mlm_Y, nsp_y) in train_iter:
        print(tokens_X.shape, segments_X.shape, valid_lens_X.shape,
              pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
              nsp_y.shape)
        break

    print(len(vocab))
