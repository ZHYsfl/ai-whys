import torch
import torch.nn.functional as F
from torch import nn
from d2l import torch as d2l
from bert import BERTModel
from bert_pretrain_dataset import load_data_wiki


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X,
                         valid_lens_x, pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    """计算一个batch的BERT预训练损失（MLM损失 + NSP损失）。"""
    # 前向：只需要mlm_Y_hat和nsp_Y_hat，编码器输出encoded_X不用
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x,
                                  pred_positions_X)
    # mlm_Y_hat: (batch_size, num_pred, vocab_size)，mlm_Y: (batch_size, num_pred)
    # 展平成 (batch_size*num_pred, vocab_size) 和 (batch_size*num_pred,)
    # 用mlm_weights只对真预测位取平均：padding位的logits不参与损失
    mlm_l = (F.cross_entropy(mlm_Y_hat.reshape(-1, vocab_size),
                             mlm_Y.reshape(-1), reduction='none')
             * mlm_weights_X.reshape(-1)).sum() / mlm_weights_X.sum()
    # nsp_Y_hat: (batch_size, 2)，nsp_y: (batch_size,)
    nsp_l = loss(nsp_Y_hat, nsp_y)
    return mlm_l, nsp_l, mlm_l + nsp_l


def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    """训练BERT（num_steps步，而非跑完整个数据集）。"""
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.001)
    step, timer = 0, d2l.Timer()
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        # DataLoader每次吐出7个分量，顺序与Dataset.__getitem__一致
        for (tokens_X, segments_X, valid_lens_x, pred_positions_X,
             mlm_weights_X, mlm_Y, nsp_y) in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y = mlm_Y.to(devices[0])
            nsp_y = nsp_y.to(devices[0]).long()  # 布尔标签要转成整数算交叉熵
            trainer.zero_grad()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            step += 1
            if step >= num_steps:
                num_steps_reached = True
                break
    print(f'MLM loss {float(mlm_l):.3f}, NSP loss {float(nsp_l):.3f}')
    print(f'{step / timer.stop():.1f} sequences/sec on {str(devices)}')


if __name__ == '__main__':
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    print(f'Vocabulary size: {len(vocab)}')
    net = BERTModel(vocab_size=len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)

    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss()

    train_bert(train_iter, net, loss, len(vocab), devices, num_steps=50)
