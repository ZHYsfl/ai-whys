# foundation

深度学习和强化学习的学习笔记与代码重实现，记录了从零实现经典算法的过程。

![架构图](pics/arch.png)

## 为什么写这份笔记

初学深度学习时，我和绝大多数同学的状态一样：**代码能跑，但不知道为什么这么写**。公式背下来了，但每个符号对应代码里的哪一行、书上那些"显然""易知"背后是什么，没人讲。

这份笔记因此有三个执念：

- **每个"为什么"都要有答案**。笔记记录了学习过程中真实提出的一个个问题——为什么要 `detach`？为什么参数的形状和直觉相反？为什么 loss 要加掩码？——以及它们被彻底搞懂之后的解答。
- **每个算法写两遍**。`code/` 下的脚本大多成对出现：`*_low_level` 从零手写（连矩阵乘法都自己实现），`*_high_level` 用 PyTorch 官方 API。手写版为了搞懂底层数学，API 版为了掌握工程惯例。
- **注释比代码多**。所有代码附带大量中文注释，解释机制、张量形状对齐和踩过的坑，而不只是复述"这段代码在干嘛"。

这个仓库是我从入门到保研上海交通大学 AI 学院直博期间的学习沉淀，也希望能帮到同样在路上的人。如果它帮你解开了一个"为什么"，欢迎给个 Star。

## 内容

- **dl.md**：深度学习笔记，16 章，配套《动手学深度学习》，覆盖线性回归 → CNN/ResNet → RNN/GRU/LSTM → 注意力/Transformer
- **rl.md**：强化学习笔记，配套《动手学强化学习》（持续更新中）
- **dl_and_rl.svg / dl_and_rl.excalidraw**：学科全景架构图，SVG 矢量版 + Excalidraw 可编辑源文件，可见 DL 与 RL 的知识脉络如何串成一体（`pics/arch.png` 为同一张图的 PNG 导出）
- **pics/**：30 张原创插图（Excalidraw 绘制）

共 **38 个实验**，配套代码在 code/ 下，编号与两份笔记中的实验声明一一对应（dl.md 实验 1–33，rl.md 实验 34–38）：

### 基础

1. **线性回归**（MBGD）— [低阶](code/linear_regression/linreg_low_level.py) / [高阶](code/linear_regression/linreg_high_level.py)
2. **Softmax 回归**（MBGD）— [低阶](code/softmax_regression/softmax_low_level.py) / [高阶](code/softmax_regression/softmax_high_level.py)
3. **Dropout** — [code](code/dropout/dropout.py)

### 卷积网络

4. **交叉相关** — [code](code/cnn/cross_correlation.py)
5. **填充与步幅** — [code](code/cnn/padding_and_stride.py)
6. **1×1 卷积**（与全连接层等价的验证）— [code](code/cnn/one_multiply_one_convolution.py)
7. **多输入/输出通道** — [code](code/cnn/multi_input_and_output_channels.py)
8. **池化层** — [code](code/cnn/polling.py)
9. **LeNet** — [code](code/cnn/lenet.py)
10. **AlexNet** — [code](code/cnn/alexnet.py)
11. **VGG** — [code](code/cnn/vgg.py)
12. **NiN** — [code](code/cnn/nin.py)
13. **GoogLeNet** — [code](code/cnn/googlenet.py)

### 归一化与并行

14. **Batch Normalization** — [低阶](code/batch_normalization/batch_normalization_low_level.py) / [高阶](code/batch_normalization/batch_normalization_high_level.py)
15. **ResNet** — [code](code/resnet/resnet.py)
16. **多 GPU 数据并行** — [低阶](code/multi_gpu_training/multi_gpu_training_low_level.py) / [高阶](code/multi_gpu_training/multi_gpu_training_high_level.py)

### 自然语言处理

17. **序列模型** — [code](code/nlp/sequence_model.py)
18. **文本预处理** — [code](code/nlp/text_pre_process.py)
19. **齐普夫定律验证** — [code](code/nlp/zipf_law_loglog.py)
20. **文本序列数据准备** — [code](code/nlp/seqdataloader.py)
21. **RNN** — [低阶](code/nlp/rnn_low_level.py) / [高阶](code/nlp/rnn_high_level.py)
22. **GRU** — [低阶](code/nlp/gru_low_level.py) / [高阶](code/nlp/gru_high_level.py)
23. **LSTM** — [code](code/nlp/lstm.py)
24. **深度循环神经网络** — [code](code/nlp/deep_lstm.py)
25. **双向 LSTM 的错误使用案例** — [code](code/nlp/bidirectional_lstm_in_wrong_scene.py)
26. **机器翻译数据集** — [code](code/nlp/machine_translation_dataset.py)
27. **编码器-解码器架构** — [code](code/nlp/encoder_decoder.py)
28. **seq2seq** — [code](code/nlp/seq2seq.py)
29. **无参注意力（Nadaraya-Watson）** — [code](code/nlp/nadaraya_watson.py)
30. **两种注意力评分函数** — [code](code/nlp/attention_score.py)
31. **Bahdanau 注意力 seq2seq** — [code](code/nlp/bahdanau_attention.py)
32. **多头注意力** — [code](code/nlp/multi_head_attention.py)
33. **Transformer** — [code](code/nlp/transformer.py)

### 强化学习

34. **Q-learning** — [code](code/q_learning/q_learning.py)
35. **DQN** — [code](code/q_learning/dqn.py)
36. **REINFORCE** — [code](code/reinforce/reinforce.py)
37. **Actor-Critic** — [code](code/actor_critic/actor_critic.py)
38. **PPO**（已迁移至 gymnasium 最新版 API）— [code](code/actor_critic/ppo.py)

## 运行环境

- Python 3.14，推荐使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境
- 深度学习部分依赖 PyTorch；强化学习部分已迁移至 [gymnasium](https://gymnasium.farama.org/) 最新版 API

## 许可证与致谢

本项目以 [Apache License 2.0](LICENSE) 开源，详见 [NOTICE](NOTICE)。

代码并非全部原创，致谢如下：

- **深度学习代码**改写自[《动手学深度学习》](https://zh.d2l.ai)（[d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)，原内容以 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 发布），在学习过程中加入了大量中文注释与工程改进；
- **强化学习代码**改写自[《动手学强化学习》](https://hrl.boyuai.com)（[boyu-ai/Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)，原代码以 Apache-2.0 发布），并迁移至 gymnasium 最新版 API；
- 训练工具依赖 [d2l PyPI 包](https://pypi.org/project/d2l/)（MIT-0）。

笔记（dl.md / rl.md）与插图（pics/）为本项目原创内容。

本项目是独立的学习项目，与上述书籍作者及出版社无任何隶属或背书关系。
