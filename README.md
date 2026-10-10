# ai-whys

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ZHYsfl/ai-whys/pulls)

> 一个系统性的 AI 学习路线：**38 个实验、16 章笔记**，从线性回归到 Transformer、从 Q-learning 到 PPO，每个算法都从零实现，并讲清每个"为什么"。
>
> **作者：周浩洋**（上海交通大学 AI 学院博士在读）· 欢迎交流：**zhouhy5523@gmail.com** · 也欢迎提交 PR、开 issue

![架构图](pics/arch.png)

## 38 个实验

共 **38 个实验**，编号与两份笔记中的实验声明一一对应（[dl.md](dl.md) 实验 1–33，[rl.md](rl.md) 实验 34–38）。每个实验都有三样东西：**数学推导笔记**（dl.md / rl.md）、**Excalidraw 插图**（pics/）、**可运行的代码**（code/）——下面清单中，每个实验都附上了笔记原文和代码的直达链接：

### 基础

1. **线性回归**（MBGD）— [笔记](dl.md#实验1线性回归算法的具体实现mbgd) / [从零手写](code/linear_regression/linreg_low_level.py) / [PyTorch 实现](code/linear_regression/linreg_high_level.py)
2. **Softmax 回归**（MBGD）— [笔记](dl.md#实验2softmax回归算法的具体实现mbgd) / [从零手写](code/softmax_regression/softmax_low_level.py) / [PyTorch 实现](code/softmax_regression/softmax_high_level.py)
3. **Dropout** — [笔记](dl.md#实验3dropout的具体实现) / [code](code/dropout/dropout.py)

### 卷积网络

4. **交叉相关** — [笔记](dl.md#实验4交叉相关的具体实现) / [code](code/cnn/cross_correlation.py)
5. **填充与步幅** — [笔记](dl.md#实验5填充和步幅的具体实现) / [code](code/cnn/padding_and_stride.py)
6. **1×1 卷积**（与全连接层等价的验证）— [笔记](dl.md#实验61x1卷积的具体实现以及和全连接层等价的验证) / [code](code/cnn/one_multiply_one_convolution.py)
7. **多输入/输出通道** — [笔记](dl.md#实验7多输入通道和多输出通道的具体实现) / [code](code/cnn/multi_input_and_output_channels.py)
8. **池化层** — [笔记](dl.md#实验8池化层的具体实现) / [code](code/cnn/polling.py)
9. **LeNet** — [笔记](dl.md#实验9lenet的具体实现) / [code](code/cnn/lenet.py)
10. **AlexNet** — [笔记](dl.md#实验10alexnet的具体实现) / [code](code/cnn/alexnet.py)
11. **VGG** — [笔记](dl.md#实验11vgg的具体实现) / [code](code/cnn/vgg.py)
12. **NiN** — [笔记](dl.md#实验12nin的具体实现) / [code](code/cnn/nin.py)
13. **GoogLeNet** — [笔记](dl.md#实验13googlenet的具体实现) / [code](code/cnn/googlenet.py)

### 归一化与并行

14. **Batch Normalization** — [笔记](dl.md#实验14batch-normalization的具体实现) / [从零手写](code/batch_normalization/batch_normalization_low_level.py) / [PyTorch 实现](code/batch_normalization/batch_normalization_high_level.py)
15. **ResNet** — [笔记](dl.md#实验15resnet的具体实现) / [code](code/resnet/resnet.py)
16. **多 GPU 数据并行** — [笔记](dl.md#实验16数据并行的具体实现) / [从零手写](code/multi_gpu_training/multi_gpu_training_low_level.py) / [PyTorch 实现](code/multi_gpu_training/multi_gpu_training_high_level.py)

### 自然语言处理

17. **序列模型** — [笔记](dl.md#实验17序列模型的具体实现) / [code](code/nlp/sequence_model.py)
18. **文本预处理** — [笔记](dl.md#实验18文本预处理的具体实现) / [code](code/nlp/text_pre_process.py)
19. **齐普夫定律验证** — [笔记](dl.md#实验19齐普夫定律的验证) / [code](code/nlp/zipf_law_loglog.py)
20. **文本序列数据准备** — [笔记](dl.md#实验20文本序列数据的准备) / [code](code/nlp/seqdataloader.py)
21. **RNN** — [笔记](dl.md#实验21rnn的具体实现) / [从零手写](code/nlp/rnn_low_level.py) / [PyTorch 实现](code/nlp/rnn_high_level.py)
22. **GRU** — [笔记](dl.md#实验22gru的具体实现) / [从零手写](code/nlp/gru_low_level.py) / [PyTorch 实现](code/nlp/gru_high_level.py)
23. **LSTM** — [笔记](dl.md#实验23lstm的具体实现) / [code](code/nlp/lstm.py)
24. **深度循环神经网络** — [笔记](dl.md#实验24深度循环神经网络的具体实现) / [code](code/nlp/deep_lstm.py)
25. **双向 LSTM 的错误使用案例** — [笔记](dl.md#实验25bidirectional_lstm的错误使用案例) / [code](code/nlp/bidirectional_lstm_in_wrong_scene.py)
26. **机器翻译数据集** — [笔记](dl.md#实验26机器翻译数据集的加载) / [code](code/nlp/machine_translation_dataset.py)
27. **编码器-解码器架构** — [笔记](dl.md#实验27编码器-解码器架构的接口) / [code](code/nlp/encoder_decoder.py)
28. **seq2seq** — [笔记](dl.md#实验28seq2seq的实现) / [code](code/nlp/seq2seq.py)
29. **无参注意力（Nadaraya-Watson）** — [笔记](dl.md#实验29无参注意力机制的实现) / [code](code/nlp/nadaraya_watson.py)
30. **两种注意力评分函数** — [笔记](dl.md#实验30两种attention-score-function的实现) / [code](code/nlp/attention_score.py)
31. **Bahdanau 注意力 seq2seq** — [笔记](dl.md#实验31注意力机制在seq2seq的应用) / [code](code/nlp/bahdanau_attention.py)
32. **多头注意力** — [笔记](dl.md#实验32多头注意力的实现) / [code](code/nlp/multi_head_attention.py)
33. **Transformer** — [笔记](dl.md#实验33transformer的实现) / [code](code/nlp/transformer.py)

### 强化学习

34. **Q-learning** — [笔记](rl.md#实验34q-learning的实现) / [code](code/q_learning/q_learning.py)
35. **DQN** — [笔记](rl.md#实验35dqn-的实现) / [code](code/q_learning/dqn.py)
36. **REINFORCE** — [笔记](rl.md#实验36reinforce-的实现) / [code](code/reinforce/reinforce.py)
37. **Actor-Critic** — [笔记](rl.md#实验37actor-critic-的实现) / [code](code/actor_critic/actor_critic.py)
38. **PPO**（已迁移至 gymnasium 最新版 API）— [笔记](rl.md#实验38ppo-的实现) / [code](code/actor_critic/ppo.py)

## 这个项目适合谁

- **想系统性学 AI、但不知道从何开始的本科生**——不需要先修成堆的理论，跟着实验 1 → 38 的顺序走，就是一条完整的深度学习 + 强化学习入门路线。
- **即将入学的研究生**——把它当作入学前自查清单：每个实验背后的"为什么"，如果你能全部讲清楚，基础就算过关；讲不清楚的，就是开学前该补的课。

## 每个实验都从三个层面拆解

同一个知识点，三种互补的打开方式：

- **数学推导**（[dl.md](dl.md) / [rl.md](rl.md)）——每个算法从公式推起，符号约定统一（大写为矩阵、小写为标量），先弄清"为什么"再谈"怎么用"
- **插图总结**（[pics/](pics)）——30 余张 Excalidraw 手绘风插图，把网络架构、数据流向和张量形状对齐画出来，一张图胜过一页文字
- **复现代码**（[code/](code)）——从零实现 + 大量中文注释，每个"为什么"都对应一段能跑、能改的代码

## 建议的学习方式

这份材料是一个**精炼的总结**：它不追求把每个细节讲满，而是给出路线、推导主干和关键问题，细节留白交给学习过程本身。因此最好的使用方式是——**把它当作地图，结合 AI coding agent（如 Kimi Code、Claude Code、Cursor 等）一起学**：

- **笔记定路线**：按实验 1 → 38 的顺序读 [dl.md](dl.md) / [rl.md](rl.md)，先建立整体脉络，推导卡住的地方直接问 agent
- **agent 陪跑细节**：每个实验的代码让 agent 逐行讲解、补齐跳步的推导、搭好环境跑通结果、改参数做对照实验——笔记里记下的那些"为什么"，就是现成的提问起点
- **输出自己的版本**：学完一个实验后合上笔记，让 agent 抽查你能否讲清背后的"为什么"；讲不清的，回到笔记补

如果你**有一定机器学习/深度学习基础、熟悉 Python 语法，但还没系统地梳理过这些知识**，这个项目就是为你准备的。希望它对各位读者有帮助，欢迎提交 PR、开 issue 与我交流。

## 为什么值得跟着学

初学深度学习时，我和绝大多数同学的状态一样：**代码能跑，但不知道为什么这么写**。公式背下来了，但每个符号对应代码里的哪一行、书上那些"显然""易知"背后是什么，没人讲。

这份笔记因此有三个执念：

- **每个"为什么"都要有答案**。笔记记录了学习过程中真实提出的一个个问题——为什么要 `detach`？为什么参数的形状和直觉相反？为什么 loss 要加掩码？——以及它们被彻底搞懂之后的解答。
- **每个算法写两遍**。`code/` 下的脚本大多成对出现：`*_low_level` 从零手写（连矩阵乘法都自己实现），`*_high_level` 用 PyTorch 官方 API。手写版为了搞懂底层数学，API 版为了掌握工程惯例。
- **注释比代码多**。所有代码附带大量中文注释，解释机制、张量形状对齐和踩过的坑，而不只是复述"这段代码在干嘛"。

仓库作者从入门到保研上海交通大学 AI 学院直博走的就是这条路线——它不是编出来的大纲，是一步一步踩出来的脚印。如果它帮你解开了一个"为什么"，欢迎给个 Star。

## 其他材料

- **dl.md**：深度学习笔记，16 章，配套《动手学深度学习》，覆盖线性回归 → CNN/ResNet → RNN/GRU/LSTM → 注意力/Transformer
- **rl.md**：强化学习笔记，配套《动手学强化学习》（持续更新中）
- **dl_and_rl.svg / dl_and_rl.excalidraw**：学科全景架构图，SVG 矢量版 + Excalidraw 可编辑源文件，可见 DL 与 RL 的知识脉络如何串成一体（`pics/arch.png` 为同一张图的 PNG 导出）
- **pics/**：30 张原创插图（Excalidraw 绘制）

## 运行环境

- Python 3.14，推荐使用 [uv](https://docs.astral.sh/uv/) 管理虚拟环境
- 安装依赖：`uv pip install -r requirements.txt`
- 深度学习部分依赖 PyTorch；强化学习部分已迁移至 [gymnasium](https://gymnasium.farama.org/) 最新版 API
- 注意：`d2l` 包对 torch/numpy 有较严格的版本 pin，若与已装的 PyTorch 冲突，改用 `uv pip install d2l --no-deps` 单独安装

## 许可证与致谢

本项目以 [Apache License 2.0](LICENSE) 开源，详见 [NOTICE](NOTICE)。

代码并非全部原创，致谢如下：

- **深度学习代码**改写自[《动手学深度学习》](https://zh.d2l.ai)（[d2l-ai/d2l-zh](https://github.com/d2l-ai/d2l-zh)，原内容以 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 发布），在学习过程中加入了大量中文注释与工程改进；
- **强化学习代码**改写自[《动手学强化学习》](https://hrl.boyuai.com)（[boyu-ai/Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)，原代码以 Apache-2.0 发布），并迁移至 gymnasium 最新版 API；
- 训练工具依赖 [d2l PyPI 包](https://pypi.org/project/d2l/)（MIT-0）。

笔记（dl.md / rl.md）与插图（pics/）为本项目原创内容。

本项目是独立的学习项目，与上述书籍作者及出版社无任何隶属或背书关系。
