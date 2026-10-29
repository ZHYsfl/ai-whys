# 参考工作与许可证说明

本目录（`code/codec/`）的教学版 codec 是**独立重写的教学实现**：代码由本项目作者编写，未从下列任何仓库复制源码；但架构选型与训练方案参照了下列经典工作。本文档说明每一项参照了什么、原仓库的许可证是什么、以及使用这些原仓库代码时需要遵守的要求。

## 各工作分别增加了什么

教学版的主干来自 SoundStream，码本学习用 VQ-VAE/DAC 式的码本损失 + 承诺损失 + STE。在这个底座上，各篇工作的增量可以一眼定位：

- **SoundStream**：确立"全卷积 codec + RVQ + 重建/对抗训练"主线，用 quantizer dropout 支持可变码率；原论文码本走 k-means 初始化、EMA 更新和死码替换，教学版先用梯度学习的码本。
- **EnCodec**：保留主干，加 MS-STFT 判别器、loss balancer 和因果模型。官方 `encodec` 仓库主要是模型与推理，完整训练入口在 AudioCraft；因果卷积只是流式计算的结构前提，不等于已有流式 API。
- **DAC**：针对码本崩溃用低维投影和 L2 归一化，码本仍由码本损失与承诺损失学习；quantizer dropout 的概率折中来自这里。
- **AudioDec**：把训练拆成 autoencoder、统计量提取、vocoder 等阶段，推理路径显式维护因果卷积 buffer，适合在学完离线 codec 后看真正的流式数据流。
- **SpeechTokenizer 与 Mimi**：语义 token 不是 RVQ 自动长出来的。SpeechTokenizer 用 HuBERT 教师对第一层量化表示做蒸馏，后续层补声学信息；Mimi 也区分语义和声学分支。没有语义教师或语义损失时，codes 只能叫声学/离散表征，不能宣称第一层就是语义 token。
- **WavTokenizer 与 HiFi-Codec**：单码本低帧率、分组残差量化（GRVQ）等路线，目标都是减少语言模型要处理的 token 数，不属于基础 RVQ 的必要组件。

建议的源码阅读顺序：DAC `dac/nn/quantize.py`（对照 VQ/RVQ）→ SoundStream 参考实现（补 EMA、死码替换、完整 GAN）→ EnCodec/AudioCraft（补 MS-STFT、balancer、因果卷积）→ AudioDec（补 streaming buffer）→ SpeechTokenizer/Mimi（补语义蒸馏与语言模型接口）。

教学版刻意不实现流式缓存、EMA 码本、语义教师、熵编码和论文规模训练。要继续时，先把想加的特性定位到数据流的某一站，再决定加哪个模块，而不是把所有技巧一次性堆进 `forward`。

## 原仓库与许可证

| 工作 | 仓库 | 许可证 |
|---|---|---|
| SoundStream（论文；无官方代码，常用参考实现 lucidrains/audiolm-pytorch 中的 `soundstream.py`） | [lucidrains/audiolm-pytorch](https://github.com/lucidrains/audiolm-pytorch) | MIT |
| EnCodec | [facebookresearch/encodec](https://github.com/facebookresearch/encodec) | MIT |
| AudioCraft（EnCodec 完整训练入口） | [facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft) | MIT |
| DAC | [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) | MIT |
| AudioDec | [facebookresearch/AudioDec](https://github.com/facebookresearch/AudioDec) | **CC BY-NC 4.0（非商业）**，部分组件 MIT |
| SpeechTokenizer | [ZhangXInFD/SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer) | Apache 2.0 |
| Mimi | [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | Apache 2.0 |
| WavTokenizer | [jishengpeng/WavTokenizer](https://github.com/jishengpeng/WavTokenizer) | MIT |
| HiFi-Codec（代码在 AcademiCodec 仓库内） | [yangdongchao/AcademiCodec](https://github.com/yangdongchao/AcademiCodec) | **未附许可证** |

## 使用原仓库代码时的合规要点

- **MIT / Apache 2.0**（SoundStream 参考实现、EnCodec、AudioCraft、DAC、SpeechTokenizer、Mimi、WavTokenizer）：可以自由使用、修改、再分发，包括商用；义务是保留原版权声明与许可证文本（Apache 2.0 额外要求声明修改、并附带专利授权条款）。若把其中的代码复制进本项目，需要在项目根目录 `NOTICE` 中署名。
- **AudioDec（CC BY-NC 4.0）**：**禁止商业用途**，且分发演绎作品时需署名。其 README 说明主体为 CC BY-NC，少量来自 ParallelWaveGAN / vector-quantize-pytorch / hifi-gan / wavenet_vocoder / SiFiGAN 的组件为 MIT。参考它的代码做学习笔记没问题；商用场景请绕开。
- **HiFi-Codec / AcademiCodec（无许可证）**：仓库没有 LICENSE 文件，按版权法默认"保留所有权利"——可以**阅读学习**，但**不要复制其代码**到任何会分发的项目里。
- 本教学实现只借鉴架构思想、未复制源码，因此不受上述许可证约束；本项目自身许可证见根目录 `LICENSE`（Apache 2.0）。
