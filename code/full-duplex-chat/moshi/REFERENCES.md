# 参考来源、派生关系与许可证

## 核心论文与官方仓库

- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)：Mimi、RQ-Transformer、多流、Inner Monologue、延迟与训练目标。
- [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi)：本次参考的本地 clone 为 `e6a55d2722a65870ef52a6c9f6ecfc0e90f38362`。Python 代码 MIT；其中若干模块同时保留 Meta/AudioCraft 的 MIT 声明。
- [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)：本次参考的本地 clone 为 `2acc879fe7c48f885a18f6cc9548bccb2674d87b`，Apache-2.0。

本项目可以阅读官方源码并保留相同算法，但运行时**不导入这两个官方包**。

## 文件级派生说明

| 本项目文件 | 来源与修改 |
|---|---|
| `streaming.py`、`transformer.py` | Moshi/AudioCraft 的流式模块、Transformer、RoPE、KV Cache；移除量化、CUDA Graph、分布式执行 mask，增加多位置输入的逐位置等价路径 |
| `convolution.py`、`seanet.py` | Moshi/AudioCraft 的因果卷积、转置卷积与 SeaNet；移除生产兼容分支，保留算法结构 |
| `quantization.py` | Moshi/AudioCraft 的 VQ/RVQ、EMA 和 Split RVQ；保留码本初始化、死码替换、STE 和 dropout |
| `mimi.py` | Moshi 的 Mimi 总装；新增教学用语义蒸馏头、显式输出数据类与训练接口 |
| `model.py`、`inference.py`、`loaders.py` | Moshi 的 LM、LMGen 和加载流程；移除 conditioner、CFG、CUDA Graph 与非核心模型分支 |
| `data.py`、`losses.py` | moshi-finetune 的 interleaver、双声道格式与加权损失；改为本地 Dataset、显式 padding 和教学数据类 |
| `lora.py`、`checkpoint.py`、`training.py` | 官方 LoRA 与微调流程；去掉 FSDP、混合精度、W&B，增加独立 adapter/merged checkpoint |

这些文件的头部保留或标明来源。完整许可证位于：

- `LICENSES/MIT-moshi.txt`
- `LICENSES/MIT-audiocraft.txt`
- `LICENSES/Apache-2.0-moshi-finetune.txt`

`NOTICE` 汇总派生关系与主要修改。本项目自身及新增代码遵循 ai-whys 根目录的 Apache-2.0，见本项目 `LICENSE`。

## 复用 ai-whys 自有模块

`mimi_training.py` 和 `scripts/generate.py` 使用本项目已有的 `teaching_codec`：

- `MultiScaleDiscriminator`、对抗与 feature-matching loss；
- 波形和多尺度 STFT 重建损失；
- WAV 读写。

`pyproject.toml` 通过 `tool.uv.sources` 指向 `../../codec`，不是复制第三方实现。

## 权重、 tokenizer 与数据

代码许可证不自动覆盖模型权重、SentencePiece 模型或音频数据：

- Moshi 官方 README 标明模型权重使用 CC-BY 4.0；下载、再分发或商用前应读取对应 Hugging Face 仓库的实际条款。
- 双声道数据集、Whisper 标注、WavLM 教师与 DailyTalk/Fisher/LibriSpeech 等数据均有各自条款。
- 本项目不重新分发上述权重和数据。
