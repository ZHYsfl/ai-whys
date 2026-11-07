# ASR 演进、现代实现与参考来源

本项目不是把某一篇旧论文缩小，而是把当前 ASR 的共同核心拆成一套可运行教学代码。

## 一、先分清三个维度

“ASR 架构是否现代”不能只看 encoder 的名字，要同时看：

1. **声学表示与 encoder**：log-Mel → Conformer/Fast Conformer/Zipformer/AuT，或 waveform → 自监督表示。
2. **对齐与输出**：CTC、RNN-T/TDT、attention encoder-decoder（AED）、Encoder-Adapter-LLM。
3. **训练范式**：小规模监督、自监督预训练 + CTC 微调、大规模弱监督、多阶段 SFT/RL。

因此，“Conformer 发表于 2020 年”不等于陈旧，“CTC 发表于 2006 年”也不等于只能配 BiLSTM。2024 年 SenseVoice-Small 仍用 encoder-only + CTC 做低延迟多语识别；Fast Conformer/Parakeet 继续提供 CTC、RNN-T、TDT 路线。真正需要避免的是把 BiGRU + CTC 当成 2026 年的完整技术图景。

## 二、经典工作分别增加了什么

| 工作 | 核心增量 | 与本教学版的关系 |
|---|---|---|
| GMM-HMM / DNN-HMM、Kaldi | 声学模型、发音词典、HMM 对齐、语言模型与 WFST 解码组成完整传统管线 | 用来理解端到端模型究竟替代了哪些组件，不实现这套重型管线 |
| [CTC (2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf) | blank、重复折叠、对所有合法单调对齐求和；无需帧级标注 | 训练目标与贪心解码的核心 |
| [RNN-T (2012)](https://arxiv.org/abs/1211.3711) | prediction network 显式依赖已经输出的非 blank token，解除 CTC 的输出独立假设 | 作为流式与语言依赖的下一站，不塞进最小版本 |
| [Towards End-to-End ASR (2014)](https://proceedings.mlr.press/v32/graves14.html) | deep BiLSTM + 字符 CTC，去掉发音词典与 forced alignment | 历史转折点，不沿用其 RNN 主干 |
| [Deep Speech (2014)](https://arxiv.org/abs/1412.5567)、[Deep Speech 2 (2015)](https://arxiv.org/abs/1512.02595) | 频谱 + 深度网络 + CTC 的清晰工程主干，变长 batch、下采样、LM beam search | 借鉴端到端数据流，不沿用旧 RNN 与外部 LM 依赖 |
| [Listen, Attend and Spell (2015)](https://arxiv.org/abs/1508.01211) | attention decoder 按 `P(y_i|x,y_<i)` 自回归输出，学习隐式语言模型 | 解释 AED 与 CTC 的根本差别 |
| [SpecAugment (2019)](https://arxiv.org/abs/1904.08779) | 直接在 log-Mel 上做时间/频率遮蔽 | 保留低成本、有效的 time/frequency mask；省略收益较小的 time warping |
| [Conformer (2020)](https://arxiv.org/abs/2005.08100) | attention 建模全局依赖，depthwise convolution 建模局部依赖，macaron FFN | 教学 encoder 的主体 |
| [wav2vec 2.0 (2020)](https://arxiv.org/abs/2006.11477) | 无标注语音上做 masked contrastive 预训练，接线性层后用 CTC 微调 | 说明 CTC 仍是现代表示学习的识别接口；不假装小数据能复现预训练收益 |
| [Whisper (2022)](https://arxiv.org/abs/2212.04356) | 680k 小时弱监督、多语多任务 AED，特殊 token 统一任务接口 | 说明数据规模/多样性与 decoder-only 生成路线 |
| [Fast Conformer (2023)](https://arxiv.org/abs/2305.05084) | 8× depthwise-separable 下采样、较小卷积核、limited-context attention；CTC 改 BPE 解决帧数约束 | 教学版采用小卷积核思想，但保留字符 + 4×，并显式检查 CTC 最小帧数 |
| [Zipformer (2023)](https://arxiv.org/abs/2310.11230) | 多帧率 U-Net 式 encoder、注意力权重复用、Bypass、BiasNorm、ScaledAdam | 作为高效流式 encoder 的进阶阅读；完整实现会掩盖 ASR 对齐主线 |
| [SenseVoice (2024)](https://arxiv.org/abs/2407.04051) | Small 用 encoder-only + CTC 做非自回归多语 ASR，并联合语言/情绪/音频事件标签 | 直接证明“现代最小 CTC”仍有现实位置；Large 则是 AED |
| [FireRedASR (2025)](https://arxiv.org/abs/2501.14350) | Conformer-AED 与 Encoder-Adapter-LLM 两路线；LLM 路线用 LoRA、只计算文本损失 | 对照 AuT/Qwen3 的语音到文本 embedding 接口和分阶段训练 |
| [Canary v2 / Parakeet v3 (2025)](https://arxiv.org/abs/2509.14128) | Fast Conformer 的多语 ASR/AST，CTC/TDT 等高效解码与标点/时间戳能力 | 说明当前生产系统的输出头与能力扩展 |
| [Qwen3-ASR (2026)](https://arxiv.org/abs/2601.21337) | 预训练 AuT encoder + projector + Qwen3 LM；动态 attention window；AuT/Omni/SFT/GSPO 多阶段训练 | 从零复现 AuT encoder、AED 预训练、projector、Qwen3-style LM 与 KV cache，不加载其权重 |

## 三、教学版具体取舍

两条路线同等地位，不把 LLM 当延伸阅读或黑盒。

**路线 A：Conformer-CTC**

- 80 维 log-Mel、utterance-level CMVN、可选 SpecAugment。
- 4× Conv2d 下采样，把 100 Hz 特征变成 25 Hz encoder 帧。
- RoPE multi-head self-attention、GLU + depthwise convolution、macaron FFN、pre-norm residual。
- 字符 CTC、forward DP、贪心与前缀 Beam Search 解码、corpus-level WER/CER。
- 所有变长张量都带真实长度；attention/convolution 显式 mask；检查 CTC 最小可行帧数。

**路线 B：AuT + Qwen3-style ASR**

- 128 维 log-Mel、三层 stride-2 Conv2d，得到 12.5 Hz 语音 token。
- AuT Transformer encoder + Transformer decoder，完整保留 AED 预训练路径。
- AuT encoder + projector + decoder-only Qwen3-style LM。
- RMSNorm、Q/K head RMSNorm、half-split RoPE、GQA、SwiGLU、causal mask。
- teacher forcing 的 next-token cross entropy，以及带 KV cache 的自回归生成。
- 全部随机初始化；可以检查每个参数、每一步数据流和每一段梯度，不调用预训练权重。

**刻意不堆入**

- RNN-T/TDT：需要 `[T,U]` 二维对齐网格、prediction/joint network 和额外 loss 实现。
- Zipformer 全部结构：多尺度帧率、NLA、Bypass 与 ScaledAdam 会增加首套代码的理解成本。
- BPE：Fast Conformer 的 8× + BPE 很合理，但教学版用字符/特殊 token 避免额外前置知识。
- 流式缓存：RoPE 本身不提供流式能力；没有 causal attention、卷积缓存和 chunk 状态就不能宣称 streaming。
- 大规模自监督/Omni/弱监督/RL：小数据无法验证其真正收益，不能用名字代替训练过程。

## 四、源码怎么读

建议顺序：

1. `deepspeech.pytorch/deepspeech_pytorch/model.py`、`loader/data_loader.py`、`decoder.py`：看早期 CTC 的 batch、长度和贪心解码。
2. `espnet/espnet2/asr/ctc.py`、`espnet_model.py`：看 CTC 如何接入统一训练框架，以及 CTC/attention 多任务学习。
3. `wenet/wenet/models/transformer/conformer/` 与 `models/transformer/attention.py`：看 Conformer、RoPE、流式 mask 和推理接口。
4. `nemo/nemo/collections/asr/parts/submodules/conformer_modules.py`、`ctc_greedy_decoding.py`、`transducer_decoding/`：看 Fast Conformer 时代的模块化实现及 CTC/RNN-T/TDT。
5. `icefall/egs/librispeech/ASR/zipformer/`：看多帧率 Zipformer、pruned transducer 与部署导向训练配方。
6. `funasr/funasr/models/sense_voice/model.py`：看现代 encoder-only + CTC 如何加入多任务标签。
7. `fireredasr/models/module/conformer_encoder.py`、`transformer_decoder.py`、`adapter.py`：看 Conformer-AED 与 Encoder-Adapter-LLM 的接口。
8. `whisper/whisper/audio.py`、`model.py`、`decoding.py`：看 log-Mel、AED、自回归解码与 KV cache。
9. `fairseq/fairseq/models/wav2vec/wav2vec2.py`、`wav2vec2_asr.py`：看自监督预训练到 CTC 微调。
10. `qwen3-asr/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py`：对照 AuT 三层卷积、projector、Qwen3 LM、动态窗口与多模态输入接口。
11. `qwen3-omni/`：继续看 AuT/Omni 预训练如何接到更大的多模态系统。
12. `kaldi/egs/librispeech/s5/run.sh`：回看传统 GMM/DNN-HMM 管线的完整阶段，理解端到端简化来自哪里。

本地对应路径均在 `asr-clone/repos/`。这些是研究用 shallow clone，不是本项目的依赖。

## 五、许可证与合规

以下以本次 clone 中实际许可证文件为准；模型权重、数据集和第三方组件可能有各自附加条款，发布模型前需另行检查：

| 仓库 | 许可证/注意事项 |
|---|---|
| SeanNaren/deepspeech.pytorch | 根目录 `LICENCE` 为 MIT；个别源文件保留 Apache-2.0 头，复制这些文件需同时遵守其声明 |
| espnet/espnet | Apache-2.0 |
| wenet-e2e/wenet | Apache-2.0 |
| NVIDIA/NeMo | Apache-2.0 |
| k2-fsa/icefall | Apache-2.0；各个 pretrained model 仍应单独核对 |
| FunAudioLLM/SenseVoice | MIT；模型权重/数据集条款另查 |
| modelscope/FunASR | MIT；另有 `MODEL_LICENSE` 与第三方组件条款 |
| FireRedTeam/FireRedASR | Apache-2.0 |
| openai/whisper | MIT |
| facebookresearch/fairseq | MIT |
| QwenLM/Qwen3-ASR、QwenLM/Qwen3-Omni | 代码仓库 Apache-2.0；模型权重与服务条款另查 |
| kaldi-asr/kaldi | `COPYING` 以 Apache-2.0 为主，并含各文件作者/版权说明 |

本项目只借鉴论文思想与数据流设计，未直接复制上述仓库源码。`LibriSpeech` 数据来自 [OpenSLR](https://www.openslr.org/12)，其原始数据与文本条款由数据发布方规定。
