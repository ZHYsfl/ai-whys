# Modern ASR：Conformer-CTC 与 AuT/Qwen3-ASR

一个以数据流、张量形状、对齐方式和训练目标为主线的现代 ASR 教学项目，对应笔记 [`../../audio-dl.md`](../../audio-dl.md) 的实验 41。两条路线地位同等，全部从零实现，**不加载任何预训练模型，不依赖 Transformers/PEFT，也不把 LLM 当黑盒**。

## 两条并列主线

### A. 非自回归：Conformer + CTC

```text
waveform [B,N]
 → log-Mel [B,F,80]
 → 4× Conv2d subsampling [B,T,64]
 → Conformer (RoPE attention + depthwise conv + macaron FFN)
 → CTC logits [B,T,29]
 → 对所有合法对齐求和（训练）
 → 折叠重复、删除 blank（推理）
```

### B. 现代 LLM ASR：AuT 预训练 → projector + Qwen3-style LM

```text
阶段 1：AuT encoder + Transformer decoder，做 AED 预训练
阶段 2：复用 AuT encoder，接 projector + decoder-only Qwen3-style LM

waveform [B,N]
 → log-Mel [B,F,128]
 → 三层 stride-2 Conv2d，8× 下采样 [B,T,64]（100 Hz → 12.5 Hz）
 → AuT Transformer encoder（正弦位置 + 局部窗口 attention）
 → projector [B,T,64]
 → Qwen3-style LM（RMSNorm、Q/K norm、RoPE、GQA、SwiGLU、causal mask）
 → next-token cross entropy（训练）
 → KV cache 自回归生成（推理）
```

“最小”指只保留两条现代路线的核心结构，不指退回最老模型。教学配置缩小了层数、隐藏维与词表，但没有删掉 AuT decoder、projector、GQA、Q/K norm 或 KV cache 来冒充 Qwen3-ASR。

## 环境

用 uv 管理（Python 3.12）。RTX 50 系 GPU 请使用 lock 中的 PyTorch/CUDA 组合：

```bash
cd code/asr
uv sync
uv run pytest                                 # 33 个测试
uv run python scripts/inspect_shapes.py       # Conformer-CTC 数据流与 CTC 交叉验证
uv run python scripts/inspect_llm_shapes.py   # AuT/projector/Qwen3/KV cache 数据流
```

## 路线 A：Conformer-CTC

### 1. 合成数据验证训练机制

```bash
uv run python scripts/train.py \
  --data synthetic --steps 200 --batch-size 4 \
  --output-dir runs/synthetic
```

### 2. 真实 LibriSpeech 训练与最佳模型选择

首次下载 `dev-clean` 约 323 MiB。已有 codec 归档时，下面命令可复用它；不加 `--archive` 则会自行下载并校验 MD5。

```bash
uv run python scripts/prepare_librispeech.py \
  --archive ../codec/data/downloads/dev-clean.tar.gz \
  --output-dir data/librispeech-full-dev \
  --max-files 2703 --eval-fraction 0.1
uv run python scripts/train.py \
  --data librispeech \
  --audio-dir data/librispeech-full-dev/train \
  --val-audio-dir data/librispeech-full-dev/val \
  --steps 8000 --batch-size 16 \
  --d-model 192 --num-heads 8 --num-layers 6 --ffn-dim 768 \
  --spec-augment --warmup-steps 200 \
  --log-every 250 --eval-every 250 --save-every 1000 \
  --output-dir runs/librispeech-full-dev
uv run python scripts/evaluate.py \
  --checkpoint runs/librispeech-full-dev/checkpoint-best.pt \
  --data librispeech --audio-dir data/librispeech-full-dev/val \
  --output runs/librispeech-full-dev/eval-best.json
```

需要从已有权重继续训练时，使用 `--init-checkpoint`。它只恢复模型权重，不恢复 optimizer；新任务应使用新的 `--output-dir` 和较小的学习率，例如：

```bash
uv run python scripts/train.py \
  --data librispeech \
  --audio-dir data/librispeech-full-dev/train \
  --val-audio-dir data/librispeech-full-dev/val \
  --init-checkpoint runs/librispeech-full-dev/checkpoint-last.pt \
  --steps 3000 --batch-size 16 \
  --d-model 192 --num-heads 8 --num-layers 6 --ffn-dim 768 \
  --spec-augment --learning-rate 2e-4 --warmup-steps 0 \
  --log-every 250 --eval-every 250 --save-every 1000 \
  --output-dir runs/librispeech-full-dev-resume
```

每 250 步同时在验证集上计算 loss/WER/CER，按验证 loss 保存 `checkpoint-best.pt`，避免把“背下训练集”误当成真实识别。该切分是确定性的 utterance-level 划分，**不是 speaker-independent**，仍不能用于论文式泛化结论。

### 3. 独立进程转录

```bash
uv run python scripts/transcribe.py \
  --checkpoint runs/librispeech-full-dev/checkpoint-best.pt \
  --input data/librispeech-full-dev/val/1272/135031/1272-135031-0022.flac \
  --output runs/librispeech-full-dev/transcript.json
```

## 路线 B：AuT + Qwen3-style ASR（从零复现）

### 1. AuT AED 预训练

```bash
uv run python scripts/train_aut.py \
  --data synthetic --steps 200 --batch-size 4 \
  --output-dir runs/aut-synthetic
```

AuT decoder 是标准 Transformer decoder：causal self-attention + cross-attention + FFN。训练时输入右移后的文本，计算 next-token cross entropy。

### 2. 迁移 AuT encoder，训练 projector + Qwen3-style LM

```bash
uv run python scripts/train_llm.py \
  --data synthetic --steps 200 --batch-size 4 \
  --aut-checkpoint runs/aut-synthetic/checkpoint.pt \
  --output-dir runs/qwen3-asr-synthetic
uv run python scripts/evaluate_llm.py \
  --checkpoint runs/qwen3-asr-synthetic/checkpoint-best.pt \
  --data synthetic --dataset-size 8 --batch-size 4 --max-new-tokens 32 \
  --output runs/qwen3-asr-synthetic/eval-best.json
```

这条路线用于学习现代 LLM ASR 的架构和多阶段数据流。随机初始化的小模型不承诺真实识别效果，也不复刻 Qwen3-ASR 的 AuT 大规模预训练、Omni 预训练、SFT 与 GSPO/RL 数据规模。

### 3. 独立生成

```bash
uv run python scripts/transcribe_llm.py \
  --checkpoint runs/qwen3-asr-synthetic/checkpoint-best.pt \
  --input data/librispeech-full-dev/val/1272/135031/1272-135031-0022.flac \
  --max-new-tokens 32 \
  --output runs/qwen3-asr-synthetic/transcript.json
```

## 架构对照

| 组件 | Conformer-CTC | AuT/Qwen3-style ASR |
|---|---|---|
| 输入特征 | 80 维 log-Mel | 128 维 log-Mel |
| 时间帧率 | 4× 下采样，25 Hz | 8× 下采样，12.5 Hz |
| 声学 encoder | RoPE Conformer | 正弦位置 AuT Transformer |
| 预训练接口 | 可选 CTC | AuT encoder + Transformer decoder（AED） |
| 文本模型 | 无 | 自己实现的 decoder-only Qwen3-style LM |
| 关键 LLM 细节 | 不适用 | RMSNorm、Q/K norm、half-split RoPE、GQA、SwiGLU |
| 训练对齐 | 对所有 CTC 路径求和 | next-token teacher forcing |
| 推理 | 所有帧并行，折叠路径 | 逐 token 生成，KV cache 复用历史 K/V |

## 代码地图

- `config.py`、`features.py` — 超参、Mel filterbank、STFT、CMVN、SpecAugment、长度/mask
- `conformer.py`、`model.py`、`ctc.py` — Conformer-CTC 主干、原生与教学版 CTC、贪心与默认前缀 Beam Search 解码
- `modern_config.py` — AuT 与 Qwen3-style LLM 的配置
- `aut.py` — 8× AuT encoder、可调局部窗口 attention（0 为全上下文，使用 SDPA 而非冒称 FlashAttention 内核）、AED decoder 与预训练损失
- `qwen3_lm.py` — RMSNorm、Q/K norm、RoPE、GQA、SwiGLU、causal attention、KV cache
- `qwen3_asr.py` — AuT encoder + projector + Qwen3-style LM，teacher forcing 与生成
- `llm_text.py` — 不依赖外部 tokenizer 的文本接口与 `language/asr_text` 特殊 token
- `data.py`、`metrics.py` — 可变长语音 batch、LibriSpeech、编辑距离与 WER/CER
- `training.py`、`evaluation.py`、`aut_training.py`、`llm_training.py` — 两条路线的训练与评估
- `checkpoint.py`、`llm_checkpoint.py` — 全部从零模型的存档与恢复

`REFERENCES.md` 记录从 CTC/RNN-T/LAS 到 Conformer、Fast Conformer、Zipformer、SenseVoice、FireRedASR、Qwen3-ASR 的演进、源码阅读顺序与许可证。

## 边界与诚实的实验结论

- 本次真实小数据实验先从零训练 5000 步，再只恢复模型权重续训 3000 步：验证 WER 从 65.96% 降至 63.74%，CER 从 26.39% 降至 24.93%。最终模型在 270 条验证音频中只有 1 条完全正确；独立转录把 `seventeen thirty fouor` 改善为正确的 `seventeen thirty four`，但最短样本仍是 `true agreed kaliko` → `tr a greed calickao`。这证明训练闭环有效，不代表已有实用识别能力。
- 续训验证 loss 最低的是第 500 步，但该 checkpoint 的 WER 为 65.57%；最终第 3000 步的 WER 更低、loss 反而更高。因此 `checkpoint-best.pt` 只表示“验证 loss 最低”，不能当作“识别错误率最低”。
- 合成集合上的 0% WER/CER 只说明机制能学会，不代表真实语音能力。
- 真实训练要同时看训练/验证 loss、WER/CER 和实际生成文本；不能只看 loss 下降。
- 本项目没有加载 Qwen 权重，现代路线只复现架构与训练/推理接口，不宣称达到 Qwen3-ASR 的能力。
- 暂未实现 RNN-T/TDT、流式卷积缓存、BPE、自监督预训练和大规模 Omni/RL 数据管线。
- `runs/` 与 `data/` 已被 Git 忽略。Checkpoint 含模型、tokenizer 与配置，不含 optimizer 和随机数状态，不承诺严格断点续训。
