# Neural Audio Codec

一个以数据流、张量形状和训练目标为主线的 neural audio codec 教学实现，对应笔记 [`../../audio-dl.md`](../../audio-dl.md) 的实验 40。只有这一套代码，不区分 low/high level 版本。

```text
waveform → encoder → latent → VQ/RVQ → quantized latent → decoder → reconstruction
                              ↓
                         codes [B, Q, F]
```

## 环境

用 uv 管理（Python 3.12）。RTX 50 系 GPU 请使用 lock 中的 PyTorch/CUDA 组合，不要绕过 lock 安装旧版：

```bash
cd code/codec
uv sync
uv run pytest                            # 23 个测试：形状、梯度、训练、CLI 契约
uv run python scripts/inspect_shapes.py  # 数据流形状与 RVQ 检查
```

## 第一步：纯重建训练（合成数据，验证机制）

```bash
uv run python scripts/train.py \
  --data synthetic --steps 200 --batch-size 4 \
  --output-dir runs/reconstruction
```

## 第二步：加入可变码率与 GAN（仍是合成数据，验证训练路径）

```bash
uv run python scripts/train.py \
  --data synthetic --steps 200 --quantizer-dropout --dropout-rate 0.5 \
  --output-dir runs/dropout
uv run python scripts/train.py \
  --data synthetic --steps 200 --use-gan \
  --output-dir runs/gan
```

## 第三步：LibriSpeech 真实语音训练与评估

评估同一验证集在训练前后的变化（首次下载约 323 MiB，之后复用归档，MD5 按 OpenSLR 官方清单校验）：

```bash
uv run python scripts/prepare_librispeech.py --max-files 64
uv run python scripts/train.py \
  --data wav --audio-dir data/librispeech-smoke/train \
  --steps 500 --output-dir runs/librispeech-reconstruction
uv run python scripts/evaluate.py \
  --checkpoint runs/librispeech-reconstruction/checkpoint-init.pt \
  --audio-dir data/librispeech-smoke/val \
  --output runs/librispeech-reconstruction/eval-init.json
uv run python scripts/evaluate.py \
  --checkpoint runs/librispeech-reconstruction/checkpoint.pt \
  --audio-dir data/librispeech-smoke/val \
  --output runs/librispeech-reconstruction/eval-final.json
```

## 第四步：WAV → codes → WAV（两个独立进程）

确认推理只依赖 codes 和 checkpoint：

```bash
uv run python scripts/encode.py \
  --checkpoint runs/reconstruction/checkpoint.pt \
  --input runs/reconstruction/original.wav \
  --output runs/reconstruction/codes.npz
uv run python scripts/decode.py \
  --checkpoint runs/reconstruction/checkpoint.pt \
  --codes runs/reconstruction/codes.npz \
  --output runs/reconstruction/decoded.wav
```

## 代码地图

`src/teaching_codec/`：

- `config.py` — 全部超参（采样率、通道数、RVQ 配置、损失权重）
- `autoencoder.py` — Encoder / Decoder（conv1d 下采样、convTranspose1d 上采样）
- `quantization.py` — 单级 VQ + 残差多级 RVQ（含 quantizer dropout）
- `codec.py` — TeachingCodec 总装：forward / encode / decode
- `losses.py` — 波形 L1 + 多尺度 STFT 重建损失
- `discriminator.py` — 多尺度判别器 + 对抗 / 特征匹配损失
- `training.py` — 判别器回合 / 生成器回合交替训练（梯度切换在这里）
- `audio_io.py`、`data.py`、`librispeech.py`、`checkpoint.py`、`runtime.py` — 读写、数据集、归档下载、存档、运行支撑

`scripts/`：`train.py`、`evaluate.py`、`encode.py`、`decode.py`、`prepare_librispeech.py`、`inspect_shapes.py`

`stft_tutorial.html` — STFT 从零到一交互教程（配套 `losses.py`；在本目录起 `python3 -m http.server` 后用浏览器打开）

`REFERENCES.md` — 各经典工作（SoundStream / EnCodec / DAC / AudioDec / SpeechTokenizer / Mimi / WavTokenizer / HiFi-Codec）的增量、源码阅读顺序，以及原仓库许可证与合规要点

## 备注

`runs/` 与 `data/` 已被 Git 忽略。Checkpoint 保存模型与判别器配置，但不含 optimizer 和随机数状态，因此不承诺严格断点续训。
