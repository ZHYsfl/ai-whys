# Neural Audio Codec

一个以数据流、张量形状和训练目标为主线的 neural audio codec 教学实现，对应笔记 [`../../audio-dl.md`](../../audio-dl.md) 的实验 40。只有这一套代码，不区分 low/high level 版本。

```text
waveform → encoder → latent → VQ/RVQ → quantized latent → decoder → reconstruction
                              ↓
                         codes [B, Q, F]
```

环境用 uv 管理（Python 3.12，RTX 50 系请使用 lock 中的 PyTorch，不要绕过 lock 装旧版）：

```bash
uv sync
uv run pytest
uv run python scripts/inspect_shapes.py
```

训练、真实语音评估、独立编码解码的完整命令见笔记的实验 40。
