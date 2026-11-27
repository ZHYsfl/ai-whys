# Teaching Moshi：Mimi + 全双工 RQ-Transformer

一个数据流优先的 Moshi 教学实现。保留 Moshi 的算法结构，缩小的是默认隐藏维、层数、词表和训练数据；删除的是 FSDP、CUDA Graph、分布式、多后端、服务端和部署量化等生产逻辑。

**不导入官方 `moshi` / `finetune` 包。** 模型代码在本项目内，可独立阅读；STFT、判别器和音频读写复用 `ai-whys` 的 `teaching_codec`。

## 架构主线

```text
用户波形 [B,1,1920]
  → Mimi: causal SeaNet → Transformer → 2× downsample → Split RVQ
  → 用户音频 tokens [B,8,1]

上一帧的 17 条流 = Moshi 文本 1 + Moshi 音频 8 + 用户音频 8
  → 每流独立 embedding，求和
  → Temporal Transformer（RoPE、RMSNorm、SwiGLU、causal attention）
  → Moshi 文本 token
  → Depth Transformer 逐码本生成 8 个 Moshi 音频 token
  → Mimi Decoder
  → Moshi 波形 [B,1,1920]
```

保留的算法细节：

- Mimi 的因果 SeaNet、瓶颈双 Transformer、额外 2× 重采样、语义 VQ 与声学 RVQ **并行量化后相加**。
- k-means 码本初始化、EMA、死码替换、STE、quantizer dropout、按序列跳过量化。
- 语义蒸馏头与 cosine loss；训练支持重建、GAN、feature matching 和语义损失。
- 17 流联合建模、每码本 Depth 参数、文本/语义/声学层级生成、acoustic delay。
- 整段 teacher forcing 与逐帧 Ring KV Cache 使用同一套模块。
- 文本 PAD/EPAD 对齐、文本 padding 降权、第一语义码本损失加权。
- 全量训练、全量微调、LoRA、adapter-only checkpoint 和 LoRA 合并。

`MoshiConfig.official()` 与 `MimiConfig.official()` 给出官方尺寸；默认配置只用于低成本教学。随机小模型和合成数据只能验证机制，不代表真实对话能力。

## 环境与检查

```bash
cd code/full-duplex-chat/moshi
uv sync
uv run pytest
uv run python scripts/inspect_shapes.py
```

`uv` 会通过 `../../codec` 以 editable 方式安装本项目已有的 `teaching_codec`。

## 训练

### 1. Mimi：RVQ + GAN + 语义蒸馏

```bash
uv run python scripts/train_mimi.py \
  --max-steps 200 --batch-size 4 --output-dir runs/mimi
```

默认采用论文最终 Mimi 的 adversarial-only 路线；`--with-reconstruction` 可加入波形与多尺度 STFT 重建损失，`--no-gan` 可单独研究非对抗路线。

合成数据使用合成教师向量，只验证蒸馏损失的计算与梯度。真实训练应由冻结的 WavLM 等非因果教师提供 `[B,semantic_dimension,T]` targets；教师不属于 Mimi 的模型结构。

### 2. Moshi：合成离散对话训练

```bash
uv run python scripts/train.py \
  --max-steps 200 --batch-size 4 --output-dir runs/moshi
```

合成数据直接生成完整的 17 流 tokens，因此可先学 RQ-Transformer，不必先训练好 Mimi。

### 3. 真实双声道数据

JSONL 每行：

```json
{"path": "dialogue.wav", "duration": 10.0}
```

每个 WAV 是同采样率的 **24 kHz 双声道**：左声道 Moshi，右声道用户。同名 JSON 提供主说话人词级对齐：

```json
{
  "alignments": [
    [" hello", [0.0, 0.4], "SPEAKER_MAIN"]
  ]
}
```

```bash
uv run python scripts/finetune.py \
  --base-checkpoint runs/moshi/checkpoint-last.pt \
  --manifest data/train.jsonl \
  --mimi-checkpoint runs/mimi/checkpoint-last.pt \
  --max-steps 2000 --learning-rate 2e-6 \
  --output-dir runs/moshi-lora
```

默认使用 LoRA 并只保存适配器。全量微调：

```bash
uv run python scripts/finetune.py \
  --base-checkpoint runs/moshi/checkpoint-last.pt \
  --full \
  --max-steps 2000 --learning-rate 2e-6 \
  --output-dir runs/moshi-full
```

真实官方模型使用 SentencePiece 时增加 `--tokenizer /path/to/tokenizer.model`。文本词表必须等于 `model.config.text_card`。

## 推理

使用教学 checkpoint：

```bash
uv run python scripts/generate.py \
  --mimi-checkpoint runs/mimi/checkpoint-last.pt \
  --model-checkpoint runs/moshi/checkpoint-last.pt \
  --input input.wav --output response.wav --greedy
```

也提供官方 safetensors / Hugging Face 加载入口：

```bash
uv run python scripts/generate.py \
  --official --hf-repository kyutai/moshiko-pytorch-bf16 \
  --device cuda --input input.wav --output response.wav
```

官方 7B 权重需要相应显存；本项目未以下载 7B 权重作为测试前提。相同小模型权重的交叉验证中，Moshi 参数键完全一致，有效 logits 最大差约 `3e-6`；Mimi 除新增蒸馏头外参数键一致，样例 codes 完全一致、解码波形差为 0。

## 源码阅读顺序

1. `src/teaching_moshi/config.py`：区分教学尺寸与官方尺寸。
2. `model.py`：先看 `MoshiLM.forward()`，再看 `forward_text()`、Depth 训练和 `MoshiGenerator.step()`。
3. `transformer.py`：RoPE、per-codebook 参数、SwiGLU、Ring KV Cache。
4. `mimi.py`：Mimi 总装和训练/编解码接口。
5. `seanet.py`、`convolution.py`：因果卷积和跨帧 overlap-add。
6. `quantization.py`：VQ/RVQ、EMA、Split RVQ。
7. `data.py`、`losses.py`：17 流数据、文本对齐、加权交叉熵。
8. `training.py`、`mimi_training.py`、`lora.py`：训练、对抗交替和微调。
9. `inference.py`：真正的 waveform → tokens → waveform 逐帧闭环。

## 边界与许可证

- 这是教学研究实现，不含实时麦克风/Web 服务、回声消除、FSDP、混合精度、CUDA Graph、MLX/Rust 和部署量化。
- 默认合成实验不证明模型获得语义、语音质量或对话能力；真实能力需要真实数据、教师模型和预训练权重。
- Moshi PyTorch、moshi-finetune 及 AudioCraft 的派生代码保留原版权与许可证。详见 `NOTICE`、`LICENSES/` 和 `REFERENCES.md`。
- 模型权重、SentencePiece 模型与数据集有独立条款，不能由代码许可证替代。
